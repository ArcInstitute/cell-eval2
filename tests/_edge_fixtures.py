"""Edge-case counts fixtures (#412): 6 knockdowns of their own genes G0..G5 over 60 genes, an
80-cell control and 60 cells per perturbation, plus one 12-perturbation scenario
(`twelve_perts`, G0..G11) so `de_*_nsig_spearman`'s gate of 10 perturbations is met. Shared by
the bounds test and, through internal:tools/metric_drift/, the release drift check -- change a
value and both move."""
import anndata as ad
import numpy as np
import pandas as pd

G, NC, NP = 60, 80, 60
GENES = [f"G{i}" for i in range(G)]
PERTS = [f"G{i}" for i in range(6)]
PERTS12 = [f"G{i}" for i in range(12)]
CONTROL = "non-targeting"
_BASE = np.random.default_rng(0).gamma(2.0, 10.0, size=G) + 1.0


def _effects(seed, perts=PERTS, sizes=None):
    # Effect genes are drawn from outside the knockdown targets; `sizes` (one per perturbation)
    # sets each one's number of responding genes. The default reproduces the 6-perturbation draw.
    r = np.random.default_rng(seed)
    eff = {}
    for j, p in enumerate(perts):
        f = np.ones(G)
        n = 20 if sizes is None else sizes[j]
        idx = r.choice(np.arange(len(perts), G), n, replace=False)
        f[idx[:n // 2]], f[idx[n // 2:]] = 4.0, 0.25
        f[GENES.index(p)] = 0.1
        eff[p] = f
    return eff


def _make(eff, seed, ctrl, override=None, perts=PERTS):
    r = np.random.default_rng(seed)
    blocks, labels = [ctrl], [CONTROL] * ctrl.shape[0]
    for p in perts:
        X = override[p] if override and p in override else r.poisson(_BASE * eff[p], size=(NP, G))
        blocks.append(X)
        labels += [p] * X.shape[0]
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(X.shape[0])])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES))


_CTRL = np.random.default_rng(1).poisson(_BASE, size=(NC, G))
_EFF_R, _EFF_P = _effects(10), _effects(20)
_NOEFF = {p: np.ones(G) for p in PERTS}
# 4, 6, ..., 26 responding genes, so the real significant-gene counts vary and the count
# correlation is defined; the prediction takes the sizes in a fixed shuffled order, so it is
# neither perfect nor anti.
_SIZES12 = [4 + 2 * j for j in range(12)]
_EFF12_R = _effects(30, PERTS12, _SIZES12)
_EFF12_P = _effects(40, PERTS12, list(np.random.default_rng(41).permutation(_SIZES12)))


def _pred(mutate=None, eff=None, override=None):
    a = _make(_EFF_P if eff is None else eff, 3, _CTRL.copy(), override)
    if mutate:
        mutate(a.X)
    return a


def _zero_gene(X):    # G1's cells lose gene G30; the control expresses it
    X[np.arange(NC + NP, NC + 2 * NP), GENES.index("G30")] = 0.0


def _zero_cell(X):
    X[NC + 2, :] = 0.0


def _zero_pert(X):    # every G2 cell all-zero
    X[NC + 2 * NP: NC + 3 * NP, :] = 0.0


SCENARIOS = ("perfect", "anti", "random", "no_effect", "copies_control", "real_zero_delta",
             "zero_gene", "zero_gene_eps0", "zero_cell", "zero_pert", "twelve_perts")


def scenario(name):
    """`(pred, real, de_overrides)` for one scenario; fresh AnnData objects on every call."""
    real = _make(_EFF_R, 2, _CTRL)
    table = {
        "perfect": lambda: (real.copy(), real, {}),
        "anti": lambda: (_pred(eff={p: 1.0 / f for p, f in _EFF_R.items()}), real, {}),
        "random": lambda: (_pred(), real, {}),
        "no_effect": lambda: (_pred(eff=_NOEFF), real, {}),
        "copies_control": lambda: (_pred(override={p: _CTRL[:NP].copy() for p in PERTS}), real, {}),
        "real_zero_delta": lambda: (_pred(), _make(_EFF_R, 2, _CTRL, {"G0": _CTRL[:NP].copy()}), {}),
        "zero_gene": lambda: (_pred(_zero_gene), real, {}),
        "zero_gene_eps0": lambda: (_pred(_zero_gene), real, {"epsilon": 0.0}),
        "zero_cell": lambda: (_pred(_zero_cell), real, {}),
        "zero_pert": lambda: (_pred(_zero_pert), real, {}),
        "twelve_perts": lambda: (_make(_EFF12_P, 5, _CTRL.copy(), perts=PERTS12),
                                 _make(_EFF12_R, 4, _CTRL, perts=PERTS12), {}),
    }
    return table[name]()


def config(profile, de_overrides=None):
    """CPU/pdex config for `profile`; `vcc2026` is the preset, so it carries the whole rule."""
    from dataclasses import replace
    from cell_eval2.config import EvalConfig
    base = EvalConfig.from_preset("vcc2026") if profile == "vcc2026" else EvalConfig(metrics=profile)
    return replace(base, pert_col="target", control=CONTROL, device="cpu", num_threads=4,
                   de=replace(base.de, backend="pdex", **(de_overrides or {})))
