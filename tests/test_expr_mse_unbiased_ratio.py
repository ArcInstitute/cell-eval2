# tests/test_expr_mse_unbiased_ratio.py
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys
import types

import numpy as np
import polars as pl
import pytest

from cell_eval2.metrics.delta import distance_unbiased, mse_unbiased, mse_unbiased_capped
from cell_eval2.moments import GroupMoments

CONTROL = "ctrl"


@pytest.fixture(autouse=True)
def _register_derived_if_absent(monkeypatch):
    """Make the aggregation test below runnable BEFORE Task 6 registers the real entries.

    `test_a_submission_omitting_a_perturbation_is_CAUGHT_not_absorbed` goes through
    `run.aggregate_metrics`, which consults `_derived_value` only for a metric the CATALOG
    declares derived. Before Task 6 there is no such entry, so the omission this test exists
    to catch would pass silently -- a test that cannot fail. Same mechanism as
    `tests/test_metric_aggregation.py`'s fixture; once Task 6 lands the catalog already
    carries the entry and this returns immediately.
    """
    from cell_eval2.catalog import CATALOG, DerivedAgg, MetricSpec
    from cell_eval2.scoring import DIAG, Scoring

    num, den = "expr_mse_unbiased_capped", "expr_distance_unbiased"
    derived = "expr_mse_unbiased_capped_norm"
    if derived in CATALOG:
        return
    patched = dict(CATALOG)
    for name in (num, den):
        patched[name] = MetricSpec(
            name=name, func=lambda **_: {}, scoring=DIAG, agg="mean", profiles=("full",),
            kind="anndata", normalization="lognorm", needs_moments=True)
    patched[derived] = MetricSpec(
        name=derived, func=None, agg="ratio_of_sums",
        derived=DerivedAgg(numerator=num, denominator=den),
        scoring=Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                        clamp_low=None, clamp_high=1.0),
        profiles=("full",), kind="anndata", normalization="lognorm")
    monkeypatch.setattr("cell_eval2.run.CATALOG", patched)


def _panel(seed=0, n_genes=64, n_cells=200):
    """Two-perturbation panel with known moments. Returns the kwargs both metrics take."""
    rng = np.random.default_rng(seed)
    perts = np.array(["p1", "p2", CONTROL])
    real_means = rng.normal(3.0, 1.0, size=(3, n_genes))
    pred_means = real_means + rng.normal(0.0, 0.2, size=(3, n_genes))
    counts = np.array([n_cells, n_cells, 5 * n_cells], dtype=float)
    # sumsq chosen so tr Sigma-hat is positive and of a realistic size
    sq = np.einsum("ij,ij->i", real_means, real_means)
    real_sumsq = counts * sq + counts * n_genes * 0.5
    pred_sumsq = counts * np.einsum("ij,ij->i", pred_means, pred_means) + counts * n_genes * 0.3
    return dict(
        pred_bulk=(perts, pred_means), real_bulk=(perts, real_means),
        pred_moments=GroupMoments(perts=perts, counts=counts, sumsq=pred_sumsq),
        real_moments=GroupMoments(perts=perts, counts=counts, sumsq=real_sumsq),
        control=CONTROL,
        comparator="lognorm",
    )


def test_the_three_metrics_drop_the_control_row():
    kw = _panel()
    for fn in (mse_unbiased, mse_unbiased_capped, distance_unbiased):
        out = fn(**kw)
        assert CONTROL not in out, f"{fn.__name__} kept the control row"
        assert sorted(out) == ["p1", "p2"]


def test_distance_unbiased_is_structurally_independent_of_the_prediction():
    """It is a property of the REFERENCE, which is what makes it cacheable per panel.

    Varying predicted VALUES alone is NOT enough to test this -- an implementation that takes
    its row set and its moments from the pred side passes that and is still broken (codex,
    checkpoint 1). Vary the LABEL SET, the label ORDER, and the moments, and drop the pred
    side entirely.
    """
    base = _panel(seed=1)
    expected = distance_unbiased(**base)
    assert expected != {}, "empty result would make every assertion below vacuous"

    perts, means = base["pred_bulk"]
    variants = {
        "different values": {"pred_bulk": (perts, means * 7.0 + 3.0)},
        "a MISSING perturbation": {"pred_bulk": (perts[1:], means[1:])},
        "renamed labels": {"pred_bulk": (np.array(["zz", "aa", CONTROL]), means)},
        "reordered labels": {"pred_bulk": (perts[::-1], means[::-1])},
        "different pred moments": {
            "pred_moments": GroupMoments(perts=perts,
                                         counts=base["pred_moments"].counts * 3.0,
                                         sumsq=base["pred_moments"].sumsq * 11.0)},
        "NO pred side at all": {"pred_bulk": None, "pred_moments": None},
    }
    for label, override in variants.items():
        got = distance_unbiased(**{**base, **override})
        assert got == expected, f"expr_distance_unbiased changed under {label}: {got} != {expected}"


def test_a_submission_omitting_a_perturbation_is_CAUGHT_not_absorbed():
    """The shard-streaming backstop, at the level where it is decidable.

    `scale.py:116-123` validates only the gene axis, so a shard-streamed submission can omit a
    real perturbation. The numerator then omits it and the denominator does not -- and
    `_derived_value` raises on the mismatch. Scoping the DENOMINATOR to the predicted labels
    instead (an earlier draft) would make both sides omit it, leaving two equal label sets and
    nothing to detect: the submission would silently choose its own cohort (codex round 3).
    """
    from cell_eval2.run import aggregate_metrics

    kw = _panel(seed=1)
    den = distance_unbiased(**kw)
    perts, means = kw["pred_bulk"]
    kw["pred_bulk"] = (perts[1:], means[1:])              # a submission omitting p1
    num = mse_unbiased_capped(**kw)
    assert set(num) < set(den), (
        f"the omission did not shrink the numerator ({sorted(num)} vs {sorted(den)}); this "
        "test cannot demonstrate anything"
    )
    rows = ([(p, "expr_mse_unbiased_capped", v) for p, v in num.items()]
            + [(p, "expr_distance_unbiased", v) for p, v in den.items()])
    frame = pl.DataFrame(rows, schema={"perturbation": pl.String, "metric": pl.String,
                                       "value": pl.Float64}, orient="row")
    with pytest.raises(ValueError, match="cover different perturbations"):
        aggregate_metrics(frame)


def test_the_capped_and_uncapped_numerators_differ_when_the_cap_bites():
    # Force the prediction's term far above the real side's so the cap is the only thing
    # that can separate them.
    kw = _panel(seed=2)
    perts, counts, sumsq = (kw["pred_moments"].perts, kw["pred_moments"].counts,
                            kw["pred_moments"].sumsq)
    kw["pred_moments"] = GroupMoments(perts=perts, counts=counts, sumsq=sumsq * 50.0)
    plain, capped = mse_unbiased(**kw), mse_unbiased_capped(**kw)
    assert plain != capped, (
        f"cap did not bite: plain={plain}, capped={capped} -- the fixture no longer "
        "exercises the branch and this test cannot fail"
    )
    # Capping SUBTRACTS LESS, so the capped value is the larger one.
    for p in plain:
        assert capped[p] > plain[p], f"{p}: capped {capped[p]} !> uncapped {plain[p]}"


def test_distance_unbiased_returns_a_negative_value_without_raising():
    # A perturbation indistinguishable from control at this depth is a NORMAL outcome, not a
    # reference defect. The old `denom <= 0` raise is what this replaces.
    kw = _panel(seed=3)
    perts, means = kw["real_bulk"]
    means = means.copy()
    means[0] = means[2]          # p1's mean IS the control's -> only sampling noise remains
    kw["real_bulk"] = (perts, means)
    gm = kw["real_moments"]
    kw["real_moments"] = GroupMoments(
        perts=gm.perts, counts=gm.counts,
        sumsq=gm.counts * np.einsum("ij,ij->i", means, means) + gm.counts * means.shape[1] * 0.5,
    )
    out = distance_unbiased(**kw)
    assert out["p1"] < 0.0, f"expected a negative value for a null perturbation, got {out['p1']}"


@pytest.mark.parametrize("fn", [mse_unbiased, mse_unbiased_capped, distance_unbiased],
                         ids=lambda f: f.__name__)
def test_every_metric_is_gene_averaged(fn):
    """DUPLICATE the genes rather than redrawing a wider panel.

    Two independent random panels of 64 and 128 genes give a ratio that a loose bound like
    `0.5 < r < 2.0` accepts even from a SUMMED implementation (codex, checkpoint 1). Exact
    duplication doubles every summed quantity and leaves every gene-averaged one identical,
    so the correct answer is exactly 1.0 and a summed one is exactly 2.0.
    """
    base = _panel(seed=4, n_genes=64)

    def doubled(bulk):
        perts, means = bulk
        return perts, np.concatenate([means, means], axis=1)

    wide = dict(base)
    wide["pred_bulk"], wide["real_bulk"] = doubled(base["pred_bulk"]), doubled(base["real_bulk"])
    for side in ("pred_moments", "real_moments"):
        gm = base[side]
        wide[side] = GroupMoments(perts=gm.perts, counts=gm.counts, sumsq=gm.sumsq * 2.0)

    small, big = fn(**base), fn(**wide)
    assert small and set(small) == set(big), "empty or mismatched results assert nothing"
    for p in small:
        assert small[p] != 0.0, f"{p} is exactly 0; the ratio below would be undefined"
        ratio = big[p] / small[p]
        assert ratio == pytest.approx(1.0, abs=1e-9), (
            f"{fn.__name__}[{p}] is not gene-averaged: ratio {ratio} "
            "(a summed implementation gives exactly 2.0)"
        )


def test_moments_are_required_and_never_silently_skipped():
    kw = _panel()
    kw["real_moments"] = None
    for fn in (mse_unbiased, mse_unbiased_capped, distance_unbiased):
        with pytest.raises(ValueError, match="requires per-group moments"):
            fn(**kw)


def test_the_missing_moments_error_names_the_driver_on_every_metric():
    # All three are dispatched with a `driver=`, and the error is what tells an operator WHICH
    # driver failed to route moments. distance_unbiased dropped it on the floor (Copilot, #262).
    kw = _panel()
    kw["real_moments"] = None
    for fn in (mse_unbiased, mse_unbiased_capped, distance_unbiased):
        with pytest.raises(ValueError, match="score_streaming_cell") as excinfo:
            fn(**kw, driver="score_streaming_cell (cell-stream)")
        assert "unknown" not in str(excinfo.value), (
            f"{fn.__name__} reported the driver as unknown while one was supplied: "
            f"{excinfo.value}"
        )


def test_distance_unbiased_raises_when_the_control_is_absent():
    kw = _panel()
    kw["control"] = "not-a-group"
    with pytest.raises(ValueError, match="control"):
        distance_unbiased(**kw)


#: The revision this characterization reproduces: the last one before the #247 cap
#: (`delta.py:176`). `810215c~1` is commit 6a40e433841fb05261cc786a4d7dcbf0c8884d2b in the
#: `ArcInstitute/cell_eval2` archive.
PRE_247 = "810215c~1"
PRE_247_COMMIT = "6a40e433841fb05261cc786a4d7dcbf0c8884d2b"

#: The two files as they stood there are VENDORED under `tests/fixtures/pre_247/` and read from
#: disk. They used to be recovered with `git show`, which tied the test to one object in one
#: `.git`: it is absent from a tree copy, from an sdist, and from any repository but the archive
#: -- and the old guard RAISED rather than skipped in a full clone that could not resolve it.
#: See that directory's README.md.
PRE_247_FIXTURES = Path(__file__).parent / "fixtures" / "pre_247"
PRE_247_SOURCES = {"moments.py": "src/cell_eval2/moments.py",
                   "delta.py": "src/cell_eval2/metrics/delta.py"}

#: SHA-256 of each vendored file. This is a CHARACTERIZATION baseline, so a drifted fixture is
#: the one failure that does not announce itself: the test would still pass while comparing
#: against something other than the pre-#247 metric. Verified on every run for that reason.
PRE_247_SHA256 = {
    "moments.py": "3c55e97a98024ffa59facc0c5e000a0c7464297974acc6672fc19af82709cf91",
    "delta.py": "cf2c10b89eef43c52a1f166dc0c98c6b0d2fe0d6d05b72ab223acd07b1412e23",
}


def _read_pre_247(name):
    """The vendored pre-#247 source for `name`, verified byte-for-byte. Never skips: the file
    is tracked in this repository, so a missing or edited one is a defect, not an environment."""
    path = PRE_247_FIXTURES / name
    # One runnable line: the explanation is a shell comment, so the hint can be pasted as it is.
    restore = (f"# from a clone of the ArcInstitute/cell_eval2 archive:\n"
               f"    git show {PRE_247_COMMIT}:{PRE_247_SOURCES[name]} > {path}")
    assert path.is_file(), (
        f"the pre-#247 characterization fixture {path} is missing; it is vendored, not "
        f"generated. Restore it verbatim:\n    {restore}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == PRE_247_SHA256[name], (
        f"{path} is not the file it characterizes: sha256 {digest}, expected "
        f"{PRE_247_SHA256[name]}. The fixture is a frozen historical source and must not be "
        f"edited, reformatted or re-linted. Restore it verbatim:\n    {restore}\n"
        "If you are deliberately re-basing the characterization onto a different revision, "
        "change PRE_247_COMMIT and this digest in the same commit and say why.")
    return raw.decode("utf-8")


def _historical_mse_unbiased(tmp_path):
    """Load `delta.mse_unbiased` as it stood at PRE_247, with its relative imports satisfied."""
    mpath = tmp_path / "moments_old.py"
    # encoding pinned: the vendored source carries non-ASCII mathematics (Σ, Σ̂), which
    # `write_text` would otherwise hand to the locale codec.
    mpath.write_text(_read_pre_247("moments.py"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("moments_old", mpath)
    mo = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the module carries `from __future__ import annotations` and a
    # dataclass, and dataclass machinery can resolve annotations through `sys.modules[name]`.
    # A scratchpad dry-run of this loader succeeded without it on this interpreter, so it is
    # defensive rather than a fix for an observed failure (codex, checkpoint 1).
    sys.modules[spec.name] = mo
    try:
        spec.loader.exec_module(mo)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    src = re.sub(r"^from \.\.?[\w.]+ import .*$", "", _read_pre_247("delta.py"), flags=re.M)
    import anndata as ad
    import numpy as np
    old = types.ModuleType("delta_pre247")
    old.__dict__.update({"trace_over_n_for": mo.trace_over_n_for,
                         "unbiased_sq_dist": mo.unbiased_sq_dist, "np": np, "ad": ad})
    exec(compile(src, "delta_pre247.py", "exec"), old.__dict__)
    return old.mse_unbiased, mo


def test_expr_mse_unbiased_reproduces_the_pre_247_metric_exactly(tmp_path):
    """CHARACTERIZATION, not a gate (spec §6). If this ever diverges, the divergence is
    measured and documented -- in the docstring and the release notes -- and raised with
    Alex. The name does NOT change: we do not carry two versions of this metric."""
    import numpy as np

    from cell_eval2.metrics.delta import mse_unbiased
    from cell_eval2.moments import GroupMoments

    historical, mo = _historical_mse_unbiased(tmp_path)
    rng = np.random.default_rng(11)
    n, g = 40, 96
    perts = np.array([f"p{i}" for i in range(n)] + ["ctrl"])
    real_means = rng.normal(3.0, 1.0, size=(n + 1, g))
    pred_means = real_means + rng.normal(0.0, 0.3, size=(n + 1, g))
    # Independent pred moments, including n < 2 groups, so #219's zero-fallback branch and a
    # wide pred_tn/real_tn range are both exercised rather than assumed.
    #
    # ⚠️ Counts must be WHOLE NUMBERS >= 1 since #227: `trace_over_n_for` now rejects anything
    # else, because a fractional or zero cell count means the moments disagree with the
    # pseudobulk they were built from. This fixture used `real_counts * uniform(0.05, 3.0)`
    # (fractional) with a literal `0.0` in the third slot, and both were incidental to what it
    # tests -- the wide pred_tn/real_tn range survives `rint`, and `n < 2` is still exercised by
    # `1.0`, which is the case #219 is actually about. `0.0` was never the #219 case (an empty
    # group has no sample mean either) and is exactly what #227 closed.
    real_counts = rng.integers(50, 900, size=n + 1).astype(float)
    pred_counts = np.maximum(1.0, np.rint(real_counts * rng.uniform(0.05, 3.0, size=n + 1)))
    pred_counts[:3] = [1.0, 1.0, 1.0]
    assert (pred_counts >= 1).all() and (pred_counts == np.rint(pred_counts)).all()
    assert pred_counts.max() / pred_counts.min() > 100, "the wide-range intent must survive rint"
    real_sumsq = real_counts * np.einsum("ij,ij->i", real_means, real_means) + real_counts * g
    pred_sumsq = pred_counts * np.einsum("ij,ij->i", pred_means, pred_means) + pred_counts * g

    kw = dict(pred_bulk=(perts, pred_means), real_bulk=(perts, real_means), control="ctrl")
    hist = historical(
        pred_moments=mo.GroupMoments(perts=perts, counts=pred_counts, sumsq=pred_sumsq),
        real_moments=mo.GroupMoments(perts=perts, counts=real_counts, sumsq=real_sumsq), **kw)
    now = mse_unbiased(
        pred_moments=GroupMoments(perts=perts, counts=pred_counts, sumsq=pred_sumsq),
        real_moments=GroupMoments(perts=perts, counts=real_counts, sumsq=real_sumsq),
        comparator="lognorm", **kw)

    assert set(hist) == set(now) and hist, "no overlap would make the comparison vacuous"
    spread = max(hist.values()) - min(hist.values())
    assert spread > 0, f"all historical values identical ({spread}); fixture cannot discriminate"
    diffs = {k: (hist[k], now[k]) for k in hist if hist[k] != now[k]}
    assert diffs == {}, (
        f"expr_mse_unbiased diverged from {PRE_247} on {len(diffs)}/{len(hist)} "
        f"perturbations: {dict(list(diffs.items())[:5])}. Per spec §6 the NAME DOES NOT "
        "CHANGE -- measure the divergence, document it in the docstring and the release "
        "notes, and raise it with Alex."
    )


N_PERT, N_GENES, GENE_SD, EFFECT_SD = 200, 200, 4.4, 0.35


def _anchor_panel(n_cells, seed=0):
    """A panel whose OBSERVED means carry sampling noise consistent with their moments.

    ⚠️ Getting this wrong is how the anchor tests become untestable. A first draft used
    noise-free means, so `D_p` carried no sampling inflation, the OLD definition scored
    0.955-0.995, and every test below passed under the very definition they exist to reject.
    The means must be DRAWN: `mu_hat = mu_true + N(0, sigma^2/n)`, with `sumsq` built from the
    same `sigma` so `trace_sigma` recovers it.

    Returns `(perts, true, obs, counts, real_moments)`.
    """
    rng = np.random.default_rng(seed)
    perts = np.array([f"p{i}" for i in range(N_PERT)] + ["ctrl"])
    ctrl_mu = rng.normal(3.0, 1.0, size=N_GENES)
    true = np.vstack([ctrl_mu + rng.normal(0.0, EFFECT_SD, size=N_GENES)
                      for _ in range(N_PERT)] + [ctrl_mu])
    counts = np.full(N_PERT + 1, float(n_cells))
    counts[-1] = 10.0 * n_cells                     # a deep control pool, as on real panels
    obs = true + rng.normal(0.0, 1.0, size=true.shape) * (GENE_SD / np.sqrt(counts))[:, None]
    sumsq = counts * np.einsum("ij,ij->i", obs, obs) + (counts - 1.0) * N_GENES * GENE_SD**2
    return perts, true, obs, counts, GroupMoments(perts=perts, counts=counts, sumsq=sumsq)


def _moments_for(perts, means, counts):
    sumsq = counts * np.einsum("ij,ij->i", means, means) + (counts - 1.0) * N_GENES * GENE_SD**2
    return GroupMoments(perts=perts, counts=counts, sumsq=sumsq)


def _aggregate(perts, pred_means, pred_counts, obs, real_m):
    kw = dict(pred_bulk=(perts, pred_means), real_bulk=(perts, obs),
              pred_moments=_moments_for(perts, pred_means, pred_counts),
              real_moments=real_m, control="ctrl", comparator="lognorm")
    num, den = mse_unbiased_capped(**kw), distance_unbiased(**kw)
    return sum(num.values()) / sum(den.values())


def _no_skill_arm(perts, true, counts, seed):
    """An independent draw from the CONTROL population, emitted at the control's depth."""
    rng = np.random.default_rng(seed)
    means = np.broadcast_to(true[-1], true.shape) + rng.normal(
        0.0, GENE_SD / np.sqrt(counts[-1]), size=true.shape)
    return means, np.full(len(perts), counts[-1])


def _replicate_arm(perts, true, counts, seed):
    """An independent draw from each perturbation's OWN population -- NOT the real sample.

    ⚠️ Reusing the real sample here is a PASTE, whose numerator is `-2 trS/n` and which
    correctly reads NEGATIVE. A first draft did exactly that and the test failed at -0.084.
    """
    rng = np.random.default_rng(seed)
    return true + rng.normal(0.0, GENE_SD / np.sqrt(counts[0]), size=true.shape), counts.copy()


@pytest.mark.parametrize("n_cells", [200, 500, 2000])
def test_the_no_skill_anchor_is_1_at_every_depth(n_cells):
    # THE POINT OF THE WHOLE CHANGE. Measured on this fixture: 1.0176 / 1.0084 / 1.0031.
    perts, true, obs, counts, real_m = _anchor_panel(n_cells, seed=5)
    means, pcounts = _no_skill_arm(perts, true, counts, seed=105)
    got = _aggregate(perts, means, pcounts, obs, real_m)
    assert got == pytest.approx(1.0, abs=0.05), f"no-skill anchor {got} at n={n_cells}"


def _no_skill_arm_at_depth(perts, true, n_pred, seed):
    """A control-emitting submission whose DRAWN noise matches its DECLARED depth.

    `_no_skill_arm` always emits at the control pool's depth. Declaring a thin `n_pred` while
    keeping those deep-pool means would be an incoherent submission -- it subtracts a
    correction its means do not carry -- and reads BELOW 1 for that reason rather than for the
    reason under test.
    """
    rng = np.random.default_rng(seed)
    means = np.broadcast_to(true[-1], true.shape) + rng.normal(
        0.0, GENE_SD / np.sqrt(n_pred), size=true.shape)
    return means, np.full(len(perts), float(n_pred))


@pytest.mark.parametrize("n_cells", [200, 500, 2000])
def test_a_no_skill_submission_THINNER_than_the_reference_reads_ABOVE_1(n_cells):
    """The anchor is 1.0 whatever the REFERENCE's depth; it is not invariant to the
    SUBMISSION's, and must not be described as if it were.

    E[capped numerator] = effect^2 + max(0, C_pred - k*C_real,p) against a denominator of
    effect^2, so a submission emitting thinner than the reference reads above 1 -- it is
    claiming a sampling correction larger than the reference earns, which is exactly what
    `PRED_TRACE_CAP_K` refuses (#247). Scoring below 0 there is the intended consequence, not a
    broken anchor: the same submission emitted at or above the reference's depth reads 1.0.

    Measured on this fixture (n_pred = n_real / 10): 8.11 at n_real=200, 3.79 at 500, 1.68 at
    2000. Raised by the checkpoint-2 review of #257.
    """
    perts, true, obs, counts, real_m = _anchor_panel(n_cells, seed=5)
    at_depth = _aggregate(perts, *_no_skill_arm_at_depth(perts, true, counts[-1], seed=105),
                          obs, real_m)
    thin = _aggregate(perts, *_no_skill_arm_at_depth(perts, true, counts[0] / 10.0, seed=105),
                      obs, real_m)
    assert at_depth == pytest.approx(1.0, abs=0.05), (
        f"the reference-depth arm must still anchor at 1.0, got {at_depth}"
    )
    assert thin > 1.5, (
        f"a 10x-thin no-skill submission read {thin}, i.e. the cap did NOT refuse its oversized "
        "correction; that is the #247 lever reopening"
    )


def _no_skill_arm_more_dispersed(perts, true, n_pred, sd_mult, seed):
    """A no-skill submission at the SAME cell count but with a wider per-cell spread."""
    rng = np.random.default_rng(seed)
    sd = GENE_SD * sd_mult
    means = np.broadcast_to(true[-1], true.shape) + rng.normal(
        0.0, sd / np.sqrt(n_pred), size=true.shape)
    counts = np.full(len(perts), float(n_pred))
    sumsq = counts * np.einsum("ij,ij->i", means, means) + (counts - 1.0) * N_GENES * sd**2
    return means, counts, GroupMoments(perts=perts, counts=counts, sumsq=sumsq)


@pytest.mark.parametrize("n_cells", [200, 500, 2000])
def test_a_no_skill_submission_MORE_DISPERSED_at_the_SAME_depth_also_reads_ABOVE_1(n_cells):
    """The cap's condition is on the CORRECTION `tr Sigma/n`, not on the cell count.

    `..._THINNER_than_the_reference_...` covers the n_pred half. This covers the other one: at
    an IDENTICAL cell count, a wider per-cell spread makes `C_pred > k C_real,p` just as
    thinning does, and the value goes above 1 for the same reason. Both halves are needed --
    a description that says only "thinner" is wrong, and was, until the checkpoint-2 review.

    Measured at 2x dispersion: 3.34 / 1.91 / 1.22 at n_real = 200 / 500 / 2000.
    """
    perts, true, obs, counts, real_m = _anchor_panel(n_cells, seed=5)
    at_ref, wide = [], []
    for mult, sink in ((1.0, at_ref), (2.0, wide)):
        means, pcounts, pm = _no_skill_arm_more_dispersed(perts, true, counts[0], mult,
                                                          seed=105)
        kw = dict(pred_bulk=(perts, means), real_bulk=(perts, obs), pred_moments=pm,
                  real_moments=real_m, control="ctrl", comparator="lognorm")
        num, den = mse_unbiased_capped(**kw), distance_unbiased(**kw)
        sink.append(sum(num.values()) / sum(den.values()))
    assert at_ref[0] == pytest.approx(1.0, abs=0.05), (
        f"matched dispersion at the reference's own depth must anchor at 1.0, got {at_ref[0]}"
    )
    assert wide[0] > 1.2, (
        f"a 2x-dispersed no-skill submission at the SAME depth read {wide[0]}; the cap did not "
        "refuse its oversized correction, so the #247 lever is open at equal cell count"
    )


@pytest.mark.parametrize("n_cells", [200, 500, 2000])
def test_the_replicate_anchor_is_0_at_every_depth(n_cells):
    # Measured on this fixture: -0.0059 / -0.0023 / -0.0006.
    perts, true, obs, counts, real_m = _anchor_panel(n_cells, seed=5)
    means, pcounts = _replicate_arm(perts, true, counts, seed=205)
    got = _aggregate(perts, means, pcounts, obs, real_m)
    assert got == pytest.approx(0.0, abs=0.05), f"replicate anchor {got} at n={n_cells}"


def test_a_paste_overshoots_below_zero_and_the_clamp_is_what_absorbs_it():
    # A paste is the SAME sample, so the numerator is -2 trS/n and the value is negative and
    # depth-dependent (-1.624 / -0.640 / -0.159 here). clamp_high=1.0 in the scale is what
    # turns that into a score of 1.0; nothing in the metric bounds it.
    perts, _, obs, counts, real_m = _anchor_panel(200, seed=5)
    got = _aggregate(perts, obs.copy(), counts.copy(), obs, real_m)
    assert got < -0.5, f"paste read {got}; expected a clear negative overshoot"


@pytest.mark.parametrize("n_cells,expected_old", [(200, 0.534), (500, 0.744), (2000, 0.922)])
def test_the_old_definition_misses_the_anchor_on_these_very_panels(n_cells, expected_old):
    """The guard that makes the two tests above non-vacuous. Recompute the SHIPPED-v0.9.0
    arithmetic -- plug-in denominator, mean of ratios -- on the same fixture and require it to
    be visibly wrong AND to move with depth, reproducing the real-panel pattern (VCC Test
    0.7643 at n=1142, CCL_2 0.2386 at n=500)."""
    perts, true, obs, counts, real_m = _anchor_panel(n_cells, seed=5)
    means, pcounts = _no_skill_arm(perts, true, counts, seed=105)
    kw = dict(pred_bulk=(perts, means), real_bulk=(perts, obs),
              pred_moments=_moments_for(perts, means, pcounts), real_moments=real_m,
              control="ctrl", comparator="lognorm")
    num = mse_unbiased_capped(**kw)
    ci = int(np.flatnonzero(perts == "ctrl")[0])
    idx = {str(p): i for i, p in enumerate(perts)}
    plugin = {p: float(np.sum((obs[idx[p]] - obs[ci]) ** 2) / N_GENES) for p in num}
    old = float(np.mean([num[p] / plugin[p] for p in num]))
    assert old == pytest.approx(expected_old, abs=0.05), (
        f"the old definition scored {old} at n={n_cells}, expected ~{expected_old}. If it is "
        "now near 1.0 the fixture has become too high-signal and the anchor tests above can "
        "no longer discriminate -- raise GENE_SD or lower EFFECT_SD, do not relax this."
    )


#: Where the real-panel moments live. The artifacts are internal cluster files and this module
#: ships in the public cut, so the DIRECTORIES are not written here. They are SEARCHED FOR, in
#: order: every directory in `$CEV2_REAL_PANEL_DIRS` (colon-separated, for artifacts staged
#: elsewhere) first, then every directory in `internal:tools/real_panel_dirs.txt`, which `tools/**` DROPs
#: from the ship set. The environment takes priority but does not suppress the sidecar. Neither
#: yielding the file -> the three tests below skip, which is what CI and any public checkout get.
#: The FILE NAMES stay here -- they are content-addressed cache keys and are what identifies the
#: panel -- and so do the NUMBERS, which are the evidence for the assertions.
REAL_PANEL_DIRS_ENV = "CEV2_REAL_PANEL_DIRS"
# Spelled as ONE literal rather than joined segment-by-segment so the publish gate that forbids a
# shipping file naming an unshipped path can SEE it; a PATHOK records the exemption, because this
# is a real filesystem lookup and cannot carry the `internal:` marker. Absent -> the three tests
# below skip, which is what CI and every public checkout get.
REAL_PANEL_DIRS_FILE = Path(__file__).parent.parent / "tools/real_panel_dirs.txt"
REAL_PANELS = [
    # (moments npz file name, control label, old-definition no-skill anchor measured for #257)
    ("pseudobulk_moments_lognorm-0d411f92b97cd2f0.moments.npz",
     "non-targeting", 0.7643, "VCC_Test"),
    ("pseudobulk_moments_lognorm-d27cd7f0f4b4918f.moments.npz",
     "__control__", 0.2386, "CCL_2"),
    ("pseudobulk_moments_lognorm-ca4c96af6db90824.moments.npz",
     "__control__", 0.2754, "H1_CGS"),
]


def _real_panel_dirs():
    """$CEV2_REAL_PANEL_DIRS first, then the internal sidecar if it is present.

    Entries are stripped: a padded `"/staged/panels : "` would otherwise contribute a directory
    named `" "`, miss silently, and let the search fall through to an internal panel the caller
    had meant to override.
    """
    env = os.environ.get(REAL_PANEL_DIRS_ENV, "")
    dirs = [d for d in (e.strip() for e in env.split(os.pathsep)) if d]
    if REAL_PANEL_DIRS_FILE.is_file():
        dirs += [ln.strip()
                 for ln in REAL_PANEL_DIRS_FILE.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    return dirs


def _find_real_panel(name):
    """The first configured directory holding `name`, or None if no configured one does."""
    for d in _real_panel_dirs():
        candidate = Path(d) / name
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.parametrize("name,ctrl,old_anchor,label",
                         REAL_PANELS, ids=[p[3] for p in REAL_PANELS])
def test_real_panels_the_old_anchor_is_reproduced_and_the_new_one_is_1(
        name, ctrl, old_anchor, label, monkeypatch):
    """Both halves on real data: the regression is reproduced, and the fix removes it.

    Skips wherever the panels are not reachable. The two assertions must travel together --
    reproducing the old number is what proves the new one is a fix rather than a
    differently-wrong number.

    ⚠️ The no-skill arm is the real control profile broadcast over every perturbation, so its
    across-perturbation spread is exactly zero and #348's bound withholds its whole correction.
    #257's calibration is what is under test here, so the bound is disabled for those two
    assertions and #348's effect on this real panel is pinned separately at the end.
    """
    path = _find_real_panel(name)
    if path is None:
        pytest.skip(f"{label} moments (internal cluster storage) not found in any directory "
                    f"from {REAL_PANEL_DIRS_FILE.name} or ${REAL_PANEL_DIRS_ENV}")
    with np.load(path, allow_pickle=False) as z:
        perts = np.asarray([str(p) for p in z["perts"]])
        real = np.asarray(z["means"], dtype=np.float64)
        gm = GroupMoments(perts=perts, counts=z["counts"], sumsq=z["sumsq"])
    ci = int(np.flatnonzero(perts == ctrl)[0])
    G = real.shape[1]

    # A no-skill submission: the real control, emitted at the control pool's own depth.
    pred = np.broadcast_to(real[ci], real.shape).copy()
    pred_counts = np.full(len(perts), float(np.asarray(gm.counts, float)[ci]))
    pred_sumsq = pred_counts * np.einsum("ij,ij->i", pred, pred) + (
        np.asarray(gm.sumsq, float)[ci] - np.asarray(gm.counts, float)[ci]
        * float(real[ci] @ real[ci])) * (pred_counts - 1.0) / (
        np.asarray(gm.counts, float)[ci] - 1.0)
    kw = dict(pred_bulk=(perts, pred), real_bulk=(perts, real),
              pred_moments=GroupMoments(perts=perts, counts=pred_counts, sumsq=pred_sumsq),
              real_moments=gm, control=ctrl, comparator="lognorm")

    with monkeypatch.context() as m:
        m.setattr("cell_eval2.metrics.delta._across_pert_budget", lambda *_a, **_k: float("inf"))
        num, den = mse_unbiased_capped(**kw), distance_unbiased(**kw)
    new = sum(num.values()) / sum(den.values())
    plugin = {p: float(np.sum((real[i] - real[ci]) ** 2) / G)
              for i, p in enumerate(perts) if p != ctrl}
    old = float(np.mean([num[p] / plugin[p] for p in num]))

    assert old == pytest.approx(old_anchor, abs=0.02), (
        f"{label}: the OLD definition scored {old}, not the {old_anchor} recorded in #257 -- "
        "either the panel changed or the reproduction is wrong; do not update this number "
        "without re-deriving it"
    )
    assert new == pytest.approx(1.0, abs=0.03), (
        f"{label}: the NEW definition scored {new}, not 1.0 (old was {old})"
    )

    # #348, on this panel: a broadcast arm has no across-perturbation spread, so the bound
    # withholds the prediction-side correction ENTIRELY -- identical to handing the metric an
    # arm whose correction is zero, per perturbation and exactly. `sumsq = n*||mu||^2` is what
    # makes `tr Sigma-hat/n` vanish under the `lognorm` comparator.
    bounded = mse_unbiased_capped(**kw)
    zero_kw = dict(kw, pred_moments=GroupMoments(
        perts=perts, counts=pred_counts,
        sumsq=pred_counts * np.einsum("ij,ij->i", pred, pred)))
    assert bounded == mse_unbiased_capped(**zero_kw)
    assert bounded != num, f"{label}: the bound did not bind on a broadcast no-skill arm"
