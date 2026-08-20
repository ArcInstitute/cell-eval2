import numpy as np
import pytest

from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.baseline import (
    build_baseline_prediction,
    generic_response_profile,
    lock_matrix_space,
)
# private on purpose: this test is ABOUT the per-side resolution run.py performs
from cell_eval2.run import _effective_input_type


def _cfg(**kw):
    base = dict(metrics=["expr_mae"], pert_col="target", control="non-targeting",
                input_type="counts", validate_input=True)
    base.update(kw)
    return EvalConfig(**base)


def _pred_for(real, cfg):
    prof = generic_response_profile(real, pert_col=cfg.pert_col, control=cfg.control,
                                    exclude_target_gene=False)
    return build_baseline_prediction(prof, real, pert_col=cfg.pert_col,
                                     control=cfg.control)


def _integer_total_reference():
    """The counterexample that kills the INFERRED rule: guess_is_lognorm tests fractional
    ROW TOTALS, not fractional entries. Non-control row totals 0.5 and 1.5 give a profile
    [0.75, 0.25]; positive control support gives r=[0.375, 0.25], whose emitted row totals
    exactly 1.0. Thus the prediction reads as `counts` while real reads as `lognorm`."""
    import anndata as ad
    import pandas as pd
    X = np.array([[0.5, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float64)
    obs = pd.DataFrame({"target": ["pA", "pB", "non-targeting"]}, index=["c0", "c1", "c2"])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1"]))


def test_v1_counts_reference_TRAP_reproduced_then_locked(synthetic_counts_pair):
    """Discriminating: assert the divergence FIRST, then that locking removes it. A test
    that only asserted the post-lock state would pass against a no-op implementation."""
    _, real = synthetic_counts_pair
    cfg = _cfg(version="v1")
    pred = _pred_for(real, cfg)

    assert _effective_input_type(real, cfg, side="real") == "counts"
    assert _effective_input_type(pred, cfg, side="pred") == "lognorm"   # the bug

    locked = lock_matrix_space(real, pred, config=cfg)
    assert locked.allow_discrete is True
    assert _effective_input_type(real, locked, side="real") == "counts"
    assert _effective_input_type(pred, locked, side="pred") == "counts"


def test_v2_autodetect_counts_reference_is_locked(synthetic_counts_pair):
    _, real = synthetic_counts_pair
    cfg = _cfg(version="v2", autodetect_input_type=True)
    pred = _pred_for(real, cfg)
    assert _effective_input_type(pred, cfg, side="pred") == "lognorm"   # the bug
    locked = lock_matrix_space(real, pred, config=cfg)
    assert locked.allow_discrete is True
    assert _effective_input_type(pred, locked, side="pred") == "counts"


def test_v2_without_autodetect_changes_nothing(synthetic_counts_pair):
    """Neither side auto-detects, so both already use the declared type and there is
    nothing to lock. Forcing an inert allow_discrete would only make run_params.yaml
    misdescribe the run -- and it is value-affecting in general, so it must not be set
    where it does nothing."""
    _, real = synthetic_counts_pair
    cfg = _cfg(version="v2")
    locked = lock_matrix_space(real, _pred_for(real, cfg), config=cfg)
    assert locked.allow_discrete is False
    assert locked == cfg


def test_lognorm_reference_is_left_alone(synthetic_pair):
    """Forcing allow_discrete here would be actively wrong: the reference resolves to
    lognorm and so does the fractional profile, so they already agree."""
    _, real = synthetic_pair            # log1p'd -> fractional
    cfg = _cfg(version="v1", input_type="lognorm")
    pred = _pred_for(real, cfg)
    assert _effective_input_type(real, cfg, side="real") == "lognorm"
    locked = lock_matrix_space(real, pred, config=cfg)
    assert locked.allow_discrete is False
    assert _effective_input_type(pred, locked, side="pred") == "lognorm"


def test_INTEGER_ROW_TOTAL_profile_is_caught_not_inferred_away():
    """The inferred rule ('real resolves to lognorm -> already consistent, leave it') would
    return this config unchanged and ship a corrupted comparison. The checked rule must
    RAISE: allow_discrete can only pull a side toward counts, and re-interpreting the
    REFERENCE is not the baseline's to do."""
    real = _integer_total_reference()
    cfg = _cfg(version="v1")
    pred = _pred_for(real, cfg)
    # the counterexample, asserted rather than assumed
    assert _effective_input_type(real, cfg, side="real") == "lognorm"
    assert _effective_input_type(pred, cfg, side="pred") == "counts"
    with pytest.raises(ValueError, match="lognorm"):
        lock_matrix_space(real, pred, config=cfg)


def _subset_trap_reference():
    """The four-matrix fixture: profile [0.75, 0.25] and r=[0.3, 0.25] emit rows totalling
    1.0, while the control row's 3.5 total keeps both FULL matrices reading `lognorm`.
    Only real[control] and pred[non-control] disagree."""
    import anndata as ad
    import pandas as pd
    X = np.array([[0.5, 0.0], [1.0, 0.5], [2.5, 1.0]], dtype=np.float64)
    obs = pd.DataFrame({"target": ["pA", "pB", "non-targeting"]}, index=["c0", "c1", "c2"])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1"]))


def test_the_lock_CHECKS_THE_DE_SUBSETS_TOO():
    """Discriminating for the four-matrix rule: here the FULL prediction and the FULL
    reference agree, so a two-matrix check returns the config unchanged -- and then
    _pred_de_input re-resolves the pred's non-control cells and the real control pool
    separately (run.py:524-546), finds them different, and dies at norm.py:441 trying the
    irreversible lognorm -> counts conversion.

    The CPU backend is PINNED, not left at 'auto'. `pred_non_control` is checked only when
    the CPU concat path is taken (`_lock_from_adata`, gated on `_use_inmem_external_ref`);
    under 'auto' the resolved backend follows the HOST, so on a CUDA machine this resolves
    to gpudge, the check is correctly skipped, and the test would fail for a reason that is
    not a defect. The gpudge branch is asserted separately below.
    """
    from cell_eval2.config import DEParams
    from cell_eval2.run import _effective_input_type as eff
    real = _subset_trap_reference()
    obs = real.obs
    de_cfg = _cfg(version="v1", metrics=["de_wilcoxon_overlap"], control_source="real",
                  de=DEParams(backend="scanpy"))
    pred = _pred_for(real, de_cfg)
    assert eff(real, de_cfg, side="real") == eff(pred, de_cfg, side="pred")   # two-matrix: OK
    ctrl = np.asarray(obs["target"]) == "non-targeting"
    assert eff(real[ctrl], de_cfg, side="real") != eff(pred[~ctrl], de_cfg, side="pred")
    with pytest.raises(ValueError, match="pred_non_control"):
        lock_matrix_space(real, pred, config=de_cfg)

    # ...and the SAME fixture must NOT raise when nothing re-detects the subsets: no DE
    # metric and a numeric target_sum means _pred_de_input never runs (run.py:834) and the
    # control pool is never re-resolved (run.py:765-773). Rejecting here would fail a run
    # that would have been correct -- checking too MANY matrices is a defect too.
    mae_cfg = _cfg(version="v1", metrics=["expr_mae"], target_sum=1e6)
    assert lock_matrix_space(real, _pred_for(real, mae_cfg), config=mae_cfg) == mae_cfg


def test_the_lock_SKIPS_pred_non_control_UNDER_gpudge():
    """The other side of the four-matrix rule, and the reason the test above must pin its
    backend. A resolved gpudge backend passes the FULL prediction as its DE target
    (`_use_inmem_external_ref`, run.py:472-519), so `_pred_de_input` never re-resolves the
    prediction's non-control rows -- and checking a matrix nothing will look at would fail a
    run that is correct. On the SAME fixture that raises under a CPU backend, the lock must
    therefore return the config untouched.

    This runs on any host, GPU or not: `_use_inmem_external_ref` short-circuits on an
    explicit (non-'auto') backend and never imports gpudge or probes for a CUDA device.
    """
    from cell_eval2.config import DEParams
    from cell_eval2.run import _use_inmem_external_ref
    real = _subset_trap_reference()
    kw = dict(version="v1", metrics=["de_wilcoxon_overlap"], control_source="real")
    cpu_cfg = _cfg(**kw, de=DEParams(backend="scanpy"))
    gpu_cfg = _cfg(**kw, de=DEParams(backend="gpudge"))
    pred = _pred_for(real, cpu_cfg)

    # discriminating: the SAME fixture + the SAME pred, differing only in the backend
    assert _use_inmem_external_ref(cpu_cfg) is False
    assert _use_inmem_external_ref(gpu_cfg) is True
    with pytest.raises(ValueError, match="pred_non_control"):
        lock_matrix_space(real, pred, config=cpu_cfg)
    assert lock_matrix_space(real, pred, config=gpu_cfg) == gpu_cfg


def test_the_lock_CHANGES_THE_NUMBERS(synthetic_counts_pair):
    """The trap is not a labelling nicety: reading the profile as lognorm against a counts
    reference skips the normalization and produces a different expr_mae. v1 is used
    because it does not run input-type validation, so the WRONG arm still computes a
    number instead of raising -- which is exactly what makes the bug silent.
    Measured after dispersed emission: 6.8057 unlocked vs 0.4738 locked."""
    _, real = synthetic_counts_pair
    cfg = _cfg(version="v1", allow_fractional_counts=True)
    pred = _pred_for(real, cfg)
    wrong = compute_metrics(pred, real, config=cfg)["value"].mean()
    right = compute_metrics(pred, real,
                            config=lock_matrix_space(real, pred, config=cfg))["value"].mean()
    assert np.isfinite(wrong) and np.isfinite(right)
    assert not np.isclose(wrong, right)


def test_accepts_a_path_reference(synthetic_counts_pair, tmp_path):
    _, real = synthetic_counts_pair
    p = tmp_path / "real.h5ad"
    real.write_h5ad(p)
    cfg = _cfg(version="v1")
    locked = lock_matrix_space(str(p), _pred_for(real, cfg), config=cfg)
    assert locked.allow_discrete is True
    real.write_h5ad(p)          # reopenable for writing -> nothing holds the file


def test_already_discrete_config_is_unchanged(synthetic_counts_pair):
    _, real = synthetic_counts_pair
    cfg = _cfg(version="v1", allow_discrete=True)
    assert lock_matrix_space(real, _pred_for(real, cfg), config=cfg) == cfg
