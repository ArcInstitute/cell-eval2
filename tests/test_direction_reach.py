import math
from decimal import Decimal

import polars as pl
import pytest

from cell_eval2.de import TargetResolution, prepare_de
from cell_eval2.metrics import direction
from cell_eval2.metrics.direction import (
    REACH_PURITY_FLOOR,
    _k_star,
    _ontarget_excluded_frame,
    _purity_curve,
    de_direction_reach,
)

SIG, NS = 0.01, 0.90


def _tbl(rows):
    return pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "feature": [r[1] for r in rows],
            "log2_fold_change": [float(r[2]) for r in rows],
            "p_adj": [float(r[3]) for r in rows],
        },
        schema={"target": pl.String, "feature": pl.String,
                "log2_fold_change": pl.Float64, "p_adj": pl.Float64},
    )


def _resolution(rows, mode):
    """mode='self' (default) -> every target maps to its own label; None -> derive."""
    # TargetResolution comes from the module-level import; a local re-import here would
    # shadow it and is the F811/F401 shape that already broke CI once on this branch.
    if mode is None:
        return None
    if mode != "self":
        return mode
    targets = sorted({r[0] for r in rows})
    return TargetResolution({t: t for t in targets}, len(targets))


def _prep(pred_rows, real_rows, *, resolution="self"):
    """Default to a SELF-MAP resolution -- see the plan's fixture-trap note."""
    return prepare_de(_tbl(pred_rows), _tbl(real_rows), control="non-targeting",
                      p_adj_threshold=0.05,
                      target_resolution=_resolution(real_rows, resolution))


def test_k_star_accepts_a_per_target_p0_mapping():
    curve = pl.DataFrame(
        {"target": ["A", "A", "B"], "k": [1, 2, 1],
         "n_denom": [1, 2, 1], "n_match": [1, 1, 1],
         "purity": [1.0, 0.5, 1.0]},
        schema={"target": pl.String, "k": pl.Int64, "n_denom": pl.Int64,
                "n_match": pl.Int64, "purity": pl.Float64},
    )
    assert _k_star(curve, p0=0.9) == {"A": 1, "B": 1}
    assert _k_star(curve, p0={"A": 0.4, "B": 0.9}) == {"A": 2, "B": 1}
    # A target absent from the mapping (undefined q) drops out entirely.
    assert _k_star(curve, p0={"A": 0.4}) == {"A": 2}


def test_k_star_accepts_a_numpy_scalar_threshold():
    """A numpy scalar is NOT a Python float -- isinstance(np.float32(0.9), float) is
    False, since only np.float64 subclasses it. Dispatching on `(int, float)` therefore
    sent a np.float32 threshold down the MAPPING branch and raised
    "'numpy.float32' object has no attribute 'keys'". Hence: dispatch on Mapping."""
    import numpy as np

    curve = pl.DataFrame(
        {"target": ["A", "A"], "k": [1, 2], "n_denom": [1, 2], "n_match": [1, 1],
         "purity": [1.0, 0.5]},
        schema={"target": pl.String, "k": pl.Int64, "n_denom": pl.Int64,
                "n_match": pl.Int64, "purity": pl.Float64},
    )
    for scalar in (np.float32(0.9), np.float64(0.9), np.float16(0.9)):
        assert _k_star(curve, p0=scalar) == {"A": 1}, repr(scalar)


def test_the_v0_5_0_sensitivity_survives_a_numpy_p_adj_threshold():
    """The end-to-end form of the above, and the regression it actually caused. The three
    v0.5.0 direction metrics must be byte-identical before and after #195, and
    `1.0 - np.float32(alpha)/2.0` stays np.float32 under NEP 50, so an ordinary
    DEParams(p_adj_threshold=np.float32(0.05)) reached the broken branch."""
    import numpy as np

    from cell_eval2.metrics.direction import de_direction_sensitivity

    rows = [("A", "B", 2.0, 0.001), ("A", "C", 1.0, 0.002)]
    for alpha in (0.05, np.float64(0.05), np.float32(0.05)):
        p = prepare_de(_tbl(rows), _tbl(rows), control="non-targeting",
                       p_adj_threshold=alpha, target_resolution=_resolution(rows, "self"))
        assert de_direction_sensitivity(p, universe="adjudicated")["A"] == pytest.approx(1.0)
        assert de_direction_reach(p, universe="adjudicated",
                                  corrected=False)["A"] == pytest.approx(1.0)


@pytest.mark.parametrize("universe", ["adjudicated", "all"])
@pytest.mark.parametrize("corrected", [True, False])
def test_reach_accepts_a_non_float_p_adj_threshold(universe, corrected):
    """`p_adj_threshold` is caller-supplied and need not be a Python float. The reach
    variants did Python arithmetic on it -- `1.0 - alpha` when corrected, `alpha / 2.0` when
    raw (the raw half is gone as of the REACH_PURITY_FLOOR constant; see
    `test_raw_reach_does_no_arithmetic_on_alpha`) -- so a Decimal raised TypeError in both
    branches until de_direction_reach started
    normalizing with float().

    ⚠️ This is NOT a #195 regression: the v0.5.0 de_direction_sensitivity raises the same
    TypeError on a Decimal on origin/main (measured). It is deliberately not fixed here --
    v0.5.0 must stay byte-identical, so changing when it raises belongs to #196. This test
    pins only that the NEW family does not inherit the trap.
    """
    import numpy as np

    rows = [("A", "A", 3.0, 0.001), ("A", "B", 2.0, 0.002), ("A", "C", 1.0, 0.003)]
    for alpha in (0.05, np.float32(0.05), Decimal("0.05")):
        p = prepare_de(_tbl(rows), _tbl(rows), control="non-targeting",
                       p_adj_threshold=alpha,
                       target_resolution=TargetResolution({"A": "A"}, 1))
        got = de_direction_reach(p, universe=universe, corrected=corrected)["A"]
        assert got == pytest.approx(1.0), (alpha, universe, corrected)


def test_corrected_threshold_equals_the_transformed_curve_threshold():
    """Spec 2.5: (P - q)/(1 - q) >= 1 - alpha  <=>  P >= q + (1-alpha)(1-q), for q < 1."""
    alpha, q = 0.05, 0.6
    raw_p0 = q + (1 - alpha) * (1 - q)
    curve = pl.DataFrame(
        {"target": ["A"] * 4, "k": [1, 2, 3, 4], "n_denom": [1, 2, 3, 4],
         "n_match": [1, 2, 3, 3],
         "purity": [1.0, 1.0, 1.0, 0.75]},
        schema={"target": pl.String, "k": pl.Int64, "n_denom": pl.Int64,
                "n_match": pl.Int64, "purity": pl.Float64},
    )
    raw_hit = _k_star(curve, p0=raw_p0)
    transformed = curve.with_columns(purity=(pl.col("purity") - q) / (1 - q))
    assert raw_hit == _k_star(transformed, p0=1 - alpha)


def test_reach_is_bounded_by_one_when_the_target_gene_is_reference_significant():
    """Spec 8: CONSTRUCTED so that RETAINING the target gene would give
    k* = N_conf + 1. On an arbitrary fixture an unfiltered pool still satisfies
    reach <= 1 and the assertion passes without exercising anything.

    Real: A's own gene A plus B, C all significant and all UP -> retained N_conf = 3,
    excluded N_conf = 2. Pred ranks A first (smallest p_adj) and all three agree, so an
    unfiltered pool gives k* = 3 over N_conf = 2 => 1.5.
    """
    pred = [("A", "A", 3.0, 0.001), ("A", "B", 2.0, 0.002), ("A", "C", 1.0, 0.003)]
    real = [("A", "A", 1.0, SIG), ("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG)]
    p = _prep(pred, real)
    from cell_eval2.metrics.direction import _reference_stats
    assert _reference_stats(p)["n_conf"].to_list() == [2]
    curve = _purity_curve(_ontarget_excluded_frame(p), universe="adjudicated", alpha=0.05)
    assert _k_star(curve, p0=0.975) == {"A": 2}
    assert de_direction_reach(p, universe="adjudicated", corrected=False)["A"] == \
        pytest.approx(1.0)
    assert de_direction_reach(p, universe="adjudicated", corrected=False)["A"] <= 1.0


def _ten_genes_last_one_wrong():
    """N_conf = 10 ranked 9-correct-then-1-miss, so purity(10) = 9/10 = 0.9 EXACTLY.

    The one fixture shape that separates the two thresholds: 0.9 admits the full depth
    (k* = 10) and 0.975 does not (k* = 9, the last all-correct prefix).
    """
    pred = [("A", f"g{i}", 10.0 - i, 0.001 + i * 1e-6) for i in range(10)]
    real = [("A", f"g{i}", 1.0, SIG) for i in range(10)]
    pred[9] = ("A", "g9", -1.0, pred[9][3])          # the sole miss, ranked last
    return pred, real


def test_raw_reach_no_longer_shares_v0_5_0_sensitivity_s_threshold():
    """The identity this file used to assert is DELIBERATELY dead.

    `direction_reach_raw` moved to REACH_PURITY_FLOOR = 0.9; `de_direction_sensitivity` is one
    of the three v0.5.0 metrics whose values must not move (spec 1/3) and keeps 1 - alpha/2.
    Asserted on a fixture that SEPARATES them -- the old two-gene, all-correct fixture reads
    1.0 under either threshold, so keeping it would have left this file green while the
    metrics diverged underneath it.
    """
    from cell_eval2.metrics.direction import de_direction_sensitivity
    pred, real = _ten_genes_last_one_wrong()
    p = _prep(pred, real)
    reach = de_direction_reach(p, universe="adjudicated", corrected=False)["A"]
    sens = de_direction_sensitivity(p, universe="adjudicated")["A"]
    assert reach == pytest.approx(1.0)      # k* = 10, purity(10) = 0.9 >= 0.9
    assert sens == pytest.approx(0.9)       # k* = 9,  purity(10) = 0.9 <  0.975
    assert reach > sens


def test_the_purity_floor_is_the_constant_and_the_metric_follows_it(monkeypatch):
    """Pins BOTH halves: the shipped value, and that the raw branch READS THE CONSTANT.

    ⚠️ The second half needs the monkeypatch. Re-deriving the fixture's answer through
    `_k_star(curve, p0=REACH_PURITY_FLOOR)` and asserting the metric agrees does NOT prove
    the metric reads the name -- a hard-coded literal `0.9` in `de_direction_reach` satisfies
    every such assertion identically. Moving the constant and requiring the METRIC to move is
    the only form that distinguishes them.
    """
    assert REACH_PURITY_FLOOR == 0.9
    pred, real = _ten_genes_last_one_wrong()
    p = _prep(pred, real)
    curve = _purity_curve(_ontarget_excluded_frame(p), universe="adjudicated", alpha=0.05)
    assert _k_star(curve, p0=REACH_PURITY_FLOOR) == {"A": 10}
    assert de_direction_reach(p, universe="adjudicated", corrected=False)["A"] == \
        pytest.approx(1.0)

    # Move the constant; the metric must follow. 0.91 is just above the fixture's
    # purity(10) = 0.9, so the full depth stops qualifying and k* drops to 9.
    monkeypatch.setattr(direction, "REACH_PURITY_FLOOR", 0.91)
    assert de_direction_reach(_prep(pred, real), universe="adjudicated",
                              corrected=False)["A"] == pytest.approx(0.9)


@pytest.mark.parametrize("alpha", ["0.05", "0.02"])
def test_raw_reach_does_no_arithmetic_on_alpha(alpha):
    """A Decimal p_adj_threshold used to reach `alpha / 2.0` and raise; with a constant floor
    the raw branch never touches alpha, so it completes. The corrected branch still does
    `1.0 - alpha`, which is why the `float()` normalization lives inside it.

    ⚠️ TWO alphas, because one proves nothing. The fixture's significant populations are
    unchanged across both (every real p_adj is 0.01, so it clears either threshold), so a raw
    reach that still DERIVED its floor from alpha -- `1 - 2*alpha`, say, which happens to equal
    0.9 at alpha = 0.05 -- would move at 0.02 and be caught here.
    """
    pred, real = _ten_genes_last_one_wrong()
    p = prepare_de(_tbl(pred), _tbl(real), control="non-targeting",
                   p_adj_threshold=Decimal(alpha),
                   target_resolution=TargetResolution({"A": "A"}, 1))
    assert de_direction_reach(p, universe="adjudicated", corrected=False)["A"] == \
        pytest.approx(1.0)


@pytest.mark.parametrize("corrected", [True, False])
@pytest.mark.parametrize("universe", ["adjudicated", "all"])
def test_reach_is_nan_when_nconf_is_zero(corrected, universe):
    """Spec 5 columns 1-2: the agreement convention does NOT transfer to the depth
    family -- reach never asks the model to make a call, and reach_unbounded's pool is
    non-empty at N_conf = 0, so k* can be positive over a zero denominator."""
    pred = [("A", "B", 1.0, SIG)]
    real = [("A", "B", 1.0, NS)]
    assert math.isnan(de_direction_reach(_prep(pred, real), universe=universe,
                                         corrected=corrected)["A"])


@pytest.mark.parametrize("universe", ["adjudicated", "all"])
def test_corrected_reach_is_nan_without_q_but_raw_reach_computes(universe):
    """Spec 5 columns 4-5: raw reach never reads q."""
    pred = [("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG)]
    real = [("A", "B", 0.0, SIG), ("A", "C", 0.0, SIG)]  # significant, no direction
    p = _prep(pred, real)
    assert math.isnan(de_direction_reach(p, universe=universe, corrected=True)["A"])
    got = de_direction_reach(p, universe=universe, corrected=False)["A"]
    # Pin the VALUE, not just "not NaN": `not isnan` also passes for an accidental 1.0 or
    # for a k* leaked from the corrected branch. N_conf = 2, but neither reference row is
    # adjudicable (both log2FC are 0), so nothing sustains the raw threshold and k* = 0 --
    # a COMPUTED zero, which is exactly what distinguishes it from the corrected NaN.
    assert got == pytest.approx(0.0)


def test_never_reaching_p0_is_a_computed_zero_not_nan():
    pred = [("A", "B", 1.0, 0.001), ("A", "C", 1.0, 0.002)]
    real = [("A", "B", -1.0, SIG), ("A", "C", -1.0, SIG)]  # every sign wrong
    p = _prep(pred, real)
    assert de_direction_reach(p, universe="adjudicated", corrected=False)["A"] == \
        pytest.approx(0.0)


def test_every_perturbation_gets_a_row():
    pred = [("A", "B", 1.0, SIG), ("B", "A", 1.0, SIG)]
    real = [("A", "B", 1.0, SIG), ("B", "A", 1.0, NS)]
    p = _prep(pred, real)
    for corrected in (True, False):
        for universe in ("adjudicated", "all"):
            assert set(de_direction_reach(p, universe=universe,
                                          corrected=corrected)) == {"A", "B"}


def test_an_unadjudicable_reference_significant_gene_shortens_the_adjudicated_reach():
    """Issue #204: `universe='adjudicated'` filters on reference SIGNIFICANCE, which does
    not imply adjudicability -- so the SCORED `direction_reach` CAN move on the right data.

    The #204 measurement found no such row on any of the six reference lines (0 of 326,832
    / 177,289 / 419,959 / 339,572 / 205,840 / 244,205 reference-significant genes), so the
    scored metric did not move there. That is an empirical result, not a structural
    guarantee, and this fixture pins the behaviour where it does not hold: B is
    reference-significant with an exactly-zero log2FC, so it counts toward N_conf but
    carries no direction and does not advance the depth.

    q = 1.0 (A is the only vote), so the corrected threshold P0 = q + (1-alpha)(1-q) = 1.0
    and the raw threshold is REACH_PURITY_FLOOR; the prefix is pure either way, so this
    fixture is threshold-agnostic. k* = 1 of N_conf = 2.
    Under the pre-#204 row-counting depth B advanced k as well, giving k* = 2 and 1.0.
    """
    pred = [("G1", "A", 1.0, SIG), ("G1", "B", 1.0, 0.02)]
    real = [("G1", "A", 1.0, SIG), ("G1", "B", 0.0, SIG)]
    p = _prep(pred, real)
    for corrected in (True, False):
        assert de_direction_reach(p, universe="adjudicated",
                                  corrected=corrected)["G1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------
# the purity floor as a SEMANTICS key -- the silent half
# ---------------------------------------------------------------------------------------

def _tiny_counts_adata(perts, genes):
    """A minimal counts AnnData whose targets name measured genes, for driver-level tests.
    Same shape as `test_target_gene_exclusion_172._tiny_adata`, which this mirrors."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    labels = np.repeat(perts, 6)
    X = rng.poisson(4.0, size=(labels.size, len(genes))).astype(np.float32)
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"perturbation": labels},
                         index=[f"c{i}" for i in range(labels.size)]),
        var=pd.DataFrame(index=list(genes)),
    )


def test_the_purity_floor_is_actually_WIRED_into_the_driver(monkeypatch, tmp_path):
    """`tests/test_cache.py` calls `_result_config_digest` with `reach_floor_used=True` by
    hand, so DELETING the `reach_floor_used=` argument at the driver's call site would leave
    it green while warm old-floor caches were served again (codex round 3). This goes through
    `compute_metrics` and separates the two halves: moving the floor must move the result
    fingerprint for a reach run and must NOT move it for a metric the floor cannot touch.
    """
    from cell_eval2.config import EvalConfig
    import cell_eval2.metrics.direction as direction_mod
    import cell_eval2.run as run_mod

    seen: list[str] = []
    real_fp = run_mod.result_fingerprint

    def spy(**kw):
        seen.append(kw["config_digest"])
        return real_fp(**kw)

    monkeypatch.setattr(run_mod, "result_fingerprint", spy)
    genes = [f"g{i}" for i in range(12)]
    perts = ["non-targeting", "g0", "g1"]

    def digests(metrics):
        seen.clear()
        # `cache_pred` is what makes the driver compute a result fingerprint at all; the key
        # this test is about only exists on that path.
        cfg = EvalConfig(metrics=metrics, pert_col="perturbation", control="non-targeting",
                         input_type="counts", device="cpu",
                         cache_pred=str(tmp_path / "cache"))
        ad_obj = _tiny_counts_adata(perts, genes)
        try:
            run_mod.compute_metrics(ad_obj, ad_obj, config=cfg)
        except Exception:
            pass                       # the digest is computed before any metric runs
        return list(seen)

    for metrics, must_move in ((["de_wilcoxon_direction_reach_raw"], True),
                               (["expr_mae"], False)):
        before = digests(metrics)
        monkeypatch.setattr(direction_mod, "REACH_PURITY_FLOOR", 0.975)
        after = digests(metrics)
        monkeypatch.setattr(direction_mod, "REACH_PURITY_FLOOR", REACH_PURITY_FLOOR)
        assert before and after, f"no fingerprint was computed for {metrics}"
        moved = before != after
        assert moved is must_move, (
            f"{metrics}: moving the purity floor {'did not move' if must_move else 'moved'} "
            f"the result cache key; the driver wiring is wrong"
        )


def test_the_purity_floor_reaches_the_anchor_s_semantic_identity(monkeypatch, toy_de_adata):
    """The anchor is a FROZEN artifact, so a bundle built before the floor moved carries a
    replicate computed under a different rule for a scored member.

    `anchor_semantic_params` is the SAME subset the anchor cache key and `validate_anchor`
    both read, so a hole here is two holes: a false cache hit AND a supplied artifact that
    validates against a run it does not belong to. Nothing else in that dict can see the
    floor -- it is a module constant, not a config knob, so `_SEMANTIC_FIELDS` and
    `config_hash` are both blind to it (codex round 3 found this binding untested).

    Both halves asserted: a reach selection MOVES, an expression-only one does NOT.
    """
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.config import EvalConfig
    import cell_eval2.metrics.direction as direction_mod
    from cell_eval2.run import _resolve_config

    # `toy_de_adata` is a FACTORY fixture -- call it. Passing the factory itself happens to
    # work here (with `input_type="counts"` fixed, this path never inspects the object), but
    # that is an accident of the code path, not the interface (codex round 4).
    real = toy_de_adata()

    def identity(metrics):
        cfg = _resolve_config(EvalConfig(metrics=metrics, pert_col="target_gene",
                                         control="non-targeting", input_type="counts",
                                         validate_input=False), {})
        names = list(resolve_metrics(cfg.metrics, version=cfg.version)[0])
        return semantic_identity(cfg, real, names)

    for metrics, must_move in ((["de_wilcoxon_direction_reach_raw"], True),
                               (["expr_mae"], False)):
        before = identity(metrics)
        monkeypatch.setattr(direction_mod, "REACH_PURITY_FLOOR", 0.975)
        after = identity(metrics)
        monkeypatch.setattr(direction_mod, "REACH_PURITY_FLOOR", REACH_PURITY_FLOOR)
        assert (before != after) is must_move, (
            f"{metrics}: moving the purity floor "
            f"{'did not move' if must_move else 'moved'} the anchor's semantic identity"
        )


# ---------------------------------------------------------------------------------------
# `purity_floor` -- the CALIBRATION argument (#327), and the unreachability it rests on
# ---------------------------------------------------------------------------------------

def _ten_genes_last_two_wrong():
    """N_conf = 10 ranked 8-correct-then-2-misses, so the purity curve crosses THREE times:
    purity(8) = 1.0, purity(9) = 8/9 = 0.888..., purity(10) = 0.8.

    `_ten_genes_last_one_wrong` separates 0.9 from 0.975 and nothing else: its prefix is
    perfectly pure below k = 10, so every floor at or under 0.9 reads 1.0 there and the
    argument could only ever be shown to move the metric UPWARD, off a saturated value.
    This fixture gives three floors three DIFFERENT numbers (0.8 / 0.9 / 1.0), so a
    `purity_floor` that reached the code path without reaching the arithmetic cannot pass.
    """
    pred = [("A", f"g{i}", 10.0 - i, 0.001 + i * 1e-6) for i in range(10)]
    real = [("A", f"g{i}", 1.0, SIG) for i in range(10)]
    for i in (8, 9):                                  # the two misses, ranked last
        pred[i] = ("A", f"g{i}", -(10.0 - i), pred[i][3])
    return pred, real


@pytest.mark.parametrize("floor,expected", [
    (0.75, 1.0),        # purity(10) = 0.8 clears it -> the full depth
    (0.8, 1.0),         # ...and the crossing is inclusive: `>=`, not `>`
    (0.85, 0.9),        # only purity(9) = 0.888... clears it
    (REACH_PURITY_FLOOR, 0.8),
    (0.95, 0.8),        # nothing past the pure prefix clears it
])
def test_the_purity_floor_ARGUMENT_moves_the_raw_metric(floor, expected):
    """The point of #327: a sweep can drive the SHIPPED metric. The 0.9 calibration could
    not -- `internal:tools/reachval/purity_threshold_sweep.py` re-assembles the metric from
    `_purity_curve`/`_k_star` with its own `p0`, because this entry point had no way in.

    ⚠️ Asserted on VALUES that differ, not on "the call accepted the argument". A parameter
    accepted and then dropped on the floor satisfies a smoke test identically; three floors
    with three different answers do not.
    """
    pred, real = _ten_genes_last_two_wrong()
    got = de_direction_reach(_prep(pred, real), universe="adjudicated", corrected=False,
                             purity_floor=floor)["A"]
    assert got == pytest.approx(expected)


def test_the_purity_floor_argument_moves_the_metric_UPWARD_too():
    """The other side of the default. `_ten_genes_last_one_wrong` reads 1.0 at the shipped
    floor because purity(10) = 0.9 exactly clears it; 0.91 does not, so k* falls to 9. Same
    crossing the constant-following test drives through the monkeypatch, driven here through
    the ARGUMENT -- which is the half a caller can actually reach."""
    pred, real = _ten_genes_last_one_wrong()
    p = _prep(pred, real)
    assert de_direction_reach(p, universe="adjudicated", corrected=False)["A"] == \
        pytest.approx(1.0)
    assert de_direction_reach(p, universe="adjudicated", corrected=False,
                              purity_floor=0.91)["A"] == pytest.approx(0.9)


def test_the_DEFAULT_is_the_constant_and_None_means_the_constant():
    """#327 must not move a single scored number, so the default and the `None` sentinel
    must both resolve to `REACH_PURITY_FLOOR` -- not to a third behaviour.

    ⚠️ Asserted on the UNSATURATED fixture (0.8, not 1.0). On a fixture that reads 1.0 the
    three calls agree at the metric's ceiling whatever the floor did, so the comparison
    would hold even if the resolution were wrong.
    """
    pred, real = _ten_genes_last_two_wrong()
    p = _prep(pred, real)
    default = de_direction_reach(p, universe="adjudicated", corrected=False)
    assert default["A"] == pytest.approx(0.8)                      # not at the ceiling
    assert de_direction_reach(p, universe="adjudicated", corrected=False,
                              purity_floor=None) == pytest.approx(default)
    assert de_direction_reach(p, universe="adjudicated", corrected=False,
                              purity_floor=REACH_PURITY_FLOOR) == pytest.approx(default)


def test_an_explicit_purity_floor_OVERRIDES_the_constant(monkeypatch):
    """The two halves are not the same claim. `test_the_purity_floor_is_the_constant_and_the
    _metric_follows_it` proves the branch reads the NAME; this proves the argument WINS when
    both are present. A `p0 = REACH_PURITY_FLOOR` that ignored the argument entirely passes
    that test and fails this one."""
    pred, real = _ten_genes_last_two_wrong()
    monkeypatch.setattr(direction, "REACH_PURITY_FLOOR", 0.5)      # would read 1.0 unaided
    assert de_direction_reach(_prep(pred, real), universe="adjudicated",
                              corrected=False)["A"] == pytest.approx(1.0)
    assert de_direction_reach(_prep(pred, real), universe="adjudicated", corrected=False,
                              purity_floor=0.85)["A"] == pytest.approx(0.9)


@pytest.mark.parametrize("floor", [0.5, 0.85])
@pytest.mark.parametrize("universe", ["adjudicated", "all"])
def test_the_purity_floor_does_NOT_reach_the_CORRECTED_branch(floor, universe):
    """⚠️ The corrected form thresholds at the per-target majority-sign null
    `P0 = q + (1-alpha)(1-q)` -- computed from the reference rather than chosen, and a
    different object from the raw floor. #327 gives the argument to the raw branch only, so
    the four `corrected=True` spellings must be bit-identical with and without it.

    NOT vacuous: both floors move the RAW form on this fixture, which the last assertion
    pins -- so a `purity_floor` leaking into the corrected `p0` would be seen here rather
    than swallowed by a fixture that could not tell the thresholds apart.
    """
    pred, real = _ten_genes_last_two_wrong()
    p = _prep(pred, real)
    base = de_direction_reach(p, universe=universe, corrected=True)
    assert base["A"] == pytest.approx(0.8)          # q = 1.0 -> P0 = 1.0, k* = 8 of N_conf 10
    assert de_direction_reach(p, universe=universe, corrected=True,
                              purity_floor=floor) == pytest.approx(base)
    assert de_direction_reach(p, universe=universe, corrected=False,
                              purity_floor=floor)["A"] != pytest.approx(base["A"])


def test_de_direction_sensitivity_does_NOT_gain_the_purity_floor():
    """`de_direction_sensitivity` is one of the three v0.5.0 metrics whose values must not
    move (spec 1/3): it keeps `1 - alpha/2` and #327 must not hand it a lever. The raw-reach
    == sensitivity identity is already broken on purpose and pinned as broken, so the two
    thresholds are now independent facts and this one must stay a derived constant.

    Both halves: the argument is absent from the signature, AND passing it raises rather
    than being swallowed by a `**kwargs` the function does not have.
    """
    import inspect

    from cell_eval2.metrics.direction import de_direction_sensitivity

    assert "purity_floor" not in inspect.signature(de_direction_sensitivity).parameters
    pred, real = _ten_genes_last_two_wrong()
    p = _prep(pred, real)
    # 1 - alpha/2 = 0.975: only the pure prefix clears it, k* = 8 of 10.
    assert de_direction_sensitivity(p, universe="adjudicated")["A"] == pytest.approx(0.8)
    with pytest.raises(TypeError):
        de_direction_sensitivity(p, universe="adjudicated", purity_floor=0.5)


def _bound_keywords(func):
    """Every keyword bound anywhere down a (possibly nested) `functools.partial` chain.

    Mirrors `run._func_is_one_of`'s unwrapping rather than reading `func.keywords` once: a
    doubly-wrapped partial would hide a binding from the single-level read, and this test
    exists precisely to catch a binding nobody meant to add.
    """
    out: dict = {}
    seen = 0
    while func is not None and seen < 8:
        out.update(getattr(func, "keywords", None) or {})
        func = getattr(func, "func", None)
        seen += 1
    return out


def test_the_purity_floor_is_NOT_reachable_from_the_CATALOG_or_the_CONFIG():
    """⚠️ THE SAFETY ARGUMENT OF #327 IN ONE TEST. The floor is a function argument and
    nothing else on purpose: a per-run knob on a scored `vcc2026` member would let two
    submissions scored at different floors compare as if they were comparable, and let a
    bundle built at one floor score a submission computed at another with nothing raising.

    That unreachability is ALSO what makes the rest of the design correct -- it is why
    `competition_payload()` may freeze the resolved default, why `run._result_config_digest`,
    `partition.result_semantics` and `anchor.anchor_semantic_params` may go on stamping the
    module constant, and why `_PARTIAL_SEMANTICS_SCHEMA` was not bumped. Break it and five
    surfaces become wrong at once, silently. This test is the tripwire on that.

    It is a GUARD, not a proof of the change: it passes before #327 too (trivially, the
    argument did not exist). What it forbids is the NEXT edit.
    """
    from cell_eval2.catalog import CATALOG
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _func_is_one_of

    spellings = [n for n, s in CATALOG.items()
                 if _func_is_one_of(s.func, (de_direction_reach,))]
    # Four variants x two backend families. Pinned so a registration that stopped resolving
    # to this func cannot empty the loop below and leave it green over nothing.
    assert len(spellings) == 8, spellings
    for name in spellings:
        assert "purity_floor" not in _bound_keywords(CATALOG[name].func), name

    # ...and no config field can supply one either. The whole nested dict, not just the top
    # level: `de.*`, `discrimination.*` and `filter.*` are config surfaces too.
    def leaves(d, prefix=""):
        for k, v in d.items():
            yield f"{prefix}{k}"
            if isinstance(v, dict):
                yield from leaves(v, f"{prefix}{k}.")

    paths = set(leaves(EvalConfig.from_preset("vcc2026").to_dict()))
    assert not [p for p in paths if "purity" in p], sorted(paths)
    # ⚠️ NOT a blanket ban on the substring `floor`: the config already carries two, and both
    # are the DE engine's AUC p-value floor -- a different quantity on a different surface.
    # Enumerating them keeps this a statement about WHICH floors the config can set, so a new
    # one has to be looked at rather than blending into a substring rule.
    assert {p for p in paths if p.split(".")[-1].endswith("floor")} == {"de.auc_pval_floor"}


@pytest.mark.parametrize("path", ["purity_floor", "de.purity_floor",
                                  "discrimination.purity_floor"])
def test_the_CLI_escape_hatch_cannot_supply_a_purity_floor(path):
    """`--set KEY.PATH=VALUE` is the generic escape hatch for any config field with no
    dedicated flag, so "there is no `--purity-floor` flag" is not on its own the claim worth
    pinning -- `--set` is the way in if there is one. It resolves the path against
    `cfg.to_dict()` and exits on an unknown one, which is what closes the door."""
    from cell_eval2.cli import _apply_set_overrides
    from cell_eval2.config import EvalConfig

    with pytest.raises(SystemExit):
        _apply_set_overrides(EvalConfig.from_preset("vcc2026"), [f"{path}=0.5"])


def test_the_DE_DRIVER_cannot_deliver_a_purity_floor(monkeypatch):
    """The tightest form of the unreachability claim, and the one the catalog check cannot
    make. `dispatch_de_metrics` builds a FIXED kwargs dict and filters it by the metric's
    signature, so what a DE metric can receive is decided by the DRIVER, not by the metric
    -- a future `de_available` entry named `purity_floor` would reach every reach spelling
    without touching the catalog at all.

    ⚠️ The spy DECLARES `purity_floor`, so the signature filter WOULD hand it one if the
    driver had one to give. That is what makes a green run evidence rather than a tautology:
    a spy without the parameter passes whatever the driver does.

    ⚠️ And its default is a SENTINEL, not `None`. `None` is the argument's own "use the
    constant" value, so a spy defaulting to it cannot tell "the driver passed nothing" from
    "the driver passed None" -- the two are indistinguishable at the callee and only one of
    them is the claim (codex checkpoint-2 P3). The live-channel half is asserted on
    `auc_pval_floor`, which the driver genuinely supplies from the config: `universe` proves
    only that the catalog partial binds, which is a different mechanism entirely.
    """
    import functools
    from dataclasses import replace

    import cell_eval2.run as run_mod
    from cell_eval2.catalog import CATALOG
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import dispatch_de_metrics

    unset = object()
    seen: list[dict] = []

    def spy(prepared=None, *, universe="adjudicated", corrected=False, purity_floor=unset,
            auc_pval_floor=None, auc_pval_floor_value=None):
        seen.append({"purity_floor": purity_floor, "auc_pval_floor": auc_pval_floor})
        return dict.fromkeys(prepared.perturbations, 1.0)

    name = "de_wilcoxon_direction_reach_raw"
    patched = dict(CATALOG)
    patched[name] = replace(CATALOG[name], func=functools.partial(
        spy, universe="adjudicated", corrected=False))
    monkeypatch.setattr(run_mod, "CATALOG", patched)

    cfg = EvalConfig()
    cfg = replace(cfg, de=replace(cfg.de, auc_pval_floor="clip"))   # a NON-default, to be seen
    pred, real = _ten_genes_last_two_wrong()
    dispatch_de_metrics([name], _prep(pred, real), cfg)
    assert seen, "the spy never ran -- the dispatch did not reach the patched spec"
    assert all(s["purity_floor"] is unset for s in seen), seen
    # ...and the channel really is live: a parameter the driver DOES supply arrives, with the
    # value this config set. So the assertion above is about the driver's kwargs dict, not
    # about a signature filter that happens to drop everything.
    assert all(s["auc_pval_floor"] == "clip" for s in seen), seen
