"""compute_lfc_nmae_reference: the split-half replicate reference for de_lfc_nmae (#208).

The unit tests here drive the pure arithmetic through `_nmae_ref_from_tables`, which takes
three DE tables and needs no cells. The one test that does need cells asserts the property
the whole reference rests on -- that the two halves' CONTROL cells are disjoint.
"""
import math

import numpy as np
import polars as pl
import pytest
from cell_eval2.lfc_nmae_ref import _nmae_ref_from_tables

_N = 12
_FEATURES = [f"g{i}" for i in range(_N)]
_REAL_LFC = [3.0, -2.0, 1.5, -1.0, 2.5, -3.5, 0.5, -0.5, 4.0, -4.0, 1.0, -1.5]
_MEAN_ABS = sum(abs(x) for x in _REAL_LFC) / _N


def _table(lfc, p_adj=None, target="A"):
    return pl.DataFrame({
        "target": [target] * _N,
        "feature": _FEATURES,
        "log2_fold_change": lfc,
        "p_value": [0.001] * _N,
        "p_adj": ([0.001] * _N) if p_adj is None else p_adj,
    })


def test_identical_halves_give_a_zero_reference():
    """A - B == 0 everywhere -> nmae_ref_sqrt2 == 0. Degenerate, but it pins the numerator."""
    out = _nmae_ref_from_tables(_table(_REAL_LFC), _table(_REAL_LFC), _table(_REAL_LFC),
                                p_adj_threshold=0.05, min_gate_size=10)
    row = out.filter(pl.col("perturbation") == "A").row(0, named=True)
    assert row["nmae_ref_raw"] == 0.0
    assert row["nmae_ref_sqrt2"] == 0.0


def test_sqrt2_correction_is_exactly_sqrt2():
    """The corrected value is the raw one divided by sqrt(2) -- the ONE arithmetic claim
    the correction makes. Spec section 3.2: Var(e_half) = 2 Var(e_full), so the ratio of
    standard deviations is sqrt(2)."""
    half_a = _table([x + 1.0 for x in _REAL_LFC])
    half_b = _table([x - 1.0 for x in _REAL_LFC])
    out = _nmae_ref_from_tables(_table(_REAL_LFC), half_a, half_b,
                                p_adj_threshold=0.05, min_gate_size=10)
    row = out.filter(pl.col("perturbation") == "A").row(0, named=True)
    # mean|A - B| = mean|2.0| = 2.0, denominator = mean|real|
    assert row["nmae_ref_raw"] == pytest.approx(2.0 / _MEAN_ABS, abs=1e-12)
    assert row["nmae_ref_sqrt2"] == pytest.approx(row["nmae_ref_raw"] / math.sqrt(2.0), abs=1e-12)


def test_gate_and_denominator_come_from_the_FULL_real_table():
    """Spec section 5.1: the gate is computed on ALL the real cells, and the denominator is
    the FULL-depth mean|lfc_real| -- not either half's. Built so a half-derived gate would
    give a different answer: the halves call every gene significant, the full table calls
    only the first 10, and the halves' own LFCs are large where the full table's are not."""
    full = _table(_REAL_LFC, p_adj=[0.001] * 10 + [0.9] * 2)
    half_a = _table([x + 1.0 for x in _REAL_LFC[:10]] + [50.0, 50.0])
    half_b = _table([x - 1.0 for x in _REAL_LFC[:10]] + [-50.0, -50.0])
    out = _nmae_ref_from_tables(full, half_a, half_b,
                                p_adj_threshold=0.05, min_gate_size=10)
    row = out.filter(pl.col("perturbation") == "A").row(0, named=True)
    mean_abs_gated = sum(abs(x) for x in _REAL_LFC[:10]) / 10
    assert row["n_gate"] == 10
    assert row["nmae_ref_raw"] == pytest.approx(2.0 / mean_abs_gated, abs=1e-12)


def test_small_gate_is_omitted():
    full = _table(_REAL_LFC, p_adj=[0.001] * 9 + [0.9] * 3)
    out = _nmae_ref_from_tables(full, _table(_REAL_LFC), _table(_REAL_LFC),
                                p_adj_threshold=0.05, min_gate_size=10)
    assert out.filter(pl.col("perturbation") == "A").height == 0


def test_genes_missing_from_a_half_are_treated_as_no_change():
    """Same convention as the member: a gene the half's DE did not report is a 0 LFC on
    that side, not a dropped gene."""
    half_a = _table(_REAL_LFC).head(6)
    out = _nmae_ref_from_tables(_table(_REAL_LFC), half_a, _table([0.0] * _N),
                                p_adj_threshold=0.05, min_gate_size=10)
    row = out.filter(pl.col("perturbation") == "A").row(0, named=True)
    expected = sum(abs(x) for x in _REAL_LFC[:6]) / _N
    assert row["nmae_ref_raw"] == pytest.approx(expected / _MEAN_ABS, abs=1e-12)


def test_the_two_halves_have_disjoint_control_cells():
    """Not a config assertion -- the actual cells. This is the property the reference
    rests on, and ceiling.py measured what its absence costs."""
    import anndata as ad
    from cell_eval2.ceiling import _disjoint_halves
    from cell_eval2.lfc_nmae_ref import _assert_disjoint_controls

    n = 40
    obs = pl.DataFrame({"target": ["non-targeting"] * 20 + ["A"] * 20}).to_pandas()
    obs.index = [f"c{i}" for i in range(n)]
    adata = ad.AnnData(X=np.ones((n, 3), dtype="float32"), obs=obs)
    a, b = _disjoint_halves(adata, "target", "non-targeting", 0)
    _assert_disjoint_controls(a, b, pert_col="target", control="non-targeting")
    ctrl_a = set(a.obs_names[a.obs["target"] == "non-targeting"])
    ctrl_b = set(b.obs_names[b.obs["target"] == "non-targeting"])
    assert ctrl_a and ctrl_b
    assert not (ctrl_a & ctrl_b)


def test_shared_control_cells_raise():
    """Mutation-proof: the guard must actually fire."""
    import anndata as ad
    from cell_eval2.lfc_nmae_ref import _assert_disjoint_controls

    def _ad(names):
        obs = pl.DataFrame({"target": ["non-targeting"] * len(names)}).to_pandas()
        obs.index = names
        return ad.AnnData(X=np.ones((len(names), 3), dtype="float32"), obs=obs)

    with pytest.raises(ValueError, match="share"):
        _assert_disjoint_controls(_ad(["c0", "c1"]), _ad(["c1", "c2"]),
                                  pert_col="target", control="non-targeting")


def test_a_gated_target_missing_from_a_half_raises():
    """REACHABLE even with every group >= 2 cells: the v2 default
    filter.filter_gene_min_cpm_cell = 5.0 inner-joins the DE frame and can remove every gene
    for a target whose CELLS survived the split. Left-joining such a target would fill both
    half vectors with 0.0 -> |A - B| = 0 -> nmae_ref_sqrt2 = 0: a perfect replicate for a target
    never measured, and then a denominator of 1 in the scaled score, while the member had
    already averaged it in."""
    full = pl.concat([_table(_REAL_LFC, target="A"), _table(_REAL_LFC, target="B")])
    half_a = _table([x + 1.0 for x in _REAL_LFC], target="A")   # B missing
    half_b = _table([x - 1.0 for x in _REAL_LFC], target="A")
    with pytest.raises(ValueError, match="missing from a split half"):
        _nmae_ref_from_tables(full, half_a, half_b,
                              p_adj_threshold=0.05, min_gate_size=10)


def test_cpm_dropout_reaches_the_raise_through_the_public_function(tmp_path, monkeypatch):
    """The same case, driven end to end with VALID cell counts -- every group has >= 2
    cells, so the precondition passes, and the raise comes from the DE backend losing the
    target on a half. Without this the raise is only ever proven on a hand-built table and
    the spec's orchestration claim is untested."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    labels = ["non-targeting"] * 8 + ["A"] * 8 + ["B"] * 8      # all >= 2 cells
    obs = pl.DataFrame({"target": labels}).to_pandas()
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)

    calls = []

    def fake_de(a, **k):
        calls.append(a.n_obs)
        targets = ["A", "B"] if len(calls) == 1 else ["A"]      # halves LOSE B
        return pl.concat([
            pl.DataFrame({"target": [t] * 12,
                          "feature": [f"g{i}" for i in range(12)],
                          "log2_fold_change": list(_REAL_LFC),
                          "p_value": [0.001] * 12, "p_adj": [0.001] * 12})
            for t in targets])

    monkeypatch.setattr(mod, "_compute_de_side", fake_de)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(ValueError, match="missing from a split half"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)


@pytest.mark.parametrize("which", ["full", "a", "b"])
def test_duplicate_rows_raise_on_each_of_the_three_tables(which):
    """A duplicate in a HALF multiplies the join and silently changes n_gate, the numerator
    AND the denominator. This function has no join-height guard, so each table is checked."""
    tables = {"full": _table(_REAL_LFC), "a": _table(_REAL_LFC), "b": _table(_REAL_LFC)}
    tables[which] = pl.concat([tables[which], tables[which].head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        _nmae_ref_from_tables(tables["full"], tables["a"], tables["b"],
                              p_adj_threshold=0.05, min_gate_size=10)


def test_half_side_non_finite_does_not_shrink_the_gate():
    """The round-1 fix applied to the member but not here. A NaN in half A and an inf in
    half B must be treated as no-change on that side, NOT row-filtered -- filtering would
    change n_gate and the denominator, so the reference would be measured over a different
    gene set than the member. Boundary case: a gate of exactly min_gate_size."""
    full = _table(_REAL_LFC, p_adj=[0.001] * 10 + [0.9] * 2)      # gate of exactly 10
    a_lfc = [float("nan")] + [x + 1.0 for x in _REAL_LFC[1:]]
    b_lfc = [float("inf")] + [x - 1.0 for x in _REAL_LFC[1:]]
    out = _nmae_ref_from_tables(full, _table(a_lfc), _table(b_lfc),
                                p_adj_threshold=0.05, min_gate_size=10)
    row = out.row(0, named=True)
    assert row["n_gate"] == 10                       # NOT 9
    # The DENOMINATOR is the point, not just the count -- filtering would have changed both,
    # so assert the exact value rather than a shape. Gate = genes 0..9; the denominator is
    # mean|real| over them, unchanged by anything on the half side. The numerator differs
    # from the clean case on gene 0 alone: |0 - 0| = 0 there instead of |1 - (-1)| = 2.
    den = sum(abs(x) for x in _REAL_LFC[:10]) / 10
    assert row["nmae_ref_raw"] == pytest.approx((2.0 * 9 / 10) / den, abs=1e-12)
    clean = _nmae_ref_from_tables(full, _table([x + 1.0 for x in _REAL_LFC]),
                                  _table([x - 1.0 for x in _REAL_LFC]),
                                  p_adj_threshold=0.05, min_gate_size=10)
    clean_row = clean.row(0, named=True)
    assert clean_row["nmae_ref_raw"] == pytest.approx(2.0 / den, abs=1e-12)
    assert row["n_gate"] == clean_row["n_gate"]


def test_compute_lfc_nmae_reference_orchestration(tmp_path, monkeypatch):
    """The PUBLIC function, not just the arithmetic helper: the three DE calls go through
    _compute_de_side with caching off, a fixed seed is reproducible and a different one is
    not, all three share the numeric target_sum resolved from the full control pool, and
    nothing is written. The aggregate is end-to-end consumable by `score_metrics`, which
    yields a `from_reference` column."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    n = 60
    obs = pl.DataFrame({"target": ["non-targeting"] * 20 + ["A"] * 40}).to_pandas()
    obs.index = [f"c{i}" for i in range(n)]
    adata = ad.AnnData(X=np.ones((n, 4), dtype="float32"), obs=obs)

    calls = []

    def fake_de(a, *, cfg, fp, store, side, reference_adata=None):
        # Derive the output from the CELLS this side actually received, not from the call
        # index. Keying on call order would make the seed-determinism assertion below pass
        # whether or not `seed` was wired to anything -- the exact class of test that
        # restates the harness instead of the behaviour.
        names = sorted(a.obs_names[a.obs["target"] == "A"])
        calls.append({"n_obs": a.n_obs, "fp": fp, "store": store, "cells": names,
                      "target_sum": cfg.target_sum})
        shift = (hash(tuple(names)) % 7) / 10.0
        return pl.DataFrame({
            "target": ["A"] * 12,
            "feature": [f"g{i}" for i in range(12)],
            "log2_fold_change": [x + shift for x in _REAL_LFC],
            "p_value": [0.001] * 12,
            "p_adj": [0.001] * 12,
        })

    monkeypatch.setattr(mod, "_compute_de_side", fake_de)
    # NON-NULL cache paths on purpose: with both None, an implementation that accidentally
    # honoured a configured cache would still pass the "nothing written" assertion below.
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path),
                     cache_real=str(tmp_path / "cache_real"),
                     cache_pred=str(tmp_path / "cache_pred"), target_sum=None)
    res, agg = mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)

    assert len(calls) == 3                                  # full + two halves
    assert all(c["fp"] is None and c["store"] is None for c in calls)
    assert all(c["target_sum"] is not None for c in calls)
    assert len({c["target_sum"] for c in calls}) == 1
    assert not (tmp_path / "cache_real").exists()
    assert not (tmp_path / "cache_pred").exists()
    assert agg["n_perturbations"][0] == 1
    assert agg["nmae_ref_sqrt2"][0] == pytest.approx(agg["nmae_ref_raw"][0] / math.sqrt(2))
    assert not list(tmp_path.iterdir())                     # nothing written
    # the two halves genuinely received DIFFERENT cells
    assert calls[1]["cells"] != calls[2]["cells"]
    assert not (set(calls[1]["cells"]) & set(calls[2]["cells"]))

    seed0_cells = [c["cells"] for c in calls]
    calls.clear()
    res2, agg2 = mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert res.equals(res2) and agg.equals(agg2)            # same seed -> same output
    assert [c["cells"] for c in calls] == seed0_cells       # ...and the SAME split

    calls.clear()
    mod.compute_lfc_nmae_reference(adata, config=cfg, seed=1)
    assert [c["cells"] for c in calls] != seed0_cells       # a different seed splits differently

    from cell_eval2.score import score_metrics
    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.9]})
    out = score_metrics(user, base, lfc_nmae_ref=agg)   # the aggregate IS consumable
    assert "from_reference" in out.columns


def test_perturbation_with_one_cell_raises_before_any_de(tmp_path, monkeypatch):
    """Spec 3.2.1, Alex 2026-08-02. _disjoint_halves drops a group iff n // 2 < 1, i.e. at
    exactly n == 1; such a target would be absent from both halves while still sitting in
    the full-real gate, so the reference would omit it while the member had already averaged
    it in. Refuse the input instead -- and refuse it BEFORE paying for any DE."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    called = []
    monkeypatch.setattr(mod, "_compute_de_side",
                        lambda *a, **k: called.append(1) or pl.DataFrame())

    labels = ["non-targeting"] * 20 + ["A"] * 20 + ["B"]      # B has ONE cell
    obs = pl.DataFrame({"target": labels}).to_pandas()
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))

    with pytest.raises(ValueError, match=r"<= 1 cell"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert called == []                                       # no DE was computed


def test_one_cell_CONTROL_also_raises(tmp_path, monkeypatch):
    """The precondition covers the control too -- `_disjoint_halves` drops it on the same
    rule, and a reference built on one control half is not a replicate of anything."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    called = []
    monkeypatch.setattr(mod, "_compute_de_side",
                        lambda *a, **k: called.append(1) or pl.DataFrame())
    labels = ["non-targeting"] + ["A"] * 20
    obs = pl.DataFrame({"target": labels}).to_pandas()
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(ValueError, match=r"<= 1 cell"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert called == []


def test_mixed_dtype_labels_are_counted_the_way_the_splitter_groups_them(tmp_path,
                                                                        monkeypatch):
    """`astype(str).value_counts()` would see one group of two and wave this through, while
    `_disjoint_halves` groups raw values and drops BOTH as one-cell groups. The precheck
    must use the splitter's own grouping, or the guarantee it exists to provide is void."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    called = []
    monkeypatch.setattr(mod, "_compute_de_side",
                        lambda *a, **k: called.append(1) or pl.DataFrame())
    labels = ["non-targeting"] * 10 + ["A"] * 10 + [1, "1"]     # two ONE-cell groups
    obs = pl.DataFrame({"target": [str(x) for x in labels]}).to_pandas()
    obs["target"] = list(labels)                                # restore the mixed dtypes
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(ValueError, match=r"<= 1 cell"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert called == []


def test_all_control_input_returns_empty_before_any_de(tmp_path, monkeypatch):
    """Classified from the FULL input. A post-split check cannot tell "there were never any
    targets" from "the split lost them", and only the second is a bug."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    called = []
    monkeypatch.setattr(mod, "_compute_de_side",
                        lambda *a, **k: called.append(1) or pl.DataFrame())
    obs = pl.DataFrame({"target": ["non-targeting"] * 8}).to_pandas()
    obs.index = [f"c{i}" for i in range(8)]
    adata = ad.AnnData(X=np.ones((8, 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    res, agg = mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert res.height == 0
    assert agg["nmae_ref_sqrt2"][0] is None and agg["n_perturbations"][0] == 0
    assert called == []


def test_null_perturbation_label_raises_before_split_or_de(tmp_path, monkeypatch):
    """Measured: pandas `groupby(..., observed=True).indices` defaults to dropna=True and
    accounts for only 4 of 6 cells when two are null -- and `_disjoint_halves` has the same
    default. So a null is invisible to the precheck AND the split, while the DE backends
    stringify labels and turn it into a target named "nan" sitting in the full-real gate
    with no group in either half. Reject it before either step can run."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    de_calls, split_calls = [], []
    monkeypatch.setattr(mod, "_compute_de_side",
                        lambda *a, **k: de_calls.append(1) or pl.DataFrame())
    monkeypatch.setattr(mod, "_disjoint_halves",
                        lambda *a, **k: split_calls.append(1) or (None, None))

    labels = ["non-targeting"] * 8 + ["A"] * 8 + [None, None]
    obs = pl.DataFrame({"target": ["x"] * len(labels)}).to_pandas()
    obs["target"] = labels
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(ValueError, match="null"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert de_calls == [] and split_calls == []


def test_stray_target_in_a_de_real_PATH_raises_before_the_all_control_return(tmp_path,
                                                                            monkeypatch):
    """Two things at once, because they share one failure: the supplied de_real must be
    NORMALIZED FROM A PATH (that is what --de-real hands us), and it must be validated
    BEFORE the all-control early return, or a stray table slips through whenever the input
    happens to be all control."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    monkeypatch.setattr(mod, "_compute_de_side", lambda *a, **k: pl.DataFrame())
    de_path = tmp_path / "de_real.csv"
    pl.DataFrame({"target": ["GHOST"] * 12, "feature": [f"g{i}" for i in range(12)],
                  "log2_fold_change": list(_REAL_LFC),
                  "p_value": [0.001] * 12, "p_adj": [0.001] * 12}).write_csv(de_path)

    obs = pl.DataFrame({"target": ["non-targeting"] * 8}).to_pandas()   # ALL control
    obs.index = [f"c{i}" for i in range(8)]
    adata = ad.AnnData(X=np.ones((8, 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(ValueError, match="no cells in the real data"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0, de_real=str(de_path))


def test_header_only_de_real_is_malformed_not_empty(tmp_path, monkeypatch):
    """A zero-row DE source must RAISE. `load_de_table` overrides only target/feature
    dtypes, so a header-only CSV loads log2_fold_change as a string and normalize_de_schema
    raises on `.abs()`. That is the right outcome -- an empty DE table cannot gate anything,
    and coercing it would produce an empty reference that reads as a data outcome."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    monkeypatch.setattr(mod, "_compute_de_side", lambda *a, **k: pl.DataFrame())
    de_path = tmp_path / "empty.csv"
    de_path.write_text("target,feature,log2_fold_change,p_value,p_adj\n")
    obs = pl.DataFrame({"target": ["non-targeting"] * 4 + ["A"] * 4}).to_pandas()
    obs.index = [f"c{i}" for i in range(8)]
    adata = ad.AnnData(X=np.ones((8, 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(Exception):        # whatever normalize_de_schema raises
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0, de_real=str(de_path))


def test_partial_de_real_warns_and_measures_only_named_targets(tmp_path, monkeypatch,
                                                               caplog):
    """A supplied full-real table may omit a target, but that narrowed coverage must be
    explicit: it cannot gate a target it never names."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    labels = ["non-targeting"] * 8 + ["A"] * 8 + ["B"] * 8
    obs = pl.DataFrame({"target": labels}).to_pandas()
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)

    def fake_de(a, **kwargs):
        return pl.concat([_table(_REAL_LFC, target="A"),
                          _table(_REAL_LFC, target="B")])

    monkeypatch.setattr(mod, "_compute_de_side", fake_de)
    de_real = _table(_REAL_LFC, target="A")
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with caplog.at_level("WARNING"):
        results, agg = mod.compute_lfc_nmae_reference(
            adata, config=cfg, seed=0, de_real=de_real,
        )

    rendered = " ".join(record.getMessage() for record in caplog.records)
    assert "no row for 1 of the 2 non-control target(s)" in rendered
    assert results["perturbation"].to_list() == ["A"]
    assert agg["n_perturbations"][0] == 1


def test_numeric_de_real_targets_are_compared_stringified(tmp_path, monkeypatch):
    """Mutation-proof for the `str(t)` in the stray check: with numeric labels on both
    sides, dropping the stringification makes every target look stray and this test fails."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    monkeypatch.setattr(mod, "_compute_de_side", lambda a, **k: pl.DataFrame({
        "target": ["7"] * 12, "feature": [f"g{i}" for i in range(12)],
        "log2_fold_change": list(_REAL_LFC),
        "p_value": [0.001] * 12, "p_adj": [0.001] * 12}))
    de_real = pl.DataFrame({"target": [7] * 12,          # INTEGER targets
                            "feature": [f"g{i}" for i in range(12)],
                            "log2_fold_change": list(_REAL_LFC),
                            "p_value": [0.001] * 12, "p_adj": [0.001] * 12})
    labels = ["non-targeting"] * 8 + [7] * 8             # integer obs labels
    obs = pl.DataFrame({"target": ["x"] * len(labels)}).to_pandas()
    obs["target"] = labels
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0, de_real=de_real)  # no raise


def test_missing_control_raises(tmp_path, monkeypatch):
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    obs = pl.DataFrame({"target": ["A"] * 4 + ["B"] * 4}).to_pandas()
    obs.index = [f"c{i}" for i in range(8)]
    adata = ad.AnnData(X=np.ones((8, 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    with pytest.raises(ValueError, match="absent from"):
        mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)


def test_two_cells_per_perturbation_is_accepted(tmp_path, monkeypatch):
    """The boundary: n == 2 gives h == 1, so the target survives into BOTH halves. Only
    n == 1 is refused -- the precondition must not be off by one."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    monkeypatch.setattr(mod, "_compute_de_side", lambda a, **k: pl.DataFrame({
        "target": ["A"] * 12, "feature": [f"g{i}" for i in range(12)],
        "log2_fold_change": list(_REAL_LFC),
        "p_value": [0.001] * 12, "p_adj": [0.001] * 12}))

    labels = ["non-targeting"] * 4 + ["A"] * 2
    obs = pl.DataFrame({"target": labels}).to_pandas()
    obs.index = [f"c{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    res, agg = mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0)
    assert agg["n_perturbations"][0] == 1


def test_backed_input_leaves_the_callers_handle_usable(tmp_path, monkeypatch):
    """`to_memory()` CLOSES the source's backing file on the anndata version pinned for
    py3.11 in CI -- baseline._materialize_reference exists to avoid exactly that. Assert the
    property it protects, not merely that materialization happened."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    labels = ["non-targeting"] * 4 + ["A"] * 4
    obs = pl.DataFrame({"target": labels}).to_pandas()
    obs.index = [f"c{i}" for i in range(len(labels))]
    path = tmp_path / "real.h5ad"
    ad.AnnData(X=np.ones((len(labels), 4), dtype="float32"), obs=obs).write_h5ad(path)

    # Through the PUBLIC function, not `_materialize_reference` directly: a direct call would
    # keep passing if compute_lfc_nmae_reference stopped using the helper, which is the whole
    # thing being guarded.
    monkeypatch.setattr(mod, "_compute_de_side", lambda a, **k: pl.DataFrame({
        "target": ["A"] * 12, "feature": [f"g{i}" for i in range(12)],
        "log2_fold_change": list(_REAL_LFC),
        "p_value": [0.001] * 12, "p_adj": [0.001] * 12}))
    backed = ad.read_h5ad(path, backed="r")
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    mod.compute_lfc_nmae_reference(backed, config=cfg, seed=0)
    assert backed.X[:].shape == (len(labels), 4)      # caller's handle still readable


def test_de_real_is_reused_when_supplied(tmp_path, monkeypatch):
    """Passing a full-real table must save exactly one DE pass -- the claim the CLI's
    --de-real passthrough and the spec's cost model both rest on."""
    import anndata as ad
    import cell_eval2.lfc_nmae_ref as mod
    from cell_eval2.config import EvalConfig

    n = 60
    obs = pl.DataFrame({"target": ["non-targeting"] * 20 + ["A"] * 40}).to_pandas()
    obs.index = [f"c{i}" for i in range(n)]
    adata = ad.AnnData(X=np.ones((n, 4), dtype="float32"), obs=obs)
    calls = []

    def fake_de(a, *, cfg, fp, store, side, reference_adata=None):
        calls.append(a.n_obs)
        return pl.DataFrame({"target": ["A"] * 12,
                             "feature": [f"g{i}" for i in range(12)],
                             "log2_fold_change": list(_REAL_LFC),
                             "p_value": [0.001] * 12, "p_adj": [0.001] * 12})

    monkeypatch.setattr(mod, "_compute_de_side", fake_de)
    cfg = EvalConfig(pert_col="target", control="non-targeting", outdir=str(tmp_path))
    supplied = fake_de(adata, cfg=cfg, fp=None, store=None, side="real")
    calls.clear()
    mod.compute_lfc_nmae_reference(adata, config=cfg, seed=0, de_real=supplied)
    assert len(calls) == 2                                  # halves only


def test_agg_emits_both_raw_and_sqrt2(graded_counts_real):
    """Both columns stay emitted so the previous number remains auditable; only the name of
    the corrected one and which one is SCORED change.

    NOT `synthetic_counts_pair`: that fixture has no differential signal, so the reference is
    EMPTY, `nmae_ref_raw` is None, and `raw / sqrt(2)` raises TypeError before asserting
    anything."""
    import math

    from cell_eval2 import EvalConfig
    from cell_eval2.lfc_nmae_ref import compute_lfc_nmae_reference

    # backend/device PINNED for the same reason tests/test_anchor.py's NMAE_KW pins them:
    # "auto" resolves differently on a GPU host and a CPU one, and these are exact-value
    # assertions over a DE-gated quantity.
    _res, agg = compute_lfc_nmae_reference(
        graded_counts_real,
        config=EvalConfig(metrics=["de_wilcoxon_lfc_nmae"], pert_col="target",
                          device="cpu", de={"backend": "pdex"}), seed=0)
    assert "nmae_ref_raw" in agg.columns
    assert "nmae_ref_sqrt2" in agg.columns
    # The one deliberate bare `nmae_ref` left in the tree: a NEGATIVE assertion that the old
    # name is gone. Task 9's rename audit expects exactly this hit and no other.
    assert "nmae_ref" not in agg.columns
    raw = agg["nmae_ref_raw"][0]
    assert raw is not None, "empty reference -- the assertion below would be vacuous"
    assert agg["nmae_ref_sqrt2"][0] == pytest.approx(raw / math.sqrt(2.0))


def test_empty_agg_still_carries_both_column_names():
    """`_empty_agg()` is the one shape `score._from_reference_column` treats as a survivable
    data outcome; it must keep the new names or that path raises 'missing column' instead."""
    from cell_eval2.lfc_nmae_ref import _empty_agg

    agg = _empty_agg()
    assert set(agg.columns) == {"statistic", "nmae_ref_raw", "nmae_ref_sqrt2",
                                "n_perturbations"}
    assert agg["nmae_ref_raw"][0] is None


@pytest.mark.parametrize("mean_raw,should_warn", [
    (1.2, True),     # 1 - 1.2 < 0: from_reference already inverts. mean_sqrt2 = 0.849, so
                     # the OLD gate (>= 1 on the sqrt2 mean) says NOTHING here.
    (1.0, True),     # exactly at the boundary: 1 - 1.0 == 0, not positive
    (0.9, False),    # healthy under the new rule; the OLD rule agreed
    # both rules warn -- keeps the parametrization from only testing the gap. NOTE 1.42, not
    # 1.41: 1.41/sqrt(2) = 0.9970 < 1, so at 1.41 the OLD rule is still silent and this case
    # would be testing the gap over again. sqrt(2) = 1.41421 is the crossover.
    (1.42, True),
])
def test_degeneracy_warning_thresholds_on_the_RAW_mean(caplog, mean_raw, should_warn):
    """The warning must fire on the quantity `score` divides by.

    `mean_raw = mean_sqrt2 * sqrt(2)`, so the old gate fires LATE: at mean_raw = 1.2 the
    reference is already past the point where `1 - nmae_ref_raw` goes non-positive and
    `from_reference` inverts, while mean_sqrt2 is only 0.849 and the old threshold is silent.

    Tested on the extracted helper rather than end to end: `compute_lfc_nmae_reference` runs
    three DE passes before reaching the warn, and a test that has to stub its way past them
    ends up asserting nothing."""
    import logging

    from cell_eval2.lfc_nmae_ref import _warn_if_degenerate

    with caplog.at_level(logging.WARNING):
        _warn_if_degenerate(mean_raw)
    fired = any("nmae_ref_raw" in r.getMessage() for r in caplog.records)
    assert fired is should_warn, caplog.text
    if should_warn:
        assert f"{mean_raw:.4f}" in caplog.text     # names the RAW value, not the sqrt2 one
