"""The DE gene gate keeps a gene on the REFERENCE group alone -- issue #351.

Under `rule_version` 2 the gate kept a (target, gene) row when the TARGET group's mean CPM
cleared the threshold **OR** the control's did. For a gene at or below the threshold in the
control, the row `(t, g)` then existed only when `g` had risen above the threshold in `t`, so

    tmean > threshold >= ref_mean  ==>  (tmean + eps)/(ref_mean + eps) > 1  ==>  log2FC(t, g) > 0

and a row's mere PRESENCE disclosed its log2FC's sign. (#351 writes the stronger bound
`log2FC > log2(threshold/ref_mean)`; the pseudocount breaks it -- `(threshold+eps)/(ref_mean+eps)
<= threshold/ref_mean` whenever `ref_mean <= threshold` -- and it is undefined at `ref_mean = 0`.
Strict positivity is what holds, and what the measurement below confirms.) Measured on the three official val
panels: `P(real log2FC > 0) = 1.000000` over 26,373 / 33,969 / 26,839 such rows, and a
submission that pasted control cells with counts added to exactly those genes -- the same block
for all 300 targets, reading no perturbation-specific information -- measured
`de_wilcoxon_direction_fidelity_yield_raw` 0.689661 / 0.764160 / 0.688485 against baselines
0.505647 / 0.522736 / 0.509365 (+0.3722 / +0.5059 / +0.3651 `from_baseline`, +0.1057 of OVERALL
`avg_score`). Under the reference-only gate those arms read 0.001249 / 0.001124 / 0.000272 with
`n_pred` 0.0 -- the honest control-paste floor.

TWO code paths implement the one semantics, because the gate that leaked was never
`_apply_cpm_filter`:

  * CPU backends (pdex/scanpy) -- `de_compute._apply_cpm_filter`, `kept = genes[ref_cpm > t]`.
  * gpudge -- `compute_de` returns from its gpudge branch BEFORE the CPU gate runs, and gpudge's
    own `filter_gene_min_cpm_cell` is the OR (`gpudge._filter.combined_keep_mask`: "AND each
    active filter's (target OR ref) mask"). cell_eval2 therefore takes the reference-only
    decision itself, in `_finalize_gpudge_de`, by one of TWO routes that `_gpudge_gate_plan`
    picks between: the FRAME route compares the returned frame's own `ref_mean * 1e6/target_sum`
    (bit-exact, and the competition's route), while the MATRIX route compares a per-cell-CPM
    vector derived from the reference cells -- needed wherever `ref_mean` is in the wrong unit
    (lognorm, whose `_to_linear` never applies `target_sum`) or is the wrong statistic (a
    geometric `mean_calc`). gpudge is unchanged; its own gate stays on ONLY where it nests
    exactly, and is muted otherwise. The official bundles are gpudge/cuda runs, so THIS is the
    path that carries the fix; a test that only exercised `_apply_cpm_filter` would pass while
    the leak stayed open.

Section F pins actual numbers end to end (#338, "no golden end-to-end scoring test"). It runs on
the pdex CPU backend, so it pins the metric arithmetic AND the gate semantics -- not the
competition number, which needs gpudge on CUDA and is #338's option (a)/(c), still open.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import false_discovery_control

from cell_eval2.de import prepare_de
from cell_eval2.de_compute import (
    _apply_cpm_filter,
    _bh_per_target,
    _finalize_gpudge_de,
    _gpudge_gate_plan,
    _ref_cpm_from_cells,
    _resolve_cpm_filter,
    _to_linear,
    compute_de,
)

CONTROL = "non-targeting"
TARGETS = ("GENE1", "GENE2", "GENE3")
#: Every cell carries exactly this library, so a gene's CPM is `count / LIB * 1e6` with no
#: normalization slack -- the gate's threshold can then be placed EXACTLY between two genes
#: rather than approximately, which is what makes "absent from every target" falsifiable.
LIB = 1000
N_CELLS = 12
#: 3 counts of a 1000-count library = 3,000 CPM. Chosen so the fixture below straddles it.
THRESHOLD = 3_000.0


# ----------------------------------------------------------------------- the shared fixture

def _counts(seed: int = 351):
    """(adata, control_cpm) -- a deterministic counts panel straddling `THRESHOLD`.

    `seed` moves only the per-cell jitter, so two seeds give two honest replicates of the same
    biology -- which is what section F needs for a submission-vs-reference pair.

    Every cell carries exactly `LIB` counts, so a gene's CPM is `count / LIB * 1e6` with no
    normalization slack: the threshold sits EXACTLY between two genes rather than approximately,
    which is what makes "absent from every target" falsifiable rather than probable.

    Five genes carry the whole argument:

      * `always_in`   -- 5 counts in every group -> 5,000 CPM. Above the threshold in the control,
        so it survives under both the old OR and the new reference-only rule.
      * `leak_gene`   -- 1 count in the control (1,000 CPM, BELOW) rising to 8 in GENE1 (8,000
        CPM, ABOVE), and `leak_gene2` -- 2 in the control rising to 7 in GENE2. These are #351's
        genes: the OR admitted each for exactly the perturbation that raised it, where its log2FC
        is positive by construction. The reference-only rule drops them from EVERY target.
      * `always_out`  -- 0 counts everywhere. Dropped under both rules; present so a test can tell
        "dropped because the control is low" from "dropped because nothing is expressed".
      * `down_gene`   -- 6 counts in the control (6,000 CPM, ABOVE) collapsing to 1 in GENE2. The
        mirror image, and the reason the OR leaked a SIGN rather than merely a set: a DOWN move is
        only ever visible when the control already clears the threshold, so the rows the OR
        admitted on the TARGET's account could only be up-regulations.

    Each target gene is expressed at 10 counts (10,000 CPM) in every group and knocked down to 1
    in its own -- both because that is what a knockdown looks like and because the target genes
    must SURVIVE the gate: `prepare_de` resolves each target to a feature of the table and raises
    if none resolves (the #195 fixture trap). `filler`/`filler2` absorb the rest of the library,
    `filler2` taking a per-cell jitter so the ranksum is not degenerate on every gene at once.
    """
    genes = (["always_in", "leak_gene", "leak_gene2", "always_out", "down_gene",
              "filler", "filler2"] + list(TARGETS))
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for grp in (CONTROL,) + TARGETS:
        for _ in range(N_CELLS):
            cell = dict.fromkeys(genes, 0)
            cell["always_in"] = 5
            cell["leak_gene"] = 8 if grp == "GENE1" else 1
            cell["leak_gene2"] = 7 if grp == "GENE2" else 2
            cell["always_out"] = 0
            cell["down_gene"] = 1 if grp == "GENE2" else 6
            for t in TARGETS:                      # knocked down in its own perturbation
                cell[t] = 1 if t == grp else 10
            cell["filler2"] = 20 + int(rng.integers(0, 3))
            cell["filler"] = LIB - sum(v for k, v in cell.items() if k != "filler")
            rows.append([cell[g] for g in genes])
            labels.append(grp)
    X = np.asarray(rows, dtype=np.float32)
    assert (X.sum(axis=1) == LIB).all(), "every cell must carry the same library"
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=genes),
    )
    ctrl = X[np.asarray(labels) == CONTROL]
    control_cpm = dict(zip(genes, ctrl.mean(axis=0) / LIB * 1e6))
    return adata, control_cpm


def _linear_and_frame():
    """(`linear` AnnData already in CPM, a full DE frame over it) for `_apply_cpm_filter`.

    `_apply_cpm_filter` takes the NORMALIZED matrix the LFC was computed from, so normalize the
    counts to `LIB` -> CPM here (cpm_factor 1.0) and hand it every (target, gene) row.
    """
    adata, control_cpm = _counts()
    X = adata.X / LIB * 1e6
    linear = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
    genes = list(adata.var_names)
    rng = np.random.default_rng(7)
    rows = []
    for t in TARGETS:
        for g in genes:
            # p_values spread over (0, 1) so BH's step-up has something to do, and DISTINCT per
            # (target, gene) so a re-adjustment is visible rather than coincidental.
            rows.append((t, g, float(rng.uniform(0.001, 0.9))))
    df = pl.DataFrame(
        {"target": [r[0] for r in rows], "feature": [r[1] for r in rows],
         "p_value": [r[2] for r in rows]},
        schema={"target": pl.Utf8, "feature": pl.Utf8, "p_value": pl.Float64},
    ).with_columns(p_adj=pl.col("p_value"), log2_fold_change=pl.lit(1.0))
    return linear, df, control_cpm


# ================================================================ A. the CPU gate

def test_kept_set_is_the_control_alone_and_identical_for_every_target():
    """The invariant #351 asks for: one gene set, a property of the evaluation data."""
    linear, df, control_cpm = _linear_and_frame()
    out = _apply_cpm_filter(df, linear, groupby="target", reference=CONTROL,
                            threshold=THRESHOLD)
    per_target = {t: set(sub["feature"].to_list())
                  for t, sub in out.group_by("target", maintain_order=True)
                  for t in [t[0]]}
    assert len(per_target) == len(TARGETS)
    assert len(set(map(frozenset, per_target.values()))) == 1, per_target
    expected = {g for g, cpm in control_cpm.items() if cpm > THRESHOLD}
    assert next(iter(per_target.values())) == expected


def test_the_leak_gene_is_absent_from_every_target_including_its_own():
    """`leak_gene` clears the threshold ONLY in GENE1, where its log2FC is positive by
    construction. The OR admitted exactly that row; the reference-only rule admits none."""
    linear, df, control_cpm = _linear_and_frame()
    assert control_cpm["leak_gene"] <= THRESHOLD          # below in the control ...
    tgt_cpm = {t: float(np.asarray(linear.X)[linear.obs["target"].to_numpy() == t]
                        [:, list(linear.var_names).index("leak_gene")].mean())
               for t in TARGETS}
    assert tgt_cpm["GENE1"] > THRESHOLD                   # ... above in GENE1's own cells
    out = _apply_cpm_filter(df, linear, groupby="target", reference=CONTROL,
                            threshold=THRESHOLD)
    assert out.filter(pl.col("feature") == "leak_gene").height == 0


def test_a_control_expressed_gene_is_kept_for_every_target_even_where_it_falls():
    """`down_gene` is above the threshold in the control and collapses in GENE2. It stays --
    which is the point: a DOWN move is only ever scoreable when the control carries the gene."""
    linear, df, _ = _linear_and_frame()
    out = _apply_cpm_filter(df, linear, groupby="target", reference=CONTROL,
                            threshold=THRESHOLD)
    kept = out.filter(pl.col("feature") == "down_gene")
    assert sorted(kept["target"].unique().to_list()) == sorted(TARGETS)
    assert out.filter(pl.col("feature") == "always_out").height == 0   # nothing expressed


def test_negative_threshold_is_still_keep_all():
    linear, df, _ = _linear_and_frame()
    out = _apply_cpm_filter(df, linear, groupby="target", reference=CONTROL, threshold=-1.0)
    assert out.height == df.height
    assert set(out["feature"].to_list()) == set(df["feature"].to_list())


def test_bh_is_recomputed_per_target_over_the_survivors():
    """Load-bearing, not bookkeeping: BH reads `i/m` over the rows actually tested, so the
    incoming `p_adj` describes the WRONG universe once rows are dropped."""
    linear, df, _ = _linear_and_frame()
    out = _apply_cpm_filter(df, linear, groupby="target", reference=CONTROL,
                            threshold=THRESHOLD)
    for t in TARGETS:
        sub = out.filter(pl.col("target") == t)
        expect = false_discovery_control(sub["p_value"].to_numpy(), method="bh")
        assert np.allclose(sub["p_adj"].to_numpy(), expect, rtol=0, atol=0)
        # and NOT the p_adj that came in (which was BH over the full gene axis)
        incoming = (df.filter(pl.col("target") == t)
                    .join(sub.select("feature"), on="feature", how="inner")["p_adj"].to_numpy())
        assert not np.allclose(sub["p_adj"].to_numpy(), incoming)


def test_the_gate_scales_by_cpm_factor_not_by_the_raw_mean():
    """`linear` is normalized to `target_sum`, so the compare needs `1e6/target_sum` (F4.1).
    Same matrix at target_sum=1e4 must keep the same genes once the factor is applied."""
    linear, df, control_cpm = _linear_and_frame()
    small = ad.AnnData(X=np.asarray(linear.X) / 100.0, obs=linear.obs.copy(),
                       var=linear.var.copy())          # normalized to 1e4 instead of 1e6
    out_1e6 = _apply_cpm_filter(df, linear, groupby="target", reference=CONTROL,
                                threshold=THRESHOLD, cpm_factor=1.0)
    out_1e4 = _apply_cpm_filter(df, small, groupby="target", reference=CONTROL,
                                threshold=THRESHOLD, cpm_factor=100.0)
    assert set(out_1e4["feature"].to_list()) == set(out_1e6["feature"].to_list())
    # ... and gating the 1e4 matrix WITHOUT the factor compares a 100x-too-small number, which
    # drops genes that clear the threshold in true CPM. `always_in` is the witness: 5,000 CPM,
    # so 50 unscaled.
    unscaled = _apply_cpm_filter(df, small, groupby="target", reference=CONTROL,
                                 threshold=THRESHOLD, cpm_factor=1.0)
    assert control_cpm["always_in"] > THRESHOLD
    assert "always_in" in set(out_1e4["feature"].to_list())
    assert "always_in" not in set(unscaled["feature"].to_list())
    assert unscaled.height < out_1e4.height


def test_bh_per_target_preserves_row_order():
    """A gated table keeps the order the engine emitted, so a caller can diff frames without
    sorting first (the group-wise concat this replaced did not)."""
    _, df, _ = _linear_and_frame()
    out = _bh_per_target(df)
    assert out["target"].to_list() == df["target"].to_list()
    assert out["feature"].to_list() == df["feature"].to_list()


def test_bh_per_target_holds_nan_p_out_of_the_set():
    """scipy's false_discovery_control RAISES on NaN, and wilcoxon emits NaN on constant genes,
    so the v2-default filter path would crash if they were passed through (Gemini re-review)."""
    df = pl.DataFrame({
        "target": ["A", "A", "A", "B"],
        "feature": ["g0", "g1", "g2", "g0"],
        "p_value": [0.01, float("nan"), 0.5, 0.2],
        "p_adj": [0.0, 0.0, 0.0, 0.0],
    })
    out = _bh_per_target(df)
    got = out["p_adj"].to_numpy()
    assert np.isnan(got[1])                                     # NaN in -> NaN out
    expect_a = false_discovery_control(np.array([0.01, 0.5]), method="bh")
    assert np.allclose(got[[0, 2]], expect_a)                   # BH over A's two valid rows
    assert got[3] == pytest.approx(0.2)                         # B alone -> m=1


# ================================================================ B. the gpudge gate plan

def _plan(threshold=5.0, *, mean_calc="arithmetic", target_sum=1e6, gpudge_normalized=True,
          ref_cells=None, var_names=()):
    return _gpudge_gate_plan(threshold, mean_calc=mean_calc, target_sum=target_sum,
                             gpudge_normalized=gpudge_normalized, ref_cells=ref_cells,
                             var_names=var_names, where="t")


def test_the_competition_settings_take_the_frame_route():
    """v2/vcc2026: counts, target_sum=1e6, arithmetic, and gpudge does the normalizing -- so the
    frame's `ref_mean` IS the array gpudge's own gate scales, the factor is exactly 1.0, and its
    own gate may stay on."""
    assert _plan() == (5.0, 1.0, None, True)


def test_the_frame_route_rescales_a_non_cpm_normalization_target():
    assert _plan(target_sum=1e4) == (5.0, 100.0, None, True)


def test_counts_prenormalized_by_cell_eval2_take_the_matrix_route():
    """A NOMINALLY uniform library is not enough for the frame route. `_to_linear` does normalize
    counts to `target_sum`, so `ref_mean * 1e6/target_sum` is ALGEBRAICALLY the per-cell CPM -- but
    gpudge downcasts the staged values to float32, so the staged row totals need not be exactly
    `target_sum`, and its gate reads a separate accumulator regardless. The frame route claims to be
    gpudge's own compare, not merely equal to it, so this case goes to the matrix."""
    adata, _ = _counts()
    thr, factor, ref_cpm, keep = _plan(target_sum=1e4, gpudge_normalized=False,
                                       ref_cells=adata.X, var_names=list(adata.var_names))
    assert thr == 5.0 and factor == 1.0 and keep is False
    assert ref_cpm is not None


def test_a_negative_threshold_is_keep_all_and_never_picks_a_route():
    """`FilterParams` documents a negative threshold as the explicit keep-all. It must not raise on
    a path where no route is available -- keeping every gene needs neither a vector nor a rescale."""
    assert _plan(-1.0, mean_calc="geometric", target_sum=None,
                 gpudge_normalized=False, ref_cells=None) == (-1.0, 1.0, None, False)


def test_no_filter_is_inert_on_every_combination():
    """v1 sets filter_gene_min_cpm_cell=None, and it is ALSO geometric and median-normalized. The
    plan must not raise, not read a matrix and not gate -- the frozen 2025 profile cannot move."""
    assert _plan(None, mean_calc="geometric", target_sum=None,
                 gpudge_normalized=False) == (None, 1.0, None, False)


def test_a_geometric_mean_calc_takes_the_matrix_route():
    """`ref_mean` is the GEOMETRIC control mean there. gpudge does emit that column -- what it
    withholds is the ARITHMETIC reference mean its own gate compares, accumulated separately -- so
    the emitted value is the wrong statistic and the plan reads the reference cells instead of
    refusing (the shipped `cell-eval-0.7.6` preset is geometric WITH the filter on, and must keep
    its gate)."""
    adata, _ = _counts()
    thr, factor, ref_cpm, keep = _plan(mean_calc="geometric", ref_cells=adata.X,
                                       var_names=list(adata.var_names))
    assert thr == 5.0 and factor == 1.0 and keep is False
    assert ref_cpm is not None and len(ref_cpm) == adata.n_vars


def test_lognorm_input_takes_the_matrix_route():
    """The starkest case: `_to_linear` only applies `expm1` to lognorm and never touches
    `target_sum`, so `ref_mean` sits in the USER's own scale and `1e6/target_sum` is not even
    algebraically its rescale."""
    adata, _ = _counts()
    _, _, ref_cpm, keep = _plan(gpudge_normalized=False, ref_cells=adata.X,
                                var_names=list(adata.var_names))
    assert ref_cpm is not None and keep is False


def test_a_streamed_reference_with_no_frame_route_refuses():
    """`compute_de_streaming`'s reference shard lives inside the archive, so there are no cells to
    derive a CPM vector from. Gating on the wrong quantity is the one thing it must not do."""
    with pytest.raises(NotImplementedError, match="decided by the REFERENCE group alone") as e:
        _plan(mean_calc="geometric", ref_cells=None)
    assert "Mode 2" in str(e.value)   # the message must not recommend something that also fails
    with pytest.raises(NotImplementedError):
        _plan(target_sum=None, ref_cells=None)


def test_the_reference_cpm_vector_is_normalization_invariant():
    """The property that makes one definition serve every gpudge sub-path: `x_ig / L_i` does not
    move under a per-cell rescaling, so raw counts, CPM, a 1e4-normalized matrix and expm1'd
    lognorm all give the same vector -- and cell_eval2 holds a different one of those on each."""
    adata, _ = _counts()
    raw = _ref_cpm_from_cells(adata.X, n_genes=adata.n_vars)
    cpm = _ref_cpm_from_cells(_to_linear(adata, "counts", 1e6).X, n_genes=adata.n_vars)
    small = _ref_cpm_from_cells(_to_linear(adata, "counts", 1e4).X, n_genes=adata.n_vars)
    lognorm = _to_linear(adata, "counts", 1e4)
    lognorm.X = np.log1p(np.asarray(lognorm.X))
    expm1 = _ref_cpm_from_cells(_to_linear(lognorm, "lognorm", 1e4).X, n_genes=adata.n_vars)
    for other in (cpm, small, expm1):
        assert np.allclose(raw, other, rtol=1e-9, atol=1e-9), (raw, other)


def test_the_two_routes_agree_on_the_competition_scale():
    """The frame route is kept only because it is bit-exact on the competition path; it must not
    be a DIFFERENT gate. On a CPM-normalized matrix the arithmetic control mean (what gpudge emits
    as `ref_mean`, factor 1.0) and the matrix route's per-cell-CPM mean are the same number."""
    adata, _ = _counts()
    linear = _to_linear(adata, "counts", 1e6)
    ctrl = np.asarray(linear.X)[linear.obs["target"].to_numpy() == CONTROL]
    frame_quantity = ctrl.mean(axis=0)                       # gpudge's ref_mean * 1.0
    matrix_quantity = _ref_cpm_from_cells(ctrl, n_genes=adata.n_vars)
    assert np.allclose(frame_quantity, matrix_quantity, rtol=1e-9, atol=1e-9)


def _rowwise_oracle(X):
    """`mean_i(x_ig / L_i * 1e6)` written out one cell at a time, as the definition reads."""
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    acc = np.zeros(X.shape[1])
    for row in X:
        lib = row.sum()
        acc += row / (lib if lib > 0 else 1.0) * 1e6
    return acc / X.shape[0]


def _heterogeneous_pool(seed=17, n_cells=40, n_genes=12):
    """A reference pool with DELIBERATELY unequal per-cell libraries, one all-zero cell, and a
    sparse twin. Equal-library cells would make the invariance claim vacuous -- the per-cell divisor
    is the whole point -- and a zero-library cell is the edge the guard exists for."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64)
    X *= rng.choice([1.0, 4.0, 25.0], size=(n_cells, 1))     # libraries spread over ~25x
    X[3] = 0.0                                               # a zero-library cell
    return X, sp.csr_matrix(X)


def test_the_reference_cpm_vector_matches_a_rowwise_oracle_on_uneven_libraries():
    """Dense and sparse, against the definition written out per cell. `w @ X` is an algebraic
    rearrangement of that loop, so it has to agree -- and unequal libraries plus a zero-library
    cell are what make the test non-vacuous."""
    dense, sparse = _heterogeneous_pool()
    want = _rowwise_oracle(dense)
    got_dense = _ref_cpm_from_cells(dense, n_genes=dense.shape[1])
    got_sparse = _ref_cpm_from_cells(sparse, n_genes=dense.shape[1])
    assert np.allclose(got_dense, want, rtol=1e-12, atol=0)
    assert np.allclose(got_sparse, want, rtol=1e-12, atol=0)


def test_the_reference_cpm_vector_is_invariant_under_HETEROGENEOUS_per_cell_rescaling():
    """The claim in the docstring, tested where it can actually fail: scale each cell by its OWN
    random factor. A global rescale would pass even if the divisor were dropped."""
    dense, sparse = _heterogeneous_pool()
    rng = np.random.default_rng(99)
    factors = rng.uniform(0.05, 20.0, size=(dense.shape[0], 1))
    rescaled = dense * factors
    base = _ref_cpm_from_cells(dense, n_genes=dense.shape[1])
    assert np.allclose(_ref_cpm_from_cells(rescaled, n_genes=dense.shape[1]), base,
                       rtol=1e-10, atol=0)
    import scipy.sparse as sp
    assert np.allclose(_ref_cpm_from_cells(sp.csr_matrix(rescaled), n_genes=dense.shape[1]), base,
                       rtol=1e-10, atol=0)


def test_the_reference_cpm_vector_chunks_the_dense_path_without_changing_the_answer(monkeypatch):
    """The dense path walks cell chunks to avoid a whole-matrix float64 transient. Chunking must be
    an implementation detail, so shrink the budget until it actually bites -- at the shipped
    20M-element budget a 7-gene pool would need ~2.9M cells before the loop ran twice, so a fixture
    alone cannot exercise it."""
    from cell_eval2 import de_compute as dc

    dense, _ = _heterogeneous_pool(n_cells=97, n_genes=7)
    want = _rowwise_oracle(dense)
    assert np.allclose(_ref_cpm_from_cells(dense, n_genes=7), want, rtol=1e-12, atol=0)
    monkeypatch.setattr(dc, "_REF_CPM_DENSE_ELEMENTS", 21)      # 3 cells per chunk -> 33 chunks
    assert np.allclose(dc._ref_cpm_from_cells(dense, n_genes=7), want, rtol=1e-12, atol=0)
    monkeypatch.setattr(dc, "_REF_CPM_DENSE_ELEMENTS", 1)        # degenerate: one cell per chunk
    assert np.allclose(dc._ref_cpm_from_cells(dense, n_genes=7), want, rtol=1e-12, atol=0)


def test_the_reference_cpm_vector_holds_on_a_high_dynamic_range_float32_sparse_pool():
    """float32 storage with a wide dynamic range, against the float64 oracle.

    ⚠️ This does NOT prove that a float32 accumulator would have failed here -- measured on scipy
    1.18 a float32 CSR's own `.sum(axis=1)` already agreed with float64 on a case where the DENSE
    float32 reduction lost 12 counts. What it pins is that the vector equals the float64 answer for
    float32 STORAGE, which is the property `_ref_cpm_from_cells` guarantees by widening `.data`
    rather than inheriting from scipy's choice of accumulator."""
    import scipy.sparse as sp

    rng = np.random.default_rng(5)
    X = rng.poisson(2.0, size=(24, 9)).astype(np.float32)
    X[:, 0] = 2.0 ** 22                      # one gene orders of magnitude above the rest
    X[7] = 0.0                               # and a zero-library cell
    got = _ref_cpm_from_cells(sp.csr_matrix(X), n_genes=9)
    assert np.allclose(got, _rowwise_oracle(X.astype(np.float64)), rtol=1e-12, atol=0)


def test_the_reference_cpm_vector_refuses_an_empty_pool():
    import scipy.sparse as sp
    with pytest.raises(ValueError, match="empty"):
        _ref_cpm_from_cells(np.zeros((0, 5)), n_genes=5)
    with pytest.raises(ValueError, match="empty"):
        _ref_cpm_from_cells(sp.csr_matrix((0, 5), dtype=np.float64), n_genes=5)


def test_the_reference_cpm_vector_refuses_a_shape_slip():
    adata, _ = _counts()
    with pytest.raises(ValueError, match="expected"):
        _ref_cpm_from_cells(adata.X, n_genes=adata.n_vars + 1)


@pytest.mark.parametrize("preset,route", [
    ("vcc2026", "frame"),            # the competition: counts, 1e6, arithmetic
    ("v2", "frame"),                 # same knobs
    ("v1", "off"),                   # filter None outright -> the frozen 2025 profile is safe
    ("cell-eval-0.7.6", "matrix"),   # filter 5.0, geometric AND lognorm -> decided from the cells
])
def test_every_shipped_preset_resolves_to_a_gate_and_none_of_them_refuses(preset, route):
    """The claim the refusal rests on, machine-checked rather than asserted in a docstring: no
    shipped preset can reach it. `cell-eval-0.7.6` is why the matrix route exists -- it ships
    `filter_gene_min_cpm_cell = 5.0` with `mean_calc='geometric'`, `input_type='lognorm'` and
    `target_sum=1e4`, so refusing (or dropping the gate) would have changed a shipped preset."""
    from cell_eval2.config import EvalConfig

    cfg = EvalConfig.from_preset(preset)
    adata, _ = _counts()
    eff = _resolve_cpm_filter(cfg.filter.filter_gene_min_cpm_cell, input_type=cfg.input_type,
                              resolved_backend="gpudge")
    thr, factor, ref_cpm, _keep = _gpudge_gate_plan(
        eff, mean_calc=cfg.de.mean_calc, target_sum=cfg.target_sum,
        # counts at 1e6 are handed to gpudge raw with cpm_normalize=True, so gpudge normalizes;
        # lognorm is pre-normalized on the CPU, which is what routes `cell-eval-0.7.6` to the matrix.
        gpudge_normalized=(cfg.input_type == "counts" and cfg.target_sum == 1e6),
        ref_cells=adata.X,
        var_names=list(adata.var_names), where=f"{preset} preset")
    if route == "off":
        assert (thr, factor, ref_cpm) == (None, 1.0, None)
    elif route == "frame":
        assert (thr, factor) == (5.0, 1.0) and ref_cpm is None
    else:
        assert thr == 5.0 and ref_cpm is not None


def test_the_resolver_policy_is_unchanged_by_this_fix():
    """`_resolve_cpm_filter` keeps its two pre-existing rules and gains none: the gate applies on
    counts for every backend, and on lognorm only for gpudge (whose gate is normalization-
    invariant). #351 is resolved downstream of it, by the plan above."""
    assert _resolve_cpm_filter(5.0, input_type="counts", resolved_backend="pdex") == 5.0
    assert _resolve_cpm_filter(5.0, input_type="lognorm", resolved_backend="gpudge") == 5.0
    assert _resolve_cpm_filter(5.0, input_type="lognorm", resolved_backend="pdex") is None
    assert _resolve_cpm_filter(None, input_type="counts", resolved_backend="gpudge") is None


# ================================================================ C. the gpudge-side gate

def _gpudge_frame():
    """gpudge's raw schema over two targets. `ref_mean` is ONE value per gene (the shared
    control group), which is what makes the gate's decision target-independent.

    `boosted` carries the #351 attack's signature: a below-threshold-in-control gene with a
    crushing p-value. `marginal` is only significant BECAUSE `boosted` is in the BH pool -- the
    miniature of "n_pred goes to 0, not to ~4".
    """
    ref_mean = {"kept_hi": 40.0, "kept_mid": 6.0, "boosted": 4.0, "dropped_lo": 0.5}
    rows = []
    for t in ("T1", "T2"):
        for g, rm in ref_mean.items():
            p = {"kept_hi": 0.030, "kept_mid": 0.040, "boosted": 1e-20,
                 "dropped_lo": 0.9}[g]
            rows.append((t, g, rm, p))
    return pl.DataFrame({
        "target": [r[0] for r in rows],
        "feature": [r[1] for r in rows],
        "target_mean": [r[2] * 2 for r in rows],
        "ref_mean": [r[2] for r in rows],
        "log2_fold_change": [1.0] * len(rows),
        "p_value": [r[3] for r in rows],
        "p_adj": [r[3] for r in rows],
    })


def test_gpudge_gate_drops_below_threshold_genes_from_every_target():
    out = _finalize_gpudge_de(_gpudge_frame(), epsilon=1e-9, clip_value=None,
                              fdr_scope="per_pert", cpm_threshold=5.0, cpm_factor=1.0)
    assert set(out["feature"].to_list()) == {"kept_hi", "kept_mid"}
    assert sorted(out["target"].unique().to_list()) == ["T1", "T2"]
    # every surviving target keeps the SAME gene set
    assert len({frozenset(sub["feature"].to_list())
                for _k, sub in out.group_by("target")}) == 1


def _bh_collapse_frame():
    """The #351 attack's BH signature at fixture scale, on one target.

    20 `boost*` rows sit BELOW the threshold in the control with crushing p-values -- the block
    the arm manufactured. `marginal` is a survivor whose raw p of 0.03 clears alpha ONLY because
    those 20 rows push it to rank 21 of 29: BH reads `m/i · p`, so removing them drops its rank
    to 1 of 9 and its adjusted p from 0.041 to 0.27. Eight `tail*` survivors supply the large
    p-values that make the step-up's min bite.
    """
    rows = [(f"boost{i}", 4.0, 1e-20) for i in range(20)]
    rows += [("marginal", 6.0, 0.03)]
    rows += [(f"tail{i}", 40.0, 0.5 + 0.05 * i) for i in range(8)]
    return pl.DataFrame({
        "target": ["T1"] * len(rows),
        "feature": [r[0] for r in rows],
        "target_mean": [r[1] * 4 for r in rows],
        "ref_mean": [r[1] for r in rows],
        "log2_fold_change": [2.0] * len(rows),
        "p_value": [r[2] for r in rows],
        "p_adj": [r[2] for r in rows],
    })


def test_gpudge_gate_readjusts_p_adj_so_the_boosted_block_stops_carrying_the_marginal():
    """The mechanism behind `n_pred` 76.4 -> 0.0, and the reason the BH recomputation is
    load-bearing rather than bookkeeping.

    ⚠️ `before` is BH over the FULL universe -- what gpudge itself computed over its OR-kept set,
    which included the boosted rows because they clear the threshold in the TARGET group. It is
    NOT the `cpm_threshold=None` passthrough, which would just hand back the p_adj the fixture
    set."""
    frame = _bh_collapse_frame()
    before = dict(zip(_bh_per_target(frame)["feature"], _bh_per_target(frame)["p_adj"]))
    assert before["marginal"] < 0.05        # significant while the 1e-20 block is in the pool

    after_df = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                                   cpm_threshold=5.0, cpm_factor=1.0)
    after = dict(zip(after_df["feature"], after_df["p_adj"]))
    assert "boost0" not in after                                   # the block is gone ...
    assert after["marginal"] >= 0.05                               # ... and so is the call
    # exactly scipy's BH over what survived -- nothing crosses a target boundary
    surv = frame.filter(pl.col("ref_mean") > 5.0)["p_value"].to_numpy()
    assert np.allclose(after_df["p_adj"].to_numpy(),
                       false_discovery_control(surv, method="bh"))


def test_gpudge_gate_is_a_passthrough_when_no_threshold_is_given():
    """v1 and every non-filtering config must be byte-identical to before the fix."""
    frame = _gpudge_frame()
    out = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                             cpm_threshold=None)
    assert out.height == frame.height
    assert out["p_adj"].to_list() == frame["p_adj"].to_list()   # p_adj untouched


def test_gpudge_gate_negative_threshold_is_keep_all():
    frame = _gpudge_frame()
    out = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                              cpm_threshold=-1.0, cpm_factor=1.0)
    assert out.height == frame.height


def test_gpudge_gate_applies_the_cpm_factor():
    """`ref_mean` sits in `target_sum`-normalized units; at target_sum=1e4 the same genes must
    survive once the 100x rescale is applied, and none survive without it."""
    frame = _gpudge_frame().with_columns(pl.col("ref_mean") / 100.0)
    out = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                              cpm_threshold=5.0, cpm_factor=100.0)
    assert set(out["feature"].to_list()) == {"kept_hi", "kept_mid"}
    assert _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                               cpm_threshold=5.0, cpm_factor=1.0).height == 0


def test_gpudge_gate_preserves_row_order():
    frame = _gpudge_frame()
    out = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                              cpm_threshold=5.0, cpm_factor=1.0)
    keep = frame.filter(pl.col("ref_mean") > 5.0)
    assert list(zip(out["target"], out["feature"])) == list(zip(keep["target"], keep["feature"]))


def test_gpudge_gate_matrix_route_drops_by_the_supplied_vector():
    """The matrix route ignores `ref_mean` entirely and gates on the vector the caller derived
    from the reference cells -- which is the whole point on paths where `ref_mean` is in the wrong
    unit (lognorm) or is the wrong statistic (geometric)."""
    frame = _gpudge_frame()
    var_names = ["kept_hi", "kept_mid", "boosted", "dropped_lo"]
    # deliberately DISAGREES with the frame's ref_mean: only `boosted` clears the threshold here
    ref_cpm = np.array([1.0, 2.0, 99.0, 0.5])
    out = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                              cpm_threshold=5.0, ref_cpm=ref_cpm, var_names=var_names)
    assert set(out["feature"].to_list()) == {"boosted"}
    # ... and the surviving rows are re-adjusted per target over what survived (m=1 -> p_adj == p)
    assert out["p_adj"].to_list() == out["p_value"].to_list()


def test_gpudge_gate_matrix_route_refuses_a_misaligned_vector():
    frame = _gpudge_frame()
    with pytest.raises(ValueError, match="cannot align"):
        _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                            cpm_threshold=5.0, ref_cpm=np.array([1.0, 2.0]),
                            var_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="cannot align"):
        _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                            cpm_threshold=5.0, ref_cpm=np.array([1.0, 2.0]), var_names=None)


def test_gpudge_gate_on_an_empty_frame():
    """gpudge can return nothing (every gene filtered upstream, or an empty batch). The gate and
    the BH pass must both no-op rather than raise -- `_apply_cpm_filter` has always had this
    guard, and the gpudge side needs it for the same reason."""
    empty = _gpudge_frame().clear()
    out = _finalize_gpudge_de(empty, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                              cpm_threshold=5.0, cpm_factor=1.0)
    assert out.height == 0
    assert _bh_per_target(empty).height == 0


def test_gpudge_gate_needs_ref_mean():
    frame = _gpudge_frame().drop("ref_mean")
    with pytest.raises(ValueError, match="missing ref_mean"):
        _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                            cpm_threshold=5.0, cpm_factor=1.0)


def test_gpudge_global_fdr_pools_over_the_survivors_only():
    """fdr_scope='global' overwrites the per-target adjustment with one pool -- over the GATED
    rows, because the gate runs first. Same order as the CPU path."""
    frame = _gpudge_frame()
    out = _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="global",
                              cpm_threshold=5.0, cpm_factor=1.0)
    expect = false_discovery_control(out["p_value"].to_numpy(), method="bh")
    assert np.allclose(out["p_adj"].to_numpy(), expect)


# ================================================================ D. end to end, CPU backend

def _de(adata, *, threshold, backend="pdex"):
    return compute_de(adata, backend=backend, groupby="target", reference=CONTROL,
                      mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                      target_sum=1e6, clip_value=None, filter_gene_min_cpm_cell=threshold,
                      fdr_scope="per_pert", threads=1)


def test_end_to_end_no_surviving_gene_sits_below_the_threshold_in_the_control():
    """The property that makes the leak unreachable: after the gate there is no row whose
    control mean is at or below the threshold, so there is no row whose sign is fixed by its
    own presence."""
    adata, control_cpm = _counts()
    df = _de(adata, threshold=THRESHOLD)
    below = {g for g, cpm in control_cpm.items() if cpm <= THRESHOLD}
    assert below, "fixture must contain below-threshold genes or it proves nothing"
    assert df.filter(pl.col("feature").is_in(list(below))).height == 0
    # and the gate really is target-independent end to end
    assert len({frozenset(sub["feature"].to_list())
                for _k, sub in df.group_by("target")}) == 1


def test_end_to_end_gate_off_is_unaffected():
    """v1's `filter_gene_min_cpm_cell=None` must be bit-identical to a no-gate run: the frozen
    2025 `vcc` profile has the filter off, so it cannot move under #351."""
    adata, _ = _counts()
    off = _de(adata, threshold=None)
    assert off.filter(pl.col("feature") == "leak_gene").height == len(TARGETS)
    # every (target, gene) present, nothing gated
    assert off.height == len(TARGETS) * adata.n_vars


def test_end_to_end_matches_the_gpudge_side_decision():
    """The two implementations are one semantics. Gate the CPU frame, and separately gate an
    un-gated CPU frame through the gpudge-side helper on its own `ref_mean`: same rows."""
    adata, _ = _counts()
    gated = _de(adata, threshold=THRESHOLD)
    ungated = _de(adata, threshold=None)
    # the CPU frame carries no ref_mean, so attach the control's CPM the gate would have used
    _, _, control_cpm = _linear_and_frame()
    with_ref = ungated.with_columns(
        pl.col("feature").replace_strict(control_cpm).alias("ref_mean"))
    via_gpudge = _finalize_gpudge_de(with_ref, epsilon=1e-9, clip_value=None,
                                     fdr_scope="per_pert", cpm_threshold=THRESHOLD,
                                     cpm_factor=1.0)
    assert (sorted(zip(gated["target"], gated["feature"]))
            == sorted(zip(via_gpudge["target"], via_gpudge["feature"])))


# ================================================================ E. the arm, in miniature

def _zero_information_arm():
    """#351's exploit at fixture scale: the prediction is the CONTROL block with counts added to
    the below-threshold genes, submitted UNCHANGED for every target. It cannot distinguish one
    perturbation from another, so anything it scores is a leak.
    """
    adata, control_cpm = _counts()
    ctrl = np.asarray(adata.X)[adata.obs["target"].to_numpy() == CONTROL].copy()
    genes = list(adata.var_names)
    below = [g for g, cpm in control_cpm.items() if 0 < cpm <= THRESHOLD]
    blocks, labels = [], []
    for t in TARGETS:
        block = ctrl.copy()
        for g in below:                     # raise every below-threshold gene, sign fixed
            block[:, genes.index(g)] += 8
        blocks.append(block)
        labels += [t] * block.shape[0]      # ... and the SAME block for every target
    blocks.append(ctrl.copy())              # the control the prediction is scored against
    labels += [CONTROL] * ctrl.shape[0]
    X = np.vstack(blocks).astype(np.float32)
    pred = ad.AnnData(X=X, var=adata.var.copy(),
                      obs=pd.DataFrame({"target": labels},
                                       index=[f"p{i}" for i in range(X.shape[0])]))
    return pred, adata, below


def _or_gated(df, adata, threshold):
    """`rule_version` 2's gate, re-implemented locally: keep a row when the TARGET group's mean
    CPM clears the threshold OR the control's does, then re-run BH per target.

    The shipped code no longer contains this rule, which is exactly why the test has to: the
    property under test is a property of the rule that was REMOVED, and asserting the new rule
    lacks it is only meaningful next to a demonstration that the old one had it.
    """
    X = np.asarray(adata.X, dtype=np.float64)
    cpm = X / X.sum(axis=1, keepdims=True) * 1e6
    labels = adata.obs["target"].to_numpy()
    genes = np.asarray(adata.var_names, dtype=str)
    means = {g: cpm[labels == g].mean(axis=0) for g in np.unique(labels)}
    ref = means[CONTROL]
    keep = [(t, genes[i])
            for t in sorted(set(df["target"].to_list()))
            for i in np.flatnonzero((means[t] > threshold) | (ref > threshold))]
    kf = pl.DataFrame({"target": [k[0] for k in keep], "feature": [k[1] for k in keep]},
                      schema={"target": pl.Utf8, "feature": pl.Utf8})
    return _bh_per_target(df.join(kf, on=["target", "feature"], how="inner"))


def test_the_or_gate_fixed_the_sign_of_every_row_it_admitted_and_the_new_gate_admits_none():
    """#351's identity at fixture scale: `P(real log2FC > 0) = 1` over the rows the OR admitted
    because the TARGET group cleared the threshold. Measured on the official panels over
    26,373 / 33,969 / 26,839 such rows; here it is exact for the same structural reason, and the
    reference-only gate leaves no such row for a submission to exploit."""
    _, real_adata, below = _zero_information_arm()
    assert below, "the fixture needs below-in-control genes or it proves nothing"

    ungated = _de(real_adata, threshold=None)
    or_gated = _or_gated(ungated, real_adata, THRESHOLD)
    admitted = or_gated.filter(pl.col("feature").is_in(below))
    assert admitted.height > 0, "the OR must admit some below-in-control row"
    assert (admitted["log2_fold_change"] > 0).all()          # the sign, fixed by presence alone
    assert admitted["log2_fold_change"].min() > 0

    # the reference-only gate: no such row exists at all, so nothing carries a fixed sign
    assert _de(real_adata, threshold=THRESHOLD).filter(
        pl.col("feature").is_in(below)).height == 0


def test_the_zero_information_arm_loses_its_credited_matches_under_the_new_gate():
    """The arm is the control block with counts added to the below-threshold genes, submitted
    UNCHANGED for every target -- it cannot tell one perturbation from another, so any credit it
    earns is leak. Under the OR its boosted rows are scored AND every one of them matches; under
    the reference-only gate they are not in the scored population at all."""
    from cell_eval2.metrics.direction import _components, de_direction_fidelity_yield_raw

    pred_adata, real_adata, below = _zero_information_arm()

    def _joined(gate):
        """(prepared, scored-rows-on-the-boosted-genes) under `gate` in ('or', 'ref')."""
        real, pred = _de(real_adata, threshold=None), _de(pred_adata, threshold=None)
        if gate == "or":
            real = _or_gated(real, real_adata, THRESHOLD)
            pred = _or_gated(pred, pred_adata, THRESHOLD)
        else:
            real = _de(real_adata, threshold=THRESHOLD)
            pred = _de(pred_adata, threshold=THRESHOLD)
        prep = prepare_de(pred, real, control=CONTROL, p_adj_threshold=0.05)
        scored = (pred.filter(pl.col("p_adj") < 0.05)
                  .join(real.select("target", "feature",
                                    pl.col("log2_fold_change").alias("real_lfc")),
                        on=["target", "feature"], how="inner")
                  .filter(pl.col("feature").is_in(below)))
        return prep, scored

    or_prep, or_scored = _joined("or")
    assert or_scored.height > 0, "under the OR the arm must score its boosted genes"
    # every one of them is a MATCH: the arm called them UP and the reference cannot disagree
    assert ((or_scored["log2_fold_change"] > 0) & (or_scored["real_lfc"] > 0)).all()

    ref_prep, ref_scored = _joined("ref")
    assert ref_scored.height == 0, "the reference-only gate must remove them from the pool"

    # and the member follows: the arm's per-target credit collapses. `k` counts matches in the
    # scored set, so it is the direct reading of "what the leak was worth" on this fixture.
    or_k = _components(or_prep)["k"].sum()
    ref_k = _components(ref_prep)["k"].sum()
    assert or_k > ref_k, (or_k, ref_k)
    or_val = np.nanmean(list(de_direction_fidelity_yield_raw(or_prep).values()))
    ref_val = np.nanmean(list(de_direction_fidelity_yield_raw(ref_prep).values()))
    assert or_val > ref_val, (or_val, ref_val)


# ================================================================ F. golden numbers (#338)

#: Section F's own panel: bigger than the gate fixture above, and built so no member is
#: degenerate. `GOLD_LIB` fixed per cell again, so `THRESHOLD` (3,000 CPM) is exactly 6 counts and
#: the seven lowest-expressed genes sit below it -- the gate has to bite, or the pinned numbers
#: would not guard #351 at all.
GOLD_LIB = 2_000
GOLD_CELLS = 30
GOLD_TARGETS = ("GENE1", "GENE2", "GENE3", "GENE4", "GENE5")
#: 16 genes move per target, comfortably above `de_lfc_nmae`'s `min_gate_size=10`, so the member
#: has a cohort instead of being omitted (the graded-fixture trap conftest records).
GOLD_MOVED = 16


def _golden_pair():
    """(real, pred, genes) counts AnnData for the golden pin.

    The reference carries a deterministic per-gene control profile: the five target genes sit at
    30 counts (they must survive the gate AND be knocked down visibly), 41 measured genes sweep
    2-23 counts, and two housekeeping-ish genes absorb the library. `THRESHOLD` is 3,000 CPM =
    6 counts at `GOLD_LIB`, so the eight lowest genes are gated out -- the gate HAS to bite here or
    the pinned numbers below would not guard #351 at all.

    Every target moves `GOLD_MOVED` genes, half up 3x and half down to a third, and is itself
    knocked down to 2 counts in its own group. The submission is deliberately PARTIAL: it
    reproduces two thirds of each target's moved genes and gets two of them backwards. A perfect
    replicate would pin 1.0 on three of the four members and pin nothing useful; a partial one
    lands them mid-range, where a dependency bump actually shows.
    """
    named = list(GOLD_TARGETS)
    measured = [f"g{i}" for i in range(41)]
    genes = named + measured + ["gfill", "gjit"]
    base = {g: 30 for g in named}
    base.update({g: 2 + (k * 22) // len(measured) for k, g in enumerate(measured)})
    base["gjit"], base["gfill"] = 40, 0          # gfill is solved for last, per cell

    def _moved(t):
        """This target's moved genes: a deterministic, target-specific slice of `measured`, so no
        two targets carry the same effect and none of them touches the library-absorbing pair."""
        j = named.index(t)
        pick = [measured[(j * 7 + 3 * k) % len(measured)] for k in range(GOLD_MOVED)]
        return pick[:GOLD_MOVED // 2], pick[GOLD_MOVED // 2:]      # (up, down)

    def _build(seed, *, fidelity):
        rng = np.random.default_rng(seed)
        rows, labels = [], []
        for grp in (CONTROL,) + GOLD_TARGETS:
            up, down = ([], []) if grp == CONTROL else _moved(grp)
            if fidelity < 1.0 and up:
                keep = int(round(len(up) * fidelity))
                up, down = up[:keep], down[:keep]
                up, down = up[:-1] + down[-1:], down[:-1] + up[-1:]   # two signs backwards
            for _ in range(GOLD_CELLS):
                cell = dict(base)
                for g in up:
                    cell[g] = cell[g] * 3
                for g in down:
                    cell[g] = max(1, cell[g] // 3)
                if grp != CONTROL:
                    cell[grp] = 2                                     # the knockdown
                cell["gjit"] = base["gjit"] + int(rng.integers(0, 3))
                cell["gfill"] = GOLD_LIB - sum(v for k, v in cell.items() if k != "gfill")
                rows.append([cell[g] for g in genes])
                labels.append(grp)
        X = np.asarray(rows, dtype=np.float32)
        assert (X.sum(axis=1) == GOLD_LIB).all(), "every cell must carry the same library"
        assert (X >= 0).all(), "gfill must absorb the effects, not go negative"
        return ad.AnnData(X=X, var=pd.DataFrame(index=genes),
                          obs=pd.DataFrame({"target": labels},
                                           index=[f"x{i}" for i in range(X.shape[0])]))

    return _build(1, fidelity=1.0), _build(2, fidelity=2 / 3), genes


#: The committed output half of #338's input/output pair. EXACT literals for three of the four: the
#: point is that a dependency bump -- polars' group ordering, `scipy.false_discovery_control`,
#: scanpy's `normalize_total`, a pdex release -- fails HERE instead of silently reissuing every
#: published number against a resolved-differently environment (`pyproject.toml` declares only
#: lower bounds and releases are git tags, so nothing else stops it). Regenerate deliberately and
#: say why in the commit; never to turn a red test green.
#:
#: ⚠️ `de_wilcoxon_lfc_nmae` is pinned to a TOLERANCE, and #338 asks for exactly this to be named
#: per member rather than blanketed. All four members read `log2_fold_change`, which comes from
#: `np.log2` -- a libm function that is not required to be bit-identical across libm builds or
#: architectures. The other three consume it only through its SIGN and through the significance
#: ORDERING, where a 1-ulp difference cannot change the answer unless the true value is exactly 0
#: (handled separately by the non-zero-LFC filter). `lfc_nmae` consumes the MAGNITUDES
#: arithmetically -- `mean|lfc_pred - lfc_real| / mean|lfc_real|` -- so a last-bit difference
#: propagates straight into the reported number. `_LFC_NMAE_RTOL` is far tighter than any real
#: change to the metric and far looser than libm noise.
#:
#: ⚠️ What this pins and what it does not. It runs the **pdex CPU backend**, so it pins the metric
#: arithmetic and the reference-only gate's semantics. It does NOT pin a competition number: the
#: official bundles carry `resolved_de_backend='gpudge'` / `resolved_device='cuda'` as submission
#: peers, `[gpudge]` is opt-in and CPU CI installs base+dev only, and the gpudge gate path is a
#: different code path (`_finalize_gpudge_de`, exercised in section C). #338's option (a)/(c) --
#: the same pin on the GPU gate -- is still open.
_GOLDEN_MEMBERS = {
    "de_wilcoxon_direction_fidelity_yield_raw": 0.5012987012987014,
    "de_wilcoxon_direction_reach_raw": 0.24175824175824173,
    "de_wilcoxon_sig_jaccard": 0.6591408591408591,
    "de_wilcoxon_lfc_nmae": 0.641203798176819,
}
#: see the note above: this member alone reads log2FC magnitudes arithmetically
_LFC_NMAE_RTOL = 1e-12
#: The gate drops these ten of the 48 genes -- the eight below 6 counts in the control plus the two
#: at exactly 6 (the compare is strict `>`). Pinned as a list because "which genes" is the #351
#: semantics: under the OR any of them could re-enter for the perturbation that raised it.
_GOLDEN_DROPPED = ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]
_GOLDEN_ROWS = 190          # 5 targets x 38 surviving genes, both sides


def test_golden_de_scoring_end_to_end_on_the_committed_pair():
    """#338: a committed input/output pair scored through the real path, compared bit-for-bit.

    Nothing else in the suite pins an actual score. `competition_digest()` pins each member's
    scoring POLICY, `rule_version` covers semantics by hand, `test_scales.py` pins the registry --
    and none of them would notice a dependency that quietly changed a number.
    """
    from cell_eval2.catalog import CATALOG

    real_adata, pred_adata, genes = _golden_pair()
    real = _de(real_adata, threshold=THRESHOLD)
    pred = _de(pred_adata, threshold=THRESHOLD)

    # (1) the gate: exactly the control-above-threshold set, on both sides, for every target
    ctrl = np.asarray(real_adata.X)[real_adata.obs["target"].to_numpy() == CONTROL]
    control_cpm = dict(zip(genes, ctrl.mean(axis=0) / GOLD_LIB * 1e6))
    expected_kept = {g for g, cpm in control_cpm.items() if cpm > THRESHOLD}
    assert sorted(set(genes) - expected_kept) == _GOLDEN_DROPPED
    assert sorted(set(real["feature"].to_list())) == sorted(expected_kept)
    assert sorted(set(pred["feature"].to_list())) == sorted(expected_kept)
    assert real.height == pred.height == _GOLDEN_ROWS

    # (2) the numbers, exactly
    prep = prepare_de(pred, real, control=CONTROL, p_adj_threshold=0.05)
    for member, expected in _GOLDEN_MEMBERS.items():
        per_target = CATALOG[member].func(prep)
        assert len(per_target) == len(GOLD_TARGETS), (member, per_target)
        got = float(np.nanmean(list(per_target.values())))
        if member == "de_wilcoxon_lfc_nmae":
            assert got == pytest.approx(expected, rel=_LFC_NMAE_RTOL, abs=0), \
                f"{member}: {got!r} != committed {expected!r}"
        else:
            assert got == expected, f"{member}: {got!r} != committed {expected!r}"


def test_the_golden_pair_is_deterministic():
    """The fixture must be reproducible from its seeds alone, or the pin above pins nothing."""
    a1, p1, _ = _golden_pair()
    a2, p2, _ = _golden_pair()
    assert np.array_equal(np.asarray(a1.X), np.asarray(a2.X))
    assert np.array_equal(np.asarray(p1.X), np.asarray(p2.X))


def test_the_golden_pin_would_notice_the_or_gate():
    """What makes the pin a #351 regression test rather than a generic dependency tripwire: score
    the SAME pair under the removed OR rule and show the pinned values do not hold there. A kept-set
    comparison alone would not establish that -- a bigger gene set could in principle leave every
    member mean untouched."""
    from cell_eval2.catalog import CATALOG

    real_adata, pred_adata, _ = _golden_pair()
    or_real = _or_gated(_de(real_adata, threshold=None), real_adata, THRESHOLD)
    or_pred = _or_gated(_de(pred_adata, threshold=None), pred_adata, THRESHOLD)
    or_kept = set(or_real["feature"].to_list())
    assert set(_GOLDEN_DROPPED) & or_kept, "the OR must re-admit some gated gene"
    assert set(_de(real_adata, threshold=THRESHOLD)["feature"].to_list()) < or_kept

    or_prep = prepare_de(or_pred, or_real, control=CONTROL, p_adj_threshold=0.05)
    moved = []
    for member, pinned in _GOLDEN_MEMBERS.items():
        vals = CATALOG[member].func(or_prep)
        got = float(np.nanmean(list(vals.values()))) if vals else float("nan")
        same = (got == pytest.approx(pinned, rel=_LFC_NMAE_RTOL, abs=0)
                if member == "de_wilcoxon_lfc_nmae" else got == pinned)
        if not same:
            moved.append(member)
    assert moved, f"the OR rule must move at least one pinned member; got none of {list(_GOLDEN_MEMBERS)}"


def test_the_all_others_sentinel_spellings_are_rejected_as_a_reference():
    """gpudge reads `__all_others__` (and the legacy `all_others`) as its 1-vs-rest sentinel rather
    than as a group label, and then emits a rest-of-panel `ref_mean` that differs PER TARGET. A
    group literally so named would pass the membership check and quietly restore a target-dependent
    kept set, so `compute_de` refuses the spelling outright."""
    adata, _ = _counts()
    for sentinel in ("__all_others__", "all_others"):
        renamed = adata.copy()
        labels = renamed.obs["target"].to_numpy().astype(object)
        labels[labels == CONTROL] = sentinel
        renamed.obs["target"] = labels.astype(str)
        with pytest.raises(ValueError, match="ALL_OTHERS sentinel"):
            compute_de(renamed, backend="pdex", groupby="target", reference=sentinel,
                       mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                       target_sum=1e6, clip_value=None, filter_gene_min_cpm_cell=THRESHOLD,
                       fdr_scope="per_pert", threads=1)


# ============================================= G. argument routing, no GPU required

def _stub_streaming_gpudge(monkeypatch, seen):
    """Install a fake `gpudge` whose `de()` records what it was handed and returns one row per
    (target, gene). No CUDA: `_resolve_backend` is stubbed too. This is the cheap half of an
    integration test -- it cannot check gpudge's arithmetic, but it DOES check which arguments each
    entry point derives, which is where the gate's correctness actually lives."""
    import sys
    import types

    from cell_eval2 import de_compute as dc

    genes = [f"g{i}" for i in range(4)]

    def fake_de(shard_archive=None, reference=None, groupby=None, mean_calc=None, epsilon=None,
                cpm_normalize=None, normalize_target_sum=None, filter_gene_min_cpm_cell=None,
                **kw):
        seen["filter"] = filter_gene_min_cpm_cell
        seen["normalize_target_sum"] = normalize_target_sum
        rows = [(t, g) for t in ("T1", "T2") for g in genes]
        return pl.DataFrame({
            "target": [t for t, _ in rows], "feature": [g for _, g in rows],
            "log2_fold_change": [0.5] * len(rows), "p_value": [0.01] * len(rows),
            "p_adj": [0.01] * len(rows),
            "target_mean": [2.0] * len(rows), "ref_mean": [1.0] * len(rows),
        })

    mod = types.ModuleType("gpudge")
    mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", mod)
    monkeypatch.setattr(dc, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(dc, "_release_gpu_pool", lambda: None)
    return genes


def _stream_reference(genes, *, per_gene):
    """An AnnData external control pool (streaming Mode 2) with a chosen per-gene CPM profile."""
    lib = 1_000
    counts = np.array([[int(round(c * lib / 1e6)) for c in per_gene]], dtype=np.float32)
    counts = np.repeat(counts, 8, axis=0)
    pad = lib - counts.sum(axis=1)
    X = np.hstack([counts, pad[:, None]])
    return ad.AnnData(X=X, var=pd.DataFrame(index=list(genes) + ["gpad"]),
                      obs=pd.DataFrame({"target": [CONTROL] * X.shape[0]},
                                       index=[f"r{i}" for i in range(X.shape[0])]))


def test_streaming_mode2_uses_its_annData_reference_cells_for_the_matrix_route(monkeypatch):
    """`compute_de_streaming` accepts an AnnData external control pool, so the matrix route IS
    available to it -- claiming otherwise made a geometric or median-normalized Mode 2 run raise,
    with an error that recommended supplying the very AnnData it was already holding."""
    from cell_eval2.de_compute import compute_de_streaming

    seen = {}
    genes = _stub_streaming_gpudge(monkeypatch, seen)
    #  g0/g1 above 5 CPM in the control, g2/g3 below -> the gate must keep exactly g0, g1
    ref = _stream_reference(genes, per_gene=[40_000.0, 20_000.0, 2_000.0, 1_000.0])
    out = compute_de_streaming("ignored.shad", backend="gpudge", reference=ref, groupby="target",
                               mean_calc="geometric", epsilon=1e-9, target_sum=1e4,
                               clip_value=None, fdr_scope="per_pert",
                               filter_gene_min_cpm_cell=5_000.0)
    assert set(out["feature"].to_list()) == {"g0", "g1"}
    assert seen["filter"] is None, "the matrix route must mute gpudge's own gate"
    # every target keeps the same set -- the #351 invariant, through the real entry point
    assert len({frozenset(sub["feature"].to_list()) for _k, sub in out.group_by("target")}) == 1


def test_streaming_mode1_still_refuses_when_no_route_is_available(monkeypatch):
    """Mode 1 (a label, or None -> the archive's own reference shard) genuinely has no cells here."""
    from cell_eval2.de_compute import compute_de_streaming

    seen = {}
    _stub_streaming_gpudge(monkeypatch, seen)
    with pytest.raises(NotImplementedError, match="decided by the REFERENCE group alone"):
        compute_de_streaming("ignored.shad", backend="gpudge", reference=CONTROL,
                             groupby="target", mean_calc="geometric", epsilon=1e-9,
                             target_sum=1e4, clip_value=None, fdr_scope="per_pert",
                             filter_gene_min_cpm_cell=5.0)


def test_streaming_keeps_gpudge_own_gate_on_the_frame_route(monkeypatch):
    """The counterpart: arithmetic + a known finite target is the frame route, and there gpudge's
    gate nests exactly, so it is forwarded rather than muted (it prunes gene chunks)."""
    from cell_eval2.de_compute import compute_de_streaming

    seen = {}
    genes = _stub_streaming_gpudge(monkeypatch, seen)
    ref = _stream_reference(genes, per_gene=[40_000.0, 20_000.0, 2_000.0, 1_000.0])
    compute_de_streaming("ignored.shad", backend="gpudge", reference=ref, groupby="target",
                         mean_calc="arithmetic", epsilon=1e-9, target_sum=1e6, clip_value=None,
                         fdr_scope="per_pert", filter_gene_min_cpm_cell=5.0)
    assert seen["filter"] == 5.0


def test_streaming_negative_threshold_keeps_everything_without_choosing_a_route(monkeypatch):
    """The keep-all short-circuit, through the entry point that has no matrix route: it must not
    raise, and it must not drop a row."""
    from cell_eval2.de_compute import compute_de_streaming

    seen = {}
    _stub_streaming_gpudge(monkeypatch, seen)
    out = compute_de_streaming("ignored.shad", backend="gpudge", reference=CONTROL,
                               groupby="target", mean_calc="geometric", epsilon=1e-9,
                               target_sum=None, clip_value=None, fdr_scope="per_pert",
                               filter_gene_min_cpm_cell=-1.0)
    assert out.height == 8   # 2 targets x 4 genes, nothing gated


def test_cell_layout_streaming_routes_the_gate_by_mean_calc(monkeypatch):
    """`compute_de_streaming_cell` drives gpudge's `refpool_de_core` directly, so its flag routing
    needs its own stub: arithmetic (gpudge normalizes to the resolved target) forwards the gate,
    geometric mutes it and gates from `ref_X`. No CUDA -- this pins which arguments the entry point
    derives, which is where the gate's correctness lives."""
    import sys
    import types

    import scipy.sparse as sp

    from cell_eval2 import de_compute as dc

    genes = [f"g{i}" for i in range(4)]
    seen = {}

    def fake_core(*, ref_X, target_source, targets, n_genes, var_names, device, mean_calc,
                  epsilon, filter_gene_min_cpm_cell, **kw):
        seen["filter"] = filter_gene_min_cpm_cell
        list(target_source(False))            # drain the generator as the real core does
        rows = [(str(t), g) for t in targets for g in genes]
        return pl.DataFrame({
            "target": [t for t, _ in rows], "feature": [g for _, g in rows],
            "log2_fold_change": [0.5] * len(rows), "p_value": [0.01] * len(rows),
            "p_adj": [0.01] * len(rows),
            "target_mean": [2.0] * len(rows), "ref_mean": [1.0] * len(rows),
        })

    monkeypatch.setitem(sys.modules, "gpudge", types.ModuleType("gpudge"))
    csr = types.ModuleType("gpudge._csr_dense")
    csr.ensure_csr = lambda X, name=None: sp.csr_matrix(X)
    csr.csr_row_sums = lambda X: np.asarray(X.sum(axis=1)).ravel()
    refpool = types.ModuleType("gpudge._refpool")
    refpool.refpool_de_core = fake_core
    monkeypatch.setitem(sys.modules, "gpudge._csr_dense", csr)
    monkeypatch.setitem(sys.modules, "gpudge._refpool", refpool)
    monkeypatch.setattr(dc, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(dc, "_release_gpu_pool", lambda: None)
    monkeypatch.setattr(dc, "_gpudge_supports_refpool_core", lambda: True)
    monkeypatch.setattr(dc, "torch", types.SimpleNamespace(device=lambda s: s), raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=lambda s: s))

    lib = 1_000
    #  g0/g1 clear 5,000 CPM in the control; g2/g3 do not
    ref = np.array([[40, 20, 2, 1]], dtype=np.float64)
    ref = np.repeat(ref, 8, axis=0)
    ref = np.hstack([ref, (lib - ref.sum(axis=1))[:, None]])
    axis = genes + ["gpad"]

    def _run(mean_calc):
        return dc.compute_de_streaming_cell(
            ref_X=sp.csr_matrix(ref),
            group_iter_factory=lambda: iter([("T1", sp.csr_matrix(ref))]),
            targets=["T1"], var_names=axis, n_genes=len(axis), backend="gpudge",
            mean_calc=mean_calc, epsilon=1e-9, target_sum=1e6, clip_value=None,
            fdr_scope="per_pert", filter_gene_min_cpm_cell=5_000.0)

    _run("arithmetic")
    assert seen["filter"] == 5_000.0, "the frame route forwards gpudge's own gate"
    out = _run("geometric")
    assert seen["filter"] is None, "the matrix route mutes it"
    # the stub emits rows for `genes` only, so `gpad` -- which also clears the threshold -- has no
    # row to survive; g2/g3 do have rows and are correctly gated out
    assert set(out["feature"].to_list()) == {"g0", "g1"}


def test_the_plan_rejects_a_non_numeric_target_sum_by_name(monkeypatch):
    """`np.isfinite` raises a bare TypeError on a string, so a `target_sum` that is neither a number
    nor None is rejected explicitly (Gemini round 1). ⚠️ A RAISE, not a coerce-to-None: silently
    re-routing a misconfiguration to the matrix route would hide it behind an unrelated message."""
    for bad in ("median", "1e6", object(), True):
        with pytest.raises(TypeError, match="must be a number or None"):
            _plan(target_sum=bad)


def test_the_alignment_guard_names_the_offending_feature():
    """The guard reduces to the unique feature set before checking, and still reports which gene is
    off-axis -- the count is of FEATURES, not of rows."""
    frame = _gpudge_frame()          # features kept_hi / kept_mid / boosted / dropped_lo, 2 targets
    with pytest.raises(ValueError, match="gene axis does not cover") as e:
        _finalize_gpudge_de(frame, epsilon=1e-9, clip_value=None, fdr_scope="per_pert",
                            cpm_threshold=5.0, ref_cpm=np.array([9.0, 9.0]),
                            var_names=["kept_hi", "kept_mid"])
    msg = str(e.value)
    assert "2 feature(s)" in msg, msg      # boosted + dropped_lo, once each, not once per target
    assert "boosted" in msg or "dropped_lo" in msg
