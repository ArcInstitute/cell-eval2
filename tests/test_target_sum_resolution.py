"""#155: target_sum=None must resolve to ONE number per run -- the real control pool's median
library size -- instead of "the median of whichever matrix normalize_total was handed".

CPU-only: the in-memory external-reference DE path that carries the defect is gpudge-only
(run._use_inmem_external_ref), so gpudge is stubbed. See the plan's "Stubbing gpudge on CPU".
"""

from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2.config import EvalConfig

N_GENES = 12


def _stub_gpudge(monkeypatch, seen, *, control="non-targeting", n_genes=12):
    """Capture what compute_de hands gpudge, on both the target and the reference side."""
    import sys
    import types

    import numpy as np
    import polars as pl

    def _totals(X):
        # SPARSE-SAFE: np.asarray(csr) yields a 0-d object array, so .sum(axis=1) is wrong.
        # Several fixtures in this repo are CSR (tests/test_pred_de_control_scale.py).
        from scipy.sparse import issparse
        return (np.asarray(X.sum(axis=1)).ravel() if issparse(X)
                else np.asarray(X).sum(axis=1)).copy()

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen.setdefault("calls", []).append({
            "target_totals": _totals(adata.X),
            "ref_totals": _totals(reference.X) if hasattr(reference, "X") else None,
            "cpm_normalize": cpm_normalize,
            "normalize_target_sum": normalize_target_sum,
        })
        # Never emit the control as a DE target -- de.assemble_prepared_de rejects it.
        targets = sorted(set(np.asarray(adata.obs[groupby]).astype(str)) - {control})
        rows = [(t, f"g{i}") for t in targets for i in range(n_genes)]
        return pl.DataFrame({
            "target": [t for t, _ in rows], "feature": [f for _, f in rows],
            "log2_fold_change": [0.5] * len(rows), "p_value": [0.01] * len(rows),
            "p_adj": [0.02] * len(rows),
            # Trap 4: v1 keeps clip_value=20.0 (config.py:150), and _finalize_gpudge_de then
            # RAISES "gpudge output is missing target_mean/ref_mean" (de_compute.py:330-333)
            # before any assertion runs. v2 has clip_value=None and does not need these, but
            # every v1 test in this plan does -- so the stub always emits them.
            "target_mean": [2.0] * len(rows), "ref_mean": [1.0] * len(rows),
        })

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute, partition_inmem
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(de_compute, "_gpudge_supports_inmem_external_ref", lambda: True)
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda b: "gpudge")


def _counts_pair(seed_real=0, seed_pred=1, ctrl_scale=1.0):
    """(pred, real) raw-count pair. ctrl_scale multiplies the REAL CONTROL library sizes only."""
    perts = ["non-targeting"] * 40 + sum(([g] * 30 for g in ["A", "B", "C"]), [])

    def mk(seed, scale_ctrl):
        r = np.random.default_rng(seed)
        X = r.poisson(5, size=(len(perts), N_GENES)).astype(np.float64)
        if scale_ctrl != 1.0:
            mask = np.array([p == "non-targeting" for p in perts])
            X[mask] = X[mask] * scale_ctrl
        # ⚠️ The gene names must be the INDEX, not a column. `var={"gene": [...]}` leaves
        # `var_names` a RangeIndex ("0", "1", ...) while the stub below emits features named
        # `g0, g1, ...`, i.e. an axis that no real gpudge would ever return. The #351 CPM gate
        # decides which genes to keep from the reference cells and matches them against the
        # frame's `feature` column, so a mismatched axis makes it drop every row -- and the
        # finalizer now raises on exactly that rather than gating silently.
        return ad.AnnData(X=X, obs={"target": perts},
                          var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]))

    return mk(seed_pred, 1.0), mk(seed_real, ctrl_scale)


def _cfg_median(**kw):
    v2 = EvalConfig.v2()
    base = replace(v2, target_sum=None, metrics=["expr_mae"], validate_input=False,
                   device="cpu", de=replace(v2.de, backend="gpudge"))
    return replace(base, **kw) if kw else base


def test_both_halves_get_one_target(monkeypatch):
    """The invariant: the two halves of the LFC ratio are normalized to the SAME per-cell
    total. NOT "log2FCs are invariant to that total" -- see spec section 1."""
    from cell_eval2 import compute_metrics
    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    pred, real = _counts_pair(ctrl_scale=4.0)  # control pool deliberately off-scale
    compute_metrics(pred, real, config=_cfg_median(metrics=["de_wilcoxon_overlap"]))
    ext = [c for c in seen["calls"] if c["ref_totals"] is not None]
    assert ext, "expected at least one external-reference DE call"
    for call in ext:
        assert np.allclose(call["target_totals"], call["target_totals"][0])
        assert np.allclose(call["ref_totals"], call["target_totals"][0])


def test_resolution_equals_passing_the_number_explicitly(monkeypatch):
    from cell_eval2 import compute_metrics
    from cell_eval2.norm import resolve_target_sum
    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    pred, real = _counts_pair(ctrl_scale=3.0)
    ctrl = real[real.obs["target"] == "non-targeting"].copy()
    resolved = resolve_target_sum(ctrl, input_type="counts", target_sum=None)
    auto = compute_metrics(pred, real, config=_cfg_median())
    explicit = compute_metrics(pred, real, config=_cfg_median(target_sum=resolved))
    assert auto.equals(explicit)


def test_caller_config_object_is_not_mutated(monkeypatch):
    from cell_eval2 import compute_metrics
    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    pred, real = _counts_pair()
    cfg = _cfg_median()
    compute_metrics(pred, real, config=cfg)
    assert cfg.target_sum is None


def test_v1_declared_lognorm_over_counts_still_resolves(monkeypatch):
    """v1 permits a declared lognorm config over data that is actually counts. Resolution must
    use the EFFECTIVE input type, or the resolver returns None while to_normalization keeps
    deriving per-matrix medians from those counts."""
    from cell_eval2 import compute_metrics
    from cell_eval2.norm import resolve_target_sum
    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    pred, real = _counts_pair(ctrl_scale=2.0)
    v1 = EvalConfig.v1()
    cfg = replace(v1, metrics=["de_wilcoxon_overlap"], validate_input=False, device="cpu",
                  control_source="real",
                  de=replace(v1.de, backend="gpudge", fdr_scope="per_pert"))
    compute_metrics(pred, real, config=cfg)
    ctrl = real[real.obs["target"] == "non-targeting"].copy()
    expected = resolve_target_sum(ctrl, input_type="counts", target_sum=None)
    ext = [c for c in seen["calls"] if c["ref_totals"] is not None]
    assert ext
    assert np.allclose(ext[0]["target_totals"], expected)
    assert np.allclose(ext[0]["ref_totals"], expected)


def _part_cfg(**kw):
    base = _cfg_median(de=replace(EvalConfig.v2().de, backend="gpudge", fdr_scope="per_pert"))
    return replace(base, **kw) if kw else base


def _read_manifest(cache_dir):
    import json
    import os
    with open(os.path.join(str(cache_dir), "reference.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_reference_manifest_records_normalize_target_sum(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem
    from cell_eval2.norm import resolve_target_sum
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _pred, real = _counts_pair(ctrl_scale=2.0)
    partition_inmem.build_reference(real, config=_part_cfg(), cache_dir=str(tmp_path),
                                    control_format="h5ad", comparator="bulk_lognorm")
    ctrl = real[real.obs["target"] == "non-targeting"].copy()
    assert _read_manifest(tmp_path)["normalize_target_sum"] == resolve_target_sum(
        ctrl, input_type="counts", target_sum=None)


def test_v2_bundle_records_the_literal_target_sum(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _pred, real = _counts_pair()
    partition_inmem.build_reference(real, config=_part_cfg(target_sum=1e6),
                                    cache_dir=str(tmp_path), control_format="h5ad",
                                    comparator="bulk_lognorm")
    assert _read_manifest(tmp_path)["normalize_target_sum"] == 1e6


def test_score_piece_rejects_a_bundle_without_the_key(tmp_path, monkeypatch):
    import json
    import os
    from cell_eval2 import partition_inmem
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    pred, real = _counts_pair()
    cfg = _part_cfg()
    partition_inmem.build_reference(real, config=cfg, cache_dir=str(tmp_path),
                                    control_format="h5ad", comparator="bulk_lognorm")
    path = os.path.join(str(tmp_path), "reference.json")
    manifest = _read_manifest(tmp_path)
    del manifest["normalize_target_sum"]                      # simulate a pre-fix bundle
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    with pytest.raises(ValueError, match="normalize_target_sum"):
        partition_inmem.score_piece(
            piece, str(tmp_path), config=cfg, piece_id="p0", comparator="bulk_lognorm"
        )


def test_score_piece_rejects_a_target_that_disagrees_with_the_bundle(tmp_path, monkeypatch):
    """A bundle built at one target and scored at another produces artifacts on two scales,
    and aggregate_partials cannot catch it: every partial records the same CALLER-derived
    config_hash, so the cross-partial guard sees agreement."""
    from cell_eval2 import partition_inmem
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    pred, real = _counts_pair()
    partition_inmem.build_reference(real, config=_part_cfg(target_sum=1e6),
                                    cache_dir=str(tmp_path), control_format="h5ad",
                                    comparator="bulk_lognorm")
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    with pytest.raises(ValueError, match="normalize_target_sum"):
        partition_inmem.score_piece(piece, str(tmp_path), config=_part_cfg(target_sum=1e4),
                                    piece_id="p0", comparator="bulk_lognorm")


def test_score_piece_accepts_an_explicit_target_that_matches(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    pred, real = _counts_pair()
    partition_inmem.build_reference(real, config=_part_cfg(target_sum=1e6),
                                    cache_dir=str(tmp_path), control_format="h5ad",
                                    comparator="bulk_lognorm")
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    out = partition_inmem.score_piece(piece, str(tmp_path), config=_part_cfg(target_sum=1e6),
                                      piece_id="p0", comparator="bulk_lognorm")
    assert out.height > 0


def _spy_compute_de(monkeypatch, calls):
    """Capture the kwargs partition_inmem hands compute_de, then delegate to the real function.

    compute_de resolves input_type internally (via _to_linear) and never forwards it to
    gpudge, so the _stub_gpudge seam CANNOT observe it. This one can.
    """
    from cell_eval2 import partition_inmem
    real = partition_inmem.compute_de

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(partition_inmem, "compute_de", spy)


def _spy_side_bulks(monkeypatch, seen):
    """Capture every _side_bulks call's kwargs (they are all keyword-only), then delegate.

    TWO seams are required (trap 5). partition_inmem imports _side_bulks at module scope
    (:46-47) -- that binding serves build_reference (:189) and score_piece (:468) -- but
    _build_reference_streaming_core (:245) and _build_pred_control_reference_core (:340)
    re-import it FUNCTION-LOCALLY, which shadows the patched module global at call time. Patch
    only partition_inmem._side_bulks and the streaming builder -- the very site the override
    exists for -- is invisible to this spy.

    Captures the whole kwargs dict rather than one field so Task 8 can reuse it to inspect
    `cfg` without a second, easily-one-seamed copy of this helper.
    """
    from cell_eval2 import partition_inmem, run
    real = run._side_bulks

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(run, "_side_bulks", spy)                 # function-local importers
    monkeypatch.setattr(partition_inmem, "_side_bulks", spy)     # module-global users


def _v1_over_counts_cfg():
    """v1 DECLARES lognorm while the data is integer counts, so the EFFECTIVE type is counts:
    norm.resolve_input_type:205 short-circuits only when `version != "v1" and not autodetect`,
    so v1 autodetects unconditionally and guess_is_lognorm is False on integers. EvalConfig
    defaults already give pert_col="target"/control="non-targeting" (config.py:175-176), which
    is what _counts_pair builds.

    metrics MUST include an anndata metric. _needed_normalizations keeps only `kind ==
    "anndata"` entries (run.py:149), so a DE-only list makes norms == [] -- score_piece then
    skips its `if anndata_names:` block entirely and never calls _side_bulks, and the builders'
    calls do no work. The override would be completely untested.
    """
    v1 = EvalConfig.v1()
    return replace(v1, metrics=["de_wilcoxon_overlap", "expr_mae"], validate_input=False,
                   device="cpu", control_source="real", target_sum=None,
                   de=replace(v1.de, backend="gpudge", fdr_scope="per_pert"))


def test_effective_input_type_reaches_de_in_build_reference(tmp_path, monkeypatch):
    """Spec section 7 item 9, site partition_inmem.py:193 (+ the _side_bulks call at :189)."""
    from cell_eval2 import partition_inmem
    calls, bulks = [], []
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _spy_compute_de(monkeypatch, calls)
    _spy_side_bulks(monkeypatch, bulks)
    _pred, real = _counts_pair()
    cfg = _v1_over_counts_cfg()
    assert cfg.input_type == "lognorm", "the config must DECLARE lognorm or this proves nothing"
    partition_inmem.build_reference(real, config=cfg, cache_dir=str(tmp_path),
                                    control_format="h5ad", comparator="lognorm")
    assert calls and calls[-1]["input_type"] == "counts"
    assert bulks and all(b.get("effective_input_type") == "counts" for b in bulks)
    assert _read_manifest(tmp_path)["effective_input_type"] == "counts"


def test_effective_input_type_reaches_de_in_the_streaming_builder(tmp_path, monkeypatch):
    """Spec section 7 item 9, site partition_inmem.py:289 -- the PER-BATCH DE call -- and the
    per-batch _side_bulks call at :284, which is the one the override exists for. Uses the
    public build_reference_streaming wrapper (partition_inmem.py:216) rather than building an
    H5adBatchSource by hand; leave its input_type= argument unset so the rebind is what binds."""
    from cell_eval2 import partition_inmem
    from cell_eval2.h5ad_manifest import MemBudget
    calls, bulks = [], []
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _spy_compute_de(monkeypatch, calls)
    _spy_side_bulks(monkeypatch, bulks)
    _pred, real = _counts_pair()
    h5 = str(tmp_path / "real.h5ad")
    real.write_h5ad(h5)
    # mem_budget is a MemBudget, NOT an int: plan_pert_batches reads .host_bytes/.gpu_bytes
    # (h5ad_manifest.py:89) and an int dies with AttributeError before compute_de is ever reached.
    #
    # The budget must be DERIVED, not guessed. plan_pert_batches charges the resident control
    # plus the batch at n_genes*itemsize*safety(=3.0) per cell and RAISES outright if one
    # perturbation cannot fit (h5ad_manifest.py:97-102) -- so too small is an error, not a finer split.
    # itemsize is the ARCHIVE's own dtype (h5ad_manifest.py:142), float64 here. cap_cells = 45 leaves
    # room for the 40 control cells plus one 30-cell perturbation but not two, giving three
    # single-perturbation batches. More than one batch is the point: a single batch cannot show
    # the per-batch autodetect this test exists to catch.
    # (Measured: a "comfortable" 1<<18 gives exactly ONE batch at either itemsize.)
    per_cell = N_GENES * np.dtype(real.X.dtype).itemsize * 3.0
    budget = int((40 + 45) * per_cell)
    partition_inmem.build_reference_streaming(
        h5, config=_v1_over_counts_cfg(), cache_dir=str(tmp_path / "ref"),
        control="non-targeting",
        mem_budget=MemBudget(host_bytes=budget, gpu_bytes=budget), comparator="lognorm")
    assert len(calls) == 3, "expected one per-batch DE call per single-perturbation batch"
    assert all(c["input_type"] == "counts" for c in calls)
    assert bulks and all(b.get("effective_input_type") == "counts" for b in bulks)


def test_effective_input_type_reaches_de_in_score_piece(tmp_path, monkeypatch):
    """Spec section 7 item 9, site partition_inmem.py:497 -- the one the builders' rebind does
    NOT reach. score_piece is a separate function, so it needs its own rebind, derived from
    piece_mat (under control_source='real' there is no pred control block to anchor on)."""
    from cell_eval2 import partition_inmem
    calls, bulks = [], []
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    pred, real = _counts_pair()
    cfg = _v1_over_counts_cfg()
    partition_inmem.build_reference(real, config=cfg, cache_dir=str(tmp_path),
                                    control_format="h5ad", comparator="lognorm")
    # Spy AFTER the build, so only the piece's own calls are captured.
    _spy_compute_de(monkeypatch, calls)
    _spy_side_bulks(monkeypatch, bulks)
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    partition_inmem.score_piece(
        piece, str(tmp_path), config=cfg, piece_id="p0", comparator="lognorm"
    )
    assert calls and all(c["input_type"] == "counts" for c in calls)
    assert bulks and all(b.get("effective_input_type") == "counts" for b in bulks)


def test_effective_input_type_reaches_the_pred_control_builder(tmp_path, monkeypatch):
    """Spec section 7 item 9, site partition_inmem.py:359.

    This core runs NO compute_de at all, so the _side_bulks override is the only thing that can
    prove the threading here: drop the Step 3f edit at :359 and every other test in this file
    stays green. It is also the second of the two function-local importers, so it is the site
    trap 5 is about.
    """
    from cell_eval2 import partition_inmem
    from cell_eval2.norm import resolve_target_sum
    bulks = []
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    # ctrl_scale=3.0 makes the REAL control pool three times deeper than the pred control, so
    # "adopted the bundle's target" and "derived its own median" are DIFFERENT numbers. With
    # the default 1.0 they nearly coincide and the adoption assertion below proves nothing.
    pred, real = _counts_pair(ctrl_scale=3.0)
    cfg = _v1_over_counts_cfg()
    # _build_pred_control_reference_core adopts the REAL bundle's target (_bundle_target_sum),
    # so the real bundle must exist in this cache_dir first.
    partition_inmem.build_reference(real, config=cfg, cache_dir=str(tmp_path),
                                    control_format="h5ad", comparator="lognorm")
    bundle_ts = _read_manifest(tmp_path)["normalize_target_sum"]
    pred_ctrl = pred[pred.obs["target"] == "non-targeting"].copy()
    pred_own_median = resolve_target_sum(pred_ctrl, input_type="counts", target_sum=None)
    assert pred_own_median != bundle_ts, \
        "fixture is not discriminating: the pred control's own median equals the bundle's"

    pred_h5 = str(tmp_path / "pred.h5ad")
    pred.write_h5ad(pred_h5)
    # Spy AFTER the real build, so only the pred-control core's call is captured.
    _spy_side_bulks(monkeypatch, bulks)
    # No input_type= argument: the rebind is what must bind it.
    partition_inmem.build_pred_control_reference(
        pred_h5, config=cfg, cache_dir=str(tmp_path), control="non-targeting",
        comparator="lognorm")
    assert bulks, "expected a pred-control pseudobulk call"
    assert all(b.get("effective_input_type") == "counts" for b in bulks)
    assert all(b["cfg"].input_type == "counts" for b in bulks)
    # The load-bearing half: it must ADOPT the real bundle's target rather than derive the pred
    # control's own median -- a second target is exactly the split #155 removes. Without this
    # assertion, deleting the _bundle_target_sum call in _build_pred_control_reference_core
    # leaves every test in this file green while the pred pseudobulk silently normalizes to a
    # different scale than the real artifacts it is compared against.
    for b in bulks:
        assert b["cfg"].target_sum == bundle_ts, (
            f"pred control normalized to {b['cfg'].target_sum!r}, not the bundle's "
            f"{bundle_ts!r} (its own median would be {pred_own_median!r})"
        )


def test_side_bulks_override_bypasses_both_recompute_sites(tmp_path):
    """A direct unit on the override itself: with effective_input_type given, NEITHER
    recomputation site (run.py:351 in full(), run.py:363 in compute()) may call
    _effective_input_type. A path source is used because :351 fires only for path/backed
    inputs, so an in-memory AnnData would leave that site unproven."""
    import cell_eval2.run as run

    _pred, real = _counts_pair()
    h5 = str(tmp_path / "real.h5ad")
    real.write_h5ad(h5)
    cfg = _v1_over_counts_cfg()

    def _boom(*a, **kw):
        raise AssertionError("_effective_input_type must not run when the override is given")

    orig = run._effective_input_type
    run._effective_input_type = _boom
    try:
        out = run._side_bulks(h5, fp=None, store=None, norms=["lognorm"], cfg=cfg,
                              side="real", effective_input_type="counts")
    finally:
        run._effective_input_type = orig
    assert "lognorm" in out


def test_precompute_cache_rejects_target_sum_none_on_the_pred_side(tmp_path):
    """#267 narrowed this to side='pred' ONLY. The pred side carries no real control pool, so
    any target it invented would key entries a resolved run could never hit -- which is the
    property the refusal exists to protect. side='real' is covered by the three tests below."""
    from cell_eval2 import compute_metrics  # noqa: F401  (package import)
    from cell_eval2.run import precompute_cache
    pred, _real = _counts_pair()
    cfg = _cfg_median(cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"))
    with pytest.raises(NotImplementedError, match="target_sum=None"):
        precompute_cache(pred, side="pred", config=cfg)


def test_precompute_cache_resolves_target_sum_none_on_the_real_side(tmp_path):
    """#267: the control pool IS inside the one side being loaded, so the real-side warm
    resolves rather than refuses, and it resolves to the SAME number compute_metrics does."""
    import os

    from cell_eval2 import norm as _norm
    from cell_eval2.run import precompute_cache

    _pred, real = _counts_pair(ctrl_scale=3.0)   # controls at a distinct library scale
    cfg = _cfg_median(cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"))
    precompute_cache(real, side="real", config=cfg, comparator="lognorm")   # must NOT raise
    assert any(os.scandir(str(tmp_path / "r"))), "the real side must write its cache"

    expected = _norm.resolve_target_sum(
        real[real.obs["target"].astype(str) == cfg.control],
        input_type="counts", target_sum=None)
    assert expected is not None and expected > 0
    # The control pool was scaled x3, so the resolved target must not be the whole-side median.
    whole = _norm.resolve_target_sum(real, input_type="counts", target_sum=None)
    assert expected != whole, "fixture must distinguish the control pool from the full side"


def test_precompute_cache_real_side_entry_is_HIT_by_a_later_target_sum_none_run(
        tmp_path, monkeypatch):
    """#267's safety property, and the only thing that makes the change worth making: the
    entries the one-sided warm writes must be the ones a subsequent `compute_metrics` at
    `target_sum=None` reads. A resolution that differed by any amount would write a warm cache
    that is never read -- silently, since a miss just recomputes.

    Measured at the L2 cache seam: `CacheStore.get_or_compute` records a HIT for the real-side
    pseudobulk key and a MISS for the (deliberately un-warmed) pred side. Counting cache
    outcomes rather than materializations is what makes this test discriminating -- a resolution
    that landed on a different number would still materialize exactly once per side, and only
    the hit/miss split shows the warm entry was read.
    """
    from cell_eval2 import compute_metrics
    from cell_eval2.cache import MISS, CacheStore
    from cell_eval2.run import precompute_cache

    pred, real = _counts_pair(ctrl_scale=3.0)
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")
    cfg = _cfg_median(cache_real=cr, cache_pred=cp)
    # The comparator must be the one the later run RESOLVES to, or the keys differ for a reason
    # that has nothing to do with #267: a v2 counts run resolves `bulk_lognorm` (#264), so a
    # warm at comparator="lognorm" writes `pseudobulk_lognorm` and the run looks for
    # `pseudobulk_bulk_lognorm`. That miss is a comparator mismatch, not a target mismatch.
    precompute_cache(real, side="real", config=cfg, comparator="bulk_lognorm")

    seen = []
    real_get = CacheStore.get

    def recording_get(self, key, *, fingerprint, params, kind):
        out = real_get(self, key, fingerprint=fingerprint, params=params, kind=kind)
        seen.append((str(self.root) if hasattr(self, "root") else "?", key,
                     "MISS" if out is MISS else "HIT", params.get("target_sum")))
        return out

    monkeypatch.setattr(CacheStore, "get", recording_get)
    compute_metrics(pred, real, config=cfg)

    pb = [s for s in seen if s[1].startswith("pseudobulk")]
    assert pb, f"no pseudobulk cache lookups observed; saw {seen}"
    real_side = [s for s in pb if cr in s[0]]
    pred_side = [s for s in pb if cp in s[0]]
    assert real_side and all(s[2] == "HIT" for s in real_side), (
        f"the warm real-side entry was NOT hit (#267): {real_side}")
    assert pred_side and all(s[2] == "MISS" for s in pred_side), (
        f"the un-warmed pred side should miss, so a HIT here means the test proves nothing: "
        f"{pred_side}")
    # And the resolved target actually reached the key, rather than a still-None placeholder.
    assert all(s[3] is not None and s[3] > 0 for s in pb), pb


def test_precompute_cache_real_side_names_a_missing_control_label(tmp_path):
    """#267: resolve_target_sum's own message does not name the label, and a wrong `control` is
    the likeliest cause of an unresolvable one-sided warm."""
    from cell_eval2.run import precompute_cache
    _pred, real = _counts_pair()
    cfg = _cfg_median(cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"),
                      control="not-a-label")
    with pytest.raises(ValueError, match="no rows with target=='not-a-label'"):
        precompute_cache(real, side="real", config=cfg, comparator="lognorm")


def test_precompute_cache_accepts_target_sum_none_on_lognorm(tmp_path):
    """Not blanket (spec section 4.2): on genuinely lognorm input target_sum is inert, the
    one-side entries stay usable by a v1 run, and raising would be a new v1 API regression."""
    import os

    from cell_eval2.run import precompute_cache

    _pred, real = _counts_pair()
    X = np.asarray(real.X, dtype=np.float64)
    lognorm = real.copy()
    lognorm.X = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    v1 = EvalConfig.v1()                       # declares lognorm, target_sum=None
    cfg = replace(v1, metrics=["expr_mae"], validate_input=False, device="cpu",
                  cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"))
    precompute_cache(
        lognorm, side="real", config=cfg, comparator="lognorm"
    )      # must NOT raise
    assert any(os.scandir(str(tmp_path / "r"))), "the lognorm side must still write its cache"


def _arith_lfc(counts_ad, target_sum, *, epsilon, control="non-targeting"):
    """log2FC per gene from arithmetic group means of a normalize_total(target_sum) matrix."""
    from cell_eval2.de_compute import _to_linear
    lin = _to_linear(counts_ad, "counts", target_sum)
    labels = np.asarray(lin.obs["target"]).astype(str)
    X = np.asarray(lin.X)
    ctrl_mean = X[labels == control].mean(axis=0)
    out = {}
    for g in sorted(set(labels) - {control}):
        out[g] = np.log2((X[labels == g].mean(axis=0) + epsilon) / (ctrl_mean + epsilon))
    return out


def test_arithmetic_lfc_at_epsilon_zero_is_target_invariant():
    """The exact form of the invariant: with arithmetic means and epsilon=0 a COMMON target
    cancels, so two different shared targets give identical log2FCs."""
    _pred, real = _counts_pair()
    a = _arith_lfc(real, 1e6, epsilon=0.0)
    b = _arith_lfc(real, 1e4, epsilon=0.0)
    for g in a:
        assert np.allclose(a[g], b[g])


def _sparse_counts_adata():
    """A fixture where epsilon is VISIBLE: several genes are zero in the perturbed groups, so
    the group mean is ~0 and a fixed additive epsilon dominates the ratio. On the dense Poisson
    fixture the means are ~5, and 1e6-vs-1e4 with epsilon=1e-9 differ by ~3e-13 -- inside
    np.allclose's default tolerance, so the assertion below would silently pass."""
    labels = ["non-targeting"] * 10 + ["A"] * 10
    X = np.zeros((20, 4), dtype=np.float64)
    X[:10, :] = [1.0, 2.0, 3.0, 4.0]      # control expresses every gene
    X[10:, 0] = 5.0                        # A expresses ONE gene; the other three are exact 0
    return ad.AnnData(X=X, obs={"target": labels},
                      var={"gene": [f"g{i}" for i in range(4)]})


def test_lfc_with_epsilon_is_not_target_invariant():
    """The limit of the invariant, pinned rather than implied: a nonzero epsilon makes the
    absolute target visible, which is why the anchor population in spec section 3 is a
    modelling choice and not only a provenance one."""
    a = _arith_lfc(_sparse_counts_adata(), 1e6, epsilon=1e-9)
    b = _arith_lfc(_sparse_counts_adata(), 1e4, epsilon=1e-9)
    assert not np.allclose(a["A"], b["A"]), "epsilon must make the absolute target visible"


def _heterogeneous_counts_adata():
    """Within-group rows must DIFFER, or a geometric mean cannot differ from an arithmetic one.

    _sparse_counts_adata makes every row of a group identical, and for identical rows
    expm1(mean(log1p(x))) == x EXACTLY -- so the shared target cancels perfectly and the test
    below would compare two equal vectors and fail against a correct implementation. Equal
    library sizes (10 per cell) keep normalize_total from flattening the within-group spread,
    and every entry is > 0 so epsilon can genuinely be zero.
    """
    labels = ["non-targeting"] * 2 + ["A"] * 2
    X = np.array([[1.0, 9.0], [9.0, 1.0],      # control: one total, opposite compositions
                  [2.0, 8.0], [4.0, 6.0]])     # A:       one total, different compositions
    return ad.AnnData(X=X, obs={"target": labels},
                      var={"gene": ["g0", "g1"]})


def test_geometric_mean_lfc_is_not_target_invariant():
    """The other non-homogeneous case: expm1(mean(log1p(x))) is not scale-invariant, so a
    geometric-mean LFC depends on the shared target even at epsilon=0.

    Worked values (hand-checked, so a failure means the maths moved, not the fixture):
    at target_sum=10 the LFCs are [-0.273, +0.999]; at 1e6 they are [-0.085, +1.208].
    """
    from cell_eval2.de_compute import _to_linear

    def _geo_lfc(target_sum):
        lin = _to_linear(_heterogeneous_counts_adata(), "counts", target_sum)
        labels = np.asarray(lin.obs["target"]).astype(str)
        X = np.asarray(lin.X)
        geo = {g: np.expm1(np.log1p(X[labels == g]).mean(axis=0))
               for g in ("non-targeting", "A")}
        return np.log2(geo["A"] / geo["non-targeting"])   # epsilon=0: every value is > 0

    hi, lo = _geo_lfc(1e6), _geo_lfc(10.0)
    assert np.all(np.isfinite(hi)) and np.all(np.isfinite(lo)), \
        "a -inf on both sides would make the assertion below pass vacuously"
    assert not np.allclose(hi, lo), "expm1(mean(log1p(x))) must not be scale-invariant"


def test_numeric_target_sum_never_computes_a_median(monkeypatch):
    """A genuine no-op proof: if a numeric target_sum ever reached the resolver's median
    branch this would raise. Stronger than running the same implementation twice."""
    from cell_eval2 import compute_metrics, norm
    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)

    def _boom(_X):
        raise AssertionError("_median_library_size must not run for a numeric target_sum")

    monkeypatch.setattr(norm, "_median_library_size", _boom)
    pred, real = _counts_pair()
    out = compute_metrics(pred, real, config=_cfg_median(target_sum=1e6))
    assert out.height > 0


def _mixed_pair(swap=False):
    """(pred, real) whose two sides have DIFFERENT effective input types.

    Reproduces the construction in tests/test_pred_de_control_scale.py:16
    (_lognorm_pred_counts_real) inline -- `tests/` is not an importable package, so it cannot be
    imported across modules -- keeping the two files in agreement about what the supported mixed
    path looks like, CSR matrices included (which is why the shared stub uses sparse-safe row
    totals). The real side's counts are large enough that a lognorm mis-detection would trip the
    scale-limit gate.

    swap=False -> lognorm pred + counts real: the SUPPORTED direction.
    swap=True  -> counts pred + lognorm real: the direction that raises inside _pred_de_input.
    """
    import pandas as pd
    import scipy.sparse as sp

    genes = [f"g{i}" for i in range(4)]
    labels = ["g1", "g1", "g2", "g2", "non-targeting", "non-targeting"]
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=genes)
    lognorm_X = np.array([[0.5, 1.2, 0.0, 2.1], [0.6, 1.0, 0.1, 2.0],
                          [1.5, 0.2, 0.3, 0.0], [1.4, 0.3, 0.2, 0.1],
                          [0.9, 0.9, 0.9, 0.9], [0.8, 1.0, 1.0, 0.7]], dtype=np.float32)
    counts_X = np.array([[10, 50, 0, 80], [12, 45, 1, 75],
                         [60, 5, 9, 0], [55, 7, 8, 2],
                         [40, 60, 30, 90], [38, 62, 28, 88]], dtype=np.float32)

    def mk(X):
        return ad.AnnData(X=sp.csr_matrix(X), obs=obs.copy(), var=var.copy())

    return (mk(counts_X), mk(lognorm_X)) if swap else (mk(lognorm_X), mk(counts_X))


def test_mixed_counts_real_lognorm_pred_still_runs_and_is_not_forced_to_one_scale(monkeypatch):
    """Spec section 8: a counts real side and a lognorm pred side still end up on two scales,
    because the pred's lognorm values encode an unrecoverable original target. Pre-existing,
    orthogonal to #155, and NOT fixed here -- failing loud would break the path
    tests/test_pred_de_control_scale.py covers as supported. Pinned END TO END (that file tests
    only the internal _pred_de_input conversion) so a future change is deliberate."""
    from cell_eval2 import compute_metrics
    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=4)
    pred, real = _mixed_pair()                        # counts real, lognorm pred
    v1 = EvalConfig.v1()
    cfg = replace(v1, pert_col="target_gene", control="non-targeting", control_source="real",
                  metrics=["de_wilcoxon_overlap"], validate_input=False, device="cpu",
                  de=replace(v1.de, backend="gpudge", fdr_scope="per_pert"))
    out = compute_metrics(pred, real, config=cfg)     # must NOT raise
    assert out.height > 0
    ext = [c for c in seen["calls"] if c["ref_totals"] is not None]
    # UNCONDITIONAL: `if ext:` would let this test pass silently the day the external-reference
    # path stops being taken, which is exactly the regression it exists to catch.
    assert ext, "expected the in-memory external-reference DE path (control_source='real')"
    # The known gap: the two halves are NOT forced onto one scale here.
    assert not np.allclose(ext[0]["ref_totals"].mean(),
                           ext[0]["target_totals"].mean(), rtol=1e-3)


def test_mixed_lognorm_real_counts_pred_anndata_only_run_leaves_target_none(monkeypatch):
    """The reverse direction raises only inside _pred_de_input, which an ANNDATA-ONLY run never
    reaches: target_sum stays None (the real side is lognorm, so it cannot be resolved) and the
    pred counts pseudobulk normalizes to its own median. Documented in spec section 8, pinned
    here, not fixed."""
    from cell_eval2 import compute_metrics
    bulks = []
    _spy_side_bulks(monkeypatch, bulks)
    pred, real = _mixed_pair(swap=True)               # lognorm real, counts pred
    v1 = EvalConfig.v1()
    cfg = replace(v1, pert_col="target_gene", control="non-targeting", control_source="real",
                  metrics=["expr_mae"], validate_input=False, device="cpu")
    out = compute_metrics(pred, real, config=cfg)     # anndata-only: no DE, so no raise
    assert out.height > 0
    # "leaves target None" is the CLAIM in the name -- assert it, do not merely assert that the
    # run completed. The pred side is counts, so this is the case where the pred pseudobulk
    # still normalizes to its own median.
    pred_calls = [b for b in bulks if b.get("side") == "pred"]
    assert pred_calls, "expected a pred-side pseudobulk call"
    assert all(b["cfg"].target_sum is None for b in pred_calls)
