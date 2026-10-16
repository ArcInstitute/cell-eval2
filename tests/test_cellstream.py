"""Tests for the cell-layout out-of-core scorer (score_cellstream) and its source."""
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pytest

from cell_eval2.h5ad_manifest import MemBudget
from _helpers import full_minus_moments

PERTS = ["control", "GENE_A", "GENE_B", "GENE_C"]
_BIG = MemBudget(host_bytes=10**12, gpu_bytes=10**12)


def _genes(n_genes):
    targets = PERTS[1:]
    return targets + [f"g{j}" for j in range(len(targets), n_genes)]


def _write_cell(tmp_path, name, seed, *, n_per=30, n_genes=50, contexts=("ctxA",), lognorm=False):
    """Write a grouped, reference-first cell archive; return (shad_path, adata)."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive
    rng = np.random.default_rng(seed)
    nctx = len(contexts)
    n = n_per * len(PERTS) * nctx
    X = rng.poisson(0.5, size=(n, n_genes)).astype(np.float32)
    if lognorm:  # CPM to 1e4 then log1p -> fractional lognorm values
        libs = X.sum(axis=1, keepdims=True)
        libs[libs == 0] = 1.0
        X = np.log1p(X * (1e4 / libs))
    X = sp.csr_matrix(X)
    pert = np.tile(np.repeat(PERTS, n_per), nctx)
    ctx = np.repeat(np.asarray(contexts, dtype=object), n_per * len(PERTS))
    obs = pd.DataFrame(
        {"perturbation": pert, "context": ctx.astype(str),
         "control_value": np.repeat("control", n), "dataset": np.repeat("dsX", n)},
        index=[f"{name}_c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=_genes(n_genes))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    shad = tmp_path / f"{name}.shad"
    ref = (obs["perturbation"] == "control").to_numpy()
    write_cell_archive(adata, shad, group_by="perturbation", reference=ref,
                       codec="zstd", overwrite=True)
    return shad, adata


def test_cellbatchsource_control_block(tmp_path):
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "real", 2)
    src = CellBatchSource(shad, pert_col="perturbation", control="control")
    try:
        ctrl = src.read_control_block()
        assert set(ctrl.obs["perturbation"].astype(str)) == {"control"}
        assert ctrl.n_obs == 30
        assert list(ctrl.var.index) == _genes(50)
    finally:
        src.close()


def test_cellbatchsource_batches_complete(tmp_path):
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "pred", 1)
    src = CellBatchSource(shad, pert_col="perturbation", control="control")
    try:
        batches = list(src.iter_pert_batches(_BIG))
        seen = [p for perts, _ in batches for p in perts]
        assert sorted(seen) == ["GENE_A", "GENE_B", "GENE_C"]   # no control, no dupes, all perts
        for perts, batch_ad in batches:
            assert "control" not in perts
            assert set(batch_ad.obs["perturbation"].astype(str)) == set(perts)
    finally:
        src.close()


def test_cellbatchsource_splits_under_tiny_budget(tmp_path):
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "pred", 1)
    src = CellBatchSource(shad, pert_col="perturbation", control="control")
    try:
        # per_cell = 50 genes * 4 bytes * 3.0 safety = 600 B; control(30)+one pert(30)=60 cells
        # budget just above two perts' worth forces >1 batch but never splits a pert
        tiny = MemBudget(host_bytes=600 * (30 + 45), gpu_bytes=10**12)
        batches = list(src.iter_pert_batches(tiny))
        assert len(batches) >= 2
        seen = [p for perts, _ in batches for p in perts]
        assert sorted(seen) == ["GENE_A", "GENE_B", "GENE_C"]
    finally:
        src.close()


def test_cellbatchsource_passthrough(tmp_path):
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "real", 2)
    src = CellBatchSource(shad, pert_col="perturbation", control="control")
    try:
        want = src.store.read_group("GENE_A").toarray()
        got = None
        for perts, batch_ad in src.iter_pert_batches(_BIG):
            if perts == ["GENE_A"] or "GENE_A" in perts:
                sub = batch_ad[batch_ad.obs["perturbation"].astype(str) == "GENE_A"]
                got = sub.X.toarray()
                break
        np.testing.assert_array_equal(np.sort(got, axis=0), np.sort(want, axis=0))
    finally:
        src.close()


def test_cellbatchsource_context_slice(tmp_path):
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "real", 2, contexts=("ctxA", "ctxB"))
    src = CellBatchSource(shad, pert_col="perturbation", control="control", context="ctxA")
    try:
        ctrl = src.read_control_block()
        assert set(ctrl.obs["context"].astype(str)) == {"ctxA"}
        assert ctrl.n_obs == 30
    finally:
        src.close()


def test_cellbatchsource_empty_context_yields_nothing(tmp_path):
    """Regression (#130, Gemini r3): an empty context slice must not crash iter_pert_batches
    (a size-0 array with the O(N) group-boundary boolean mask -> IndexError)."""
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "real", 2, contexts=("ctxA",))
    src = CellBatchSource(shad, pert_col="perturbation", control="control", context="ZZZ_absent")
    try:
        assert src._rows_all.size == 0                    # empty context slice
        assert list(src.iter_pert_batches(_BIG)) == []    # no crash, yields no batches
    finally:
        src.close()


def test_cellbatchsource_missing_control_raises(tmp_path):
    from cell_eval2.cellstream import CellBatchSource
    shad, _ = _write_cell(tmp_path, "real", 2)
    src = CellBatchSource(shad, pert_col="perturbation", control="nope")
    try:
        with pytest.raises(ValueError, match="control"):
            src.read_control_block()
    finally:
        src.close()


def test_control_label_autodetects_from_control_value(tmp_path):
    from cell_eval2.cellstream import _control_label
    from cell_eval2.config import EvalConfig
    from cell_eval2._cell_archive import open_cell_store
    shad, _ = _write_cell(tmp_path, "real", 2)              # obs.control_value == "control"
    store = open_cell_store(shad)
    # pert_col must match the fixture's obs column: #177 made _control_label VERIFY the label it
    # resolves, and it cannot do that without the column the label has to appear in. The old
    # EvalConfig() default ("target") happened not to matter while nothing was checked.
    cfg = replace(EvalConfig(), pert_col="perturbation")
    try:  # cfg default control is "non-targeting"; control_value column overrides it
        assert _control_label(store, cfg) == "control"
    finally:
        store.close()


def test_control_label_refuses_a_fallback_label_absent_from_the_archive(tmp_path):
    """#177: the fallback to cfg.control was silent, so a preset's label could be adopted for a
    dataset that names its controls something else and the run continued against a control that
    does not exist. The membership check turns that into an error at the point of decision."""
    from cell_eval2.cellstream import _control_label
    from cell_eval2.config import EvalConfig
    from cell_eval2._cell_archive import open_cell_store
    shad, adata = _write_cell(tmp_path, "real", 2)
    del adata.obs["control_value"]                          # the missing-column silent path
    shad2 = tmp_path / "nocv.shad"
    from cellstream.cell import write_cell_archive
    write_cell_archive(adata, str(shad2), group_by="perturbation", reference="control",
                       overwrite=True, codec="zstd")
    store = open_cell_store(shad2)
    cfg = replace(EvalConfig(), pert_col="perturbation", control="non-targeting")
    try:
        with pytest.raises(ValueError, match="does not appear in"):
            _control_label(store, cfg)
    finally:
        store.close()


def test_control_label_accepts_a_correct_cfg_control_without_the_column(tmp_path):
    """The membership check must not turn the LEGITIMATE fallback into an error: a hand-built
    archive with no control_value column and a correct cfg.control still resolves."""
    from cell_eval2.cellstream import _control_label
    from cell_eval2.config import EvalConfig
    from cell_eval2._cell_archive import open_cell_store
    shad, adata = _write_cell(tmp_path, "real", 2)
    del adata.obs["control_value"]
    shad2 = tmp_path / "nocv_ok.shad"
    from cellstream.cell import write_cell_archive
    write_cell_archive(adata, str(shad2), group_by="perturbation", reference="control",
                       overwrite=True, codec="zstd")
    store = open_cell_store(shad2)
    cfg = replace(EvalConfig(), pert_col="perturbation", control="control")
    try:
        assert _control_label(store, cfg) == "control"
    finally:
        store.close()


def test_control_label_warns_and_falls_back_on_a_non_uniform_control_value(tmp_path, caplog):
    """The second silent path: the column is PRESENT but varies. Falling back is still the right
    resolution, but cellstream scores one context, so a varying control_value is anomalous and
    must be said out loud."""
    import logging

    from cell_eval2.cellstream import _control_label
    from cell_eval2.config import EvalConfig
    from cell_eval2._cell_archive import open_cell_store
    # `_write_cell` FIRST: it carries this module's `importorskip("cellstream")`, and CI installs no
    # `scale` extra. Importing cellstream ABOVE the call raises ModuleNotFoundError there instead of
    # skipping -- invisible locally (the dev venv has cellstream) and red in CI. Every other test in
    # this file orders it this way; this one did not.
    shad, adata = _write_cell(tmp_path, "real", 2)
    from cellstream.cell import write_cell_archive
    cv = adata.obs["control_value"].astype(str).to_numpy().copy()
    cv[: len(cv) // 2] = "other"                            # deliberately non-uniform
    adata.obs["control_value"] = cv
    shad2 = tmp_path / "nonuniform.shad"
    write_cell_archive(adata, str(shad2), group_by="perturbation", reference="control",
                       overwrite=True, codec="zstd")
    store = open_cell_store(shad2)
    cfg = replace(EvalConfig(), pert_col="perturbation", control="control")
    try:
        with caplog.at_level(logging.WARNING, logger="cell_eval2.cellstream"):
            assert _control_label(store, cfg) == "control"   # falls back, correctly
        assert "NOT uniform" in caplog.text, caplog.text
        assert any(r.levelname == "WARNING" for r in caplog.records), caplog.text
    finally:
        store.close()


def test_resolve_input_type_counts_and_lognorm(tmp_path):
    from cell_eval2.cellstream import _resolve_input_type_cell
    from cell_eval2.config import EvalConfig
    from cell_eval2._cell_archive import open_cell_store
    counts_shad, _ = _write_cell(tmp_path, "counts", 2, lognorm=False)
    ln_shad, _ = _write_cell(tmp_path, "ln", 3, lognorm=True)
    for path, expect in ((counts_shad, "counts"), (ln_shad, "lognorm")):
        store = open_cell_store(path)
        try:
            assert _resolve_input_type_cell(store, EvalConfig()) == expect
        finally:
            store.close()


def _skip_if_no_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("score_cellstream requires a CUDA device (gpudge external-reference DE)")
    pytest.importorskip("gpudge")


def test_score_cellstream_smoke_gpu(tmp_path):
    _skip_if_no_gpu()
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.h5ad_manifest import ScoreResult
    from cell_eval2.config import EvalConfig
    pred, _ = _write_cell(tmp_path, "pred", 1)
    real, _ = _write_cell(tmp_path, "real", 2)
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    cfg = EvalConfig(device="cuda", pert_col="perturbation",
                     metrics=full_minus_moments())   # control auto-detected -> "control"
    res = score_cellstream(pred, real, config=cfg, mem_budget=_BIG)
    assert isinstance(res, ScoreResult)
    assert sorted(res.per_pert["perturbation"].unique().to_list()) == ["GENE_A", "GENE_B", "GENE_C"]
    assert res.per_pert.height > 0 and res.overall.height > 0
    assert set(res.per_context["context"].to_list()) == {"ctxA"}


def test_score_cellstream_exported():
    import cell_eval2
    assert hasattr(cell_eval2, "score_cellstream")


def test_sanitize_cell_adata_makes_h5ad_writable(tmp_path):
    """Regression (#130): parquet-backed cell archives return a pandas StringArray var index
    (gene names) and can return StringArray obs columns; anndata (<0.11 write semantics) refuses
    to write these to the partition reference-cache h5ad (real_control.h5ad / pred_control.h5ad,
    partition_inmem.py:131/365). _sanitize_cell_adata must cast them to object so the AnnData
    round-trips through write_h5ad, matching h5ad-sourced batches."""
    from cell_eval2.cellstream import _sanitize_cell_adata
    a = ad.AnnData(
        X=sp.csr_matrix(np.ones((3, 4), dtype=np.float32)),
        obs=pd.DataFrame({"perturbation": ["control", "control", "GENE_A"]},
                         index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    # Force the pandas StringArray dtypes that cellstream returns from its parquet obs/var members.
    a.var.index = pd.Index([f"g{j}" for j in range(4)], dtype="string", name="gene")
    a.obs["perturbation"] = a.obs["perturbation"].astype("string")
    assert pd.api.types.is_extension_array_dtype(a.var_names.dtype)      # precondition: the bug
    assert pd.api.types.is_extension_array_dtype(a.obs["perturbation"].dtype)

    out = _sanitize_cell_adata(a)
    assert out is a                                                     # in place
    assert a.var_names.dtype == object
    assert a.obs["perturbation"].dtype == object
    assert list(a.var_names) == [f"g{j}" for j in range(4)]             # values preserved
    a.write_h5ad(tmp_path / "roundtrip.h5ad")   # the real check: no allow_write_nullable RuntimeError


def _assert_parity(got, want):
    """rank/DE metrics bit-exact; continuous rtol/atol 1e-7 (tests/test_cell_source.py pattern)."""
    got = got.sort(["perturbation", "metric"])
    want = want.sort(["perturbation", "metric"])
    assert got["perturbation"].to_list() == want["perturbation"].to_list()
    assert got["metric"].to_list() == want["metric"].to_list()
    metrics = np.asarray(got["metric"].to_list())
    g, w = got["value"].to_numpy(), want["value"].to_numpy()
    is_rank = np.array([m.startswith("pds") or m.startswith("de_") for m in metrics])
    np.testing.assert_array_equal(g[is_rank], w[is_rank])
    np.testing.assert_allclose(g[~is_rank], w[~is_rank], rtol=1e-7, atol=1e-7)


def _pinned(cfg):
    """Pin the AUC floor + per_pert FDR on both sides so pr_auc/roc_auc match AND the partition
    guard is satisfied (partition_inmem._require_partition_config requires fdr_scope='per_pert')."""
    from dataclasses import replace
    return replace(cfg, de=replace(cfg.de, auc_pval_floor="replace_zero",
                                   auc_pval_floor_value=1e-10, fdr_scope="per_pert"))


@pytest.mark.parametrize("target_sum,lognorm", [
    (1e6, False),   # v2 counts@1e6
    (1e4, False),   # v2 counts@1e4
    (1e4, True),    # v2 lognorm@1e4 — exercises the expm1 normalization path
])
def test_score_cellstream_matches_compute_metrics_gpu(tmp_path, target_sum, lognorm):
    _skip_if_no_gpu()
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.run import compute_metrics
    from cell_eval2.config import EvalConfig
    pred, _ = _write_cell(tmp_path, "pred", 1, lognorm=lognorm)
    real, _ = _write_cell(tmp_path, "real", 2, lognorm=lognorm)
    base = EvalConfig(device="cuda", pert_col="perturbation", control="control",
                      target_sum=target_sum, input_type=("lognorm" if lognorm else "counts"),
                      # Moment-consuming expression metrics are unavailable on this driver (#198).
                      metrics=full_minus_moments())
    cfg = _pinned(base)
    got = score_cellstream(pred, real, config=cfg, mem_budget=_BIG).per_pert.drop(
        ["dataset", "panel_id", "context"])
    want = compute_metrics(pred, real, config=cfg)
    _assert_parity(got, want)


def test_score_cellstream_matches_compute_metrics_0_7_6_gpu(tmp_path):
    """Real downstream config: v1 + control_source='pred' (exercises the pred-control reference
    branch) via the cell-eval-0.7.6 preset. _pinned forces fdr_scope='per_pert'."""
    _skip_if_no_gpu()
    from dataclasses import replace
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.run import compute_metrics
    from cell_eval2.config import EvalConfig
    pred, _ = _write_cell(tmp_path, "pred", 1, lognorm=True)
    real, _ = _write_cell(tmp_path, "real", 2, lognorm=True)
    # v1: the profile string lets resolve_metrics filter v2-native metrics silently; an explicit list would raise (#198).
    base = replace(EvalConfig.from_preset("cell-eval-0.7.6"),
                   device="cuda", pert_col="perturbation", control="control", input_type="lognorm")
    cfg = _pinned(base)
    assert cfg.control_source == "pred"          # the branch under test
    got = score_cellstream(pred, real, config=cfg, mem_budget=_BIG).per_pert.drop(
        ["dataset", "panel_id", "context"])
    want = compute_metrics(pred, real, config=cfg)
    _assert_parity(got, want)


class _FakeCellStore:
    """Minimal CellStore stand-in for the #149 wiring guard.

    cellstream is an optional extra CI does not install, so every test gated on
    importorskip("cellstream") is SKIPPED in CI -- including, without this, the central
    regression guard for the change. Implements only what CellBatchSource touches.
    """

    def __init__(self, n_per=100, n_vars=5, perts=("control", "A", "B")):
        self.n_obs = n_per * len(perts)
        self.n_vars = n_vars
        self.manifest = {"group_by": "perturbation", "value_dtype_on_disk": "float32"}
        self._x_out_dtype = np.dtype("float32")
        self._perts = list(perts)
        self.obs = pd.DataFrame(
            {"perturbation": np.repeat(np.asarray(perts, dtype=object), n_per)},
            index=[f"c{i}" for i in range(self.n_obs)],
        )
        # CellBatchSource.__init__ builds stream_tag from cell_fingerprint(store), which goes
        # through cell_metadata -> store.var.index.values + store.group_labels(). Omitting
        # either makes construction raise before a single gather happens.
        self.var = pd.DataFrame(index=[f"g{j}" for j in range(n_vars)])
        self.calls: list[tuple[int, int]] = []      # (n_rows, n_threads)
        self.closed = False

    def group_labels(self):
        return list(self._perts)

    def gather_rows_adata(self, row_ids, n_threads=1):
        rows = np.atleast_1d(np.asarray(row_ids))
        self.calls.append((int(rows.size), n_threads))
        return ad.AnnData(
            X=sp.csr_matrix((rows.size, self.n_vars), dtype=np.float32),
            obs=self.obs.iloc[rows],
            var=pd.DataFrame(index=[f"g{j}" for j in range(self.n_vars)]),
        )

    def close(self):
        self.closed = True


def _sentinel_resolver(monkeypatch, module, value=7):
    """Replace `module.resolve_gather_threads` with one that returns a SENTINEL and records its
    (n_rows, gather_threads) arguments. Patching the name in the CONSUMING module is what
    matters: cellstream/cell_source do `from ._threads import resolve_gather_threads` at import
    time, so patching `_threads` itself would not be seen."""
    seen: list[tuple] = []

    def fake(n_rows, gather_threads):
        seen.append((n_rows, gather_threads))
        return value

    monkeypatch.setattr(module, "resolve_gather_threads", fake)
    return seen


def test_cellbatchsource_plumbs_n_threads_to_every_gather(monkeypatch):
    """#149's central regression guard, and the one that catches the SILENT no-op: an unwired
    call site records cellstream's default n_threads=1, which is indistinguishable from a correctly
    resolved small read. The sentinel makes them distinguishable. cellstream-free -> runs in CI."""
    import cell_eval2.cellstream as cs

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    seen = _sentinel_resolver(monkeypatch, cs, value=7)

    src = cs.CellBatchSource("ignored.shad", pert_col="perturbation", control="control",
                             gather_threads=5)
    try:
        src.read_control_block()
        batches = list(src.iter_pert_batches(_BIG))
    finally:
        src.close()

    assert batches, "no perturbation batches were yielded"
    assert len(store.calls) >= 2, f"expected control + >=1 batch gather, saw {store.calls}"
    assert all(nt == 7 for _, nt in store.calls), (
        f"a gather site is not passing n_threads (got {store.calls})")
    # the resolver saw the REAL row count of each gather, and the configured value verbatim
    assert [n for n, _ in store.calls] == [n_rows for n_rows, _ in seen]
    assert all(gt == 5 for _, gt in seen), seen


def test_cellbatchsource_stores_gather_threads_verbatim(monkeypatch):
    """The config value is stored uncoerced (-1 stays -1) and resolved PER GATHER. int()
    coercion here would silently turn True/1.5 into 1 -- a new silent-serial path."""
    import cell_eval2.cellstream as cs

    store = _FakeCellStore()
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    src = cs.CellBatchSource("ignored.shad", pert_col="perturbation", control="control")
    try:
        assert src.gather_threads == -1
    finally:
        src.close()


def test_resolve_input_type_cell_plumbs_n_threads(monkeypatch):
    """The 2000-row autodetect peek routes through the same resolver (negligible cost, but a
    bare call here would be the one un-wired site). cellstream-free -> runs in CI: the fake store
    returns an all-zero block, so the counts-vs-lognorm verdict is stubbed out — this test is
    about PLUMBING, and the real detection logic is covered by the existing
    test_resolve_input_type_counts_and_lognorm."""
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)                 # 300 rows total, peek asks for 300
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    seen = _sentinel_resolver(monkeypatch, cs, value=7)

    assert cs._resolve_input_type_cell(store, EvalConfig(gather_threads=2)) == "counts"
    assert [nt for _, nt in store.calls] == [7], store.calls
    assert seen == [(min(store.n_obs, 2000), 2)], seen


def test_score_cellstream_wires_gather_threads_to_all_three_sources(monkeypatch):
    """score_cellstream builds THREE CellBatchSources (real reference, pred-control with
    control_source='pred', pred batches) and must hand cfg.gather_threads to every one, plus to
    the input-type peek. cellstream-free and CPU-only: the store, the GPU/gpudge partition guard,
    the two core builders and the aggregator are all stubbed.

    NOTE on the monkeypatch targets: score_cellstream imports _require_partition_config /
    _build_*_core / aggregate_partials INSIDE its own body, so those `from X import Y` statements
    run AFTER this patching and pick up the patched attributes. Patching the modules is correct
    here precisely because the imports are function-local.
    """
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    class _Done(Exception):
        """Reached the aggregator -- everything under test has already run."""

    def _stop(*a, **k):
        raise _Done

    monkeypatch.setattr(partition, "aggregate_partials", _stop)

    peeked: list[int] = []
    orig_peek = cs._resolve_input_type_cell

    def peek_spy(store_, cfg, **kw):
        peeked.append(cfg.gather_threads)
        return orig_peek(store_, cfg, **kw)

    monkeypatch.setattr(cs, "_resolve_input_type_cell", peek_spy)

    seen: list[object] = []

    class _RecordingSource:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("gather_threads"))

        def iter_pert_batches(self, mem_budget):
            return iter(())

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _RecordingSource)

    # pert_col="perturbation", NOT "target": score_cellstream rewrites the literal "target" to
    # "perturbation" (its `if cfg.pert_col == "target"` guard), which would then not match this
    # fake's obs column.
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    # control="control" matches _FakeCellStore's labels. #177 made _control_label verify the
    # label it resolves, and the old default ("non-targeting") appears nowhere in that store --
    # i.e. these wiring guards were themselves running against a control that did not exist,
    # which passed only because everything downstream of the resolution is stubbed out.
    cfg = EvalConfig(pert_col="perturbation", control="control", gather_threads=6,
                     control_source="pred", metrics=full_minus_moments())
    with pytest.raises(_Done):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)

    # Comparator resolution inspects BOTH archives independently; a one-sided peek would miss
    # an asymmetric counts/lognorm pair and can be passed by reading the prediction twice.
    assert peeked == [6, 6]
    assert seen == [6, 6, 6], f"expected all three CellBatchSources wired, got {seen}"


def test_cellstream_output_unchanged_by_threading(tmp_path, monkeypatch):
    """Parity on a REAL archive: threading is a pure speed change. Uses n_per=200 so the reads
    are big enough that gather_threads=8 actually resolves to >1 -- with the default 30-row
    fixture both arms resolve to 1 and the test would prove nothing."""
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cellstream import CellBatchSource

    shad, _ = _write_cell(tmp_path, "big", 2, n_per=200)
    out, threads_used = {}, {}
    for gt in (1, 8):
        probe = open_cell_store(shad)
        cls = type(probe)
        probe.close()
        seen: list[int] = []
        orig = cls.gather_rows_adata

        def spy(self, row_ids, n_threads=1, _seen=seen, _orig=orig):
            _seen.append(n_threads)
            return _orig(self, row_ids, n_threads=n_threads)

        monkeypatch.setattr(cls, "gather_rows_adata", spy)
        src = CellBatchSource(shad, pert_col="perturbation", control="control",
                              gather_threads=gt)
        try:
            ctrl = src.read_control_block().X.toarray()
            batches = [(perts, a.X.toarray()) for perts, a in src.iter_pert_batches(_BIG)]
        finally:
            src.close()
            monkeypatch.setattr(cls, "gather_rows_adata", orig)
        out[gt] = (ctrl, batches)
        threads_used[gt] = seen

    assert max(threads_used[8]) > 1, (
        f"gather_threads=8 never engaged >1 thread ({threads_used[8]}) -- this parity test "
        "would be vacuous; enlarge the fixture")
    assert max(threads_used[1]) == 1, threads_used[1]
    ctrl1, b1 = out[1]
    ctrl8, b8 = out[8]
    np.testing.assert_array_equal(ctrl1, ctrl8)
    assert [p for p, _ in b1] == [p for p, _ in b8]
    for (_, x1), (_, x8) in zip(b1, b8):
        np.testing.assert_array_equal(x1, x8)


def test_score_cellstream_passes_one_bundle_to_every_piece(monkeypatch):
    """The point of #153's second fix: ONE bundle per context, not one per batch. Same CPU-only
    stubbing as the gather_threads test -- the store, the partition guard, both core builders and
    the aggregator are stubbed, so neither a GPU nor cellstream is involved. mem_budget is the
    module's _BIG MemBudget (an int would blow up in plan_pert_batches); the fake source ignores
    it and yields a fixed three batches, so 'more than one piece' is guaranteed, not hoped for."""
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    built = []

    def _fake_bundle(cache_dir, cfg):
        built.append(cache_dir)
        return ("bundle", cache_dir, len(built))

    monkeypatch.setattr(pinmem, "_RefBundle", _fake_bundle)

    class _ThreeBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([((f"p{i}",), object()) for i in range(3)])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _ThreeBatchSource)

    bundles = []
    monkeypatch.setattr(pinmem, "score_piece",
                        lambda *a, **k: bundles.append(k.get("bundle")) or None)

    class _Done(Exception):
        pass

    def _stop(*a, **k):
        raise _Done

    monkeypatch.setattr(partition, "aggregate_partials", _stop)

    # Moment-consuming expression metrics are unavailable on this driver (#198).
    cfg = EvalConfig(pert_col="perturbation", control="control", control_source="real",
                     metrics=full_minus_moments())
    with pytest.raises(_Done):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)

    assert len(bundles) == 3, f"expected 3 pieces, got {len(bundles)}"
    assert all(b is bundles[0] for b in bundles) and bundles[0] is not None
    assert len(built) == 1, f"_RefBundle must be built ONCE per context, was built {len(built)}x"


# --- #387: opt-in DE-table export from the streaming path -------------------------------------

def test_score_cellstream_write_de_requires_outdir():
    """The one combination that cannot be honoured must fail fast, before any archive I/O:
    write_de asks for de_real/de_pred to be written, and they have nowhere to land without an
    outdir. The check precedes every open_cell_store, so no stubbing is needed here."""
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig
    with pytest.raises(ValueError, match=r"write_de=True requires outdir"):
        score_cellstream("pred.shad", "real.shad",
                         config=EvalConfig(pert_col="perturbation", control="control"),
                         mem_budget=_BIG, write_de=True)


def _de_piece_frame(target, n=2):
    """A small, canonical-column pred/real DE frame -- the same columns compute_de produces --
    for the write_de tests below."""
    import polars as pl
    return pl.DataFrame({
        "target": [target] * n,
        "feature": [f"g{i}" for i in range(n)],
        "target_mean": [1.0 + i for i in range(n)],
        "ref_mean": [0.5] * n,
        "target_ncells": [10] * n,
        "ref_ncells": [8] * n,
        "log2_fold_change": [1.5 - i for i in range(n)],
        "p_value": [0.01 * (i + 1) for i in range(n)],
        "U": [5.0 - i for i in range(n)],
        "p_adj": [0.02 * (i + 1) for i in range(n)],
        "abs_log2_fold_change": [abs(1.5 - i) for i in range(n)],
    })


def test_score_cellstream_write_de_emits_de_tables(tmp_path, monkeypatch):
    """write_de=True must (a) thread de_out to every score_piece and (b) publish de_real.parquet
    + de_pred.parquet to outdir beside the three ScoreResult frames. de_real holds exactly the
    reference builder's real_de.parquet table, re-encoded through de_export's pinned writer;
    de_pred is de_export.merge_de_pieces over the per-piece pred DE tables, in generation order.
    Same CPU-only stubbing as the wiring tests above (store, partition guard, both core builders,
    RefBundle, score_piece and the aggregator are stubbed), so neither a GPU nor cellstream is
    involved."""
    import io
    import os
    import polars as pl
    from polars.testing import assert_frame_equal
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.h5ad_manifest as h5m
    from cell_eval2.config import EvalConfig
    from cell_eval2.de_export import DE_PARQUET_ROW_GROUP_ROWS, read_de_parquet_metadata

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    # The reference builder writes the full real DE table into cache_dir (=ref_dir). Captured
    # bytes are the "bundle file" the published de_real.parquet must exactly match.
    captured: dict[str, bytes] = {}

    def _fake_ref(source, *, config, cache_dir, **k):
        real_path = os.path.join(cache_dir, "real_de.parquet")
        _de_piece_frame("REAL_ALL").write_parquet(real_path)
        with open(real_path, "rb") as fh:
            captured["real_de_bytes"] = fh.read()
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _fake_ref)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class _TwoBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([(("GENE_A",), object()), (("GENE_B",), object())])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _TwoBatchSource)

    # score_piece writes one pred DE table per piece into de_out, and records what it was handed.
    de_outs: list[object] = []

    def _fake_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                    bundle=None, comparator=None, de_out=None, native_gpu_normalize=False):
        de_outs.append(de_out)
        if de_out is not None:
            _de_piece_frame(f"PRED_{piece_id}").write_parquet(
                os.path.join(de_out, f"{piece_id}__de_pred.parquet"))
        return None
    monkeypatch.setattr(pinmem, "score_piece", _fake_piece)

    monkeypatch.setattr(partition, "aggregate_partials",
                        lambda *a, **k: (pl.DataFrame({"perturbation": ["GENE_A"]}), None))

    class _FakeResult:
        per_pert = pl.DataFrame({"perturbation": ["GENE_A"]})
        per_context = pl.DataFrame({"context": ["all"]})
        overall = pl.DataFrame({"metric": ["x"]})
    monkeypatch.setattr(h5m, "_assemble_score_result", lambda frames: _FakeResult())

    outdir = tmp_path / "out"
    cfg = EvalConfig(pert_col="perturbation", control="control", control_source="real",
                     metrics=full_minus_moments())
    cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                        outdir=str(outdir), write_de=True)

    # Every piece was handed the SAME de_out directory (not None).
    assert len(de_outs) == 2 and all(d is not None for d in de_outs)
    assert de_outs[0] is de_outs[1]

    de_real_path = outdir / "de_real.parquet"
    de_pred_path = outdir / "de_pred.parquet"

    # de_real is the bundle's table -- same rows, values, types and order -- re-encoded under the
    # pinned contract rather than copied with whatever writer produced the bundle file.
    assert_frame_equal(pl.read_parquet(de_real_path),
                       pl.read_parquet(io.BytesIO(captured["real_de_bytes"])), check_exact=True)

    de_pred = pl.read_parquet(de_pred_path)
    assert de_pred.height == 4  # 2 pieces x 2 rows each
    # de_pred concatenates both pieces in generation order (ctx_0 before ctx_1).
    assert de_pred["target"].to_list() == ["PRED_ctx_0"] * 2 + ["PRED_ctx_1"] * 2
    for path in (de_real_path, de_pred_path):
        meta = read_de_parquet_metadata(str(path))
        assert meta["codecs"] == ["ZSTD"], path
        assert meta["max_row_group_rows"] <= DE_PARQUET_ROW_GROUP_ROWS, path

    # All five files published together, and nothing else: no temp or backup debris, and no
    # provenance sidecar (that belongs to the ground-truth cache, not to this export).
    assert sorted(p.name for p in outdir.iterdir()) == [
        "de_pred.parquet", "de_real.parquet", "overall.parquet", "per_context.parquet",
        "per_pert.parquet"]


def test_score_cellstream_write_de_false_writes_no_de_tables(tmp_path, monkeypatch):
    """Off by default: write_de=False (the existing callers' path) passes de_out=None to every
    piece and emits no de_real/de_pred, only the ScoreResult parquets."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.h5ad_manifest as h5m
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class _OneBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([(("GENE_A",), object())])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _OneBatchSource)

    de_outs: list[object] = []
    monkeypatch.setattr(pinmem, "score_piece",
                        lambda *a, **k: de_outs.append(k.get("de_out")) or None)
    monkeypatch.setattr(partition, "aggregate_partials",
                        lambda *a, **k: (pl.DataFrame({"perturbation": ["GENE_A"]}), None))

    class _FakeResult:
        per_pert = pl.DataFrame({"perturbation": ["GENE_A"]})
        per_context = pl.DataFrame({"context": ["all"]})
        overall = pl.DataFrame({"metric": ["x"]})
    monkeypatch.setattr(h5m, "_assemble_score_result", lambda frames: _FakeResult())

    outdir = tmp_path / "out"
    cfg = EvalConfig(pert_col="perturbation", control="control", control_source="real",
                     metrics=full_minus_moments())
    cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                        outdir=str(outdir))

    assert de_outs == [None]
    assert sorted(os.listdir(outdir)) == [   # the unchanged ScoreResult output, and only that
        "overall.parquet", "per_context.parquet", "per_pert.parquet"]


def test_score_cellstream_write_de_piece_failure_publishes_nothing(tmp_path, monkeypatch):
    """A score_piece failure mid-loop must propagate, and outdir must be left completely
    untouched -- no de_pred.parquet, no de_real.parquet, no .tmp. debris. The failure happens
    INSIDE the pred_src loop, well before anything is staged, so outdir is never even created."""
    import os
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    def _fake_ref(source, *, config, cache_dir, **k):
        _de_piece_frame("REAL_ALL").write_parquet(os.path.join(cache_dir, "real_de.parquet"))
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _fake_ref)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class _TwoBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([(("GENE_A",), object()), (("GENE_B",), object())])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _TwoBatchSource)

    calls: list[str] = []

    def _fake_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                    bundle=None, comparator=None, de_out=None, native_gpu_normalize=False):
        calls.append(piece_id)
        if piece_id == "ctx_1":
            raise RuntimeError("boom: piece scoring failed")
        if de_out is not None:
            _de_piece_frame(f"PRED_{piece_id}").write_parquet(
                os.path.join(de_out, f"{piece_id}__de_pred.parquet"))
        return None
    monkeypatch.setattr(pinmem, "score_piece", _fake_piece)

    outdir = tmp_path / "out"
    cfg = EvalConfig(pert_col="perturbation", control="control", control_source="real",
                     metrics=full_minus_moments())
    with pytest.raises(RuntimeError, match="boom"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(outdir), write_de=True)

    assert calls == ["ctx_0", "ctx_1"]
    assert not outdir.exists()


def test_score_cellstream_write_de_requires_a_de_metric(tmp_path, monkeypatch):
    """write_de=True with a metric selection that resolves to no DE-kind metric must fail
    BEFORE any archive I/O -- open_cell_store is stubbed to raise if reached, so a call that
    slipped past the check would fail loudly here instead of deep inside the with-block."""
    import cell_eval2.cellstream as cs
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.config import EvalConfig

    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)

    def _must_not_be_reached(*a, **k):
        raise AssertionError("open_cell_store must not be reached")
    monkeypatch.setattr(cs, "open_cell_store", _must_not_be_reached)

    # expr_mae is 'anndata'-kind (CATALOG), not 'de'-kind -- an explicit list of only this
    # resolves to zero DE metrics.
    cfg = EvalConfig(pert_col="perturbation", control="control", metrics=["expr_mae"])
    with pytest.raises(ValueError, match=r"write_de=True requires at least one DE metric"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(tmp_path / "out"), write_de=True)


def test_score_cellstream_write_de_preserves_batch_generation_order_beyond_ten_pieces(
        tmp_path, monkeypatch):
    """[P1] Pieces are named ctx_0, ctx_1, ..., ctx_10, ctx_11 (12 batches). A lexicographic sort
    of those FILE NAMES orders them ctx_0, ctx_1, ctx_10, ctx_11, ctx_2, ..., ctx_9 -- not
    generation order. de_pred.parquet's row order must be generation order (the order pred_parts
    is appended in score_cellstream's batch loop), so with each piece carrying a distinct target
    P00..P11 (its own generation index, zero-padded), the merged table's target column must read
    P00, P01, ..., P11 in that exact order."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.h5ad_manifest as h5m
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    def _fake_ref(source, *, config, cache_dir, **k):
        _de_piece_frame("REAL_ALL").write_parquet(os.path.join(cache_dir, "real_de.parquet"))
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _fake_ref)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    n_pieces = 12

    class _TwelveBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([((f"p{i}",), object()) for i in range(n_pieces)])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _TwelveBatchSource)

    def _fake_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                    bundle=None, comparator=None, de_out=None, native_gpu_normalize=False):
        if de_out is not None:
            k = int(piece_id.rsplit("_", 1)[1])
            _de_piece_frame(f"P{k:02d}", n=1).write_parquet(
                os.path.join(de_out, f"{piece_id}__de_pred.parquet"))
        return None
    monkeypatch.setattr(pinmem, "score_piece", _fake_piece)

    monkeypatch.setattr(partition, "aggregate_partials",
                        lambda *a, **k: (pl.DataFrame({"perturbation": ["p0"]}), None))

    class _FakeResult:
        per_pert = pl.DataFrame({"perturbation": ["p0"]})
        per_context = pl.DataFrame({"context": ["all"]})
        overall = pl.DataFrame({"metric": ["x"]})
    monkeypatch.setattr(h5m, "_assemble_score_result", lambda frames: _FakeResult())

    outdir = tmp_path / "out"
    cfg = EvalConfig(pert_col="perturbation", control="control", control_source="real",
                     metrics=full_minus_moments())
    cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                        outdir=str(outdir), write_de=True)

    got = pl.read_parquet(outdir / "de_pred.parquet")["target"].to_list()
    expected = [f"P{k:02d}" for k in range(n_pieces)]
    assert got == expected, (
        f"got {got}, expected generation order {expected} -- a lexicographic-by-filename sort "
        "would instead read P00, P01, P10, P11, P02, P03, ..., P09"
    )


# --- [P1]/[P2] _StagedOutputs rollback-on-error and partial-staging cleanup --------------------


def _staging_fixture(tmp_path, *, pre_existing=False):
    """A minimal real_de_path + two piece paths for _stage_de_outputs, and (when pre_existing)
    OLD bytes already sitting at three of the destinations in outdir."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    real_de_path = str(src_dir / "real_de.parquet")
    _de_piece_frame("REAL_NEW").write_parquet(real_de_path)
    piece_paths = []
    for i, tgt in enumerate(["NEW_A", "NEW_B"]):
        p = src_dir / f"piece_{i}.parquet"
        _de_piece_frame(tgt, n=1).write_parquet(p)
        piece_paths.append(str(p))

    outdir = tmp_path / "out"
    outdir.mkdir()
    old_bytes = {}
    if pre_existing:
        for name, tgt in (("de_real.parquet", "OLD_REAL"), ("de_pred.parquet", "OLD_PRED"),
                          ("per_pert.parquet", "OLD_SCORES")):
            _de_piece_frame(tgt).write_parquet(outdir / name)
            old_bytes[name] = (outdir / name).read_bytes()
    return outdir, real_de_path, piece_paths, old_bytes


def _debris(outdir):
    return [p.name for p in outdir.iterdir() if ".tmp." in p.name or ".bak." in p.name]


@pytest.mark.parametrize("pre_existing", [True, False])
def test_staged_outputs_roll_back_a_failed_publish(tmp_path, monkeypatch, pre_existing):
    """[P1] A failure on a LATER os.replace (de_pred.parquet's) must roll back the file already
    published (de_real.parquet) and leave the not-yet-published one (per_pert.parquet) alone --
    otherwise one new file would be left beside a stale or missing other. With pre-existing
    files every destination must hold its OLD bytes afterward; without, outdir must be empty. No
    .tmp./.bak. debris either way."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs

    outdir, real_de_path, piece_paths, old_bytes = _staging_fixture(
        tmp_path, pre_existing=pre_existing)
    staging = cs._StagedOutputs(str(outdir))
    cs._stage_de_outputs(staging, real_de_path=real_de_path, piece_paths=piece_paths)
    staging.stage("per_pert.parquet", pl.DataFrame({"perturbation": ["NEW"]}).write_parquet)

    real_replace = os.replace
    fired = {"done": False}

    def flaky(src, dst, *a, **kw):
        # Fires exactly ONCE, on the first os.replace targeting de_pred.parquet -- i.e. the
        # publish of the NEW content. A later restore-from-backup call for the same destination
        # (rollback replacing it with the saved OLD bytes) must still succeed, or this injection
        # would itself defeat the rollback it is supposed to be testing.
        if str(dst).endswith("de_pred.parquet") and not fired["done"]:
            fired["done"] = True
            raise OSError("disk full")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", flaky)

    with pytest.raises(OSError, match="disk full"):
        staging.publish()
    monkeypatch.undo()

    assert fired["done"]
    assert _debris(outdir) == []
    if pre_existing:
        for name, data in old_bytes.items():
            assert (outdir / name).read_bytes() == data, name
        assert sorted(p.name for p in outdir.iterdir()) == sorted(old_bytes)
    else:
        assert list(outdir.iterdir()) == []


@pytest.mark.parametrize("failing_call", [0, 1], ids=["de_real", "de_pred"])
def test_stage_de_outputs_cleans_up_a_partial_write(tmp_path, monkeypatch, failing_call):
    """[P2] merge_de_pieces creates its destination and then raises mid-write -- for de_real
    (the first call) or for de_pred (the second, after de_real is fully staged). Neither the
    partial file nor an already-staged one may survive discard(), and nothing is published."""
    from cell_eval2 import de_export
    import cell_eval2.cellstream as cs

    outdir, real_de_path, piece_paths, _ = _staging_fixture(tmp_path)
    real_merge = de_export.merge_de_pieces
    calls = {"n": 0}

    def flaky_merge(paths, dest, **kw):
        k = calls["n"]
        calls["n"] += 1
        assert kw == {"report": False}   # staging never pays for the full-output hash
        if k == failing_call:
            with open(dest, "wb") as fh:
                fh.write(b"PARTIAL")
            raise ValueError("merge exploded")
        return real_merge(paths, dest, **kw)

    monkeypatch.setattr(de_export, "merge_de_pieces", flaky_merge)
    staging = cs._StagedOutputs(str(outdir))
    with pytest.raises(ValueError, match="merge exploded"):
        cs._stage_de_outputs(staging, real_de_path=real_de_path, piece_paths=piece_paths)
    staging.discard()
    monkeypatch.undo()

    assert calls["n"] == failing_call + 1
    assert list(outdir.iterdir()) == []


def test_score_cellstream_write_de_score_file_failure_publishes_nothing(tmp_path, monkeypatch):
    """[#388 review, P2] The DE tables are staged inside the with-block and the ScoreResult
    frames after it. A failure writing a score frame -- after both DE tables are fully staged --
    must publish NOTHING. Before the five files were published together, the DE tables were
    already in outdir by then, so the run left its own de_real/de_pred beside an older (or
    missing) per_context.parquet. Pre-existing outputs must survive untouched, with no debris."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.h5ad_manifest as h5m
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    def _fake_ref(source, *, config, cache_dir, **k):
        _de_piece_frame("REAL_ALL").write_parquet(os.path.join(cache_dir, "real_de.parquet"))
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _fake_ref)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class _OneBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([(("GENE_A",), object())])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _OneBatchSource)

    def _fake_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                    bundle=None, comparator=None, de_out=None, native_gpu_normalize=False):
        if de_out is not None:
            _de_piece_frame(f"PRED_{piece_id}").write_parquet(
                os.path.join(de_out, f"{piece_id}__de_pred.parquet"))
        return None
    monkeypatch.setattr(pinmem, "score_piece", _fake_piece)
    monkeypatch.setattr(partition, "aggregate_partials",
                        lambda *a, **k: (pl.DataFrame({"perturbation": ["GENE_A"]}), None))

    class _BrokenFrame:
        def write_parquet(self, path):
            with open(path, "wb") as fh:
                fh.write(b"PARTIAL")
            raise OSError("score frame write failed")

    class _FakeResult:
        per_pert = pl.DataFrame({"perturbation": ["GENE_A"]})
        per_context = _BrokenFrame()
        overall = pl.DataFrame({"metric": ["x"]})
    monkeypatch.setattr(h5m, "_assemble_score_result", lambda frames: _FakeResult())

    outdir = tmp_path / "out"
    outdir.mkdir()
    old = {}
    for name in ("de_real.parquet", "de_pred.parquet", "per_pert.parquet"):
        _de_piece_frame(f"OLD_{name}").write_parquet(outdir / name)
        old[name] = (outdir / name).read_bytes()

    cfg = EvalConfig(pert_col="perturbation", control="control", control_source="real",
                     metrics=full_minus_moments())
    with pytest.raises(OSError, match="score frame write failed"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(outdir), write_de=True)

    assert sorted(p.name for p in outdir.iterdir()) == sorted(old)
    for name, data in old.items():
        assert (outdir / name).read_bytes() == data, name


@pytest.mark.parametrize("pre_existing", [True, False])
@pytest.mark.parametrize("when", ["before_rename", "after_rename"])
def test_staged_outputs_roll_back_an_interrupt(tmp_path, monkeypatch, pre_existing, when):
    """[P1, checkpoint-2 review] KeyboardInterrupt is not an Exception, and an interrupt can land
    right AFTER a rename, before its bookkeeping. Either way publish() must roll back: every
    destination holds its OLD bytes (or is absent), no debris, and the interrupt propagates."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs

    outdir, real_de_path, piece_paths, old_bytes = _staging_fixture(
        tmp_path, pre_existing=pre_existing)
    staging = cs._StagedOutputs(str(outdir))
    cs._stage_de_outputs(staging, real_de_path=real_de_path, piece_paths=piece_paths)
    staging.stage("per_pert.parquet", pl.DataFrame({"perturbation": ["NEW"]}).write_parquet)

    real_replace = os.replace
    fired = {"done": False}

    def interrupted(src, dst, *a, **kw):
        # On the publish of the NEW de_pred.parquet (a temp moving INTO place), exactly once.
        if str(dst).endswith("de_pred.parquet") and ".tmp." in str(src) and not fired["done"]:
            fired["done"] = True
            if when == "after_rename":
                real_replace(src, dst, *a, **kw)
            raise KeyboardInterrupt
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        staging.publish()
    monkeypatch.undo()

    assert fired["done"]
    assert _debris(outdir) == []
    if pre_existing:
        for name, data in old_bytes.items():
            assert (outdir / name).read_bytes() == data, name
        assert sorted(p.name for p in outdir.iterdir()) == sorted(old_bytes)
    else:
        assert list(outdir.iterdir()) == []


def _staged_three(tmp_path, cs):
    """A staging of de_real + de_pred + per_pert over three pre-existing OLD outputs."""
    import polars as pl

    outdir, real_de_path, piece_paths, old_bytes = _staging_fixture(tmp_path, pre_existing=True)
    staging = cs._StagedOutputs(str(outdir))
    cs._stage_de_outputs(staging, real_de_path=real_de_path, piece_paths=piece_paths)
    new_scores = pl.DataFrame({"perturbation": ["NEW"]})
    staging.stage("per_pert.parquet", new_scores.write_parquet)
    return outdir, staging, old_bytes, new_scores


def _assert_all_new(outdir, old_bytes, new_scores):
    import polars as pl

    for name, data in old_bytes.items():
        assert (outdir / name).read_bytes() != data, f"{name} was rolled back after the commit"
    assert pl.read_parquet(outdir / "per_pert.parquet").equals(new_scores)


def test_staged_outputs_an_interrupt_after_the_last_rename_is_not_rolled_back(tmp_path,
                                                                             monkeypatch):
    """[Codex round 3 on #398] The publish COMMITS when its last rename completes. An interrupt
    landing just after that rename, before publish() returns, must not roll back: every
    destination keeps its NEW file, the backups are removed, and the interrupt propagates."""
    import os
    import cell_eval2.cellstream as cs

    outdir, staging, old_bytes, new_scores = _staged_three(tmp_path, cs)
    real_replace = os.replace

    def interrupted(src, dst, *a, **kw):
        out = real_replace(src, dst, *a, **kw)
        if str(dst).endswith("per_pert.parquet") and ".tmp." in str(src):   # the LAST rename
            raise KeyboardInterrupt
        return out

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        staging.publish()
    monkeypatch.undo()

    assert _debris(outdir) == []
    assert sorted(p.name for p in outdir.iterdir()) == sorted(old_bytes)
    _assert_all_new(outdir, old_bytes, new_scores)


@pytest.mark.parametrize("vanishing", ["de_pred.parquet", "per_pert.parquet"],
                         ids=["intermediate", "final"])
def test_staged_outputs_a_vanished_temp_is_rolled_back_not_committed(tmp_path, monkeypatch,
                                                                    vanishing):
    """[Codex round 4 on #398] A staged temp that vanishes before its rename makes os.replace
    raise. Reading "temp gone" as "published" would declare the FINAL one committed and delete
    the backups -- losing the previous outputs beside a partial new set. Every destination must
    hold its OLD bytes afterward, with no debris."""
    import os
    import cell_eval2.cellstream as cs

    outdir, staging, old_bytes, _new = _staged_three(tmp_path, cs)
    real_replace = os.replace

    def vanish_then_replace(src, dst, *a, **kw):
        if str(dst).endswith(vanishing) and ".tmp." in str(src):
            os.remove(src)          # gone before its rename: os.replace raises FileNotFoundError
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", vanish_then_replace)
    with pytest.raises(FileNotFoundError):
        staging.publish()
    monkeypatch.undo()

    assert _debris(outdir) == []
    assert sorted(p.name for p in outdir.iterdir()) == sorted(old_bytes)
    for name, data in old_bytes.items():
        assert (outdir / name).read_bytes() == data, name


def test_staged_outputs_an_unreadable_published_file_is_reported_not_skipped(tmp_path,
                                                                            monkeypatch):
    """[Codex round 5 on #398] In a fresh outdir, de_real is published, then de_pred's rename
    fails -- and de_real cannot even be stat'ed during the rollback. Nobody can say whether it
    holds this run's file, so the rollback must NAME it in its RuntimeError rather than drop it
    from the cleanup silently (the original failure is chained)."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs

    outdir, real_de_path, piece_paths, _ = _staging_fixture(tmp_path)        # no old outputs
    staging = cs._StagedOutputs(str(outdir))
    cs._stage_de_outputs(staging, real_de_path=real_de_path, piece_paths=piece_paths)
    staging.stage("per_pert.parquet", pl.DataFrame({"perturbation": ["NEW"]}).write_parquet)
    unreadable = os.path.join(str(outdir), "de_real.parquet")
    real_replace, real_stat = os.replace, os.stat

    def failing_replace(src, dst, *a, **kw):
        if str(dst).endswith("de_pred.parquet") and ".tmp." in str(src):
            raise OSError("disk full")
        return real_replace(src, dst, *a, **kw)

    def failing_stat(path, *a, **kw):
        if str(path) == unreadable and kw.get("follow_symlinks") is False:
            raise PermissionError("cannot stat")
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(os, "replace", failing_replace)
    monkeypatch.setattr(os, "stat", failing_stat)
    with pytest.raises(RuntimeError, match=r"could not be read.*de_real\.parquet") as info:
        staging.publish()
    monkeypatch.undo()
    assert isinstance(info.value.__cause__, OSError) and "disk full" in str(info.value.__cause__)


def test_staged_outputs_an_unreadable_backup_is_still_restored(tmp_path, monkeypatch):
    """[Copilot round 5 on #398] A backup whose metadata cannot be read must not be mistaken for
    one that was never moved aside: the rollback still restores it, so the OLD output returns."""
    import os
    import cell_eval2.cellstream as cs

    outdir, staging, old_bytes, _new = _staged_three(tmp_path, cs)
    real_replace, real_lstat = os.replace, os.lstat

    def failing_replace(src, dst, *a, **kw):
        if str(dst).endswith("de_pred.parquet") and ".tmp." in str(src):
            raise OSError("disk full")
        return real_replace(src, dst, *a, **kw)

    def unreadable_backup(path, *a, **kw):
        if ".bak." in str(path) and "de_real" in os.path.basename(str(path)):
            raise PermissionError("cannot lstat")
        return real_lstat(path, *a, **kw)

    monkeypatch.setattr(os, "replace", failing_replace)
    monkeypatch.setattr(os, "lstat", unreadable_backup)
    with pytest.raises(OSError, match="disk full"):
        staging.publish()
    monkeypatch.undo()

    assert (outdir / "de_real.parquet").read_bytes() == old_bytes["de_real.parquet"]
    assert _debris(outdir) == []


def test_score_cellstream_write_de_refuses_an_empty_outdir_before_any_io(monkeypatch):
    """[Copilot round 5 on #398] outdir="" does not exist, so it passed the up-front checks and
    failed only at os.makedirs after both archives had been scored."""
    import cell_eval2.cellstream as cs

    cfg = _stub_write_de_run(monkeypatch)

    def _must_not_be_reached(*a, **k):
        raise AssertionError("an archive was opened before the empty outdir was refused")
    monkeypatch.setattr(cs, "open_cell_store", _must_not_be_reached)
    with pytest.raises(ValueError, match="write_de=True requires outdir"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir="", write_de=True)


def test_staged_outputs_an_interrupt_during_backup_cleanup_keeps_the_commit(tmp_path, monkeypatch):
    """[Copilot / Codex on #398] An interrupt while the backups are removed after the commit is
    not rolled back either: the new outputs stand, the interrupt propagates, and what it leaves
    behind is only stale .bak copies -- never a temp, never a half-published set."""
    import os
    import cell_eval2.cellstream as cs

    outdir, staging, old_bytes, new_scores = _staged_three(tmp_path, cs)
    real_remove = os.remove

    def interrupted(path, *a, **kw):
        if ".bak." in str(path):
            raise KeyboardInterrupt
        return real_remove(path, *a, **kw)

    monkeypatch.setattr(os, "remove", interrupted)
    with pytest.raises(KeyboardInterrupt):
        staging.publish()
    monkeypatch.undo()

    debris = _debris(outdir)
    assert debris and all(".bak." in name for name in debris), debris
    _assert_all_new(outdir, old_bytes, new_scores)


def test_staged_outputs_a_backup_that_cannot_be_removed_is_logged_not_swallowed(tmp_path,
                                                                                monkeypatch,
                                                                                caplog):
    """[Copilot round 3 on #398] A backup whose removal fails after the commit is a stale copy,
    not a failed publish: publish() returns normally with the new outputs in place, and the
    leftover is LOGGED -- before, the failure was swallowed silently."""
    import logging
    import os
    import cell_eval2.cellstream as cs

    outdir, staging, old_bytes, new_scores = _staged_three(tmp_path, cs)
    real_remove = os.remove

    def failing(path, *a, **kw):
        if os.path.basename(str(path)).startswith("de_pred.parquet.bak."):
            raise PermissionError("cannot remove")
        return real_remove(path, *a, **kw)

    monkeypatch.setattr(os, "remove", failing)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.cellstream"):
        published = staging.publish()
    monkeypatch.undo()

    assert len(published) == 3
    _assert_all_new(outdir, old_bytes, new_scores)
    debris = _debris(outdir)
    assert len(debris) == 1 and debris[0].startswith("de_pred.parquet.bak."), debris
    assert "could not remove the backup" in caplog.text and "de_pred.parquet.bak." in caplog.text


def test_score_cellstream_write_de_empty_prediction_is_named_before_aggregation(tmp_path,
                                                                                monkeypatch):
    """[Copilot on #398] A prediction that yields no batches has no pred DE table. aggregate_partials
    would fail first with an opaque "no partial sidecars" error, so the write_de diagnostic must
    run BEFORE it -- the aggregator is made to explode here, and nothing may be written."""
    import cell_eval2.cellstream as cs
    import cell_eval2.partition as partition

    cfg = _stub_write_de_run(monkeypatch)

    class _NoBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _NoBatchSource)

    def _must_not_be_reached(*a, **k):
        raise AssertionError("aggregate_partials ran before the empty-prediction check")
    monkeypatch.setattr(partition, "aggregate_partials", _must_not_be_reached)

    outdir = tmp_path / "out"
    with pytest.raises(ValueError, match="yielded no perturbation batches"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(outdir), write_de=True)
    assert not outdir.exists()


def _stub_write_de_run(monkeypatch, *, result=None):
    """The CPU-only stubbing the write_de tests above use (store, partition guard, both core
    builders, RefBundle, a one-batch source, score_piece writing its DE piece, the aggregator and
    the result assembly), returned as the config to call score_cellstream with."""
    import os
    import polars as pl
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.h5ad_manifest as h5m
    from cell_eval2.config import EvalConfig

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core",
                        lambda source, *, config, cache_dir, **k: _de_piece_frame(
                            "REAL_ALL").write_parquet(os.path.join(cache_dir, "real_de.parquet")))
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class _OneBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([(("GENE_A",), object())])

        def close(self):
            pass

    monkeypatch.setattr(cs, "CellBatchSource", _OneBatchSource)

    def _fake_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                    bundle=None, comparator=None, de_out=None, native_gpu_normalize=False):
        if de_out is not None:
            _de_piece_frame(f"PRED_{piece_id}").write_parquet(
                os.path.join(de_out, f"{piece_id}__de_pred.parquet"))
        return None
    monkeypatch.setattr(pinmem, "score_piece", _fake_piece)
    monkeypatch.setattr(partition, "aggregate_partials",
                        lambda *a, **k: (pl.DataFrame({"perturbation": ["GENE_A"]}), None))

    class _FakeResult:
        per_pert = pl.DataFrame({"perturbation": ["GENE_A"]})
        per_context = pl.DataFrame({"context": ["all"]})
        overall = pl.DataFrame({"metric": ["x"]})
    monkeypatch.setattr(h5m, "_assemble_score_result", lambda frames: result or _FakeResult())
    return EvalConfig(pert_col="perturbation", control="control", control_source="real",
                      metrics=full_minus_moments())


@pytest.mark.parametrize("blocker", ["outdir_is_a_file", "destination_is_a_directory",
                                     "outdir_is_a_dangling_symlink",
                                     "destination_is_a_dangling_symlink",
                                     "destination_is_a_symlink_to_a_file"])
def test_score_cellstream_write_de_refuses_an_unpublishable_outdir_before_any_io(
        tmp_path, monkeypatch, blocker):
    """[P2, checkpoint-2 review] Publishing over a DIRECTORY would move it aside and leave it as
    undeletable backup debris (run._write_de_tables refuses it), and a file where outdir should
    be can never become one. Both are refused before any archive is opened -- open_cell_store
    is made to explode, so a check that ran later would surface as that AssertionError."""
    import cell_eval2.cellstream as cs

    cfg = _stub_write_de_run(monkeypatch)

    def _must_not_be_reached(*a, **k):
        raise AssertionError("an archive was opened before the outdir was checked")
    monkeypatch.setattr(cs, "open_cell_store", _must_not_be_reached)

    outdir = tmp_path / "out"
    if blocker == "outdir_is_a_file":
        outdir.write_text("not a directory")
        match = "exists and is not a directory"
    elif blocker == "outdir_is_a_dangling_symlink":   # os.path.exists is False for it
        outdir.symlink_to(tmp_path / "nowhere")
        match = "exists and is not a directory"
    elif blocker == "destination_is_a_dangling_symlink":
        outdir.mkdir()
        (outdir / "de_real.parquet").symlink_to(tmp_path / "nowhere.parquet")
        match = "exists and is not a regular file"
    elif blocker == "destination_is_a_symlink_to_a_file":   # isfile() is True through the link
        outdir.mkdir()
        target = tmp_path / "elsewhere.parquet"
        target.write_bytes(b"someone else's file")
        (outdir / "de_pred.parquet").symlink_to(target)
        match = "exists and is not a regular file"
    else:
        (outdir / "per_context.parquet").mkdir(parents=True)
        match = "exists and is not a regular file"
    with pytest.raises(ValueError, match=match):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(outdir), write_de=True)
    if blocker == "destination_is_a_directory":
        assert sorted(p.name for p in outdir.iterdir()) == ["per_context.parquet"]
        assert (outdir / "per_context.parquet").is_dir()


def test_score_cellstream_write_de_temp_dir_teardown_failure_leaves_no_temps(tmp_path,
                                                                             monkeypatch):
    """[P2, checkpoint-2 review] The DE tables are staged INSIDE the temporary-directory block, so
    its teardown runs after they are already sitting in outdir as temps. A teardown failure must
    still remove them and publish nothing."""
    import tempfile
    import cell_eval2.cellstream as cs

    cfg = _stub_write_de_run(monkeypatch)
    real_tmpdir = tempfile.TemporaryDirectory

    class _FailingTeardown(real_tmpdir):
        def __exit__(self, *exc):
            super().__exit__(*exc)
            raise OSError("temp dir teardown failed")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _FailingTeardown)
    outdir = tmp_path / "out"
    with pytest.raises(OSError, match="temp dir teardown failed"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(outdir), write_de=True)
    assert list(outdir.iterdir()) == []


# --- #179: the public entry point a downstream caller needed ---------------------------------

def test_cell_archive_input_type_is_public_and_takes_a_PATH(tmp_path):
    """#179: a downstream caller imported cellstream._resolve_input_type_cell AND open_cell_store
    to ask
    this. The public function takes a path, so a consumer needs neither private."""
    import cell_eval2
    from cell_eval2.config import EvalConfig
    assert "cell_archive_input_type" in cell_eval2.__all__
    counts, _ = _write_cell(tmp_path, "counts", 2, lognorm=False)
    ln, _ = _write_cell(tmp_path, "ln", 3, lognorm=True)
    cfg = replace(EvalConfig(), pert_col="perturbation")
    assert cell_eval2.cell_archive_input_type(counts, config=cfg) == "counts"
    assert cell_eval2.cell_archive_input_type(ln, config=cfg) == "lognorm"


def test_cell_archive_input_type_accepts_an_already_open_store(tmp_path):
    from cell_eval2 import cell_archive_input_type
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.config import EvalConfig
    counts, _ = _write_cell(tmp_path, "counts", 2)
    store = open_cell_store(counts)
    try:
        assert cell_archive_input_type(store, config=EvalConfig()) == "counts"
        # ...and did not close a store it did not open.
        assert store.gather_rows_adata(np.arange(1, dtype=np.int64)).n_obs == 1
    finally:
        store.close()


def test_cell_archive_input_type_strict_is_the_agreement_check_the_consumer_wanted(tmp_path):
    """The outcome half of #179: the caller's real question is whether the archive agrees with
    what it declared, which it previously had to peek and compare itself."""
    from cell_eval2 import cell_archive_input_type
    from cell_eval2.config import EvalConfig
    ln, _ = _write_cell(tmp_path, "ln", 3, lognorm=True)
    declared_counts = replace(EvalConfig(), input_type="counts")
    with pytest.raises(ValueError, match=r"resolve to input_type='lognorm'.*declares 'counts'"):
        cell_archive_input_type(ln, config=declared_counts, strict=True)
    # Non-strict still answers, and agreement never raises.
    assert cell_archive_input_type(ln, config=declared_counts) == "lognorm"
    assert cell_archive_input_type(ln, config=replace(EvalConfig(), input_type="lognorm"),
                                   strict=True) == "lognorm"


def test_cell_archive_input_type_strict_is_not_vacuous_under_autodetect_off(tmp_path):
    """strict= must be a real check, not a tautology. The peek autodetects regardless of
    cfg.autodetect_input_type (every cell/shard driver does -- the accumulators take no
    input_type), so the resolved value comes from the DATA even when the config says otherwise."""
    from cell_eval2 import cell_archive_input_type
    from cell_eval2.config import EvalConfig
    ln, _ = _write_cell(tmp_path, "ln", 3, lognorm=True)
    cfg = replace(EvalConfig(), input_type="counts", autodetect_input_type=False)
    assert cell_archive_input_type(ln, config=cfg) == "lognorm"      # data wins, not the flag
    with pytest.raises(ValueError):
        cell_archive_input_type(ln, config=cfg, strict=True)


def test_private_resolve_input_type_cell_still_exists_for_the_recorded_consumer(tmp_path):
    """#179 asks for the private name to survive as a thin alias -- the caller is a RELEASED
    consumer that pins an exact cell_eval2 version, so removing it would break an artifact that
    already exists."""
    from cell_eval2.cellstream import _resolve_input_type_cell, cell_archive_input_type
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.config import EvalConfig
    counts, _ = _write_cell(tmp_path, "counts", 2)
    store = open_cell_store(counts)
    try:
        assert _resolve_input_type_cell(store, EvalConfig()) == \
            cell_archive_input_type(store, config=EvalConfig())
    finally:
        store.close()
