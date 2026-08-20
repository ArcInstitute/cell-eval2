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
