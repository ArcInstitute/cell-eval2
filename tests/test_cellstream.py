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
        self.close_count = 0

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

    def gather_rows(self, row_ids, n_threads=1, out_dtype=None):
        X = self.gather_rows_adata(row_ids, n_threads).X
        return X if out_dtype is None else X.astype(out_dtype)

    @property
    def closed(self):
        return self.close_count > 0

    def close(self):
        self.close_count += 1


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
    import polars as pl
    from polars.testing import assert_frame_equal
    import cell_eval2.cellstream as cs
    from cell_eval2.config import EvalConfig
    from cell_eval2.de_export import DE_PARQUET_ROW_GROUP_ROWS, read_de_parquet_metadata

    captured, de_outs = _stub_streaming(
        monkeypatch, batches=[("GENE_A",), ("GENE_B",)],
        manifest={"normalize_target_sum": 1e6}, frame=_tidy("GENE_A"))

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

    # #412: the document joins the five existing files, with no temp or backup debris, and no
    # provenance sidecar (that belongs to the ground-truth cache, not to this export).
    assert sorted(p.name for p in outdir.iterdir()) == [
        "de_pred.parquet", "de_real.parquet", "evaluation_results.json", "overall.parquet", "per_context.parquet",
        "per_pert.parquet"]


def test_score_cellstream_write_de_false_writes_no_de_tables(tmp_path, monkeypatch):
    """Off by default: write_de=False (the existing callers' path) passes de_out=None to every
    piece and emits no de_real/de_pred, the ScoreResult parquets and the #412 document."""
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
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core",
                        lambda *a, **k: {"normalize_target_sum": 1e6})
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
                        lambda *a, **k: (_tidy("GENE_A"), None))

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
    assert sorted(os.listdir(outdir)) == [   # #412: document beside the ScoreResult output
        "evaluation_results.json", "overall.parquet", "per_context.parquet", "per_pert.parquet"]


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
        return {"normalize_target_sum": 1e6}
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
                        lambda *a, **k: (_tidy("p0"), None))

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
        return {"normalize_target_sum": 1e6}
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
                        lambda *a, **k: (_tidy("GENE_A"), None))

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

    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={"normalize_target_sum": 1e6}, frame=_tidy("GENE_A"))
    cfg = EvalConfig(**_CFG)
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


# --- The cell pass: cell metrics score after the partitioned loop -----------------------------
# CPU-only: the store, CellBatchSource, the input-type resolver, the partition guard, both core
# builders, RefBundle, score_piece, the aggregator, the GPU pool release and the shared
# cell-metric helper are stubbed, so these run in CI. The `*_gpu` tests below run the real thing.

_CELL_SEL = ["expr_mae", "pds_l1", "cell_cov_ccc"]


class _Built(Exception):
    """Raised by a stub reference builder: proves a run got past the fail-fast checks."""


def _sentinel_ref(*a, **k):
    raise _Built()


def _stub_cell_pass_run(monkeypatch, *, itypes=("counts", "counts"), n_vars=5, store=None,
                        stores=None, helper=None, calls=None):
    """Stub everything score_cellstream calls except the split, the fail-fast checks, the cell
    pass and the result assembly; return a dict recording what each stub was handed.
    ``itypes`` is (real, pred), in the order score_cellstream resolves them; ``None`` keeps the
    real resolver, which then reads ``store``'s values. ``stores`` maps each archive path to the
    store a ``CellStoreSource`` opens there (default: a copy of ``store`` per path);
    ``rec["archive_opens"]`` records each ``(path, store)`` opened. ``calls``, when given,
    receives "release" and "helper" in the order the pool release and the default helper run."""
    import copy
    import os
    import polars as pl
    import cell_eval2._cell_archive as archive
    import cell_eval2.cellstream as cs
    import cell_eval2.gpu as gpu
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.run as run

    rec = {"ref_cfgs": [], "piece_cfgs": [], "helper": [], "released": 0, "staged": 0,
           "archive_opens": []}
    if store is None:
        store = _FakeCellStore(n_per=100, n_vars=n_vars)
        # The archive names its control, so _control_label resolves it from the data, not from
        # cfg.control -- the tests leave cfg.control at its default.
        store.obs["control_value"] = "control"
    if stores is None:
        stores = {"pred.shad": copy.deepcopy(store), "real.shad": copy.deepcopy(store)}
    rec["stores"] = stores
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)

    def _open_archive(path):
        rec["archive_opens"].append((path, stores[path]))
        return stores[path]
    monkeypatch.setattr(archive, "open_cell_store", _open_archive)
    if itypes is not None:
        seq = iter(itypes)
        monkeypatch.setattr(cs, "_resolve_input_type_cell", lambda st, cfg, **k: next(seq))
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    def _fake_ref(source, *, config, cache_dir, **k):
        rec["ref_cfgs"].append(config)
        _de_piece_frame("REAL_ALL").write_parquet(os.path.join(cache_dir, "real_de.parquet"))
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _fake_ref)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class _OneBatchSource:
        def __init__(self, *a, **k):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter([(("A", "B"), object())])

        def close(self):
            pass
    monkeypatch.setattr(cs, "CellBatchSource", _OneBatchSource)

    def _fake_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                    bundle=None, comparator=None, de_out=None, native_gpu_normalize=False):
        rec["piece_cfgs"].append(config)
        if de_out is not None:
            _de_piece_frame(f"PRED_{piece_id}").write_parquet(
                os.path.join(de_out, f"{piece_id}__de_pred.parquet"))
    monkeypatch.setattr(pinmem, "score_piece", _fake_piece)

    full = pl.DataFrame({"perturbation": ["A", "B", "A", "B"],
                         "metric": ["expr_mae", "expr_mae", "pds_l1", "pds_l1"],
                         "value": [0.25, float("nan"), 0.5, 0.75]})
    monkeypatch.setattr(partition, "aggregate_partials", lambda *a, **k: (full, None))

    def _fake_helper(pred_src, real_src, cfg, names, *, effective_types, driver):
        rec["helper"].append({"pred_src": pred_src, "real_src": real_src, "cfg": cfg,
                              "names": list(names), "effective_types": dict(effective_types),
                              "driver": driver})
        if helper is not None:
            return helper()
        if calls is not None:
            calls.append("helper")
        return pl.DataFrame({"perturbation": ["A", "B"], "metric": ["cell_cov_ccc"] * 2,
                             "value": [0.9, 0.7]}, schema=run._TIDY_SCHEMA)
    monkeypatch.setattr(run, "_score_cell_metrics_on_sources", _fake_helper)

    def _fake_release():
        rec["released"] += 1
        if calls is not None:
            calls.append("release")
    monkeypatch.setattr(gpu, "_release_gpu_pool", _fake_release)
    return rec


def _cell_pass_records(caplog):
    return [r for r in caplog.records
            if r.name == "cell_eval2.cellstream" and r.getMessage().startswith("cell pass:")]


def _cell_pass_fields(record):
    import re
    msg = record.getMessage()
    fields = dict(re.findall(r"(\w+)=(\S+)", msg))
    assert set(fields) >= {"metrics", "wall_s", "peak_torch_bytes", "peak_cupy_bytes"}, msg
    return {"metrics": fields["metrics"], "wall_s": float(fields["wall_s"]),
            "peak_torch_bytes": int(fields["peak_torch_bytes"]),
            "peak_cupy_bytes": int(fields["peak_cupy_bytes"])}


def test_cell_pass_wiring_scores_cell_metrics_after_the_partitioned_loop(monkeypatch, caplog):
    """A mixed selection: the partitioned stages see only the non-cell metrics; the shared helper
    gets open sources on the right archives, the archive's own control label (cfg.control stays
    at its default), the cell names and the resolved input types; every archive opened is closed
    as often as it was opened; the cell rows reach all three frames. Exactly one `cell pass:`
    record, and the GPU pool is released twice: before the pass and after it."""
    import logging
    from collections import Counter
    from cell_eval2.catalog import CATALOG, resolve_metrics
    from cell_eval2.cell_rowsource import CellStoreSource
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    rec = _stub_cell_pass_run(monkeypatch)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    assert cfg.control != "control"     # the label must come from the archive
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        res = score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)

    (call,) = rec["helper"]
    stores = rec["stores"]
    assert isinstance(call["pred_src"], CellStoreSource)
    assert isinstance(call["real_src"], CellStoreSource)
    assert call["pred_src"]._store is stores["pred.shad"]
    assert call["real_src"]._store is stores["real.shad"]
    assert [path for path, _ in rec["archive_opens"]] == ["pred.shad", "real.shad"] * 2
    for path, n_opens in Counter(path for path, _ in rec["archive_opens"]).items():
        assert stores[path].close_count == n_opens, path
    assert call["cfg"].control == "control"
    assert call["cfg"].pert_col == "perturbation"
    assert call["names"] == ["cell_cov_ccc"]
    assert call["effective_types"] == {"pred": "counts", "real": "counts"}
    assert call["driver"] == "score_cellstream (cell pass)"
    assert rec["ref_cfgs"] and rec["piece_cfgs"]
    for seen in rec["ref_cfgs"] + rec["piece_cfgs"]:
        names, _ = resolve_metrics(seen.metrics, version=seen.version)
        assert names and all(CATALOG[n].kind != "cell" for n in names), names
    assert rec["released"] == 2
    for frame in (res.per_pert, res.per_context, res.overall):
        assert "cell_cov_ccc" in frame["metric"].to_list()
    ctx = res.per_context.filter(res.per_context["metric"] == "cell_cov_ccc")
    assert ctx["value"].to_list() == [pytest.approx(0.8)]
    (record,) = _cell_pass_records(caplog)
    assert record.levelname == "INFO"
    fields = _cell_pass_fields(record)
    assert fields["metrics"] == "cell_cov_ccc" and fields["wall_s"] >= 0


def test_cell_pass_does_not_run_without_a_cell_metric(monkeypatch, caplog):
    """No cell metric selected: no helper call, no `cell pass:` record, no extra GPU free, and
    the partitioned stages get the selection object exactly as given."""
    import logging
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    calls: list[str] = []
    _fake_torch(monkeypatch, calls, lambda: calls.append("max"))
    rec = _stub_cell_pass_run(monkeypatch, calls=calls)
    selection = ["expr_mae", "pds_l1"]
    cfg = EvalConfig(pert_col="perturbation", metrics=selection)
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        res = score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert rec["helper"] == [] and rec["released"] == 0 and calls == []
    assert rec["archive_opens"] == []
    assert _cell_pass_records(caplog) == []
    assert rec["ref_cfgs"] and all(c.metrics is selection
                                   for c in rec["ref_cfgs"] + rec["piece_cfgs"])
    assert "cell_cov_ccc" not in res.per_pert["metric"].to_list()


def test_cell_pass_leaves_every_other_metric_unchanged(monkeypatch):
    """The partitioned stages get the same metric list (resolved, in order) and an otherwise
    equal config with or without a cell metric -- the check that can fail here, since the stub
    aggregator returns a fixed frame; the frame half guards the concat and the assembly. The
    bit-exact proof on real data is the GPU test."""
    from dataclasses import replace as dc_replace
    import polars as pl
    from polars.testing import assert_frame_equal
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    def _run(selection):
        rec = _stub_cell_pass_run(monkeypatch)
        cfg = EvalConfig(pert_col="perturbation", metrics=selection)
        return rec, score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)

    rec0, base = _run(["expr_mae", "pds_l1"])
    rec1, with_cell = _run(list(_CELL_SEL))
    keys = {"per_pert": ["perturbation", "metric"], "per_context": ["context", "metric"],
            "overall": ["metric"]}
    for name, by in keys.items():
        b = getattr(base, name).sort(by)
        w = getattr(with_cell, name).filter(pl.col("metric") != "cell_cov_ccc").sort(by)
        assert b.height > 0
        assert_frame_equal(b, w)
    seen0 = rec0["ref_cfgs"] + rec0["piece_cfgs"]
    seen1 = rec1["ref_cfgs"] + rec1["piece_cfgs"]
    assert len(seen0) == len(seen1) > 0
    for got0, got1 in zip(seen0, seen1):
        assert resolve_metrics(got0.metrics, version=got0.version)[0] == \
            resolve_metrics(got1.metrics, version=got1.version)[0]
        assert dc_replace(got0, metrics=None) == dc_replace(got1, metrics=None)


@pytest.mark.parametrize("side", ["pred", "real"])
def test_cell_pass_refuses_a_side_that_is_not_counts_before_the_builder(monkeypatch, side):
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    itypes = ("lognorm", "counts") if side == "real" else ("counts", "lognorm")
    rec = _stub_cell_pass_run(monkeypatch, itypes=itypes)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    with pytest.raises(ValueError, match=rf"cell_cov_ccc.*need raw counts.*the {side} side"):
        score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert rec["ref_cfgs"] == [] and rec["helper"] == []


@pytest.mark.parametrize("allow_discrete", [False, True])
def test_cell_pass_counts_check_reads_the_resolved_type(monkeypatch, allow_discrete):
    """The counts check reads the RESOLVED type. Fractional archives autodetect as lognorm and
    are refused up front -- unless allow_discrete, which makes the resolver answer "counts"
    without reading a value: the check then passes. What remains are the value checks: with
    validate_input off the call reaches the builder; with it on, the early value check refuses
    the fractional prediction before the builder."""
    import anndata as ad_
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    class _FracStore(_FakeCellStore):
        def gather_rows_adata(self, row_ids, n_threads=1):
            rows = np.atleast_1d(np.asarray(row_ids))
            return ad_.AnnData(X=np.full((rows.size, self.n_vars), 0.37, dtype=np.float32),
                               obs=self.obs.iloc[rows], var=self.var.copy())

    store = _FracStore(n_per=100)
    store.obs["control_value"] = "control"
    _stub_cell_pass_run(monkeypatch, itypes=None, store=store)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel_ref)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL),
                     allow_discrete=allow_discrete)
    if allow_discrete:
        with pytest.raises(_Built):
            score_cellstream("pred.shad", "real.shad",
                             config=replace(cfg, validate_input=False), mem_budget=_BIG)
        with pytest.raises(ValueError, match="fractional"):
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    else:
        with pytest.raises(ValueError, match="need raw counts"):
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)


def test_cell_pass_scratch_budget_check(monkeypatch):
    """On CUDA, 21 * G**2 bytes above mem_budget.gpu_bytes is refused before the builder; at the
    boundary, and with no budget at all, the builder is reached (a sentinel stops the run
    there)."""
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig
    from cell_eval2.h5ad_manifest import MemBudget

    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL), device="cuda")
    need = 21 * 5 ** 2
    rec = _stub_cell_pass_run(monkeypatch, n_vars=5)
    with pytest.raises(ValueError, match=rf"cell_cov_ccc needs .*{need} bytes.*G=5"):
        score_cellstream("pred.shad", "real.shad", config=cfg,
                         mem_budget=MemBudget(host_bytes=10**12, gpu_bytes=need - 1))
    assert rec["ref_cfgs"] == []
    for budget in (MemBudget(host_bytes=10**12, gpu_bytes=need), None):
        _stub_cell_pass_run(monkeypatch, n_vars=5)
        monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel_ref)
        with pytest.raises(_Built):
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=budget)
    # A selection without cell_cov_ccc has no budget to meet.
    _stub_cell_pass_run(monkeypatch, n_vars=5)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel_ref)
    with pytest.raises(_Built):
        score_cellstream("pred.shad", "real.shad",
                         config=EvalConfig(pert_col="perturbation",
                                           metrics=["expr_mae", "cell_cov_corr"], device="cuda",
                                           target_gene_map={"A": "g0", "B": "g1"}),
                         mem_budget=MemBudget(host_bytes=10**12, gpu_bytes=1))


@pytest.mark.parametrize("device,resolved,refused", [
    ("cpu", None, False), ("auto", "cpu", False), ("auto", "cuda", True)])
def test_cell_pass_scratch_budget_check_runs_only_on_cuda(monkeypatch, device, resolved,
                                                          refused):
    """The budget check follows the device cell_cov_ccc runs on: off CUDA a budget below
    21 * G**2 bytes is no refusal."""
    import cell_eval2.partition_inmem as pinmem
    import cell_eval2.run as run
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig
    from cell_eval2.h5ad_manifest import MemBudget

    built = []

    def _sentinel(*a, **k):
        built.append(True)
        raise _Built()
    _stub_cell_pass_run(monkeypatch, n_vars=5)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel)
    if resolved is not None:
        monkeypatch.setattr(run, "resolve_device", lambda d: resolved)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL), device=device)
    budget = MemBudget(host_bytes=10**12, gpu_bytes=21 * 5 ** 2 - 1)
    if refused:
        with pytest.raises(ValueError, match="cell_cov_ccc needs"):
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=budget)
        assert built == []
    else:
        with pytest.raises(_Built):
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=budget)


class _ValueStore(_FakeCellStore):
    """A reference-first fake with ``sizes`` cells per label (control first) whose rows hold
    ``fill[label]`` (a scalar or a per-gene row; integer zeros for an absent label). Every
    gather appends ``(name, rows)`` to the shared ``log``."""

    def __init__(self, name, log, sizes, fill):
        super().__init__(n_per=1, n_vars=5, perts=tuple(sizes))
        labels = np.concatenate([[p] * n for p, n in sizes.items()]).astype(object)
        self.n_obs = labels.size
        self.obs = pd.DataFrame({"perturbation": labels, "control_value": "control"},
                                index=[f"c{i}" for i in range(labels.size)])
        self.name, self._log, self._labels, self._fill = name, log, labels, fill

    def rows_of(self, label):
        return np.flatnonzero(self._labels == label)

    def gather_rows_adata(self, row_ids, n_threads=1):
        rows = np.atleast_1d(np.asarray(row_ids))
        self._log.append((self.name, rows.copy()))
        X = np.zeros((rows.size, self.n_vars), dtype=np.float32)
        for i, r in enumerate(rows):
            X[i] = self._fill.get(self._labels[r], 0.0)
        return ad.AnnData(X=X, obs=self.obs.iloc[rows], var=self.var.copy())


_SIZES_600 = {"control": 600, "A": 600, "B": 600}
_FRAC = {"A": 0.37, "B": 0.37}
_INT = {"A": 1.0, "B": 1.0}

#: case -> (pred sizes, pred fill, real fill, config overrides, the error's match or None when
#: the builder is reached, the perturbation whose rows the early check reads or None). The real
#: archive always has 600 cells per label.
_PRECHECK_CASES = {
    "pred_fractional": (_SIZES_600, _FRAC, _INT, {}, "fractional", "A"),
    "allow_fractional": (_SIZES_600, _FRAC, _INT, {"allow_fractional_counts": True}, None, "A"),
    "validation_off": (_SIZES_600, _FRAC, _INT, {"validate_input": False}, None, None),
    "below_min_cells": ({"control": 30, "A": 30, "B": 30}, _FRAC, _INT, {}, None, None),
    "real_fractional": (_SIZES_600, _INT, _FRAC, {}, None, "A"),
    "nan_with_allow_fractional": (_SIZES_600, {"A": [0.37, np.nan, 0.37, 0.37, 0.37]}, _INT,
                                  {"allow_fractional_counts": True}, "per-cell total is NaN",
                                  "A"),
    "over_max_counts_with_allow_fractional": (
        _SIZES_600, {"A": 3.5, "B": 3.5}, _INT,
        {"allow_fractional_counts": True, "max_counts_per_cell": 10.0},
        "exceeds max_counts_per_cell", "A"),
    "first_scored_is_B": ({"control": 600, "A": 30, "B": 600}, {"B": 0.37}, _INT, {},
                          "fractional", "B"),
}


@pytest.mark.parametrize("case", list(_PRECHECK_CASES))
def test_cell_pass_value_checks_one_predicted_perturbation_before_the_builder(monkeypatch,
                                                                             caplog, case):
    """Before the builder, the pass's own value check runs on the predicted cells the pass
    samples for the first perturbation it scores (depth >= min_cells), never on the control rows
    a reference-first archive starts with: 500 rows, real gathered first, then pred. The real
    side is not checked early, validate_input=False opens nothing, and a failure leaves no
    helper call, no GPU release and no `cell pass:` record."""
    import logging
    from collections import Counter
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    pred_sizes, pred_fill, real_fill, overrides, match, pert = _PRECHECK_CASES[case]
    log: list = []
    stores = {"pred.shad": _ValueStore("pred", log, pred_sizes, pred_fill),
              "real.shad": _ValueStore("real", log, _SIZES_600, real_fill)}
    rec = _stub_cell_pass_run(monkeypatch, stores=stores)
    built = []

    def _sentinel(*a, **k):
        built.append(True)
        raise _Built()
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL), **overrides)
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        if match is None:
            with pytest.raises(_Built):
                score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
        else:
            with pytest.raises(ValueError, match=match):
                score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
            assert built == []
    assert rec["helper"] == [] and rec["released"] == 0
    assert _cell_pass_records(caplog) == []
    for path, n_opens in Counter(path for path, _ in rec["archive_opens"]).items():
        assert stores[path].close_count == n_opens, path
    if case == "validation_off":
        assert rec["archive_opens"] == []
    if pert is None:
        assert log == []
        return
    assert [name for name, _ in log] == ["real", "pred"]
    for name, rows in log:
        assert rows.size == 500, name
        assert np.isin(rows, stores[f"{name}.shad"].rows_of(pert)).all(), name


_GENE_MAP = {"A": "g0", "B": "g1"}

#: case -> (cell metrics after expr_mae, config overrides, predicted fill, the error's match or
#: None when the builder is reached).
_TARGET_GENE_CASES = {
    "corr_no_map": (["cell_cov_corr"], {}, _INT, "target-gene exclusion"),
    "eig_ratio_no_map": (["cell_cov_eig_ratio"], {}, _INT, "target-gene exclusion"),
    "ccc_no_map": (["cell_cov_ccc"], {}, _INT, None),
    "corr_with_map": (["cell_cov_corr"], {"target_gene_map": _GENE_MAP}, _INT, None),
    "corr_no_map_validation_off": (["cell_cov_corr"], {"validate_input": False}, _INT,
                                   "target-gene exclusion"),
    "corr_first_fractional": (["cell_cov_corr", "cell_cov_ccc"], {}, _FRAC,
                              "target-gene exclusion"),
    "ccc_first_fractional": (["cell_cov_ccc", "cell_cov_corr"], {}, _FRAC, "fractional"),
}


@pytest.mark.parametrize("case", list(_TARGET_GENE_CASES))
def test_cell_pass_resolves_the_target_gene_exclusion_before_the_builder(monkeypatch, caplog,
                                                                         case):
    """cell_cov_corr and cell_cov_eig_ratio resolve each perturbation's target gene through the
    pass's own resolver call before the builder, so a construct-ID panel without a
    target_gene_map (labels A, B; genes g0..g4) fails before any DE work, also with
    validate_input off; cell_cov_ccc needs no map. Two failing early checks raise in the
    selection's order, as the pass would."""
    import logging
    from collections import Counter
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    cell, overrides, pred_fill, match = _TARGET_GENE_CASES[case]
    log: list = []
    stores = {"pred.shad": _ValueStore("pred", log, _SIZES_600, pred_fill),
              "real.shad": _ValueStore("real", log, _SIZES_600, _INT)}
    rec = _stub_cell_pass_run(monkeypatch, stores=stores)
    built = []

    def _sentinel(*a, **k):
        built.append(True)
        raise _Built()
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel)
    cfg = EvalConfig(pert_col="perturbation", metrics=["expr_mae"] + cell, **overrides)
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        if match is None:
            with pytest.raises(_Built):
                score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
        else:
            with pytest.raises(ValueError, match=match):
                score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
            assert built == []
    assert rec["helper"] == [] and rec["released"] == 0
    assert _cell_pass_records(caplog) == []
    assert rec["archive_opens"]
    for path, n_opens in Counter(path for path, _ in rec["archive_opens"]).items():
        assert stores[path].close_count == n_opens, path


def test_cell_pass_precheck_matches_the_cell_metric_catalog():
    """The early check builds one CellsPrep from the first cell metric's max_cells / seed /
    min_cells defaults, so it samples what the pass samples only while every cell metric shares
    them; and its target-gene list is exactly the cell metrics built on cov_metric, the family
    that calls CellsPrep.scored_genes."""
    import functools
    import inspect
    from cell_eval2.catalog import CATALOG
    from cell_eval2.cellstream import _TARGET_EXCLUDING_CELL_METRICS
    from cell_eval2.metrics.cells import cov_metric

    cell = {name: spec for name, spec in CATALOG.items() if spec.kind == "cell"}
    knobs = {name: tuple(inspect.signature(spec.func).parameters[k].default
                         for k in ("max_cells", "seed", "min_cells"))
             for name, spec in cell.items()}
    assert knobs and len(set(knobs.values())) == 1, knobs

    def _base(func):
        return func.func if isinstance(func, functools.partial) else func
    assert set(_TARGET_EXCLUDING_CELL_METRICS) == {
        name for name, spec in cell.items() if _base(spec.func) is cov_metric}


def test_cell_pass_failure_publishes_nothing_and_still_logs(monkeypatch, tmp_path, caplog):
    """write_de=True: a failure inside the cell pass propagates before anything is staged, so
    outdir is as it was, and the pass still logs its one `cell pass:` record."""
    import logging
    import cell_eval2.cellstream as cs
    from cell_eval2.config import EvalConfig

    def _boom():
        raise RuntimeError("boom in the cell pass")

    rec = _stub_cell_pass_run(monkeypatch, helper=_boom)
    real_staged = cs._StagedOutputs

    class _RecordingStaged(real_staged):
        def __init__(self, *a, **k):
            rec["staged"] += 1
            super().__init__(*a, **k)
    monkeypatch.setattr(cs, "_StagedOutputs", _RecordingStaged)

    outdir = tmp_path / "out"
    outdir.mkdir()
    old = {}
    for name in cs._WRITE_DE_OUTPUTS:
        _de_piece_frame(f"OLD_{name}").write_parquet(outdir / name)
        old[name] = (outdir / name).read_bytes()
    cfg = EvalConfig(pert_col="perturbation", metrics=full_minus_moments() + ["cell_cov_ccc"])
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        with pytest.raises(RuntimeError, match="boom in the cell pass"):
            cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                                outdir=str(outdir), write_de=True)
    assert len(rec["helper"]) == 1 and rec["staged"] == 0
    assert sorted(p.name for p in outdir.iterdir()) == sorted(old)
    for name, data in old.items():
        assert (outdir / name).read_bytes() == data, name
    assert len(_cell_pass_records(caplog)) == 1


def _fake_torch(monkeypatch, calls, max_memory_allocated):
    import sys
    import types
    cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: calls.append("empty_cache"),
        reset_peak_memory_stats=lambda: calls.append("reset"),
        max_memory_allocated=max_memory_allocated)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_cell_pass_log_never_masks_the_original_error(monkeypatch, caplog):
    """A torch whose peak read itself fails (e.g. after a CUDA error in the pass) must not
    replace the pass's own exception; the record then carries peak_torch_bytes=-1."""
    import logging
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    def _max():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    calls: list[str] = []
    _fake_torch(monkeypatch, calls, _max)

    def _boom():
        raise MemoryError("out of memory in the cell pass")

    rec = _stub_cell_pass_run(monkeypatch, helper=_boom)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        with pytest.raises(MemoryError, match="out of memory in the cell pass"):
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert rec["released"] == 2
    assert calls == ["empty_cache", "reset", "empty_cache"]
    (record,) = _cell_pass_records(caplog)
    assert _cell_pass_fields(record)["peak_torch_bytes"] == -1


def test_a_one_shot_selection_without_a_cell_metric_is_still_frozen(monkeypatch):
    """The freeze is unconditional: a generator selection with no cell metric must reach every
    partitioned stage intact, not exhausted by the split's own resolution."""
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    rec = _stub_cell_pass_run(monkeypatch)
    cfg = EvalConfig(pert_col="perturbation", metrics=iter(["expr_mae", "pds_l1"]))
    score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    seen = rec["ref_cfgs"] + rec["piece_cfgs"]
    assert seen
    for c in seen:
        assert resolve_metrics(c.metrics, version=c.version)[0] == ["expr_mae", "pds_l1"]


class _FakeMemoryHook:
    """Stand-in for ``cupy.cuda.MemoryHook``: a registry of entered hooks by name, refusing a
    second entry under one name, as CuPy does; ``_fake_malloc`` reaches only registered hooks."""
    log: list = []
    active: dict = {}

    def __enter__(self):
        if self.name in _FakeMemoryHook.active:
            raise KeyError(self.name)
        _FakeMemoryHook.active[self.name] = self
        _FakeMemoryHook.log.append("enter")
        return self

    def __exit__(self, *exc):
        _FakeMemoryHook.active.pop(self.name, None)
        _FakeMemoryHook.log.append("exit")


def _fake_malloc():
    """What CuPy's pool does after an allocation: call every REGISTERED hook."""
    for hook in list(_FakeMemoryHook.active.values()):
        hook.malloc_postprocess(device_id=0, size=1, mem_size=1)


def _fake_cupy(monkeypatch, used_bytes):
    """A fake ``cupy`` whose default pool reports ``used_bytes()`` from a script."""
    import sys
    import types
    script = iter(used_bytes)
    pool = types.SimpleNamespace(used_bytes=lambda: next(script))
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(MemoryHook=_FakeMemoryHook),
                                 get_default_memory_pool=lambda: pool)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(_FakeMemoryHook, "log", [])
    monkeypatch.setattr(_FakeMemoryHook, "active", {})


@pytest.mark.parametrize("fails", [False, True])
def test_cell_pass_cupy_peak_is_the_high_water_mark_and_the_hook_always_exits(
        monkeypatch, caplog, fails):
    """The pool's used bytes go 100 (at creation) -> 900 -> 0 (after the arrays are freed) -> 300
    across the pass's allocations: the logged peak is 900, not the final reading. The allocations
    reach only a REGISTERED hook, so the peak also proves the hook brackets the pass; it exits on
    failure too, or it would stay registered on this thread and refuse every later entry."""
    import logging
    import polars as pl
    import cell_eval2.cellstream as cs
    import cell_eval2.run as run
    from cell_eval2.config import EvalConfig

    _fake_cupy(monkeypatch, [100, 900, 0, 300])

    def _helper():
        for _ in range(3):
            _fake_malloc()
        if fails:
            raise RuntimeError("boom after allocating")
        return pl.DataFrame({"perturbation": ["A"], "metric": ["cell_cov_ccc"], "value": [0.5]},
                            schema=run._TIDY_SCHEMA)

    _stub_cell_pass_run(monkeypatch, helper=_helper)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        if fails:
            with pytest.raises(RuntimeError, match="boom after allocating"):
                cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
        else:
            cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert _FakeMemoryHook.log == ["enter", "exit"]
    assert _FakeMemoryHook.active == {}
    (record,) = _cell_pass_records(caplog)
    assert _cell_pass_fields(record)["peak_cupy_bytes"] == 900


def test_cell_pass_hook_that_cannot_enter_logs_minus_one_and_still_scores(monkeypatch, caplog):
    """A hook that cannot be entered is a measurement failure, not a scoring failure."""
    import logging
    import cell_eval2.cellstream as cs
    from cell_eval2.config import EvalConfig

    _fake_cupy(monkeypatch, [0])
    _FakeMemoryHook.active["cell_eval2_cell_pass_peak"] = object()   # taken: entry raises
    rec = _stub_cell_pass_run(monkeypatch)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        res = cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert len(rec["helper"]) == 1
    assert "cell_cov_ccc" in res.per_pert["metric"].to_list()
    assert _FakeMemoryHook.log == []
    (record,) = _cell_pass_records(caplog)
    assert _cell_pass_fields(record)["peak_cupy_bytes"] == -1


def test_cell_pass_reports_the_torch_peak_after_its_own_reset(monkeypatch, caplog):
    """The frees and the torch reading bracket the pass: CuPy pool released, then torch's cache
    emptied and its peak reset, all before the helper runs; the peak is read after it, and then
    both are freed again."""
    import logging
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    calls: list[str] = []

    def _max():
        calls.append("max")
        return 4096
    _fake_torch(monkeypatch, calls, _max)
    _stub_cell_pass_run(monkeypatch, calls=calls)
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    (record,) = _cell_pass_records(caplog)
    assert _cell_pass_fields(record)["peak_torch_bytes"] == 4096
    assert calls == ["release", "empty_cache", "reset", "helper", "max", "release",
                     "empty_cache"]


@pytest.mark.parametrize("case", ["frees_ok", "frees_fail", "no_cuda"])
def test_cell_pass_frees_the_gpu_pools_after_a_failed_pass(monkeypatch, caplog, case):
    """A pass that fails still frees the CuPy pool and torch's cache after it, and logs its one
    record; a free that itself fails never replaces the pass's error. Without CUDA (no torch
    reset) only the CuPy release runs, before and after."""
    import logging
    import sys
    import types
    import cell_eval2.gpu as gpu
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.config import EvalConfig

    calls: list[str] = []
    counts = {"release": 0, "empty_cache": 0}

    def _counted(name):
        def _call():
            calls.append(name)
            counts[name] += 1
            if case == "frees_fail" and counts[name] == 2:
                raise RuntimeError(f"{name} failed after the pass")
        return _call

    def _max():
        calls.append("max")
        return 0
    cuda = types.SimpleNamespace(
        is_available=lambda: case != "no_cuda", empty_cache=_counted("empty_cache"),
        reset_peak_memory_stats=lambda: calls.append("reset"), max_memory_allocated=_max)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    err = RuntimeError("boom in the cell pass")

    def _boom():
        raise err

    _stub_cell_pass_run(monkeypatch, helper=_boom)
    monkeypatch.setattr(gpu, "_release_gpu_pool", _counted("release"))
    cfg = EvalConfig(pert_col="perturbation", metrics=list(_CELL_SEL))
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        with pytest.raises(RuntimeError) as excinfo:
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert excinfo.value is err
    if case == "no_cuda":
        assert calls == ["release", "release"]
    else:
        assert calls == ["release", "empty_cache", "reset", "max", "release", "empty_cache"]
    assert len(_cell_pass_records(caplog)) == 1


def test_score_cell_metrics_on_sources_returns_the_tidy_schema_and_leaves_the_sources_open(
        monkeypatch):
    """The shared helper: an empty result keeps the tidy schema (the cell pass concatenates it
    strictly), and the caller's two sources stay open."""
    import polars as pl
    import cell_eval2._cell_archive as archive
    import cell_eval2.run as run
    from cell_eval2.cell_rowsource import CellStoreSource
    from cell_eval2.config import EvalConfig

    stores = {"pred.shad": _FakeCellStore(), "real.shad": _FakeCellStore()}
    monkeypatch.setattr(archive, "open_cell_store", lambda p: stores[p])
    cfg = EvalConfig(pert_col="perturbation", control="control")
    pred_src, real_src = CellStoreSource("pred.shad"), CellStoreSource("real.shad")
    out = run._score_cell_metrics_on_sources(pred_src, real_src, cfg, ["expr_mae"],
                                             effective_types={"pred": "counts", "real": "counts"},
                                             driver="test")
    assert out.height == 0 and out.schema == pl.Schema(run._TIDY_SCHEMA)
    assert not stores["pred.shad"].closed and not stores["real.shad"].closed


def test_cell_pass_closes_the_pred_source_when_the_real_archive_fails_to_open(monkeypatch,
                                                                              caplog):
    """The cell pass opens its own two sources: a REAL archive that fails to open still closes
    the pred one, the error propagates, and the pass logs its one `cell pass:` record."""
    import logging
    import cell_eval2._cell_archive as archive
    import cell_eval2.cellstream as cs
    import cell_eval2.gpu as gpu
    from cell_eval2.config import EvalConfig

    pred_store = _FakeCellStore()

    def _open(path):
        if path == "real.shad":
            raise OSError("real archive unreadable")
        return pred_store
    monkeypatch.setattr(archive, "open_cell_store", _open)
    monkeypatch.setattr(gpu, "_release_gpu_pool", lambda: None)
    cfg = EvalConfig(pert_col="perturbation", control="control")
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        with pytest.raises(OSError, match="real archive unreadable"):
            cs._run_cell_pass("pred.shad", "real.shad", cfg, ["cell_cov_ccc"],
                              {"pred": "counts", "real": "counts"})
    assert pred_store.close_count == 1
    assert len(_cell_pass_records(caplog)) == 1


@pytest.mark.parametrize("fails", [False, True])
def test_cell_metrics_streaming_scores_on_the_handles_it_validated(monkeypatch, tmp_path, fails):
    """compute_metrics' cell-only fast path opens each archive once: the dispatch scores on the
    handles validate_pair checked, both close after it, and run_params.yaml is written. A failing
    dispatch propagates, still closes both, and writes no run_params.yaml."""
    import cell_eval2._cell_archive as archive
    import cell_eval2.run as run
    from cell_eval2.config import EvalConfig

    stores = {"pred.shad": _FakeCellStore(), "real.shad": _FakeCellStore()}
    opens = []

    def _open(path):
        opens.append(path)
        return stores[path]
    monkeypatch.setattr(archive, "open_cell_store", _open)
    seen = {}

    def _validate_pair(pred_src, real_src, **k):
        seen["validated"] = (pred_src, real_src)

    def _dispatch(names, pred_src, real_src, cfg, **k):
        seen["dispatched"] = (pred_src, real_src)
        seen["closes_at_dispatch"] = [st.close_count for st in stores.values()]
        if fails:
            raise RuntimeError("boom in the dispatch")
        return [{"perturbation": "A", "metric": "cell_cov_ccc", "value": 0.5}]
    monkeypatch.setattr(run, "validate_pair", _validate_pair)
    monkeypatch.setattr(run, "dispatch_cell_metrics", _dispatch)
    cfg = EvalConfig(pert_col="perturbation", control="control", metrics=["cell_cov_ccc"],
                     outdir=str(tmp_path))
    if fails:
        with pytest.raises(RuntimeError, match="boom in the dispatch"):
            run._run_cell_metrics_streaming("pred.shad", "real.shad", cfg)
    else:
        df = run._run_cell_metrics_streaming("pred.shad", "real.shad", cfg)
        assert df["metric"].to_list() == ["cell_cov_ccc"] and df["value"].to_list() == [0.5]
    assert opens == ["pred.shad", "real.shad"]
    assert seen["dispatched"][0] is seen["validated"][0]
    assert seen["dispatched"][1] is seen["validated"][1]
    assert seen["closes_at_dispatch"] == [0, 0]
    assert [st.close_count for st in stores.values()] == [1, 1]
    assert (tmp_path / "run_params.yaml").exists() is not fails


def test_cell_metrics_streaming_reports_facts_before_dispatch(monkeypatch):
    """#412 / design §8.3: the cell-layout fast path reports in-memory facts before scoring."""
    import cell_eval2._cell_archive as archive
    import cell_eval2.run as run
    from cell_eval2.config import EvalConfig

    stores = {"pred.shad": _FakeCellStore(), "real.shad": _FakeCellStore()}
    opens = []

    def _open(path):
        opens.append(path)
        return stores[path]
    monkeypatch.setattr(archive, "open_cell_store", _open)
    seen = {}
    calls = []

    def _validate_pair(pred_src, real_src, **k):
        seen["validated"] = (pred_src, real_src)

    def _dispatch(names, pred_src, real_src, cfg, **k):
        calls.append("dispatch")
        seen["dispatched"] = (pred_src, real_src)
        seen["closes_at_dispatch"] = [st.close_count for st in stores.values()]
        return [{"perturbation": "A", "metric": "cell_cov_ccc", "value": 0.5}]

    def _on_resolved(facts):
        calls.append("resolved")
        seen["facts"] = facts

    monkeypatch.setattr(run, "validate_pair", _validate_pair)
    monkeypatch.setattr(run, "dispatch_cell_metrics", _dispatch)
    cfg = EvalConfig(pert_col="perturbation", control="control", metrics=["cell_cov_ccc"],
                     device="cpu")
    df = run._run_cell_metrics_streaming("pred.shad", "real.shad", cfg,
                                          on_resolved=_on_resolved)
    facts = seen["facts"]
    assert isinstance(facts, run.RunFacts)
    assert facts.scoring_path == "in_memory"
    assert facts.labels == ("A", "B")
    assert facts.de_backend_used is False
    assert calls == ["resolved", "dispatch"]
    assert df["metric"].to_list() == ["cell_cov_ccc"] and df["value"].to_list() == [0.5]
    assert opens == ["pred.shad", "real.shad"]
    assert seen["dispatched"][0] is seen["validated"][0]
    assert seen["dispatched"][1] is seen["validated"][1]
    assert seen["closes_at_dispatch"] == [0, 0]
    assert [st.close_count for st in stores.values()] == [1, 1]


# --- The cell pass on a GPU: the real pipeline, archives written by cellstream ----------------

#: Genes that share one latent factor per cell, in every group (control included), so enough
#: gene pairs pass cell_cov_ccc's significance gate on both sides. Clear of the three target
#: genes at columns 0-2 and of the responders below.
_LATENT = slice(3, 19)


def _responders(k):
    """The 12 genes that perturbation ``PERTS[k]`` (k >= 1) up-regulates."""
    return slice(20 + 6 * (k - 1), 32 + 6 * (k - 1))


def _corr_adata(name, seed, *, n_per=60, n_genes=50, rate=3.0, alpha=1.6):
    """Counts with gene-gene structure over `_LATENT` and a strong per-perturbation response
    (target knock-down plus 12 up-regulated genes), so cell_cov_ccc and the DE metrics score.
    ``n_per`` is a cell count for every label, or a ``{label: n}`` override of 60."""
    rng = np.random.default_rng(seed)
    sizes = {p: (n_per.get(p, 60) if isinstance(n_per, dict) else n_per) for p in PERTS}
    alpha_v = np.zeros(n_genes)
    alpha_v[_LATENT] = alpha
    blocks = []
    for k, p in enumerate(PERTS):
        rate_v = np.full(n_genes, 1.0)
        rate_v[_LATENT] = rate
        if p != "control":
            rate_v[k - 1] *= 0.2
            rate_v[_responders(k)] *= 4.0
        # exp(a*z - a**2/2) has mean 1 for z ~ N(0, 1): only the across-gene dependence changes.
        z = rng.normal(size=(sizes[p], 1))
        lam = rate_v[None, :] * np.exp(alpha_v[None, :] * z - 0.5 * alpha_v[None, :] ** 2)
        blocks.append(rng.poisson(lam).astype(np.float32))
    labels = np.concatenate([[p] * sizes[p] for p in PERTS])
    n = labels.size
    obs = pd.DataFrame({"perturbation": labels, "context": np.repeat("ctxA", n),
                        "control_value": np.repeat("control", n), "dataset": np.repeat("dsX", n)},
                       index=[f"{name}_c{i}" for i in range(n)])
    return ad.AnnData(X=sp.csr_matrix(np.vstack(blocks)), obs=obs,
                      var=pd.DataFrame(index=_genes(n_genes)))


def _mean_baseline_pred(real, seed):
    """A prediction built as a mean baseline builds one: each predicted cell starts from a
    real control cell drawn with replacement, ``expm1(log1p(CP10K(c)) + shift_p) * lib(c) / 1e4``
    -- fractional, non-negative, and varying across cells. ``shift_p >= 0`` lifts the genes that
    respond to ``p``; the control group is drawn the same way with no shift."""
    rng = np.random.default_rng(seed)
    labels = real.obs["perturbation"].to_numpy()
    ctrl = real.X[labels == "control"].toarray()
    lib = np.maximum(ctrl.sum(axis=1, keepdims=True), 1.0)
    n_genes = ctrl.shape[1]
    blocks = []
    for k, p in enumerate(PERTS):
        n = int((labels == p).sum())
        pick = rng.integers(0, ctrl.shape[0], size=n)
        shift = np.zeros(n_genes)
        if p != "control":
            shift[_responders(k)] = 1.0
        x = np.log1p(ctrl[pick] * (1e4 / lib[pick])) + shift[None, :]
        blocks.append((np.expm1(x) * lib[pick] / 1e4).astype(np.float32))
    pred = real.copy()
    pred.X = sp.csr_matrix(np.vstack(blocks))
    pred.obs_names = [f"pred_c{i}" for i in range(pred.n_obs)]
    return pred


def _pred_with_fractional_gene_c():
    """``_corr_adata("pred", 1)`` with GENE_C's cells halved: fractional there, integer counts in
    every other group."""
    pred = _corr_adata("pred", 1)
    X = pred.X.toarray()
    X[(pred.obs["perturbation"] == "GENE_C").to_numpy()] *= 0.5
    pred.X = sp.csr_matrix(X)
    return pred


def _write_adata(tmp_path, name, adata):
    """Write ``adata`` as a grouped, reference-first cell archive; return its path."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive
    shad = tmp_path / f"{name}.shad"
    ref = (adata.obs["perturbation"] == "control").to_numpy()
    write_cell_archive(adata, shad, group_by="perturbation", reference=ref, codec="zstd",
                       overwrite=True)
    return str(shad)


def _gpu_cfg(**kw):
    """The GPU tests' config. `control` is left at its default: the archives name theirs in
    `control_value`, which is what score_cellstream must use."""
    from cell_eval2.config import EvalConfig
    return _pinned(EvalConfig(device="cuda", pert_col="perturbation", input_type="counts", **kw))


def _assert_bit_equal(x, y, keys, cols=("value",)):
    """Same key set on both sides (no row lost to a join), then every column bit-equal."""
    assert x.height == y.height
    assert x.join(y, on=keys, how="anti").height == 0
    assert y.join(x, on=keys, how="anti").height == 0
    j = x.join(y, on=keys, how="inner", suffix="_y")
    assert j.height == x.height
    for c in cols:
        np.testing.assert_array_equal(j[c].to_numpy(), j[f"{c}_y"].to_numpy(), err_msg=c)


def test_cell_pass_fixture_scores_finite_ccc_on_cpu():
    """The GPU tests' fixture is fit for purpose: cell_cov_ccc (CPU path) is finite for every
    perturbation at 60 cells a side, and leaves out one with 30."""
    from cell_eval2.cell_rowsource import AnnDataSource
    from cell_eval2.metrics.cells import cov_ccc_metric

    real = _corr_adata("real", 2)
    for pred, scored in ((_corr_adata("pred", 1), {"GENE_A", "GENE_B", "GENE_C"}),
                         (_corr_adata("pred", 1, n_per={"GENE_C": 30}), {"GENE_A", "GENE_B"}),
                         (_mean_baseline_pred(real, 5), {"GENE_A", "GENE_B", "GENE_C"})):
        got = cov_ccc_metric(pred_source=AnnDataSource(pred), real_source=AnnDataSource(real),
                             _cell_metric_cache={}, pert_col="perturbation", control="control",
                             device="cpu")
        assert set(got) == scored
        assert all(np.isfinite(v) for v in got.values()), got


def test_cell_pass_value_check_fixtures_fail_where_the_gpu_tests_need():
    """The GPU tests' failing predictions under the validator that allow_discrete alone leaves
    on: the GENE_C-fractional one passes the check for GENE_A (the first perturbation scored)
    and GENE_B and fails it for GENE_C, so the early check passes and the pass fails; the
    mean-baseline one fails it for the first perturbation, so the early check fails."""
    from cell_eval2.cell_rowsource import AnnDataSource
    from cell_eval2.cells_prep import CellsPrep
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _cell_block_validator

    cfg = EvalConfig(pert_col="perturbation", control="control", allow_discrete=True)
    validate = _cell_block_validator(cfg, {"pred": "counts", "real": "counts"})
    real = _corr_adata("real", 2)

    def _prep(pred):
        return CellsPrep(AnnDataSource(pred), AnnDataSource(real), pert_col="perturbation",
                         control="control", validate_cells=validate)

    prep = _prep(_pred_with_fractional_gene_c())
    assert prep.perts == ["GENE_A", "GENE_B", "GENE_C"] and prep.depth("GENE_A") >= 50
    prep.pair("GENE_A")
    prep.pair("GENE_B")
    with pytest.raises(ValueError, match="fractional"):
        prep.pair("GENE_C")
    prep = _prep(_mean_baseline_pred(real, 5))
    with pytest.raises(ValueError, match="fractional"):
        prep.pair(prep.perts[0])


def _ccc_rows(frame):
    import polars as pl
    return frame.filter(pl.col("metric") == "cell_cov_ccc")


def test_cell_pass_ccc_equals_the_compute_metrics_fast_path_gpu(tmp_path):
    """cell_cov_ccc from score_cellstream's cell pass equals compute_metrics' cell-only fast path
    on the same two archives, bit for bit, perturbation by perturbation."""
    _skip_if_no_gpu()
    from dataclasses import replace as dc_replace
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.run import compute_metrics

    pred = _write_adata(tmp_path, "pred", _corr_adata("pred", 1))
    real = _write_adata(tmp_path, "real", _corr_adata("real", 2))
    cfg = _gpu_cfg(metrics=full_minus_moments() + ["cell_cov_ccc"])
    got = _ccc_rows(score_cellstream(pred, real, config=cfg, mem_budget=_BIG).per_pert).select(
        ["perturbation", "metric", "value"])
    fast = dc_replace(cfg, metrics=["cell_cov_ccc"], control="control")
    want = compute_metrics(pred, real, config=fast)
    # The fast path's own run-to-run control, so a mismatch below is the cell pass, not CuPy.
    _assert_bit_equal(want, compute_metrics(pred, real, config=fast), ["perturbation", "metric"])
    assert got.height == 3
    assert np.isfinite(got["value"].to_numpy()).all()
    _assert_bit_equal(got, want, ["perturbation", "metric"])


def test_cell_pass_leaves_every_other_metric_bit_exact_gpu(tmp_path):
    """The arcbench-nf selection (`full` without the moments metrics; PDS is a `full` member)
    with and without cell_cov_ccc: every other value of all three frames is bit-equal, and all
    three frames hold cell_cov_ccc. The selection without it runs twice first, so GPU
    run-to-run determinism is checked before the comparison is trusted, and once more after the
    cell pass, which must leave nothing behind that moves a later call."""
    _skip_if_no_gpu()
    import polars as pl
    from cell_eval2.cellstream import score_cellstream

    pred = _write_adata(tmp_path, "pred", _corr_adata("pred", 1))
    real = _write_adata(tmp_path, "real", _corr_adata("real", 2))
    sel = full_minus_moments()
    base = score_cellstream(pred, real, config=_gpu_cfg(metrics=sel), mem_budget=_BIG)
    again = score_cellstream(pred, real, config=_gpu_cfg(metrics=sel), mem_budget=_BIG)
    cell = score_cellstream(pred, real, config=_gpu_cfg(metrics=sel + ["cell_cov_ccc"]),
                            mem_budget=_BIG)
    after = score_cellstream(pred, real, config=_gpu_cfg(metrics=sel), mem_budget=_BIG)
    keys = {"per_pert": (["dataset", "panel_id", "context", "perturbation", "metric"], ("value",)),
            "per_context": (["dataset", "panel_id", "context", "metric"], ("value", "agg")),
            "overall": (["metric"], ("value",))}
    for name, (by, cols) in keys.items():
        _assert_bit_equal(getattr(base, name), getattr(again, name), by, cols)   # the control
        _assert_bit_equal(getattr(base, name), getattr(after, name), by, cols)
    for name, (by, cols) in keys.items():
        with_cell = getattr(cell, name)
        assert _ccc_rows(with_cell).height > 0, name
        _assert_bit_equal(getattr(base, name),
                          with_cell.filter(pl.col("metric") != "cell_cov_ccc"), by, cols)
    finite = np.isfinite(base.per_pert["value"].to_numpy())
    assert finite.sum() > base.per_pert.height / 2, "too few finite values to prove anything"


def test_cell_pass_scores_a_mean_baseline_fractional_pred_gpu(tmp_path, monkeypatch, caplog):
    """A mean-baseline fractional prediction scores finite cell_cov_ccc values with
    allow_discrete + allow_fractional_counts. With allow_discrete alone, the early value check
    fails the call before the reference builder and the aggregation, so the cell pass never
    runs."""
    _skip_if_no_gpu()
    import logging
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream

    real_ad = _corr_adata("real", 2)
    real = _write_adata(tmp_path, "real", real_ad)
    pred = _write_adata(tmp_path, "pred", _mean_baseline_pred(real_ad, 5))
    sel = full_minus_moments() + ["cell_cov_ccc"]
    res = score_cellstream(pred, real, config=_gpu_cfg(
        metrics=sel, allow_discrete=True, allow_fractional_counts=True), mem_budget=_BIG)
    ccc = _ccc_rows(res.per_pert)
    assert ccc.height == 3 and np.isfinite(ccc["value"].to_numpy()).all()

    aggregated, built = [], []
    aggregate, build = partition.aggregate_partials, pinmem._build_reference_streaming_core

    def _spy_aggregate(*a, **k):
        aggregated.append(True)
        return aggregate(*a, **k)

    def _spy_build(*a, **k):
        built.append(True)
        return build(*a, **k)
    monkeypatch.setattr(partition, "aggregate_partials", _spy_aggregate)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _spy_build)
    caplog.clear()   # the first call's record may already be captured (e.g. under log_cli)
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        with pytest.raises(ValueError, match="fractional"):
            score_cellstream(pred, real, config=_gpu_cfg(metrics=sel, allow_discrete=True),
                             mem_budget=_BIG)
    assert aggregated == [] and built == []
    assert _cell_pass_records(caplog) == []


def test_cell_pass_leaves_out_a_perturbation_below_50_cells_gpu(tmp_path):
    """GENE_C has 30 predicted cells: no cell_cov_ccc value for it (by definition, not an error),
    the other two score, and GENE_C still scores every other metric."""
    _skip_if_no_gpu()
    import polars as pl
    from cell_eval2.cellstream import score_cellstream

    pred = _write_adata(tmp_path, "pred", _corr_adata("pred", 1, n_per={"GENE_C": 30}))
    real = _write_adata(tmp_path, "real", _corr_adata("real", 2))
    res = score_cellstream(pred, real, config=_gpu_cfg(
        metrics=full_minus_moments() + ["cell_cov_ccc"]), mem_budget=_BIG)
    ccc = _ccc_rows(res.per_pert)
    assert set(ccc["perturbation"].to_list()) == {"GENE_A", "GENE_B"}
    assert np.isfinite(ccc["value"].to_numpy()).all()
    def _metrics(pert):
        rows = res.per_pert.filter((pl.col("perturbation") == pert)
                                   & (pl.col("metric") != "cell_cov_ccc"))
        return set(rows["metric"].to_list())
    assert _metrics("GENE_C") and _metrics("GENE_C") == _metrics("GENE_A")


def test_cell_pass_fail_fast_checks_raise_before_the_reference_builder_gpu(tmp_path,
                                                                           monkeypatch):
    """On real archives: a lognorm side, a cell-only selection and a GPU budget below
    21 * G**2 each raise before the reference builder runs. At the boundary budget the call
    completes; with mem_budget=None the budget check does not run (the builder is reached --
    the partitioned planner itself needs a budget, so a sentinel stops the run there)."""
    _skip_if_no_gpu()
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.h5ad_manifest import MemBudget

    n_genes = 200   # 21 * G**2 bytes must also leave room to plan a batch at the boundary
    real = _write_adata(tmp_path, "real", _corr_adata("real", 2, n_genes=n_genes))
    pred = _write_adata(tmp_path, "pred", _corr_adata("pred", 1, n_genes=n_genes))
    built = []
    original = pinmem._build_reference_streaming_core

    def _recording(*a, **k):
        built.append(True)
        return original(*a, **k)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _recording)
    sel = full_minus_moments() + ["cell_cov_ccc"]

    ln_pred, _ = _write_cell(tmp_path, "ln_pred", 3, n_per=60, n_genes=n_genes, lognorm=True)
    with pytest.raises(ValueError, match=r"cell_cov_ccc.*need raw counts.*the pred side"):
        score_cellstream(str(ln_pred), real, config=_gpu_cfg(metrics=sel), mem_budget=_BIG)
    with pytest.raises(NotImplementedError, match="compute_metrics"):
        score_cellstream(pred, real, config=_gpu_cfg(metrics=["cell_cov_ccc"]), mem_budget=_BIG)
    need = 21 * n_genes ** 2
    with pytest.raises(ValueError, match=rf"cell_cov_ccc needs .*{need} bytes"):
        score_cellstream(pred, real, config=_gpu_cfg(metrics=sel),
                         mem_budget=MemBudget(host_bytes=10**12, gpu_bytes=need - 1))
    assert built == []

    res = score_cellstream(pred, real, config=_gpu_cfg(metrics=sel),
                           mem_budget=MemBudget(host_bytes=10**12, gpu_bytes=need))
    assert built == [True]
    assert np.isfinite(_ccc_rows(res.per_pert)["value"].to_numpy()).all()

    def _sentinel(*a, **k):
        built.append(None)
        raise _Built()
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", _sentinel)
    with pytest.raises(_Built):
        score_cellstream(pred, real, config=_gpu_cfg(metrics=sel), mem_budget=None)
    assert built == [True, None]


def test_cell_pass_write_de_publishes_together_or_not_at_all_gpu(tmp_path, caplog):
    """write_de=True: a failure inside the cell pass leaves a pre-seeded outdir exactly as it
    was; a good call then publishes the five outputs together, cell_cov_ccc included. The
    prediction is fractional in GENE_C only, without allow_fractional_counts: the early value
    check reads GENE_A and passes, and the pass's own check fails at GENE_C, past the
    partitioned loop."""
    _skip_if_no_gpu()
    import logging
    import os
    import polars as pl
    import cell_eval2.cellstream as cs

    real = _write_adata(tmp_path, "real", _corr_adata("real", 2))
    pred = _write_adata(tmp_path, "pred", _pred_with_fractional_gene_c())
    outdir = tmp_path / "out"
    outdir.mkdir()
    old = {}
    for name in cs._WRITE_DE_OUTPUTS:
        _de_piece_frame(f"OLD_{name}").write_parquet(outdir / name)
        old[name] = (outdir / name).read_bytes()
    sel = full_minus_moments() + ["cell_cov_ccc"]
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        with pytest.raises(ValueError, match="fractional"):
            cs.score_cellstream(pred, real, config=_gpu_cfg(metrics=sel, allow_discrete=True),
                                mem_budget=_BIG, outdir=str(outdir), write_de=True)
    assert len(_cell_pass_records(caplog)) == 1
    assert sorted(os.listdir(outdir)) == sorted(old)
    for name, data in old.items():
        assert (outdir / name).read_bytes() == data, name

    cs.score_cellstream(pred, real, config=_gpu_cfg(metrics=sel, allow_discrete=True,
                                                    allow_fractional_counts=True),
                        mem_budget=_BIG, outdir=str(outdir), write_de=True)
    assert sorted(os.listdir(outdir)) == sorted(cs._WRITE_DE_OUTPUTS)
    for name, data in old.items():
        assert (outdir / name).read_bytes() != data, name
    per_pert = pl.read_parquet(outdir / "per_pert.parquet")
    assert _ccc_rows(per_pert).height == 3


def test_cell_pass_log_line_gpu(tmp_path, caplog):
    """Exactly one `cell pass:` INFO record per call that runs the pass, with true GPU peaks:
    torch measured (>= 0) and CuPy, which runs cell_cov_ccc, above zero. A call without a cell
    metric emits none."""
    _skip_if_no_gpu()
    import logging
    from cell_eval2.cellstream import score_cellstream

    pred = _write_adata(tmp_path, "pred", _corr_adata("pred", 1))
    real = _write_adata(tmp_path, "real", _corr_adata("real", 2))
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        score_cellstream(pred, real, config=_gpu_cfg(
            metrics=full_minus_moments() + ["cell_cov_ccc"]), mem_budget=_BIG)
    (record,) = _cell_pass_records(caplog)
    assert record.levelname == "INFO"
    fields = _cell_pass_fields(record)
    assert fields["metrics"] == "cell_cov_ccc"
    assert fields["wall_s"] > 0
    assert fields["peak_torch_bytes"] >= 0
    assert fields["peak_cupy_bytes"] > 0, record.getMessage()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="cell_eval2.cellstream"):
        score_cellstream(pred, real, config=_gpu_cfg(metrics=full_minus_moments()),
                         mem_budget=_BIG)
    assert _cell_pass_records(caplog) == []


# #412 / design §6 and §8: keep document fixtures after the existing streaming tests
# so downstream references to the archive helpers retain their positions. The E402
# exemptions keep these imports here without shifting those referenced definitions.
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import jsonschema  # noqa: E402
import polars as pl  # noqa: E402

import cell_eval2.cellstream as cs  # noqa: E402
from cell_eval2.config import EvalConfig  # noqa: E402
from cell_eval2.evaluation_results import FILENAME, load_document  # noqa: E402


def _tidy(*perts):
    """Supply valid metric rows so #412's writer exercises the real document contract."""
    return pl.DataFrame({"perturbation": list(perts),
                         "metric": ["de_wilcoxon_overlap"] * len(perts),
                         "value": [0.5] * len(perts)},
                        schema={"perturbation": pl.String, "metric": pl.String,
                                "value": pl.Float64})


def _validate(doc):
    """Check the saved §14 schema independently of the #412 runtime validator."""
    schema = json.loads((Path(__file__).parent /
                         "fixtures/arcbench/evaluation_results_v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(doc)


def _stub_streaming(monkeypatch, *, batches, manifest, frame, real_partition_config=False):
    """Isolate GPU work while exercising real score frames and #412 document publication."""
    import cell_eval2.de_compute as de_compute
    import cell_eval2.norm as norm
    import cell_eval2.partition as partition
    import cell_eval2.partition_inmem as pinmem

    store = _FakeCellStore(n_per=100)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(norm, "resolve_input_type", lambda *a, **k: "counts")
    if real_partition_config:
        monkeypatch.setattr(pinmem, "_resolve_backend", lambda backend: "gpudge")
        monkeypatch.setattr(de_compute, "_resolve_backend", lambda backend: "gpudge")
    else:
        monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    captured, de_outs = {}, []

    def reference(source, *, cache_dir, **kw):
        path = Path(cache_dir) / "real_de.parquet"
        _de_piece_frame("REAL_ALL", n=1).write_parquet(path)
        captured["real_de_bytes"] = path.read_bytes()
        return manifest

    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", reference)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_RefBundle", lambda cache_dir, cfg: ("bundle", cache_dir))

    class Source:
        def __init__(self, *a, **kw):
            pass

        def iter_pert_batches(self, mem_budget):
            return iter((perts, object()) for perts in batches)

        def close(self):
            pass

    def piece(pred_piece, cache_dir, *, piece_id, de_out=None, **kw):
        de_outs.append(de_out)
        if de_out is not None:
            _de_piece_frame(f"PRED_{piece_id}").write_parquet(
                Path(de_out) / f"{piece_id}__de_pred.parquet")

    monkeypatch.setattr(cs, "CellBatchSource", Source)
    monkeypatch.setattr(pinmem, "score_piece", piece)
    monkeypatch.setattr(partition, "aggregate_partials", lambda *a, **k: (frame, None))
    return captured, de_outs


_CFG = dict(pert_col="perturbation", control="control", control_source="real",
            metrics=full_minus_moments())


def test_score_cellstream_publishes_a_valid_document_with_its_outputs(tmp_path, monkeypatch):
    _stub_streaming(monkeypatch, batches=[("GENE_A",), ("GENE_B",)],
                    manifest={"normalize_target_sum": 54611.0}, frame=_tidy("GENE_A", "GENE_B"))
    out = tmp_path / "out"
    cs.score_cellstream("pred.shad", "real.shad",                # None: the manifest resolves it
                        config=EvalConfig(**{**_CFG, "target_sum": None}), mem_budget=_BIG,
                        outdir=str(out), write_de=True)
    doc = load_document(out / FILENAME)
    _validate(doc)                                              # saved §14 schema
    s = doc["suite"]["settings"]
    assert doc["axes"]["perturbation"]["labels"] == ["GENE_A", "GENE_B"]
    assert s["scoring_path"] == "streaming" and s["target_sum"] == 54611.0
    assert (s["pert_col"], s["control"]) == ("perturbation", "control")
    assert sorted(doc["metrics"]) == sorted(full_minus_moments())
    assert doc["metrics"]["de_wilcoxon_overlap"]["values"] == [0.5, 0.5]
    assert sorted(p.name for p in out.iterdir()) == sorted(cs._WRITE_DE_OUTPUTS)


def test_score_cellstream_records_an_explicit_target_sum_as_held(tmp_path, monkeypatch):
    # the manifest stores float(...); an asked-for int must not come back as 1000000.0
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={"normalize_target_sum": 1e6},
                    frame=_tidy("GENE_A"))
    cs.score_cellstream("pred.shad", "real.shad",
                        config=EvalConfig(**{**_CFG, "target_sum": 1_000_000}), mem_budget=_BIG,
                        outdir=str(tmp_path / "out"))
    s = load_document(tmp_path / "out" / FILENAME)["suite"]["settings"]
    assert s["target_sum"] == 1_000_000 and type(s["target_sum"]) is int


def test_score_cellstream_records_the_auc_floor_it_used(tmp_path, monkeypatch):
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={"normalize_target_sum": 1e6},
                    frame=_tidy("GENE_A"), real_partition_config=True)
    cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG), mem_budget=_BIG,
                        outdir=str(tmp_path / "out"))
    s = load_document(tmp_path / "out" / FILENAME)["suite"]["settings"]
    assert s["de"]["auc_pval_floor"] == "preserve"              # min_nonzero was asked for
    assert s["de"]["backend"] == "gpudge" and s["de"]["engine"]["name"] == "gpudge"


def test_without_write_de_the_document_is_written_after_the_parquets(tmp_path, monkeypatch):
    import cell_eval2.evaluation_results as er
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={"normalize_target_sum": 1e6},
                    frame=_tidy("GENE_A"))
    out, seen, real_write = tmp_path / "out", {}, er.write_document

    def spy(doc, path):
        seen.update({n: (out / n).exists()
                     for n in ("per_pert.parquet", "per_context.parquet", "overall.parquet")})
        real_write(doc, path)
    monkeypatch.setattr(er, "write_document", spy)
    cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG), mem_budget=_BIG,
                        outdir=str(out))
    assert seen == dict.fromkeys(seen, True) and len(seen) == 3
    assert sorted(p.name for p in out.iterdir()) == [
        FILENAME, "overall.parquet", "per_context.parquet", "per_pert.parquet"]


def test_a_failing_document_write_publishes_nothing(tmp_path, monkeypatch):
    import cell_eval2.evaluation_results as er
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={"normalize_target_sum": 1e6},
                    frame=_tidy("GENE_A"))
    out = tmp_path / "out"
    out.mkdir()
    for n in cs._WRITE_DE_OUTPUTS:
        (out / n).write_text(f"previous {n}")

    def boom(doc, path):
        raise OSError("document write failed")
    monkeypatch.setattr(er, "write_document", boom)
    with pytest.raises(OSError, match="document write failed"):
        cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG), mem_budget=_BIG,
                            outdir=str(out), write_de=True)
    assert sorted(p.name for p in out.iterdir()) == sorted(cs._WRITE_DE_OUTPUTS)   # no debris
    assert all((out / n).read_text() == f"previous {n}" for n in cs._WRITE_DE_OUTPUTS)


def test_a_failed_rename_of_the_document_rolls_every_output_back(tmp_path, monkeypatch):
    import os
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={"normalize_target_sum": 1e6},
                    frame=_tidy("GENE_A"))
    out = tmp_path / "out"
    out.mkdir()
    for n in cs._WRITE_DE_OUTPUTS:
        (out / n).write_text(f"previous {n}")
    real_replace = os.replace

    def _replace(src, dst):
        if ".tmp." in os.path.basename(str(src)) and os.path.basename(str(dst)) == FILENAME:
            raise OSError("boom: the document's rename failed")
        return real_replace(src, dst)
    monkeypatch.setattr(os, "replace", _replace)
    with pytest.raises(OSError, match="boom"):
        cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG), mem_budget=_BIG,
                            outdir=str(out), write_de=True)
    assert sorted(p.name for p in out.iterdir()) == sorted(cs._WRITE_DE_OUTPUTS)
    assert all((out / n).read_text() == f"previous {n}" for n in cs._WRITE_DE_OUTPUTS)


def test_a_v1_call_writes_no_document_and_drops_a_stale_one(tmp_path, monkeypatch, caplog):
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={"normalize_target_sum": 1e6},
                    frame=_tidy("GENE_A"))
    out = tmp_path / "out"
    out.mkdir()
    (out / FILENAME).write_text("stale")
    with caplog.at_level(logging.WARNING):
        cs.score_cellstream("pred.shad", "real.shad",
                            config=EvalConfig(**{**_CFG, "metrics": "de", "version": "v1"}),
                            mem_budget=_BIG, outdir=str(out))
    assert not (out / FILENAME).exists()
    assert sum("writes no evaluation_results.json" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("fail_publish", [False, True])
def test_a_v1_write_de_call_drops_a_stale_document_and_a_failed_publish_leaves_none(
        tmp_path, monkeypatch, fail_publish):
    """#412 / design §6.6: rollback must not revive a document for a legacy run."""
    import os

    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={"normalize_target_sum": 1e6}, frame=_tidy("GENE_A"))
    out = tmp_path / "out"
    out.mkdir()
    for name in cs._WRITE_DE_OUTPUTS:
        (out / name).write_text(f"previous {name}")
    real_replace = os.replace

    def failing(src, dst):
        if ".tmp." in os.path.basename(str(src)):
            assert not (out / FILENAME).exists()
            raise OSError("first publish failed")
        return real_replace(src, dst)

    def score():
        cs.score_cellstream("pred.shad", "real.shad",
                            config=EvalConfig(**{**_CFG, "metrics": "de", "version": "v1"}),
                            mem_budget=_BIG, outdir=str(out), write_de=True)

    if fail_publish:
        monkeypatch.setattr(os, "replace", failing)
        with pytest.raises(OSError, match="first publish failed"):
            score()
    else:
        score()
    others = set(cs._WRITE_DE_OUTPUTS) - {FILENAME}
    assert {p.name for p in out.iterdir()} == others
    if fail_publish:
        assert all((out / n).read_text() == f"previous {n}" for n in others)


@pytest.mark.parametrize("write_de", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_document_build_refusal_leaves_outdir_untouched(tmp_path, monkeypatch, write_de, existing):
    """#412 / design §2.1: validate after the cell pass, before staging any output."""
    import cell_eval2.evaluation_results as er

    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={"normalize_target_sum": 1e6}, frame=_tidy("GENE_A"))
    out = tmp_path / "out"
    old = {name: f"previous {name}".encode() for name in cs._WRITE_DE_OUTPUTS}
    if existing:
        out.mkdir()
        for name, data in old.items():
            (out / name).write_bytes(data)
    calls = []
    monkeypatch.setattr(cs, "_check_cell_pass_inputs", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_precheck_cell_pass", lambda *a, **k: None)

    def cell_pass(*a, **kw):
        calls.append("cell")
        return _tidy("GENE_A").with_columns(pl.lit("cell_cov_ccc").alias("metric"))

    def refuse(frame, facts, *, metrics):
        assert calls == ["cell"]
        assert "cell_cov_ccc" in metrics and "cell_cov_ccc" in frame["metric"].to_list()
        raise ValueError("document refused")

    monkeypatch.setattr(cs, "_run_cell_pass", cell_pass)
    monkeypatch.setattr(er, "build_document", refuse)
    cfg = EvalConfig(**{**_CFG, "metrics": full_minus_moments() + ["cell_cov_ccc"]})
    with pytest.raises(ValueError, match="document refused"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(out), write_de=write_de)
    if existing:
        assert {p.name: p.read_bytes() for p in out.iterdir()} == old
    else:
        assert not out.exists()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_document_is_removed_before_a_plain_parquet_write_fails(tmp_path, monkeypatch, version):
    """#412 / design §6.6: a partial rerun must not retain the earlier document."""
    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={"normalize_target_sum": 1e6}, frame=_tidy("GENE_A"))
    out = tmp_path / "out"
    out.mkdir()
    (out / FILENAME).write_text("stale")

    def fail(frame, path, *a, **kw):
        if Path(path).parent == out:
            assert not (out / FILENAME).exists()
            raise OSError("parquet write failed")
        return real_write(frame, path, *a, **kw)

    real_write = pl.DataFrame.write_parquet
    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail)
    cfg = EvalConfig(**{**_CFG, "version": version, "metrics": "de"})
    with pytest.raises(OSError, match="parquet write failed"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(out))
    assert not (out / FILENAME).exists()


@pytest.mark.parametrize("refuse", [False, True])
def test_document_replay_checks_resolved_facts_before_cell_prechecks_and_reference(
        tmp_path, monkeypatch, refuse):
    """#412 / design §9: replay checks resolved labels before any expensive scoring."""
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.run import RunFacts

    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={"normalize_target_sum": 54611.0}, frame=_tidy("GENE_A"))
    store = _FakeCellStore(perts=("control", "GENE_A"))
    store.obs["control_value"] = "control"
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    calls = []
    real_reference = pinmem._build_reference_streaming_core

    def reference(*a, **kw):
        calls.append("reference")
        return real_reference(*a, **kw)

    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", reference)
    monkeypatch.setattr(cs, "_check_cell_pass_inputs", lambda *a, **k: calls.append("inputs"))
    monkeypatch.setattr(cs, "_precheck_cell_pass", lambda *a, **k: calls.append("precheck"))
    monkeypatch.setattr(cs, "_run_cell_pass", lambda *a, **k: _tidy("GENE_A").with_columns(
        pl.lit("cell_cov_ccc").alias("metric")))

    def checker(facts):
        assert isinstance(facts, RunFacts)
        assert calls == []
        calls.append("check")
        assert (facts.config.pert_col, facts.config.control) == ("perturbation", "control")
        assert facts.labels == () and facts.config.target_sum is None
        assert facts.input_type_real == facts.input_type_pred == "counts"
        assert facts.comparator == "bulk_lognorm" and facts.scoring_path == "streaming"
        assert facts.de_backend_used
        if refuse:
            raise ValueError("replay refused")

    cfg = EvalConfig(**{**_CFG, "pert_col": "target", "control": "non-targeting",
                       "target_sum": None, "metrics": full_minus_moments() + ["cell_cov_ccc"]})
    out = tmp_path / "out"
    if refuse:
        with pytest.raises(ValueError, match="replay refused"):
            cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                                outdir=str(out), check_facts=checker)
        assert calls == ["check"] and not out.exists()
    else:
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(out), check_facts=checker)
        assert calls == ["check", "inputs", "precheck", "reference"]
        doc = load_document(out / FILENAME)
        assert doc["suite"]["settings"]["control"] == "control"
        assert doc["metrics"]["cell_cov_ccc"]["values"] == [0.5]


def test_document_labels_are_sorted_prediction_labels_even_when_rows_are_missing(tmp_path, monkeypatch):
    """#412 / design §6.4: neither reference labels nor sparse metric rows define the axis."""
    _stub_streaming(monkeypatch, batches=[("GENE_B",), ("GENE_A",)],
                    manifest={"normalize_target_sum": 1e6}, frame=_tidy("GENE_B"))
    cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG), mem_budget=_BIG,
                        outdir=str(tmp_path / "out"))
    doc = load_document(tmp_path / "out" / FILENAME)
    assert doc["axes"]["perturbation"]["labels"] == ["GENE_A", "GENE_B"]
    assert doc["metrics"]["de_wilcoxon_overlap"]["values"] == [None, 0.5]


def test_document_without_outdir_does_no_output_io(tmp_path, monkeypatch):
    """#412 / design §6.6: an in-memory result never invokes document or output writers."""
    import cell_eval2.evaluation_results as er
    import cell_eval2.settings as settings

    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={}, frame=_tidy("GENE_A"))

    def unexpected(*a, **kw):
        raise AssertionError("unexpected document/output I/O")

    for name in ("build_document", "write_document", "remove_stale_document", "no_document_reason"):
        monkeypatch.setattr(er, name, unexpected)
    monkeypatch.setattr(settings, "check_recordable", unexpected)
    monkeypatch.setattr(cs, "_StagedOutputs", unexpected)
    # The reference builder's private scratch file is required by streaming; no result is saved.
    result = cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG),
                                mem_budget=_BIG)
    assert result.per_pert.height == 1 and list(tmp_path.iterdir()) == []


def test_document_rejects_unrecordable_settings_before_archive_io(tmp_path, monkeypatch):
    """#412 / design §8.2: refuse lossy settings before touching the archives."""
    _stub_streaming(monkeypatch, batches=[("GENE_A",)], manifest={}, frame=_tidy("GENE_A"))

    def unexpected(*a, **kw):
        raise AssertionError("archive opened before settings refusal")

    monkeypatch.setattr(cs, "open_cell_store", unexpected)
    cfg = EvalConfig(**{**_CFG, "target_gene_map": {123: ["GENE_A"]}})
    with pytest.raises(ValueError, match="quote the label"):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG,
                            outdir=str(tmp_path / "out"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path, value", [
    ("scoring_path", "in_memory"),
    ("control", "other_control"),
])
def test_score_cellstream_refuses_a_differing_fact_before_any_de_work(
        tmp_path, monkeypatch, path, value):
    """Archive-resolved facts must gate reference DE work (#412, design §9)."""
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2 import facts_checker, settings_to_config
    from cell_eval2.settings import SettingsMismatch

    _stub_streaming(monkeypatch, batches=[("GENE_A",)],
                    manifest={"normalize_target_sum": 54611.0}, frame=_tidy("GENE_A"))
    store = _FakeCellStore(perts=("control", "GENE_A"))
    store.obs["control_value"] = "control"
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    out = tmp_path / "first"
    cs.score_cellstream("pred.shad", "real.shad", config=EvalConfig(**_CFG),
                        mem_budget=_BIG, outdir=str(out))
    doc = load_document(out / FILENAME)
    calls = []
    real_reference = pinmem._build_reference_streaming_core

    def reference(*a, **kw):
        calls.append("reference")
        return real_reference(*a, **kw)

    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", reference)
    cs.score_cellstream("pred.shad", "real.shad", config=settings_to_config(doc),
                        mem_budget=_BIG, check_facts=facts_checker(doc))
    assert calls == ["reference"]
    calls.clear()
    doc["suite"]["settings"][path] = value
    with pytest.raises(SettingsMismatch, match=path):
        cs.score_cellstream("pred.shad", "real.shad", config=settings_to_config(doc),
                            mem_budget=_BIG, check_facts=facts_checker(doc))
    assert calls == []
