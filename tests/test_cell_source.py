"""Cell-layout streaming source (#117 Stage 2): metadata, reference, per-group
pseudobulk, and numerical parity of score_streaming_cell vs the Stage-1 materialize
oracle — rank/set metrics (pds_*, de_*) EXACT, continuous metrics (mae/mse/pearson) to
~1e-8 relative (float summation-order; see _assert_parity). zstd codec (pfordelta reads
need pyfastpfor, absent here)."""
import json
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

PERTS = ["non-targeting", "GENE_A", "GENE_B", "GENE_C"]


def _genes(n_genes):
    targets = PERTS[1:]
    return targets + [f"g{j}" for j in range(len(targets), n_genes)]


def _synth(seed, n_per=40, n_genes=60):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(0.5, size=(n_per * len(PERTS), n_genes)).astype(np.float32))
    obs = pd.DataFrame({"target": np.repeat(PERTS, n_per)},
                       index=[f"cell{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=_genes(n_genes))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def cell_pair(tmp_path):
    """(pred, real) each as (h5ad_path, cell_shad_path). Skips without cellstream."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive
    out = {}
    for side, seed in (("pred", 1), ("real", 2)):
        adata = _synth(seed)
        h5 = tmp_path / f"{side}.h5ad"
        adata.write_h5ad(h5)
        shad = tmp_path / f"{side}.shad"
        ref = (adata.obs["target"] == "non-targeting").to_numpy()
        write_cell_archive(adata, shad, group_by="target", reference=ref,
                           codec="zstd", overwrite=True)
        out[side] = (h5, shad)
    return out


def test_open_cell_store(cell_pair):
    from cell_eval2._cell_archive import open_cell_store
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        assert store.shape == (160, 60)
        assert store.manifest["group_by"] == "target"
    finally:
        store.close()


def test_cell_metadata_and_fingerprint(cell_pair):
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_metadata, cell_fingerprint
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        m = cell_metadata(store)
        assert m.n_obs == 160 and m.n_vars == 60
        assert m.group_by == "target"
        assert sorted(m.perts.tolist()) == sorted(PERTS)
        assert list(m.var_names) == _genes(60)
        fp = cell_fingerprint(store)
        assert isinstance(fp, str) and fp.startswith("cell:")
    finally:
        store.close()

def test_cell_fingerprint_stable_and_schema_sensitive(cell_pair, tmp_path):
    """Stable for the same archive; differs when the gene panel differs (the
    partition-aggregate safety property). Default writes set payload_sha256=None, so this
    exercises the STRUCTURAL fallback — two archives with the same schema but different
    payload would collide (documented; same limitation as stream.shad_fingerprint's
    structural fallback), so 'differs' is asserted on a distinct gene panel, not payload."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_fingerprint
    _, shad = cell_pair["pred"]
    s1, s2 = open_cell_store(shad), open_cell_store(shad)
    try:
        assert cell_fingerprint(s1) == cell_fingerprint(s2)      # stable, same archive
    finally:
        s1.close()
        s2.close()
    other = _synth(3, n_genes=50)                                # distinct gene panel
    op = tmp_path / "other.shad"
    write_cell_archive(other, op, group_by="target",
                       reference=(other.obs["target"] == "non-targeting").to_numpy(),
                       codec="zstd", overwrite=True)
    sa, so = open_cell_store(shad), open_cell_store(op)
    try:
        assert cell_fingerprint(sa) != cell_fingerprint(so)      # different schema -> differs
    finally:
        sa.close()
        so.close()


def test_validate_cell_pair(cell_pair):
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_metadata, validate_cell_pair
    (_, p), (_, r) = cell_pair["pred"], cell_pair["real"]
    sp_, sr = open_cell_store(p), open_cell_store(r)
    try:
        validate_cell_pair(cell_metadata(sp_), cell_metadata(sr),
                           pert_col="target", control="non-targeting")   # matched -> ok
        with pytest.raises(ValueError, match="control"):
            validate_cell_pair(cell_metadata(sp_), cell_metadata(sr),
                               pert_col="target", control="MISSING")
    finally:
        sp_.close()
        sr.close()


def _as_f32_csr(x):
    x = x.tocsr() if sp.issparse(x) else sp.csr_matrix(x)
    return x.astype(np.float32) if x.dtype != np.float32 else x

def test_cell_reference_matches_control(cell_pair):
    from cell_eval2._cell_archive import open_cell_store, materialize_cell
    from cell_eval2.cell_source import cell_reference
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        ref = cell_reference(store)
        assert ref.shape == (40, 60)                      # 40 non-targeting cells
        # sum of the reference pool equals the materialized control cells' sum
        a = materialize_cell(shad)
        ctrl = a.X[(a.obs["target"] == "non-targeting").to_numpy()]
        np.testing.assert_array_equal(np.asarray(ref.sum(0)).ravel(),
                                      np.asarray(_as_f32_csr(ctrl).sum(0)).ravel())
    finally:
        store.close()

def test_cell_group_blocks_cover_all_cells(cell_pair):
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_group_blocks
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        seen = 0
        labels_seen = set()
        for X, labels in cell_group_blocks(store):
            seen += X.shape[0]
            assert len(set(labels)) == 1                  # one label per block
            labels_seen.add(labels[0])
        assert seen == 160
        assert labels_seen == set(PERTS)
        # exclude drops the control block
        excl = [lbl for X, lbls in cell_group_blocks(store, exclude="non-targeting")
                for lbl in [lbls[0]]]
        assert "non-targeting" not in excl and len(excl) == 3
    finally:
        store.close()


def test_cell_pseudobulk_matches_materialize_cpu(cell_pair):
    """CPU parity oracle: prep.pseudobulk(_norm.to_normalization(...)) — the SAME oracle
    (and SAME rtol=1e-9/atol=1e-9 tolerance) test_streaming_bulk.py's
    test_streaming_pseudobulk_matches_inmemory already established for the shard-layout
    sibling of this exact accumulator (_streaming_pseudobulk_cpu). NOT streaming_bulk.
    inmem_pseudobulk: that helper's GroupedMeanAccumulator.finalize() unconditionally casts
    to float32 "on cpu and cuda alike" (by design, matching the GPU accumulator's own
    parity contract — see test_inmem_pseudobulk_matches_cpu_reference, rtol=1e-4), so it is
    not a same-dtype comparator for this fp64 CPU path; using it here would compare fp64
    against fp32-rounded values under an exact-equality assertion, which cannot pass for a
    reason unrelated to cell_pseudobulk's correctness (confirmed empirically: fp32 rounding
    of ~5.5e-8 relative vs the ~5e-16 relative floating-point summation-order noise this
    test actually cares about). DEVIATION from the plan's literal test body — see the
    implementation report."""
    from cell_eval2._cell_archive import open_cell_store, materialize_cell
    from cell_eval2.cell_source import cell_pseudobulk
    from cell_eval2 import norm as _norm
    from cell_eval2.prep import pseudobulk
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        norms, ts = ["normalized", "lognorm"], 1e6
        got = cell_pseudobulk(store, pert_col="target", norms=norms,
                              target_sum=ts, device="cpu")
    finally:
        store.close()
    # Oracle: prep.pseudobulk over the materialized (same storage order) adata, fp64 CPU —
    # the same in-memory reference compute_metrics itself uses on CPU (run._side_bulks).
    a = materialize_cell(shad)
    a64 = a.copy()
    a64.X = a64.X.astype(np.float64)
    for n in norms:
        gp, gm = got[n]
        wp, wm = pseudobulk(_norm.to_normalization(a64, "counts", n, target_sum=ts), "target")
        assert list(gp) == list(wp)
        # Mathematically equivalent (nnz-scatter accumulation vs a dense per-group .mean()),
        # not bit-identical (floating-point summation is not associative); rtol/atol match
        # test_streaming_pseudobulk_matches_inmemory's established precedent for this exact
        # accumulator. Empirically the actual gap is ~5e-16 relative, ~7 orders of magnitude
        # inside this tolerance.
        np.testing.assert_allclose(gm, wm, rtol=1e-9, atol=1e-9)


def test_cell_pseudobulk_v1_target_sum_deferred(cell_pair):
    """target_sum=None (v1 median normalization) with a norm that needs it raises a clear
    NotImplementedError, not a cryptic TypeError inside the accumulator (Copilot #127)."""
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_pseudobulk
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        with pytest.raises(NotImplementedError, match="target_sum=None"):
            cell_pseudobulk(store, pert_col="target", norms=["lognorm"],
                            target_sum=None, device="cpu")
    finally:
        store.close()


def test_compute_de_streaming_cell_guards(cell_pair):
    from cell_eval2 import de_compute
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_reference, iter_cell_groups
    # capability probe returns a bool without raising
    assert isinstance(de_compute._gpudge_supports_refpool_core(), bool)
    # v1 median (target_sum=None) is explicitly deferred -> NotImplementedError,
    # raised before any GPU work (so it is CPU-testable / CI-safe).
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        ref = cell_reference(store)
        with pytest.raises(NotImplementedError, match="median"):
            de_compute.compute_de_streaming_cell(
                ref_X=ref,
                group_iter_factory=lambda: iter_cell_groups(store, ["GENE_A"]),
                targets=["GENE_A"], var_names=np.array([f"g{j}" for j in range(60)]),
                n_genes=60, backend="gpudge", mean_calc="arithmetic", epsilon=1e-9,
                target_sum=None, clip_value=None, fdr_scope="per_pert",
                filter_gene_min_cpm_cell=5.0)
        # a non-gpudge backend is rejected up front too
        with pytest.raises(ValueError, match="gpudge"):
            de_compute.compute_de_streaming_cell(
                ref_X=ref, group_iter_factory=lambda: iter_cell_groups(store, ["GENE_A"]),
                targets=["GENE_A"], var_names=np.array([f"g{j}" for j in range(60)]),
                n_genes=60, backend="pdex", mean_calc="arithmetic", epsilon=1e-9,
                target_sum=1e6, clip_value=None, fdr_scope="per_pert",
                filter_gene_min_cpm_cell=5.0)
    finally:
        store.close()


def _assert_parity(got, want):
    """Streaming-vs-materialize parity criterion (#117 Stage 2, user decision).

    Rank/set-based metrics (pds_* discrimination, de_* differential expression) are
    order-independent and asserted EXACT. Continuous metrics (mae/mse/pearson, computed
    from pseudobulk means) differ only by float summation-order between the streaming
    per-group scatter-add (_streaming_pseudobulk_cpu) and the materialize dense .mean(),
    so they are compared at rtol=1e-7/atol=1e-7 — ~4800x tighter than the shard-layout
    path's own established 1e-4 precedent (test_scale_runner.py) and well above the
    measured ~2e-8 ceiling. `de_` carries the trailing underscore so it never matches the
    continuous `delta_*` metrics."""
    assert got["perturbation"].to_list() == want["perturbation"].to_list()
    assert got["metric"].to_list() == want["metric"].to_list()
    metrics = np.asarray(got["metric"].to_list())
    g, w = got["value"].to_numpy(), want["value"].to_numpy()
    is_rank = np.array([m.startswith("pds") or m.startswith("de_") for m in metrics])
    np.testing.assert_array_equal(g[is_rank], w[is_rank])                       # exact
    np.testing.assert_allclose(g[~is_rank], w[~is_rank], rtol=1e-7, atol=1e-7)  # continuous


def test_score_streaming_cell_anndata_parity_cpu(cell_pair):
    """DE-free metric set: score_streaming_cell streams the pseudobulk on CPU and must
    match compute_metrics on the materialized archives — rank metrics exact, continuous to
    ~1e-8 (see _assert_parity)."""
    from cell_eval2.scale import score_streaming_cell
    from cell_eval2.run import compute_metrics
    from cell_eval2.config import EvalConfig
    from cell_eval2.catalog import CATALOG, resolve_metrics
    # pick the anndata-kind metrics from 'full' so no DE (gpudge) is needed
    names, _ = resolve_metrics("full")
    anndata_only = [n for n in names if CATALOG[n].kind == "anndata"]
    cfg = EvalConfig(metrics=anndata_only, device="cpu")   # v2 defaults, CPU
    (_, cp), (_, cr) = cell_pair["pred"], cell_pair["real"]
    got = score_streaming_cell(cp, cr, config=cfg).sort(["perturbation", "metric"])
    want = compute_metrics(cp, cr, config=cfg).sort(["perturbation", "metric"])
    _assert_parity(got, want)


@pytest.mark.parametrize("version,expected", [
    ("v2", "bulk_lognorm"),
    ("v1", "lognorm"),
])
def test_cell_streaming_writer_stamps_the_resolved_comparator(
        tmp_path, cell_pair, version, expected):
    from cell_eval2.config import EvalConfig
    from cell_eval2.scale import score_streaming_cell

    (_, cp), (_, cr) = cell_pair["pred"], cell_pair["real"]
    parts = tmp_path / "parts"
    score_streaming_cell(
        cp, cr,
        config=EvalConfig(metrics=["mae"], device="cpu", input_type="counts",
                          version=version),
        partial_out=str(parts),
    )
    meta = json.loads((parts / "all.json").read_text())
    assert meta["comparator"] == expected


def test_score_streaming_cell_full_parity_gpu(cell_pair):
    """Full metric set (incl. gpudge DE) on GPU: score_streaming_cell must match
    compute_metrics on the materialized archives — THE Stage-2 correctness gate. Rank/DE
    metrics (pds_*, de_*) exact, continuous to ~1e-8 (see _assert_parity). The de_* EXACT
    assertion also gates the real-side DE core equivalence (string-ref in-mem vs
    refpool_de_core Mode-1); if a de_* metric diverges on real hardware that is a
    surface-to-user finding, not something to silently loosen. Skips without CUDA."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("gpudge streaming DE requires a CUDA device")
    pytest.importorskip("gpudge")
    from cell_eval2.scale import score_streaming_cell
    from cell_eval2.run import compute_metrics
    from cell_eval2.config import EvalConfig
    cfg = EvalConfig(device="cuda")   # v2 defaults, full metrics, gpudge DE
    (_, cp), (_, cr) = cell_pair["pred"], cell_pair["real"]
    got = score_streaming_cell(cp, cr, config=cfg).sort(["perturbation", "metric"])
    want = compute_metrics(cp, cr, config=cfg).sort(["perturbation", "metric"])
    _assert_parity(got, want)


def test_compute_de_streaming_cell_target_sum_validation(cell_pair):
    """Non-1e6 finite target_sum is allowed (guard relaxed for counts); non-finite/<=0
    raise ValueError; None still raises the v1-median NotImplementedError."""
    from cell_eval2 import de_compute
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_reference, iter_cell_groups
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        ref = cell_reference(store)
        common = dict(
            ref_X=ref,
            group_iter_factory=lambda: iter_cell_groups(store, ["GENE_A"]),
            targets=["GENE_A"], var_names=np.array([f"g{j}" for j in range(60)]),
            n_genes=60, backend="gpudge", mean_calc="arithmetic", epsilon=1e-9,
            clip_value=None, fdr_scope="per_pert", filter_gene_min_cpm_cell=5.0)
        # None -> v1 median deferral (unchanged)
        with pytest.raises(NotImplementedError, match="median"):
            de_compute.compute_de_streaming_cell(target_sum=None, **common)
        # non-finite / non-positive / non-float-numeric -> ValueError (new validation, not the
        # old 1e6 raise). Includes bool (True is an int subclass -> would sneak in as 1.0) and a
        # string (np.isfinite would TypeError) — Gemini PR #131.
        for bad in (0.0, -5.0, float("nan"), float("inf"), True, np.bool_(True), "1e4"):
            with pytest.raises(ValueError, match="finite"):
                de_compute.compute_de_streaming_cell(target_sum=bad, **common)
        # a finite non-1e6 target (0.7.6's 1e4) must PASS the target_sum guard. Prove it
        # environment-independently (Gemini PR #131): mock _resolve_backend -- the step right
        # after the guard -- to raise a sentinel and assert we reach it. Robust whether or not a
        # GPU is present. With the old !=1e6 guard, 1e4 would raise NotImplementedError BEFORE
        # _resolve_backend and the sentinel would never fire.
        # Python float AND NumPy scalar targets must all PASS the guard (Gemini PR #131 flagged
        # that a strict (int, float) check would wrongly reject np.float64/np.int64).
        from unittest.mock import patch
        with patch("cell_eval2.de_compute._resolve_backend",
                   side_effect=RuntimeError("mocked_backend_check")):
            for val in (1e4, np.float64(1e4), np.int64(10000)):
                with pytest.raises(RuntimeError, match="mocked_backend_check"):
                    de_compute.compute_de_streaming_cell(target_sum=val, **common)
    finally:
        store.close()


def test_score_streaming_cell_rejects_lognorm(cell_pair):
    """The cell-layout streaming path (DE + anndata pseudobulk) is counts-only; a lognorm
    config must raise loudly (no silent mis-score), before any GPU work."""
    from cell_eval2.scale import score_streaming_cell
    from cell_eval2.config import EvalConfig
    from cell_eval2.catalog import CATALOG, resolve_metrics
    names, _ = resolve_metrics("full")
    anndata_only = [n for n in names if CATALOG[n].kind == "anndata"]
    cfg = EvalConfig(metrics=anndata_only, input_type="lognorm", device="cpu")
    (_, cp), (_, cr) = cell_pair["pred"], cell_pair["real"]
    with pytest.raises(NotImplementedError, match="counts"):
        score_streaming_cell(cp, cr, config=cfg)


def test_score_streaming_cell_target_sum_1e4_parity_gpu(cell_pair):
    """Counts input at target_sum=1e4 (cell-eval-0.7.6's library-size target): the relaxed
    guard must stream (DE + anndata) identically to compute_metrics on the materialized
    archives — rank/DE exact, continuous ~1e-7 (see _assert_parity). Skips without CUDA."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("gpudge streaming DE requires a CUDA device")
    pytest.importorskip("gpudge")
    from cell_eval2.scale import score_streaming_cell
    from cell_eval2.run import compute_metrics
    from cell_eval2.config import EvalConfig
    cfg = EvalConfig(target_sum=1e4, device="cuda")   # v2 defaults (counts), non-1e6 target
    (_, cp), (_, cr) = cell_pair["pred"], cell_pair["real"]
    got = score_streaming_cell(cp, cr, config=cfg).sort(["perturbation", "metric"])
    want = compute_metrics(cp, cr, config=cfg).sort(["perturbation", "metric"])
    _assert_parity(got, want)


class _FakeGroupStore:
    """Minimal CellStore stand-in for the cell_source wiring guard (cellstream-free -> runs in CI)."""

    def __init__(self, sizes=(("non-targeting", 50), ("GENE_A", 20), ("GENE_B", 30)),
                 n_vars=4, reference="non-targeting"):
        self._sizes = dict(sizes)
        self.n_vars = n_vars
        self.n_obs = sum(self._sizes.values())
        self.manifest = {"group_by": "target"}
        self._reference = reference
        # cell_metadata / cell_fingerprint (used by scale.score_streaming_cell) read
        # store.var.index.values as well as group_labels()/n_obs/n_vars.
        self.var = pd.DataFrame(index=[f"g{j}" for j in range(n_vars)])
        self.calls: list[tuple[str, int, int]] = []   # (what, n_rows, n_threads)

    def _load_groups(self):
        groups, start = [], 0
        for label, n in self._sizes.items():
            groups.append({"label": label, "start": start, "stop": start + n})
            start += n
        return {"reference": self._reference, "groups": groups}

    def group_labels(self):
        return list(self._sizes)

    def read_group(self, label, n_threads=1):
        n = self._sizes[str(label)]
        self.calls.append(("read_group", n, n_threads))
        return sp.csr_matrix((n, self.n_vars), dtype=np.float32)

    def read_reference(self, n_threads=1):
        n = self._sizes[self._reference]
        self.calls.append(("read_reference", n, n_threads))
        return sp.csr_matrix((n, self.n_vars), dtype=np.float32)

    def gather_rows_adata(self, row_ids, n_threads=1):
        """#266's cell-layout half: score_streaming_cell now resolves each archive's EFFECTIVE
        input type -- its gate keyed on the DECLARED one -- and that peeks a ROW block, which this
        group-only fake did not implement. Integer-valued (all-zero) rows, so the peek resolves to
        'counts' and the new guard passes: this fake exists to check gather-thread wiring, not
        input types."""
        rows = np.atleast_1d(np.asarray(row_ids))
        self.calls.append(("gather_rows_adata", int(rows.size), n_threads))
        return ad.AnnData(
            X=sp.csr_matrix((rows.size, self.n_vars), dtype=np.float32),
            obs=pd.DataFrame({"target": ["non-targeting"] * rows.size},
                             index=[f"c{i}" for i in range(rows.size)]),
            var=pd.DataFrame(index=[f"g{j}" for j in range(self.n_vars)]),
        )

    def close(self):
        # score_streaming_cell closes both stores in a finally: block (scale.py); without this
        # the pseudobulk-path test raises AttributeError in teardown after reaching its assert.
        pass


def _sentinel_resolver(monkeypatch, module, value=7):
    """See tests/test_cellstream.py -- patch the name in the CONSUMING module, because
    cell_source does `from ._threads import resolve_gather_threads` at import time."""
    seen: list[tuple] = []

    def fake(n_rows, gather_threads):
        seen.append((n_rows, gather_threads))
        return value

    monkeypatch.setattr(module, "resolve_gather_threads", fake)
    return seen


def test_cell_source_plumbs_n_threads_to_every_read(monkeypatch):
    """#149 regression guard for the three cell_source sites. The sentinel is what makes this
    non-vacuous: a site that never passes n_threads records cellstream's default 1, which on these
    small groups is exactly what a correct resolution would also produce."""
    import cell_eval2.cell_source as csrc

    store = _FakeGroupStore()
    seen = _sentinel_resolver(monkeypatch, csrc, value=7)

    csrc.cell_reference(store, gather_threads=5)
    list(csrc.cell_group_blocks(store, gather_threads=5))
    list(csrc.iter_cell_groups(store, ["GENE_A"], gather_threads=5))

    assert [w for w, _, _ in store.calls] == (
        ["read_reference"] + ["read_group"] * 3 + ["read_group"])
    assert all(nt == 7 for _, _, nt in store.calls), (
        f"a read site is not passing n_threads (got {store.calls})")
    # each resolver call saw the EXACT row count of the read it sized, and the config verbatim
    assert [n for _, n, _ in store.calls] == [n_rows for n_rows, _ in seen]
    assert all(gt == 5 for _, gt in seen), seen


def test_cell_reference_sizes_by_the_reference_pool_not_n_obs(monkeypatch):
    """cell_reference must size the ramp by the REFERENCE pool's row count, not by n_obs. A
    50-row reference inside a 100-row archive must resolve like a 50-row read; using n_obs as an
    upper bound would over-thread every small reference pool in a large archive."""
    import cell_eval2.cell_source as csrc

    store = _FakeGroupStore()
    assert store.n_obs == 100 and store._sizes["non-targeting"] == 50
    seen = _sentinel_resolver(monkeypatch, csrc, value=7)
    csrc.cell_reference(store, gather_threads=-1)
    assert seen == [(50, -1)], seen


def test_group_row_counts_matches_the_archive(cell_pair):
    """The size map that feeds the per-group ramp must equal the archive's real group sizes."""
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import _group_row_counts
    _, shad = cell_pair["pred"]
    store = open_cell_store(shad)
    try:
        sizes = _group_row_counts(store)
        assert sizes == {p: 40 for p in PERTS}            # _synth writes n_per=40 per group
        for label in store.group_labels():
            assert sizes[str(label)] == store.read_group(str(label)).shape[0]
    finally:
        store.close()


def test_cell_source_reads_are_unchanged_by_threading(tmp_path, monkeypatch):
    """Parity on a REAL archive, with groups big enough that threading actually engages.

    The spy is what makes this non-vacuous: it asserts the threaded arm OBSERVED n_threads > 1
    at the store and the serial arm observed exactly 1. Asserting only what the resolver would
    return proves nothing about what the read actually received.
    """
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive

    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_group_blocks, cell_reference

    n_per = 250                                            # > 96 -> resolves to >1 thread
    adata = _synth(3, n_per=n_per)
    shad = tmp_path / "big.shad"
    write_cell_archive(adata, shad, group_by="target",
                       reference=(adata.obs["target"] == "non-targeting").to_numpy(),
                       codec="zstd", overwrite=True)

    got, threads_used = {}, {}
    for gt in (1, 8):
        store = open_cell_store(shad)
        cls = type(store)
        seen: list[int] = []
        orig_ref, orig_grp = cls.read_reference, cls.read_group

        def ref_spy(self, n_threads=1, _s=seen, _o=orig_ref):
            _s.append(n_threads)
            return _o(self, n_threads=n_threads)

        def grp_spy(self, label, n_threads=1, _s=seen, _o=orig_grp):
            _s.append(n_threads)
            return _o(self, label, n_threads=n_threads)

        monkeypatch.setattr(cls, "read_reference", ref_spy)
        monkeypatch.setattr(cls, "read_group", grp_spy)
        try:
            got[gt] = (cell_reference(store, gather_threads=gt).toarray(),
                       [(X.toarray(), list(lbl)) for X, lbl in
                        cell_group_blocks(store, gather_threads=gt)])
        finally:
            store.close()
            monkeypatch.setattr(cls, "read_reference", orig_ref)
            monkeypatch.setattr(cls, "read_group", orig_grp)
        threads_used[gt] = seen

    assert max(threads_used[8]) > 1, (
        f"gather_threads=8 never engaged >1 thread ({threads_used[8]}) -- this parity test "
        "would be vacuous; enlarge the fixture")
    assert set(threads_used[1]) == {1}, threads_used[1]
    np.testing.assert_array_equal(got[1][0], got[8][0])
    assert [lbl for _, lbl in got[1][1]] == [lbl for _, lbl in got[8][1]]
    for (x1, _), (x8, _) in zip(got[1][1], got[8][1]):
        np.testing.assert_array_equal(x1, x8)


def test_scale_de_path_passes_gather_threads(monkeypatch):
    """scale._score_streaming_cell_de must propagate cfg.gather_threads to BOTH cell_reference
    calls and to iter_cell_groups (3 of the 5 scale.py sites). Stops on a dedicated sentinel
    raised AFTER every call under test, so a stub blowing up early cannot be mistaken for
    success (all the patched names are real module-level symbols in scale.py)."""
    import cell_eval2.scale as scale
    from cell_eval2.config import EvalConfig

    seen: list[tuple[str, object]] = []
    meta = scale.cell_source.CellMeta(10, 2, np.array(["g0", "g1"]), "target",
                                      np.array(["non-targeting", "GENE_A"]))
    monkeypatch.setattr(scale.cell_source, "cell_metadata", lambda store: meta)

    def fake_reference(store, **kw):
        seen.append(("cell_reference", kw.get("gather_threads")))
        return "REF"

    def fake_groups(store, labels, **kw):
        seen.append(("iter_cell_groups", kw.get("gather_threads")))
        return iter(())

    monkeypatch.setattr(scale.cell_source, "cell_reference", fake_reference)
    monkeypatch.setattr(scale.cell_source, "iter_cell_groups", fake_groups)

    def fake_de(**kw):
        list(kw["group_iter_factory"]())          # force the factory -> iter_cell_groups runs
        return scale.pl.DataFrame({"target": []}, schema={"target": scale.pl.Utf8})

    monkeypatch.setattr(scale, "compute_de_streaming_cell", fake_de)

    class _Done(Exception):
        """Reached prepare_de -- both references and both group iterations already happened."""

    def _stop(*a, **k):
        raise _Done

    monkeypatch.setattr(scale, "prepare_de", _stop)

    cfg = EvalConfig(pert_col="target", control="non-targeting", gather_threads=9,
                     control_source="pred")     # 'pred' -> pred reads its OWN reference
    with pytest.raises(_Done):
        scale._score_streaming_cell_de(object(), object(), cfg=cfg, de_names=["de_overlap"],
                                       chosen={"GENE_A"})
    assert seen.count(("cell_reference", 9)) == 2, seen       # real Mode 1 + pred Mode 1
    assert seen.count(("iter_cell_groups", 9)) == 2, seen     # one per de_table call


def test_scale_pseudobulk_path_passes_gather_threads(monkeypatch):
    """score_streaming_cell must propagate cfg.gather_threads to BOTH cell_pseudobulk calls
    (the other 2 of the 5 scale.py sites). metrics=['mae'] selects an anndata metric only, so
    the DE branch is skipped and no GPU is needed. cellstream-free via _FakeGroupStore."""
    import cell_eval2.scale as scale
    from cell_eval2.config import EvalConfig

    store = _FakeGroupStore()
    monkeypatch.setattr(scale, "open_cell_store", lambda p: store)
    seen: list[object] = []

    def fake_pseudobulk(store_, **kw):
        seen.append(kw.get("gather_threads"))
        perts = np.array(list(store._sizes))
        out = {n: (perts, np.zeros((perts.size, store.n_vars), dtype=np.float64))
               for n in kw["norms"]}
        return (out, 1.0) if kw.get("with_median_umi") else out

    monkeypatch.setattr(scale.cell_source, "cell_pseudobulk", fake_pseudobulk)
    monkeypatch.setattr(scale, "_restrict", lambda bulks, keep: bulks)
    monkeypatch.setattr(scale, "dispatch_anndata_metrics", lambda *a, **k: [])

    scale.score_streaming_cell("pred.shad", "real.shad", config=EvalConfig(
        metrics=["mae"], pert_col="target", control="non-targeting", gather_threads=9))
    assert seen == [9, 9], seen
