"""Streaming integration tests for the ground-truth DEG cache (docs/ground_truth_cache.md).

Exercises the seam added to the h5ad-manifest-free streaming path: ``partition_inmem``'s
``RealDEArtifact`` / ``real_de_provider`` protocol, ``_build_reference_streaming_core``'s
provider-aware stages, and ``cellstream.score_cellstream``'s wiring of a real
``GroundTruthDEGCache`` lookup/populate around the real-DE compute.

CPU-only throughout: every DE compute is a deterministic fake (the real gpudge/pdex/scanpy
backends are never invoked), and ``partition_inmem._resolve_backend`` /
``cell_eval2.de_compute._resolve_backend`` are forced to report ``"gpudge"`` so
``_require_partition_config`` accepts the config without a CUDA device. ``pyarrow`` is required
(the ground-truth cache's pinned parquet writer).
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("pyarrow")

import anndata as ad  # noqa: E402

import cell_eval2.cellstream as cellstream  # noqa: E402
import cell_eval2.de_compute as de_compute  # noqa: E402
import cell_eval2.ground_truth_cache as gtcache  # noqa: E402
import cell_eval2.partition_inmem as partition_inmem  # noqa: E402
from cell_eval2.config import EvalConfig  # noqa: E402
from cell_eval2.de_export import sha256_file  # noqa: E402
from cell_eval2.ground_truth_cache import (  # noqa: E402
    GroundTruthCacheCorruptError,
    GroundTruthCacheFilesystemError,
    GroundTruthCacheMissError,
    GroundTruthContractError,
    GroundTruthDEGCache,
    GroundTruthDEGCacheRequest,
    GroundTruthFingerprint,
    GroundTruthFingerprintError,
    build_real_de_semantics,
    resolve_de_precision,
)
from cell_eval2.h5ad_manifest import MemBudget  # noqa: E402
from cell_eval2.partition_inmem import RealDEArtifact  # noqa: E402
from _helpers import full_minus_moments, resolved_comparator  # noqa: E402

_VERSIONS = {"cell-eval2": "0.17.0", "gpudge": "9.9.9", "cellstream": "0.10.0"}
_BIG = MemBudget(host_bytes=10**12, gpu_bytes=10**12)
_PAYLOAD = "cd" * 32
# The first two "genes" double as the FakeSource perturbation labels ("A"/"B"), matching the
# target==feature-symbol convention every direct-dispatch test in this repo relies on
# (target-gene resolution refuses a target that resolves to no feature -- see
# metrics/de.py::require_resolution / distances.py::resolve_exclusion_columns).
GENES = ["A", "B", "g0", "g1", "g2", "g3"]

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "ground_truth_cache")
FIXTURE_REAL = os.path.join(FIXTURE_DIR, "archives", "real.shad")
FIXTURE_PRED = os.path.join(FIXTURE_DIR, "archives", "pred_a.shad")
FIXTURE_MANIFEST = os.path.join(FIXTURE_DIR, "fixture_manifest.json")


@pytest.fixture(autouse=True)
def _gtc_streaming_env(monkeypatch):
    """Force the partition-eligibility gate to accept a CPU host, and make the ground-truth
    cache's semantic-document builder deterministic without gpudge/cellstream actually
    installed at a particular version.

    Two DIFFERENT ``_resolve_backend`` bindings need patching, not one: ``partition_inmem.py``
    imports it at MODULE load time (``from .de_compute import _resolve_backend``), so patching
    ``de_compute._resolve_backend`` alone would not reach ``partition_inmem``'s already-bound
    name. ``cellstream.score_cellstream`` instead imports it FRESH, function-locally, on every
    call (``from .de_compute import _resolve_backend``) specifically so a test can patch
    ``de_compute._resolve_backend`` and have it take effect there.
    """
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda backend: "gpudge")
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda backend: "gpudge")
    monkeypatch.setattr(gtcache, "producer_versions", lambda **kw: dict(_VERSIONS))


# ── Deterministic fake DE ───────────────────────────────────────────────────────────────


def make_fake_compute_de():
    """A ``partition_inmem.compute_de`` replacement returning the 11 canonical DE columns, with
    values derived deterministically from each target's own batch means (so re-running on the
    same input data reproduces the same table byte-for-byte -- the whole point of the
    cold/warm/off equality assertions below). Returns ``(fn, calls)``; ``calls`` records one
    entry per invocation for call-count assertions.
    """
    calls: list[int] = []

    def fake_compute_de(adata, *, reference, groupby, control_group, **kw):
        calls.append(1)
        obs_t = adata.obs[groupby].astype(str).to_numpy()
        targets = sorted(set(obs_t) - {control_group})
        features = list(adata.var_names)
        import scipy.sparse as sp

        X = adata.X
        Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
        rows_t, rows_f, tmeans, refm, lfc, pval, uvals, padj = [], [], [], [], [], [], [], []
        for t in targets:
            sub = Xd[obs_t == t]
            tmean = sub.mean(axis=0) if sub.shape[0] else np.zeros(len(features))
            for j, f in enumerate(features):
                tm = float(tmean[j])
                rm = 0.5
                rows_t.append(t)
                rows_f.append(f)
                tmeans.append(tm)
                refm.append(rm)
                lval = tm - rm
                lfc.append(lval)
                pval.append(1.0 / (1.0 + abs(lval)))
                uvals.append(tm * 2.0)
                padj.append(min(1.0, (1.0 / (1.0 + abs(lval))) * 2))
        n = len(rows_t)
        return pl.DataFrame({
            "target": rows_t, "feature": rows_f,
            "target_mean": tmeans, "ref_mean": refm,
            "target_ncells": [10] * n, "ref_ncells": [8] * n,
            "log2_fold_change": lfc, "p_value": pval, "U": uvals, "p_adj": padj,
            "abs_log2_fold_change": [abs(v) for v in lfc],
        })

    return fake_compute_de, calls


# ── FakeSource: a minimal PertBatchSource for direct _build_reference_streaming_core tests ──


class FakeSource:
    """A ``PertBatchSource`` over a tiny synthetic panel: an 8-cell control block plus two
    5-cell perturbation batches ("A", "B"), all Poisson counts over ``GENES``. ``seed`` controls
    the underlying data -- two ``FakeSource(seed=N)`` instances with the same ``N`` produce
    byte-identical batches, which is what lets the cold/warm/off equality assertions compare
    runs built from *separate* source instances (mirroring separate archive reads) rather than
    one re-used object.
    """

    control = "ctrl"

    def __init__(self, seed: int = 0):
        self.stream_tag = f"fake-{seed}"
        rng = np.random.default_rng(seed)
        n_genes = len(GENES)
        self._control_X = rng.poisson(3, size=(8, n_genes)).astype(np.float32)
        self._batches = [
            ("A", rng.poisson(3, size=(5, n_genes)).astype(np.float32)),
            ("B", rng.poisson(3, size=(5, n_genes)).astype(np.float32)),
        ]
        self.iter_calls = 0
        self.control_calls = 0

    def read_control_block(self):
        self.control_calls += 1
        obs = pd.DataFrame(
            {"perturbation": [self.control] * self._control_X.shape[0]},
            index=[f"r{i}" for i in range(self._control_X.shape[0])],
        )
        return ad.AnnData(X=self._control_X.copy(), obs=obs, var=pd.DataFrame(index=GENES))

    def iter_pert_batches(self, mem_budget):
        self.iter_calls += 1
        for pert, X in self._batches:
            obs = pd.DataFrame(
                {"perturbation": [pert] * X.shape[0]},
                index=[f"r{pert}{i}" for i in range(X.shape[0])],
            )
            yield [pert], ad.AnnData(X=X.copy(), obs=obs, var=pd.DataFrame(index=GENES))

    def pert_universe(self) -> list[str]:
        return sorted(p for p, _ in self._batches)

    def close(self):
        pass


def _cfg_full():
    """DE + anndata metrics (needs the real-side pseudobulk)."""
    return replace(EvalConfig.v2(), device="cpu", pert_col="perturbation", control="ctrl",
                   metrics=full_minus_moments())


def _cfg_de_only():
    """DE-kind metrics only -- resolves to an empty pseudobulk normalization set, so a cache HIT
    never needs to stream a single batch (no ``pert_universe()`` avoidance needed)."""
    return replace(EvalConfig.v2(), device="cpu", pert_col="perturbation", control="ctrl",
                   metrics=["de_wilcoxon_lfc_spearman", "de_wilcoxon_lfc_nmae",
                           "de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred"])


def _fake_fingerprint(payload=_PAYLOAD):
    return GroundTruthFingerprint.from_axes(
        payload_sha256=payload, var_names=GENES,
        pert_labels=["ctrl"] * 8 + ["A"] * 5 + ["B"] * 5, pert_col="perturbation",
    )


def _make_provider(policy, cache_root, gt_fp, *, state=None):
    """A ``real_de_provider`` mirroring ``cellstream.score_cellstream``'s own closure, for
    driving ``_build_reference_streaming_core`` directly (cases a-d below). Like the real one it
    exists only with the cache ON -- with the cache off there is no provider at all."""
    state = {} if state is None else state

    def provider(*, cfg_resolved, control_ad, compute):
        sem = build_real_de_semantics(
            cfg_resolved, control=cfg_resolved.control, input_type=cfg_resolved.input_type,
            target_sum=cfg_resolved.target_sum, device="cpu", de_backend="gpudge",
            native_gpu_normalize=True,
            precision=resolve_de_precision(
                control_ad.X.dtype, input_type=cfg_resolved.input_type,
                native_gpu_normalize=True, target_sum=cfg_resolved.target_sum,
            ),
            versions=_VERSIONS,
        )
        state["semantics"] = sem
        cache = GroundTruthDEGCache(str(cache_root), policy=policy)
        req = GroundTruthDEGCacheRequest(fingerprint=gt_fp, semantics=sem)
        res = cache.get_or_compute(req, compute)
        state["result"] = res
        return RealDEArtifact(path=res.table_path,
                              provenance={"kind": res.status, "cache_id": res.cache_id})

    return provider, state


# ── (a)-(d): _build_reference_streaming_core driven directly with FakeSource ────────────────


def test_provider_none_cold_warm_produce_identical_real_de(tmp_path, monkeypatch):
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    gt_fp = _fake_fingerprint()
    cfg = _cfg_full()
    comparator = resolved_comparator(cfg)

    def _run(cache_dir, provider=None):
        return partition_inmem._build_reference_streaming_core(
            FakeSource(seed=0), config=cfg, cache_dir=str(cache_dir), mem_budget=_BIG,
            comparator=comparator, real_de_provider=provider)

    d_none = tmp_path / "none"
    _run(d_none)
    df_none = pl.read_parquet(d_none / "real_de.parquet")
    man_none = json.loads((d_none / "reference.json").read_text())
    # Spec §5 / T6: a provider-less bundle (the cache off) keeps the manifest it always had.
    assert "real_de_source" not in man_none
    assert not os.path.islink(str(d_none / "real_de.parquet"))

    prov_cold, _ = _make_provider("readwrite", cache_root, gt_fp)
    d_cold = tmp_path / "cold"
    _run(d_cold, prov_cold)
    df_cold = pl.read_parquet(d_cold / "real_de.parquet")
    man_cold = json.loads((d_cold / "reference.json").read_text())
    assert man_cold["real_de_source"]["kind"] == "miss_computed"
    cold_link = d_cold / "real_de.parquet"
    assert os.path.islink(str(cold_link))
    assert os.path.realpath(str(cold_link)).startswith(os.path.realpath(str(cache_root)))

    prov_warm, _ = _make_provider("readwrite", cache_root, gt_fp)
    d_warm = tmp_path / "warm"
    _run(d_warm, prov_warm)
    df_warm = pl.read_parquet(d_warm / "real_de.parquet")
    man_warm = json.loads((d_warm / "reference.json").read_text())
    assert man_warm["real_de_source"]["kind"] == "hit"
    assert man_warm["real_de_source"]["cache_id"] == man_cold["real_de_source"]["cache_id"]
    warm_link = d_warm / "real_de.parquet"
    assert os.path.islink(str(warm_link))
    assert os.path.realpath(str(warm_link)).startswith(os.path.realpath(str(cache_root)))

    # pairwise .equals(), order included -- the same underlying FakeSource(seed=0) data feeds
    # every one of the three runs.
    assert df_none.equals(df_cold)
    assert df_cold.equals(df_warm)


def test_real_de_artifact_requires_a_path():
    """Spec §7 / T6: a provider hands back a path -- there is no in-memory variant."""
    with pytest.raises(ValueError, match="non-empty path"):
        RealDEArtifact(path="", provenance={})
    with pytest.raises(ValueError, match="non-empty path"):
        RealDEArtifact(path=None, provenance={})


def test_warm_bundle_calls_no_real_de_but_score_piece_still_scores_pred(tmp_path, monkeypatch):
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    gt_fp = _fake_fingerprint()
    cfg = _cfg_full()
    comparator = resolved_comparator(cfg)

    # Cold: populate the cache.
    prov_cold, _ = _make_provider("readwrite", cache_root, gt_fp)
    partition_inmem._build_reference_streaming_core(
        FakeSource(seed=0), config=cfg, cache_dir=str(tmp_path / "cold"), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_cold)

    calls.clear()
    prov_warm, _ = _make_provider("readwrite", cache_root, gt_fp)
    warm_dir = tmp_path / "warm"
    partition_inmem._build_reference_streaming_core(
        FakeSource(seed=0), config=cfg, cache_dir=str(warm_dir), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_warm)
    assert calls == []  # zero real-side fake_compute_de calls on the warm build

    bundle = partition_inmem._RefBundle(str(warm_dir), cfg)
    piece = ad.AnnData(
        X=np.random.default_rng(2).poisson(3, size=(4, len(GENES))).astype(np.float32),
        obs=pd.DataFrame({"perturbation": ["A"] * 4}, index=[f"p{i}" for i in range(4)]),
        var=pd.DataFrame(index=GENES),
    )
    scored = partition_inmem.score_piece(piece, str(warm_dir), config=cfg, bundle=bundle,
                                         comparator=comparator)
    assert scored.height > 0
    assert calls == [1]  # exactly one pred-side compute_de call


def test_warm_de_only_hit_never_streams_a_batch(tmp_path, monkeypatch):
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    gt_fp = _fake_fingerprint()
    cfg = _cfg_de_only()
    comparator = resolved_comparator(cfg)

    prov_cold, _ = _make_provider("readwrite", cache_root, gt_fp)
    partition_inmem._build_reference_streaming_core(
        FakeSource(seed=5), config=cfg, cache_dir=str(tmp_path / "cold"), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_cold)

    warm_source = FakeSource(seed=5)
    prov_warm, _ = _make_provider("readwrite", cache_root, gt_fp)
    manifest = partition_inmem._build_reference_streaming_core(
        warm_source, config=cfg, cache_dir=str(tmp_path / "warm"), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_warm)

    assert warm_source.iter_calls == 0
    assert warm_source.control_calls == 1
    assert manifest["perturbation_universe"] == sorted(warm_source.pert_universe())


def test_warm_full_profile_hit_still_streams_once_for_pseudobulk(tmp_path, monkeypatch):
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    gt_fp = _fake_fingerprint()
    cfg = _cfg_full()
    comparator = resolved_comparator(cfg)

    prov_cold, _ = _make_provider("readwrite", cache_root, gt_fp)
    cold_dir = tmp_path / "cold"
    partition_inmem._build_reference_streaming_core(
        FakeSource(seed=7), config=cfg, cache_dir=str(cold_dir), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_cold)
    npz_cold = dict(np.load(cold_dir / "real_pseudobulk_bulk_lognorm.npz"))

    calls.clear()
    warm_source = FakeSource(seed=7)
    prov_warm, _ = _make_provider("readwrite", cache_root, gt_fp)
    warm_dir = tmp_path / "warm"
    partition_inmem._build_reference_streaming_core(
        warm_source, config=cfg, cache_dir=str(warm_dir), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_warm)
    npz_warm = dict(np.load(warm_dir / "real_pseudobulk_bulk_lognorm.npz"))

    assert warm_source.iter_calls == 1
    assert np.array_equal(npz_cold["perts"], npz_warm["perts"])
    assert np.array_equal(npz_cold["means"], npz_warm["means"])
    assert calls == []  # zero real DE calls -- the pass ran with with_de=False


def test_hit_with_no_norms_and_no_pert_universe_raises_typeerror(tmp_path, monkeypatch):
    """A HIT with an empty pseudobulk selection needs ``source.pert_universe()``; a source
    lacking it (H5adBatchSource/RowStoreBatchSource today) must fail with a clear TypeError
    naming the concrete source type, not an AttributeError deep in ``sorted(None())``."""
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    gt_fp = _fake_fingerprint()
    cfg = _cfg_de_only()
    comparator = resolved_comparator(cfg)

    prov_cold, _ = _make_provider("readwrite", cache_root, gt_fp)
    partition_inmem._build_reference_streaming_core(
        FakeSource(seed=9), config=cfg, cache_dir=str(tmp_path / "cold"), mem_budget=_BIG,
        comparator=comparator, real_de_provider=prov_cold)

    class _NoUniverseSource(FakeSource):
        pert_universe = None  # not callable -- simulates a source without the optional member

    prov_warm, _ = _make_provider("readwrite", cache_root, gt_fp)
    with pytest.raises(TypeError, match="pert_universe"):
        partition_inmem._build_reference_streaming_core(
            _NoUniverseSource(seed=9), config=cfg, cache_dir=str(tmp_path / "warm"),
            mem_budget=_BIG, comparator=comparator, real_de_provider=prov_warm)


# ── _write_reference_bundle unit tests ──────────────────────────────────────────────────────


def _tiny_control_ad():
    return ad.AnnData(X=np.ones((2, len(GENES)), dtype=np.float32),
                      obs=pd.DataFrame({"perturbation": ["ctrl"] * 2}),
                      var=pd.DataFrame(index=GENES))


def test_write_reference_bundle_rejects_both_de_df_and_real_de_source(tmp_path):
    df = pl.DataFrame({"target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0]})
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    with pytest.raises(ValueError, match="exactly one"):
        partition_inmem._write_reference_bundle(
            str(tmp_path / "bundle"), cfg=_cfg_de_only(), bulks={}, control_ad=_tiny_control_ad(),
            real_ref_fingerprint="fp", var_index=GENES, universe=["A"], control_format="h5ad",
            comparator="lognorm", de_df=df, real_de_source=str(src))


def test_write_reference_bundle_rejects_neither_de_df_nor_real_de_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        partition_inmem._write_reference_bundle(
            str(tmp_path / "bundle"), cfg=_cfg_de_only(), bulks={}, control_ad=_tiny_control_ad(),
            real_ref_fingerprint="fp", var_index=GENES, universe=["A"], control_format="h5ad",
            comparator="lognorm")


def test_write_reference_bundle_real_de_source_symlinks(tmp_path):
    df = pl.DataFrame({"target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0]})
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    dest = tmp_path / "bundle"
    os.makedirs(dest, exist_ok=True)
    partition_inmem._write_reference_bundle(
        str(dest), cfg=_cfg_de_only(), bulks={}, control_ad=_tiny_control_ad(),
        real_ref_fingerprint="fp", var_index=GENES, universe=["A"], control_format="h5ad",
        comparator="lognorm", real_de_source=str(src))
    linked = dest / "real_de.parquet"
    assert os.path.islink(str(linked))
    assert os.path.realpath(str(linked)) == os.path.realpath(str(src))
    man = json.loads((dest / "reference.json").read_text())
    assert "real_de_source" not in man  # no provenance given -> no key (spec §7, D4)


@pytest.mark.parametrize("branch", ["source-symlink", "source-copy-fallback", "in-memory"])
def test_write_reference_bundle_never_writes_through_an_existing_leaf(tmp_path, monkeypatch,
                                                                     branch):
    """A bundle directory reused after a provider-backed build holds real_de.parquet as a symlink
    to a published cache entry. Writing the new table -- linked, copied when symlinks fail, or
    written from memory -- must replace that leaf, never write through it: shutil.copy2 and
    write_parquet both follow a destination symlink and would overwrite the immutable entry
    (Copilot, PR #399)."""
    entry_table = tmp_path / "published_entry.parquet"
    pl.DataFrame({"target": ["OLD"], "feature": ["g0"], "log2_fold_change": [9.0]}).write_parquet(
        entry_table)
    before = entry_table.read_bytes()
    new_df = pl.DataFrame({"target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0]})
    src = tmp_path / "src.parquet"
    new_df.write_parquet(src)
    dest = tmp_path / "bundle"
    dest.mkdir()
    os.symlink(str(entry_table), str(dest / "real_de.parquet"))
    if branch == "source-copy-fallback":
        def _no_new_symlinks(*a, **k):
            raise OSError("no symlinks on this filesystem")

        monkeypatch.setattr(os, "symlink", _no_new_symlinks)
    kw = dict(de_df=new_df) if branch == "in-memory" else dict(real_de_source=str(src))
    partition_inmem._write_reference_bundle(
        str(dest), cfg=_cfg_de_only(), bulks={}, control_ad=_tiny_control_ad(),
        real_ref_fingerprint="fp", var_index=GENES, universe=["A"], control_format="h5ad",
        comparator="lognorm", **kw)
    assert entry_table.read_bytes() == before
    assert pl.read_parquet(str(dest / "real_de.parquet")).equals(new_df)


def test_write_reference_bundle_symlink_failure_falls_back_to_copy(tmp_path, monkeypatch):
    df = pl.DataFrame({"target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0]})
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    dest = tmp_path / "bundle"
    os.makedirs(dest, exist_ok=True)

    def _boom(*a, **k):
        raise OSError("no symlinks on this filesystem")

    monkeypatch.setattr(os, "symlink", _boom)

    partition_inmem._write_reference_bundle(
        str(dest), cfg=_cfg_de_only(), bulks={}, control_ad=_tiny_control_ad(),
        real_ref_fingerprint="fp", var_index=GENES, universe=["A"], control_format="h5ad",
        comparator="lognorm", real_de_source=str(src),
        real_de_provenance={"kind": "hit", "cache_id": "abc123"})
    linked = dest / "real_de.parquet"
    assert not os.path.islink(str(linked))
    assert pl.read_parquet(str(linked)).equals(df)
    man = json.loads((dest / "reference.json").read_text())
    assert man["real_de_source"] == {"kind": "hit", "cache_id": "abc123"}


# ── (e)-(h): score_cellstream driven with a fake cell store (cellstream-free) ───────────────


class _GTFakeCellStore:
    """Minimal cell-store stand-in for ``cellstream.CellBatchSource`` -- same shape as
    ``test_cellstream.py``'s ``_FakeCellStore``, plus an optional ``payload_sha256`` in
    ``manifest`` so ``fingerprint_ground_truth_cell_archive`` can (or, when omitted,
    deliberately cannot) fingerprint it. Perturbation labels double as gene symbols (target-gene
    resolution, as in ``FakeSource`` above); ``id_prefix`` keeps a real- and a pred-side store's
    obs_names from colliding when both back the same ``score_cellstream`` call.
    """

    def __init__(self, *, n_per=20, perts=("control", "A", "B"), payload_sha256=None,
                seed=0, id_prefix="c"):
        self.n_obs = n_per * len(perts)
        self.n_vars = len(perts) - 1 + 4  # one "gene" per non-control pert, plus 4 anonymous
        self.manifest = {"group_by": "perturbation", "value_dtype_on_disk": "float32"}
        if payload_sha256 is not None:
            self.manifest["payload_sha256"] = payload_sha256
        self._x_out_dtype = np.dtype("float32")
        self._perts = list(perts)
        self.obs = pd.DataFrame(
            {"perturbation": np.repeat(np.asarray(perts, dtype=object), n_per)},
            index=[f"{id_prefix}{i}" for i in range(self.n_obs)],
        )
        gene_names = [p for p in perts if p != "control"] + \
            [f"g{j}" for j in range(self.n_vars - (len(perts) - 1))]
        self.var = pd.DataFrame(index=gene_names)
        rng = np.random.default_rng(seed)
        self._X = rng.poisson(3, size=(self.n_obs, self.n_vars)).astype(np.float32)
        self.closed = False

    def group_labels(self):
        return list(self._perts)

    def gather_rows_adata(self, row_ids, n_threads=1):
        rows = np.atleast_1d(np.asarray(row_ids))
        return ad.AnnData(X=self._X[rows].copy(), obs=self.obs.iloc[rows].copy(),
                          var=self.var.copy())

    def close(self):
        self.closed = True


def _gt_cfg_de_only(**kw):
    base = dict(device="cpu", pert_col="perturbation", control="control",
               metrics=["de_wilcoxon_lfc_spearman", "de_wilcoxon_lfc_nmae",
                       "de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred"])
    base.update(kw)
    return replace(EvalConfig.v2(), **base)


def _open_fake_stores(monkeypatch, *, real_payload=_PAYLOAD, pred_payload=None,
                      perts=("control", "A", "B"), pred_seed=1):
    real_store = _GTFakeCellStore(perts=perts, payload_sha256=real_payload, seed=0,
                                  id_prefix="r")
    pred_store = _GTFakeCellStore(perts=perts, payload_sha256=pred_payload, seed=pred_seed,
                                  id_prefix="p")
    stores = {"real.path": real_store, "pred.path": pred_store}
    monkeypatch.setattr(cellstream, "open_cell_store", lambda p: stores[p])
    return real_store, pred_store


def test_score_cellstream_control_source_pred_writes_no_pred_artifacts_to_the_cache(
        tmp_path, monkeypatch):
    """(e): control_source='pred' still only ever writes de_real.parquet/manifest.json/.lock
    files into the ground-truth cache root -- pred_control.h5ad/pred_pseudobulk_*.npz live only
    in the (torn-down) temp ref_dir."""
    _open_fake_stores(monkeypatch, pred_payload=None)
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    cfg = _gt_cfg_de_only(control_source="pred")
    result = cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                         ground_truth_cache=str(cache_root))
    assert result.per_pert.height > 0

    names = set()
    for _root, _dirs, files in os.walk(str(cache_root)):
        names.update(files)
    assert names, "the cache root must have been populated"
    for name in names:
        assert name in ("de_real.parquet", "manifest.json") or name.endswith(".lock"), name
    assert not any(n.startswith("pred_") for n in names)


def test_score_cellstream_write_de_cold_then_warm(tmp_path, monkeypatch):
    """(f): cold then warm into two outdirs -> sidecar statuses miss_computed then hit, same
    cache_id, de_real.parquet bytes identical to the cache entry's own file in both runs, and
    the sidecar's de_real_sha256 matches sha256_file of that cache entry."""
    cache_root = tmp_path / "gtcache"
    cfg = _gt_cfg_de_only()
    gt = {"ground_truth_cache": str(cache_root)}

    _open_fake_stores(monkeypatch)
    fake_de1, _c1 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de1)
    out1 = tmp_path / "out1"
    res1 = cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                       outdir=str(out1), write_de=True, **gt)
    sidecar1 = json.loads((out1 / "ground_truth_de_cache.json").read_text())
    assert sidecar1["status"] == "miss_computed"
    assert sidecar1["policy"] == "readwrite"

    _open_fake_stores(monkeypatch)
    fake_de2, calls2 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de2)
    out2 = tmp_path / "out2"
    res2 = cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                       outdir=str(out2), write_de=True, **gt)
    sidecar2 = json.loads((out2 / "ground_truth_de_cache.json").read_text())
    assert sidecar2["status"] == "hit"
    assert sidecar2["policy"] == "readwrite"
    assert sidecar2["cache_id"] == sidecar1["cache_id"]

    de_real_1 = (out1 / "de_real.parquet").read_bytes()
    de_real_2 = (out2 / "de_real.parquet").read_bytes()
    assert de_real_1 == de_real_2

    entry_dir = os.path.join(str(cache_root), gtcache.entry_relpath(sidecar1["cache_id"]))
    entry_path = os.path.join(entry_dir, "de_real.parquet")
    assert sha256_file(entry_path) == sidecar1["de_real_sha256"] == sidecar2["de_real_sha256"]
    # T4: the published de_real.parquet is byte-identical to the entry it came from.
    assert sha256_file(str(out1 / "de_real.parquet")) == sidecar1["de_real_sha256"]

    assert res1.per_pert.height == res2.per_pert.height > 0

    # write_de=False emits none of the three files.
    _open_fake_stores(monkeypatch)
    fake_de3, _c3 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de3)
    out3 = tmp_path / "out3"
    cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                outdir=str(out3), **gt)
    assert not (out3 / "de_real.parquet").exists()
    assert not (out3 / "de_pred.parquet").exists()
    assert not (out3 / "ground_truth_de_cache.json").exists()


def test_cache_off_is_mains_path(tmp_path, monkeypatch):
    """T1 / spec §5 (ruling 6): with ground_truth_cache=None the reference builder gets no
    provider, nothing is fingerprinted, no semantic document or store is built, outdir holds
    exactly main's five write_de files, and reference.json carries no real_de_source."""
    _open_fake_stores(monkeypatch)
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    def _no_cache_work(*a, **k):
        raise AssertionError("ground-truth cache work with ground_truth_cache=None")

    monkeypatch.setattr(gtcache, "fingerprint_ground_truth_cell_archive", _no_cache_work)
    monkeypatch.setattr(gtcache, "producer_versions", _no_cache_work)
    monkeypatch.setattr(gtcache, "GroundTruthDEGCache", _no_cache_work)

    seen = {}
    real_builder = partition_inmem._build_reference_streaming_core

    def _spy_builder(source, **kw):
        seen["provider"] = kw.get("real_de_provider")
        seen["manifest"] = real_builder(source, **kw)
        return seen["manifest"]

    monkeypatch.setattr(partition_inmem, "_build_reference_streaming_core", _spy_builder)

    out = tmp_path / "out"
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, outdir=str(out), write_de=True)
    assert seen["provider"] is None
    assert "real_de_source" not in seen["manifest"]
    assert sorted(os.listdir(out)) == sorted(cellstream._WRITE_DE_OUTPUTS)


def test_score_cellstream_semantics_failure_is_fatal_only_with_the_cache_on(tmp_path,
                                                                          monkeypatch):
    """With the cache on, the semantic document KEYS the lookup, so a producer_versions()
    failure (no distribution metadata -- a source-tree/PYTHONPATH execution) is fatal and
    surfaces BEFORE any compute_de call. With the cache off no semantic document is built at
    all (spec §5), so the same failure cannot reach the run."""
    _open_fake_stores(monkeypatch)
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    def _boom(**kw):
        raise GroundTruthContractError("boom: no cell-eval2 distribution metadata")
    monkeypatch.setattr(gtcache, "producer_versions", _boom)

    out = tmp_path / "out"
    result = cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                         mem_budget=_BIG, outdir=str(out), write_de=True)
    assert result.per_pert.height > 0
    assert calls, "compute_de must still have run with the cache off"
    assert not (out / cellstream._GT_SIDECAR).exists()

    calls.clear()
    _open_fake_stores(monkeypatch)
    with pytest.raises(GroundTruthContractError, match="boom"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, ground_truth_cache=str(tmp_path / "gtcache"))
    assert calls == [], "the cache-on run must fail before any compute_de call"


def test_score_cellstream_piece_failure_leaves_cache_entry_intact(tmp_path, monkeypatch):
    """(g): a score_piece failure on the second piece propagates and leaves outdir untouched,
    but the cache entry the reference build already published survives and still validates --
    cache publication happens before, and independently of, per-piece output publication."""
    _open_fake_stores(monkeypatch)
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    # Force one perturbation per batch regardless of mem_budget, so there are >= 2 pieces to
    # fail the second of -- plan_pert_batches would otherwise happily pack both into one batch
    # under a generous budget.
    def _one_pert_per_batch(sizes, **kw):
        for pert, _n in sizes:
            yield [pert]

    monkeypatch.setattr(cellstream, "plan_pert_batches", _one_pert_per_batch)

    piece_ids: list[str] = []
    real_score_piece = partition_inmem.score_piece

    def _failing_score_piece(pred_piece, cache_dir, *, config, piece_id=None, **kw):
        piece_ids.append(piece_id)
        if len(piece_ids) == 2:
            raise RuntimeError("boom: piece scoring failed")
        return real_score_piece(pred_piece, cache_dir, config=config, piece_id=piece_id, **kw)

    monkeypatch.setattr(partition_inmem, "score_piece", _failing_score_piece)

    cache_root = tmp_path / "gtcache"
    cfg = _gt_cfg_de_only()
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="boom"):
        cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                    outdir=str(out), write_de=True,
                                    ground_truth_cache=str(cache_root))
    assert len(piece_ids) == 2
    assert not (out / "de_real.parquet").exists()
    assert not (out / "de_pred.parquet").exists()
    assert not (out / "ground_truth_de_cache.json").exists()

    # The cache entry published during the (successful) reference build survives and validates.
    found_entries = []
    for root, _dirs, files in os.walk(str(cache_root)):
        if "manifest.json" in files and "de_real.parquet" in files:
            found_entries.append(root)
    assert len(found_entries) == 1
    with open(os.path.join(found_entries[0], "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    cache_id = man["cache_id"]
    cache = GroundTruthDEGCache(str(cache_root), policy="readonly")
    entry = cache.entry_dir(cache_id)
    assert os.path.isdir(entry)
    from cell_eval2.ground_truth_cache import RealDESemantics

    request = GroundTruthDEGCacheRequest(
        fingerprint=GroundTruthFingerprint(document=man["ground_truth_fingerprint"],
                                           sha256=man["ground_truth_fingerprint_sha256"]),
        semantics=RealDESemantics(document=man["de_semantics"],
                                  sha256=man["de_semantics_sha256"]),
    )
    cache.validate_entry(entry, request)  # raises on any corruption


def test_score_cellstream_checksumless_archive_fails_before_any_compute(tmp_path, monkeypatch):
    """(h), first half: a checksum-less real archive under a cache policy raises
    GroundTruthFingerprintError before any compute_de call; the same store with policy off
    runs fine (no fingerprint attempted)."""
    _open_fake_stores(monkeypatch, real_payload=None)
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    cfg = _gt_cfg_de_only()
    with pytest.raises(GroundTruthFingerprintError):
        cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                    ground_truth_cache=str(cache_root))
    assert calls == []

    _open_fake_stores(monkeypatch, real_payload=None)
    fake_de2, calls2 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de2)
    cfg_off = _gt_cfg_de_only()  # ground_truth_cache=None -> off
    result = cellstream.score_cellstream("pred.path", "real.path", config=cfg_off,
                                         mem_budget=_BIG)
    assert result.per_pert.height > 0


def test_score_cellstream_readonly_miss_raises_with_cache_id(tmp_path, monkeypatch):
    """(h), second half: readonly against an empty root raises GroundTruthCacheMissError
    naming the cache id, with no compute."""
    _open_fake_stores(monkeypatch)
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    cache_root = tmp_path / "gtcache"
    os.makedirs(cache_root, exist_ok=True)
    cfg = _gt_cfg_de_only()
    with pytest.raises(GroundTruthCacheMissError) as excinfo:
        cellstream.score_cellstream("pred.path", "real.path", config=cfg, mem_budget=_BIG,
                                    ground_truth_cache=str(cache_root),
                                    ground_truth_cache_policy="readonly")
    assert calls == []
    # The cache id (64 lowercase hex chars) appears in the error message.
    import re

    assert re.search(r"[0-9a-f]{64}", str(excinfo.value))


# ── (k): the cache-off path, the arguments, the sixth staged file (spec §4-§6) ────────────


def _no_archive_io(path):
    raise AssertionError(f"an archive was opened ({path!r}) before the argument checks ran")


@pytest.mark.parametrize("root, policy, exc, match", [
    (None, "bogus", ValueError, "must be one of"),
    ("abs", "off", ValueError, "must be one of"),
    (None, "readonly", ValueError, "needs ground_truth_cache"),
    ("relative/root", "readwrite", GroundTruthContractError, "absolute path"),
], ids=["unknown-policy-cache-off", "off-is-not-a-policy", "readonly-without-root",
        "relative-root"])
def test_cache_arguments_are_checked_before_any_archive_io(tmp_path, monkeypatch, root, policy,
                                                           exc, match):
    """T2 / spec §4: both arguments are refused before any archive is opened, and nothing is
    created -- 'off' is spelled ground_truth_cache=None here, never a policy."""
    monkeypatch.chdir(tmp_path)   # where a relative root would land if it were ever created
    monkeypatch.setattr(cellstream, "open_cell_store", _no_archive_io)
    if root == "abs":
        root = str(tmp_path / "abs_root")
    with pytest.raises(exc, match=match):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, ground_truth_cache=root,
                                    ground_truth_cache_policy=policy)
    assert os.listdir(tmp_path) == []


def test_missing_readonly_root_fails_before_any_archive_io(tmp_path, monkeypatch):
    """T2: a readonly root that does not exist is refused by the store's constructor, which runs
    before the real archive is opened -- and readonly creates nothing."""
    monkeypatch.setattr(cellstream, "open_cell_store", _no_archive_io)
    absent = tmp_path / "absent"
    with pytest.raises(GroundTruthCacheFilesystemError, match="does not exist"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, ground_truth_cache=str(absent),
                                    ground_truth_cache_policy="readonly")
    assert not absent.exists()


def _previous_outputs(out):
    out.mkdir()
    names = [*cellstream._WRITE_DE_OUTPUTS, cellstream._GT_SIDECAR]
    for n in names:
        (out / n).write_text(f"previous {n}")
    return names


def test_a_failing_sidecar_write_publishes_nothing(tmp_path, monkeypatch):
    """T3 / spec §6.6: the sidecar is staged in the same _StagedOutputs as the five write_de
    files, so a failure while writing it leaves outdir exactly as it was -- the previous six
    files intact, no temp debris."""
    out = tmp_path / "out"
    names = _previous_outputs(out)
    _open_fake_stores(monkeypatch)
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    # json.dumps raises inside the sidecar's writer, after its temp file was opened.
    monkeypatch.setattr(cellstream, "_producer_dict", lambda: {"cell_eval2": object()})
    with pytest.raises(TypeError, match="not JSON serializable"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out), write_de=True,
                                    ground_truth_cache=str(tmp_path / "gtcache"))
    assert sorted(os.listdir(out)) == sorted(names)
    assert all((out / n).read_text() == f"previous {n}" for n in names)


def test_a_failed_publish_rolls_the_sidecar_back_too(tmp_path, monkeypatch):
    """T3: the sidecar publishes before the three score frames, and a failure on the LAST
    rename rolls every published file back -- the sidecar included -- restoring the previous
    outputs."""
    out = tmp_path / "out"
    names = _previous_outputs(out)
    _open_fake_stores(monkeypatch)
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)

    real_replace = os.replace
    renamed = []

    def _replace(src, dst):
        if ".tmp." in os.path.basename(str(src)) and os.path.dirname(str(dst)) == str(out):
            renamed.append(os.path.basename(str(dst)))
            if os.path.basename(str(dst)) == "overall.parquet":
                raise OSError("boom: the last rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _replace)
    with pytest.raises(OSError, match="boom"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out), write_de=True,
                                    ground_truth_cache=str(tmp_path / "gtcache"))
    assert renamed.index(cellstream._GT_SIDECAR) < renamed.index("overall.parquet")
    assert sorted(os.listdir(out)) == sorted(names)
    assert all((out / n).read_text() == f"previous {n}" for n in names)


def test_sidecar_destination_is_checked_only_while_the_cache_is_on(tmp_path, monkeypatch):
    """T5 / spec §4: with the cache on, a directory where the sidecar would land is refused
    before any archive I/O and before the store creates anything; with the cache off that path
    is none of the run's outputs, so it is left alone."""
    out = tmp_path / "out"
    (out / cellstream._GT_SIDECAR).mkdir(parents=True)
    monkeypatch.setattr(cellstream, "open_cell_store", _no_archive_io)
    with pytest.raises(ValueError, match="not a regular file"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out), write_de=True,
                                    ground_truth_cache=str(tmp_path / "gtcache"))
    assert not (tmp_path / "gtcache").exists()

    _open_fake_stores(monkeypatch)
    fake_de, _calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, outdir=str(out), write_de=True)
    assert (out / cellstream._GT_SIDECAR).is_dir()


def test_a_different_prediction_hits_the_same_entry(tmp_path, monkeypatch):
    """Review focus: the cache key excludes the prediction. A DIFFERENT prediction scored
    against the same ground truth is a hit on the same entry: identical de_real bytes, a
    different de_pred."""
    cache_root = tmp_path / "gtcache"
    gt = {"ground_truth_cache": str(cache_root)}
    outs = []
    for pred_seed in (1, 7):
        _open_fake_stores(monkeypatch, pred_seed=pred_seed)
        fake_de, _c = make_fake_compute_de()
        monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
        out = tmp_path / f"out{pred_seed}"
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out), write_de=True, **gt)
        outs.append(out)
    s1, s2 = (json.loads((o / cellstream._GT_SIDECAR).read_text()) for o in outs)
    assert (s1["status"], s2["status"]) == ("miss_computed", "hit")
    assert s1["cache_id"] == s2["cache_id"]
    assert (outs[0] / "de_real.parquet").read_bytes() == (outs[1] / "de_real.parquet").read_bytes()
    assert (outs[0] / "de_pred.parquet").read_bytes() != (outs[1] / "de_pred.parquet").read_bytes()


def test_a_corrupt_entry_fails_the_run_and_stays_in_place(tmp_path, monkeypatch):
    """Review focus / ruling 1: a committed entry whose bytes changed is fatal at the API --
    GroundTruthCacheCorruptError before anything is staged -- and the entry is neither removed
    nor overwritten (no self-healing)."""
    cache_root = tmp_path / "gtcache"
    gt = {"ground_truth_cache": str(cache_root)}
    _open_fake_stores(monkeypatch)
    fake_de, _c = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    out1 = tmp_path / "out1"
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, outdir=str(out1), write_de=True, **gt)
    cache_id = json.loads((out1 / cellstream._GT_SIDECAR).read_text())["cache_id"]
    table = os.path.join(str(cache_root), gtcache.entry_relpath(cache_id), "de_real.parquet")
    with open(table, "rb") as fh:
        damaged = bytearray(fh.read())
    damaged[len(damaged) // 2] ^= 0xFF
    with open(table, "wb") as fh:
        fh.write(damaged)

    _open_fake_stores(monkeypatch)
    fake_de2, _c2 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de2)
    out2 = tmp_path / "out2"
    with pytest.raises(GroundTruthCacheCorruptError, match=cache_id):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out2), write_de=True, **gt)
    assert not out2.exists()
    with open(table, "rb") as fh:
        assert fh.read() == bytes(damaged)


def test_a_readonly_consumer_reads_a_populated_root_without_writing(tmp_path, monkeypatch):
    """Review focus: a readonly consumer of a populated root gets a hit, reports policy
    'readonly', and adds nothing under the root -- no lock file, no staging directory."""
    cache_root = tmp_path / "gtcache"
    _open_fake_stores(monkeypatch)
    fake_de, _c = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, ground_truth_cache=str(cache_root))

    def _tree():
        return sorted(os.path.relpath(os.path.join(r, n), str(cache_root))
                      for r, dirs, files in os.walk(str(cache_root)) for n in dirs + files)

    before = _tree()
    _open_fake_stores(monkeypatch)
    fake_de2, _c2 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de2)
    out = tmp_path / "out"
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, outdir=str(out), write_de=True,
                                ground_truth_cache=str(cache_root),
                                ground_truth_cache_policy="readonly")
    sidecar = json.loads((out / cellstream._GT_SIDECAR).read_text())
    assert (sidecar["status"], sidecar["policy"]) == ("hit", "readonly")
    assert _tree() == before


def test_the_root_may_be_a_path_object_or_start_with_a_tilde(tmp_path, monkeypatch):
    """Review focus: ground_truth_cache accepts os.PathLike, and expands a leading '~'."""
    versioned = os.path.join("ground-truth-deg", f"v{gtcache.CACHE_FORMAT_VERSION}")
    _open_fake_stores(monkeypatch)
    fake_de, _c = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, ground_truth_cache=tmp_path / "as_path")
    assert (tmp_path / "as_path" / versioned).is_dir()

    monkeypatch.setenv("HOME", str(tmp_path))
    _open_fake_stores(monkeypatch)
    fake_de2, _c2 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de2)
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, ground_truth_cache="~/as_tilde")
    assert (tmp_path / "as_tilde" / versioned).is_dir()


def test_the_root_may_live_inside_outdir(tmp_path, monkeypatch):
    """Review focus: a root inside outdir coexists with the six published files, and a second
    run into the same outdir republishes over the first run's files and hits the entry."""
    out = tmp_path / "out"
    for expected in ("miss_computed", "hit"):
        _open_fake_stores(monkeypatch)
        fake_de, _c = make_fake_compute_de()
        monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out), write_de=True,
                                    ground_truth_cache=str(out))
        assert json.loads((out / cellstream._GT_SIDECAR).read_text())["status"] == expected
        assert sorted(os.listdir(out)) == sorted(
            [*cellstream._WRITE_DE_OUTPUTS, cellstream._GT_SIDECAR, "ground-truth-deg"])


@pytest.mark.parametrize("rel, write_de, via_link", [
    ("de_real.parquet", True, False),
    ("ground_truth_de_cache.json", True, False),
    ("de_pred.parquet/nested", True, False),
    ("overall.parquet", False, False),
    ("per_pert.parquet", True, True),
], ids=["de_real", "sidecar", "inside-de_pred", "frames-only-overall", "through-symlinked-outdir"])
def test_a_root_aliasing_an_output_is_refused_before_any_archive_io(tmp_path, monkeypatch, rel,
                                                                    write_de, via_link):
    """T2 / spec §4: a root that IS, or lies inside, one of the six files a cache-on run
    publishes into outdir is refused before any archive I/O and before the store creates
    anything, whatever write_de is -- otherwise readwrite would make a directory there and the
    clash would surface only at publication, after the whole scoring run. A symlinked outdir
    cannot hide the alias."""
    real_out = tmp_path / "out"
    real_out.mkdir()
    out = real_out
    if via_link:
        out = tmp_path / "out_link"
        out.symlink_to(real_out)
    monkeypatch.setattr(cellstream, "open_cell_store", _no_archive_io)
    with pytest.raises(ValueError, match="lies inside"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir=str(out), write_de=write_de,
                                    ground_truth_cache=str(real_out / rel))
    assert os.listdir(real_out) == []


def test_an_empty_outdir_is_still_checked_for_an_aliasing_root(tmp_path, monkeypatch):
    """The publication path treats any outdir that is not None as a destination, "" included
    (its files would land relative to the working directory). The alias check uses the same
    condition, so a root at one of those files is refused before archive I/O (Copilot, PR #399)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cellstream, "open_cell_store", _no_archive_io)
    with pytest.raises(ValueError, match="lies inside"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, outdir="",
                                    ground_truth_cache=str(tmp_path / "overall.parquet"))
    assert os.listdir(tmp_path) == []


def test_a_root_on_another_drive_is_not_an_alias(tmp_path, monkeypatch):
    """Windows: os.path.commonpath raises ValueError for paths on different drives (a readonly
    cache on D:, outputs on C:). Such paths cannot overlap, so the alias check passes instead of
    leaking commonpath's ValueError (codex, checkpoint 2)."""
    import ntpath

    with pytest.raises(ValueError, match="drive"):
        ntpath.commonpath([r"D:\gt-cache", r"C:\results\de_real.parquet"])

    def _different_drives(paths):
        raise ValueError("Paths don't have the same drive")

    with monkeypatch.context() as m:
        m.setattr(os.path, "commonpath", _different_drives)
        cellstream._check_root_outside_outputs(str(tmp_path / "gtcache"), str(tmp_path / "out"))


def test_a_real_archive_replaced_mid_run_is_refused(tmp_path, monkeypatch):
    """The reference builder streams from CellBatchSource's OWN open of the real archive, not
    the handle fingerprinted first. An archive replaced between the two opens must neither key a
    new entry with the first archive's identity nor hit an old one: the second handle is
    re-fingerprinted, and a mismatch fails before any DE runs (Copilot, PR #399)."""
    first = _GTFakeCellStore(payload_sha256=_PAYLOAD, seed=0, id_prefix="r")
    replaced = _GTFakeCellStore(payload_sha256="ef" * 32, seed=0, id_prefix="r")
    pred = _GTFakeCellStore(payload_sha256=None, seed=1, id_prefix="p")
    real_opens = []

    def _open(path):
        if path == "pred.path":
            return pred
        real_opens.append(path)
        return first if len(real_opens) == 1 else replaced

    monkeypatch.setattr(cellstream, "open_cell_store", _open)
    fake_de, calls = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    cache_root = tmp_path / "gtcache"
    with pytest.raises(GroundTruthFingerprintError, match="changed while this run read it"):
        cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                    mem_budget=_BIG, ground_truth_cache=str(cache_root))
    assert len(real_opens) == 2
    assert calls == []
    assert [f for _d, _s, fs in os.walk(cache_root) for f in fs if f == "manifest.json"] == []


def test_published_de_real_is_the_cache_entry_byte_for_byte(tmp_path, monkeypatch):
    """With the cache on, de_real.parquet is a verified BYTE COPY of the entry, never a
    re-encoding -- so the sidecar's de_real_sha256 names the published file even when the entry
    came from another writer (another pyarrow version writes another footer). Simulated by
    re-encoding the entry with dictionary encoding off, which a rewrite through de_export would
    not reproduce."""
    import pyarrow.parquet as pq

    cache_root = tmp_path / "gtcache"
    gt = {"ground_truth_cache": str(cache_root)}
    _open_fake_stores(monkeypatch)
    fake_de, _c = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de)
    out1 = tmp_path / "out1"
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, outdir=str(out1), write_de=True, **gt)
    cache_id = json.loads((out1 / cellstream._GT_SIDECAR).read_text())["cache_id"]
    entry = os.path.join(str(cache_root), gtcache.entry_relpath(cache_id))
    table = os.path.join(entry, "de_real.parquet")
    manifest_path = os.path.join(entry, "manifest.json")
    pq.write_table(pq.read_table(table), table, compression="zstd", compression_level=3,
                   write_statistics=True, use_dictionary=False)
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["artifact"]["sha256"] = sha256_file(table)
    manifest["artifact"]["size_bytes"] = os.path.getsize(table)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    with open(table, "rb") as fh:
        entry_bytes = fh.read()

    _open_fake_stores(monkeypatch)
    fake_de2, _c2 = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de2)
    out2 = tmp_path / "out2"
    cellstream.score_cellstream("pred.path", "real.path", config=_gt_cfg_de_only(),
                                mem_budget=_BIG, outdir=str(out2), write_de=True, **gt)
    sidecar = json.loads((out2 / cellstream._GT_SIDECAR).read_text())
    assert sidecar["status"] == "hit"
    assert (out2 / "de_real.parquet").read_bytes() == entry_bytes
    assert sha256_file(str(out2 / "de_real.parquet")) == sidecar["de_real_sha256"]


def test_copy_verified_refuses_bytes_that_changed(tmp_path):
    """The byte copy is checked against the digest the lookup validated: bytes that changed in
    between raise GroundTruthCacheCorruptError, and the caller's staging then publishes nothing."""
    src = tmp_path / "entry.parquet"
    src.write_bytes(b"entry bytes")
    ok = tmp_path / "ok"
    cellstream._copy_verified(str(src), str(ok), sha256_file(str(src)))
    assert ok.read_bytes() == b"entry bytes"
    with pytest.raises(GroundTruthCacheCorruptError, match="changed after this run validated it"):
        cellstream._copy_verified(str(src), str(tmp_path / "bad"), "0" * 64)


# ── (j): cellstream-gated tier on the frozen fixture archives ───────────────────────────────

pytestmark_cellstream = pytest.mark.skipif(
    not os.path.isfile(FIXTURE_REAL), reason="ground_truth_cache fixture archives not present")


@pytestmark_cellstream
def test_fixture_archives_off_cold_warm_agree(tmp_path, monkeypatch):
    """Repeats (a)/(b) against the real frozen fixture archives through score_cellstream,
    exercising the REAL CellBatchSource.pert_universe, real fingerprinting (compared against
    the fixture's own recorded ground_truth_fingerprint_sha256) and real _control_label --
    nothing about the store is faked here, only compute_de."""
    pytest.importorskip("cellstream")

    cfg = _gt_cfg_de_only(control="non-targeting")

    fake_de_off, _c = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de_off)
    out_off = tmp_path / "off"
    cellstream.score_cellstream(FIXTURE_PRED, FIXTURE_REAL, config=cfg, mem_budget=_BIG,
                                outdir=str(out_off), write_de=True)
    # Cache off: exactly main's five write_de outputs, no sidecar (ruling 6a).
    assert sorted(os.listdir(out_off)) == sorted(cellstream._WRITE_DE_OUTPUTS)

    with open(FIXTURE_MANIFEST, encoding="utf-8") as fh:
        fixture_manifest = json.load(fh)
    real_entry = next(a for a in fixture_manifest["archives"] if a["file"] == "real.shad")
    expected_fp_sha = real_entry["ground_truth_fingerprint_sha256"]

    cache_root = tmp_path / "gtcache"
    gt = {"ground_truth_cache": str(cache_root)}

    fake_de_cold, calls_cold = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de_cold)
    out_cold = tmp_path / "cold"
    cellstream.score_cellstream(FIXTURE_PRED, FIXTURE_REAL, config=cfg, mem_budget=_BIG,
                                outdir=str(out_cold), write_de=True, **gt)
    sidecar_cold = json.loads((out_cold / "ground_truth_de_cache.json").read_text())
    assert sidecar_cold["status"] == "miss_computed"
    assert sidecar_cold["ground_truth_fingerprint_sha256"] == expected_fp_sha
    # T4 on the real archives: the published table is the entry's bytes.
    assert sha256_file(str(out_cold / "de_real.parquet")) == sidecar_cold["de_real_sha256"]

    fake_de_warm, calls_warm = make_fake_compute_de()
    monkeypatch.setattr(partition_inmem, "compute_de", fake_de_warm)
    out_warm = tmp_path / "warm"
    cellstream.score_cellstream(FIXTURE_PRED, FIXTURE_REAL, config=cfg, mem_budget=_BIG,
                                outdir=str(out_warm), write_de=True, **gt)
    sidecar_warm = json.loads((out_warm / "ground_truth_de_cache.json").read_text())
    assert sidecar_warm["status"] == "hit"
    assert sidecar_warm["cache_id"] == sidecar_cold["cache_id"]

    b_off = (out_off / "de_real.parquet").read_bytes()
    b_cold = (out_cold / "de_real.parquet").read_bytes()
    b_warm = (out_warm / "de_real.parquet").read_bytes()
    assert b_off == b_cold == b_warm

    # Fewer total compute_de calls on the warm run than on the cold run: the real-side pass
    # that the cold run had to perform is skipped entirely on the warm hit.
    assert len(calls_warm) < len(calls_cold)


@pytestmark_cellstream
def test_fixture_real_archive_pert_universe(tmp_path):
    """The real CellBatchSource.pert_universe() implementation against the frozen fixture: the
    six non-control perturbation labels, sorted, without gathering a single row."""
    pytest.importorskip("cellstream")
    from cell_eval2.cellstream import CellBatchSource

    src = CellBatchSource(FIXTURE_REAL, pert_col="perturbation", control="non-targeting")
    try:
        universe = src.pert_universe()
    finally:
        src.close()
    assert universe == sorted(f"TG0{i}" for i in range(1, 7))
