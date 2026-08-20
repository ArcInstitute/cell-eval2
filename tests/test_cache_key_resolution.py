"""Cache-key correctness: resolved device/backend must enter the keys, and a failed real
load must not leak the pred handle. Regression tests for ultrareview F2.1/F2.2/F2.3."""
import json

import pytest

from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.cache import config_hash
from cell_eval2.gpu import resolve_device


def test_pseudobulk_cache_params_include_resolved_device(synthetic_pair, tmp_path):
    # F2.1: the in-memory pseudobulk cache picks fp32 (GPU) vs fp64 (CPU) means by the resolved
    # device, so the cache key MUST include it -- else a cpu run can be served fp32 GPU means.
    pred, real = synthetic_pair
    cache = tmp_path / "pred_cache"
    compute_metrics(pred, real, metrics=["mae"], input_type="lognorm", cache_pred=str(cache))
    manifest = json.loads((cache / "manifest.json").read_text())
    pb = {k: v for k, v in manifest["artifacts"].items() if k.startswith("pseudobulk_")}
    assert pb, f"expected a pseudobulk_* artifact; got {list(manifest['artifacts'])}"
    for k, entry in pb.items():
        assert entry["params"].get("device") == resolve_device("auto"), \
            f"{k} params missing resolved device: {entry['params']}"


def test_de_table_cache_params_use_resolved_backend(synthetic_counts_pair, tmp_path):
    # F2.2: the DE-table cache must key on the RESOLVED backend (gpudge/pdex/scanpy), never the
    # literal "auto", else a GPU node (gpudge) and a CPU node (pdex) collide on a shared cache
    # despite their DE numbers differing (~1e-5).
    pred, real = synthetic_counts_pair
    cache = tmp_path / "pred_cache"
    compute_metrics(pred, real, metrics=["de_wilcoxon_roc_auc"], input_type="counts",
                    cache_pred=str(cache))  # de.backend defaults to "auto"
    manifest = json.loads((cache / "manifest.json").read_text())
    de = {k: v for k, v in manifest["artifacts"].items() if k.startswith("de_") and k.endswith("_table")}
    assert de, f"expected a de_*_table artifact; got {list(manifest['artifacts'])}"
    for k, entry in de.items():
        assert entry["params"]["backend"] != "auto", f"{k} stored unresolved backend: {entry['params']}"
        assert entry["params"]["backend"] in ("gpudge", "pdex", "scanpy")


def test_result_config_digest_resolves_device(monkeypatch):
    # F2.2: the result cache's config digest must reflect the RESOLVED device (and backend),
    # so a GPU node (auto->cuda, fp32) and a CPU node (auto->cpu, fp64) never collide on the
    # top-level result cache.
    import cell_eval2.run as run

    cfg = EvalConfig(metrics=["mae"], device="auto")
    monkeypatch.setattr(run, "resolve_device", lambda d: "cpu")
    d_cpu = run._result_config_digest(cfg, de_backend_used=False, comparator="lognorm")
    monkeypatch.setattr(run, "resolve_device", lambda d: "cuda")
    d_gpu = run._result_config_digest(cfg, de_backend_used=False, comparator="lognorm")
    assert d_cpu != d_gpu, "result digest ignores the resolved device (cpu vs cuda collide)"
    # and it must differ from the naive unresolved hash -> proves resolution actually happened
    assert d_cpu != config_hash(cfg.to_dict())


def test_result_config_digest_skips_de_backend_when_engine_not_used(monkeypatch):
    # F2.2/Copilot #114: when no DE engine runs (de_backend_used=False -- a non-DE run, OR a DE run
    # with both tables supplied) the digest must NOT resolve the DE backend, so result caching never
    # introduces a hard dependency on a DE backend being installed. Simulate a backend-less env by
    # making _resolve_backend raise.
    import cell_eval2.de_compute as de_compute
    import cell_eval2.run as run

    cfg = EvalConfig(metrics=["mae"], device="auto")

    def _boom(_backend):
        raise RuntimeError("no DE backend available")

    monkeypatch.setattr(de_compute, "_resolve_backend", _boom)
    digest = run._result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm",
    )  # must not touch the DE backend
    assert isinstance(digest, str) and digest
    with pytest.raises(RuntimeError):  # when the engine runs the backend IS resolved (only then)
        run._result_config_digest(cfg, de_backend_used=True, comparator="lognorm")


def test_cache_keys_use_explicit_device_verbatim(monkeypatch):
    # Copilot #114 R2: only "auto" is machine-dependent (cuda vs cpu). An explicit device must be
    # used verbatim in cache keys, never passed through resolve_device -- which RAISES for
    # device="cuda" in a no-cupy install and would make result caching break an otherwise-fine run.
    import cell_eval2.run as run

    def _boom(_d):
        raise RuntimeError("cupy unavailable")

    monkeypatch.setattr(run, "resolve_device", _boom)
    cfg = EvalConfig(metrics=["mae"], device="cuda")
    digest = run._result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm",
    )  # explicit device -> must not resolve
    assert isinstance(digest, str) and digest


def test_pred_handle_closed_when_real_load_fails(synthetic_pair, tmp_path, monkeypatch):
    # F2.3: pred_ad is opened (backed) before real_ad; if the real load raises, the pred handle
    # must still be closed (finally must cover it), or long-running/batch callers leak fds.
    import cell_eval2.run as run

    pred, _real = synthetic_pair
    pred_path = tmp_path / "pred.h5ad"
    pred.write_h5ad(pred_path)

    closed_sources = []
    orig = run._close_backed
    monkeypatch.setattr(run, "_close_backed",
                        lambda ad, src: (closed_sources.append(src), orig(ad, src))[1])

    with pytest.raises(Exception):
        compute_metrics(str(pred_path), str(tmp_path / "missing_real.h5ad"),
                        metrics=["mae"], input_type="lognorm")
    assert str(pred_path) in closed_sources, \
        "pred handle was not closed when the real input failed to load"


def test_result_cache_omits_de_backend_when_both_de_tables_supplied(synthetic_pair, tmp_path, monkeypatch):
    # Copilot #114 R3: when a DE metric is requested but BOTH de tables are supplied, no DE engine
    # runs (compute_de is skipped for both sides), so the result-cache config digest must NOT
    # resolve the DE backend. Otherwise a minimal install (no gpudge/pdex/scanpy) raises purely
    # because cache_pred is enabled, even though the run computes no DE. This is the has_de=True but
    # de_backend_used=False case: the digest gate is has_de AND (a side is computed), not just has_de.
    import polars as pl

    import cell_eval2.de_compute as de_compute

    pred, real = synthetic_pair
    de = pl.DataFrame([
        {"target": t, "feature": f, "log2_fold_change": lfc, "p_adj": p}
        for t in ("GENE1", "GENE2", "GENE3")
        for f, lfc, p in (("g0", 3.0, 0.001), ("g1", -2.0, 0.02), ("g2", 0.05, 0.8))
    ])

    def _boom(_backend):
        raise RuntimeError("no DE backend available")

    monkeypatch.setattr(de_compute, "_resolve_backend", _boom)  # cfg.de.backend defaults to "auto"
    # Both tables supplied -> compute_de is never called AND the digest must not resolve the backend.
    df = compute_metrics(pred, real, metrics=["overlap_at_N"], control="non-targeting",
                         pert_col="target", input_type="lognorm",
                         de_pred=de, de_real=de, cache_pred=str(tmp_path / "p"))
    assert df.height > 0, "run with both DE tables supplied must complete without a DE backend"


def test_thread_counts_excluded_from_config_hash():
    """Thread counts cannot change numerics (cellstream asserts parallel decode is byte-identical
    to serial), so they must not invalidate a cache. gather_threads joins num_threads in the
    skip set -- #149."""
    from dataclasses import replace

    base = EvalConfig(metrics=["mae"])
    h = config_hash(base.to_dict())
    assert config_hash(replace(base, gather_threads=8).to_dict()) == h
    assert config_hash(replace(base, num_threads=8).to_dict()) == h


def test_validate_input_is_in_the_pseudobulk_and_de_table_cache_params(monkeypatch,
                                                                       tmp_path):
    """#161. Its neighbours allow_fractional_counts / autodetect_input_type /
    max_counts_per_cell were added for exactly this reason; the master switch was missed.

    Driven through `compute_metrics`, NOT `precompute_cache(de=True)`: that API's `de=` is a
    SUPPLIED DE table (a path or frame), so `de=True` would be passed straight into
    `prep_de_side` -- and on the None default it never writes a DE artifact at all, so the
    test would assert nothing about the DE dict.

    Asserts the KEY is present in each captured params dict -- not merely that two runs
    produce different digests, which a different bug could also satisfy. The `_rank` key is
    excluded: it is documented as always-strict and carries no validation knobs."""
    from cell_eval2 import EvalConfig
    from cell_eval2.cache import CacheStore
    from cell_eval2.run import compute_metrics

    seen = {}
    real_put = CacheStore.put

    def spy(self, key, value, *, fingerprint, params, kind):
        seen[key] = dict(params)
        return real_put(self, key, value, fingerprint=fingerprint, params=params, kind=kind)

    monkeypatch.setattr(CacheStore, "put", spy)
    from _helpers import _counts_adata_fp64
    adata = _counts_adata_fp64(seed=0, per_group=6, g=8)
    # input_type="counts", NOT "lognorm": `_counts_adata_fp64` builds integer Poisson counts
    # (tests/_helpers.py:15), and with validate_input=True a declared "lognorm" is REJECTED
    # at run.py:988 -- the test would die before reaching a single cache assertion.
    # The DE backend is pinned so this measures the cache PARAMS rather than which engine
    # the host happens to have.
    cfg = EvalConfig(metrics=["expr_mae", "de_wilcoxon_overlap"], pert_col="target",
                     input_type="counts", validate_input=True, device="cpu",
                     de={"backend": "pdex"},
                     cache_real=str(tmp_path / "cr"), cache_pred=str(tmp_path / "cp"))
    compute_metrics(adata, adata, config=cfg)

    assert seen, "no cache artifacts were written; the spy never fired"
    targets = [k for k in seen if k.startswith("pseudobulk") or k.endswith("_table")]
    assert targets, f"neither a pseudobulk nor a DE-table artifact was written: {sorted(seen)}"
    missing = [k for k in targets if "validate_input" not in seen[k]]
    assert not missing, (
        f"{len(missing)} of {len(targets)} value-affecting cache param dicts omit "
        f"'validate_input': {[(k, sorted(seen[k])) for k in missing]}"
    )
