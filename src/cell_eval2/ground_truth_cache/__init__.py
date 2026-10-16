"""The ground-truth DEG cache (see docs/ground_truth_cache.md) -- package entry point.

Historically a single ~1400-line module (``cell_eval2/ground_truth_cache.py``); split (PR
#389 review) into:

* ``identity.py`` -- canonical JSON/digests, the fingerprint + semantics documents,
  cache-id construction, the identity dataclasses (``GroundTruthFingerprint``,
  ``RealDESemantics``, ``GroundTruthDEGCacheRequest``, ``GroundTruthDEGCacheResult``),
  ``fingerprint_ground_truth_cell_archive``, ``resolve_de_precision``,
  ``producer_versions``, ``build_real_de_semantics``. Stdlib + numpy only -- no
  ``polars``, ``pyarrow``, or ``fcntl`` at import time, so fingerprinting and ordinary
  scoring stay importable on a platform without POSIX advisory locks.
* ``integrity.py`` -- entry-file validation (``inspect_entry``, ``_validate_entry_files``,
  the ``_check_*`` functions), shared by the ``cache inspect`` CLI and
  ``GroundTruthDEGCache.validate_entry``. ``polars`` at module scope (a base dependency of
  ``cell_eval2``); ``pyarrow`` only via ``de_export``'s own lazy imports.
* ``store.py`` -- ``GroundTruthDEGCache`` (flock-guarded lookup/populate, staging, atomic
  publish) and ``doctor``. ``fcntl`` is imported LAZILY, only inside the two code paths
  that actually take a lock (see ``store._flock_module``) -- so importing this package
  never requires POSIX advisory locks, even on a platform (e.g. Windows) that doesn't have
  them; only an actual write/lock attempt does.

This module re-exports every name that was importable from the single-file
``cell_eval2.ground_truth_cache`` module before the split (including several
private-but-used-by-tests names), so ``from cell_eval2.ground_truth_cache import X`` and
``cell_eval2.ground_truth_cache.X`` (the ``gtcache.X`` spelling the test suite and
``cellstream.py`` use) both keep working unchanged. Two supported monkeypatch points live on
the ``store`` SUBMODULE specifically, not on this package namespace, because
``store.get_or_compute``/``store.doctor`` resolve them by bare module-level name at call
time: ``cell_eval2.ground_truth_cache.store._fault_hook`` and
``cell_eval2.ground_truth_cache.store._flock_module`` (see their docstrings in ``store.py``).
"""
from __future__ import annotations

from ..de_export import (
    DE_PARQUET_COMPRESSION_LEVEL,
    DE_PARQUET_ROW_GROUP_ROWS,
    read_de_parquet_metadata,
    sha256_file,
    write_de_parquet,
)
from .identity import (
    ARCHIVE_FORMAT,
    CACHE_FORMAT,
    CACHE_FORMAT_VERSION,
    CACHE_KEY_ALGORITHM,
    CACHE_POLICIES,
    CACHE_STATUSES,
    DOMAIN_SEQUENCE,
    ENTRY_MANIFEST_NAME,
    ENTRY_TABLE_NAME,
    FINGERPRINT_ALGORITHM,
    FINGERPRINT_DOCUMENT_KEYS_V1,
    REAL_DE_SEMANTICS_VERSION,
    SEMANTIC_DOCUMENT_KEYS_V1,
    STALE_STAGING_AGE_S,
    GroundTruthCacheCorruptError,
    GroundTruthCacheError,
    GroundTruthCacheFilesystemError,
    GroundTruthCacheMissError,
    GroundTruthContractError,
    GroundTruthDEGCacheRequest,
    GroundTruthDEGCacheResult,
    GroundTruthFingerprint,
    GroundTruthFingerprintError,
    RealDESemantics,
    build_real_de_semantics,
    cache_id,
    cache_key_document,
    canonical_json_bytes,
    document_sha256,
    entry_relpath,
    fingerprint_ground_truth_cell_archive,
    producer_versions,
    resolve_de_precision,
    sequence_sha256,
    validate_fingerprint_document,
    validate_semantic_document,
)
from .integrity import (
    _check_perturbation_universe,
    _check_target_feature_content,
    _collect_streaming,
    _validate_entry_files,
    inspect_entry,
)
from . import store
from .store import (
    GroundTruthDEGCache,
    _atomic_write_json_line,
    _fault_hook,
    _fsync_dir,
    _fsync_file,
    _producer_versions_best_effort,
    doctor,
)

__all__ = [
    "ARCHIVE_FORMAT",
    "CACHE_FORMAT",
    "CACHE_FORMAT_VERSION",
    "CACHE_KEY_ALGORITHM",
    "CACHE_POLICIES",
    "CACHE_STATUSES",
    "DE_PARQUET_COMPRESSION_LEVEL",
    "DE_PARQUET_ROW_GROUP_ROWS",
    "DOMAIN_SEQUENCE",
    "ENTRY_MANIFEST_NAME",
    "ENTRY_TABLE_NAME",
    "FINGERPRINT_ALGORITHM",
    "FINGERPRINT_DOCUMENT_KEYS_V1",
    "REAL_DE_SEMANTICS_VERSION",
    "SEMANTIC_DOCUMENT_KEYS_V1",
    "STALE_STAGING_AGE_S",
    "GroundTruthCacheCorruptError",
    "GroundTruthCacheError",
    "GroundTruthCacheFilesystemError",
    "GroundTruthCacheMissError",
    "GroundTruthContractError",
    "GroundTruthDEGCache",
    "GroundTruthDEGCacheRequest",
    "GroundTruthDEGCacheResult",
    "GroundTruthFingerprint",
    "GroundTruthFingerprintError",
    "RealDESemantics",
    "_atomic_write_json_line",
    "_check_perturbation_universe",
    "_check_target_feature_content",
    "_collect_streaming",
    "_fault_hook",
    "_fsync_dir",
    "_fsync_file",
    "_producer_versions_best_effort",
    "_validate_entry_files",
    "build_real_de_semantics",
    "cache_id",
    "cache_key_document",
    "canonical_json_bytes",
    "doctor",
    "document_sha256",
    "entry_relpath",
    "fingerprint_ground_truth_cell_archive",
    "inspect_entry",
    "producer_versions",
    "read_de_parquet_metadata",
    "resolve_de_precision",
    "sequence_sha256",
    "sha256_file",
    "store",
    "validate_fingerprint_document",
    "validate_semantic_document",
    "write_de_parquet",
]
