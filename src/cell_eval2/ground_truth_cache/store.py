"""Storage for the ground-truth DEG cache (see docs/ground_truth_cache.md).

``GroundTruthDEGCache``: flock-guarded, content-addressed store for ``de_real.parquet``
cache entries, built against the identity contract in ``identity.py`` and the checks in
``integrity.py``. An entry lives at ``root/entry_relpath(cache_id)`` (``ground-truth-deg/
v{V}/<2-hex>/<cache_id>/{manifest.json, de_real.parquet}``). Every write is staged in a
sibling ``.tmp-<cache_id>-<pid>-<uuid>`` directory, validated in place, then published with
one ``os.rename`` -- so any concurrent reader only ever sees either nothing or a fully valid,
already-validated-by-its-writer entry.

**This module owns ``root`` and ``policy``.** ``GroundTruthDEGCache`` stores both; every
read/write decision (whether ``lookup``/``get_or_compute`` may call ``compute()``, what
``GroundTruthDEGCacheResult.policy`` reports) is made from ``self.policy`` -- never from
anything a ``GroundTruthDEGCacheRequest`` carries (it no longer carries root/policy at all;
see ``identity.py``). A ``readonly`` instance can never write, lock, or create a directory,
regardless of what any request "asks for".

**``fcntl`` is imported LAZILY**, only inside the two code paths that actually take a lock
(``get_or_compute``'s writer path, and ``doctor``'s lock-exclusion probe), via
``_flock_module()``. On a platform without POSIX advisory locks (e.g. Windows),
``_flock_module()`` raises ``GroundTruthCacheFilesystemError`` naming ``fcntl`` explicitly,
instead of failing at import time -- so importing this module (and therefore
``cell_eval2`` as a whole) never requires POSIX locking, and a ``readonly`` ``lookup()`` of
an already-published entry works with no lock at all. ``_flock_module`` is called by bare
module-level name (not ``self._flock_module()``), the same pattern ``_fault_hook`` uses, so a
test can monkeypatch ``cell_eval2.ground_truth_cache.store._flock_module`` (this module) and
have it take effect everywhere this module calls it.
"""
from __future__ import annotations

import errno
import importlib.metadata as _importlib_metadata
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import polars as pl

from ..de_export import (
    DE_PARQUET_COMPRESSION_LEVEL,
    DE_PARQUET_ROW_GROUP_ROWS,
    read_de_parquet_metadata,
    write_de_parquet,
)
from .identity import (
    CACHE_FORMAT,
    CACHE_FORMAT_VERSION,
    CACHE_POLICIES,
    ENTRY_MANIFEST_NAME,
    ENTRY_TABLE_NAME,
    STALE_STAGING_AGE_S,
    GroundTruthCacheCorruptError,
    GroundTruthCacheFilesystemError,
    GroundTruthCacheMissError,
    GroundTruthContractError,
    GroundTruthDEGCacheRequest,
    GroundTruthDEGCacheResult,
    canonical_json_bytes,
    entry_relpath,
    sequence_sha256,
)
from .integrity import _is_within, _validate_entry_files

logger = logging.getLogger(__name__)


def _flock_module():
    """Import ``fcntl`` lazily. POSIX-only -- an ``ImportError`` (Windows, or any platform
    without it) becomes a targeted ``GroundTruthCacheFilesystemError`` naming ``fcntl``,
    raised only when a caller actually tries to lock, not at module import time."""
    try:
        import fcntl
    except ImportError as e:
        raise GroundTruthCacheFilesystemError(
            "POSIX advisory locks (fcntl) are unavailable on this platform; the "
            "ground-truth cache cannot write safely here -- turn the cache off "
            "(ground_truth_cache=None), or read an existing entry with the 'readonly' "
            "policy, which never locks."
        ) from e
    return fcntl


_DOCTOR_LOCK_CHILD_SRC = (
    "import fcntl, sys\n"
    "fh = open(sys.argv[1], 'a')\n"
    "fcntl.flock(fh.fileno(), fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "sys.stdin.readline()\n"
)

#: How long ``doctor`` waits for the lock-holder child to report ``locked``. A blocking
#: ``flock`` that never returns -- a broken mount, what ``doctor`` diagnoses -- must not hang it.
_DOCTOR_READY_TIMEOUT_S = 60.0


def doctor(root: str) -> dict:
    """Self-check a cache root's filesystem primitives -- flock exclusion, the parquet
    zstd round-trip, and a same-filesystem atomic directory rename -- WITHOUT computing any
    DE. Runs entirely inside a scratch directory under
    ``root/ground-truth-deg/v{CACHE_FORMAT_VERSION}/.doctor-<pid>-<uuid>`` (created if
    missing), removed again before returning.

    Never raises for a failed check: every probe is recorded in the returned report
    (``{"root", "checks": [{"name", "ok", "detail"}, ...], "ok": all(...)}``) so a caller
    (the ``cell-eval2 cache doctor`` CLI) can print it and choose an exit code. A missing
    ``fcntl`` (see ``_flock_module``) is recorded the same way -- as a failed
    ``flock_probe`` check, not a raised exception.
    """
    root_expanded = os.path.expanduser(root)
    # Filesystem work uses the canonical root, as GroundTruthDEGCache does: makedirs on a raw
    # ".../missing/../cache" would create "missing" on the way. The report keeps the input.
    root_real = os.path.realpath(root_expanded)
    versioned = os.path.join(root_real, "ground-truth-deg", f"v{CACHE_FORMAT_VERSION}")
    doctor_dir = os.path.join(versioned, f".doctor-{os.getpid()}-{uuid4().hex}")
    checks: list[dict] = []

    def _record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    # The store's own containment rule, before anything is created: a symlinked namespace
    # component that leaves the root is one GroundTruthDEGCache refuses, so doctor must too.
    versioned_real = os.path.realpath(versioned)
    if not _is_within(versioned_real, root_real):
        _record("mkdir", False,
                f"{versioned!r} resolves to {versioned_real!r}, outside the cache root "
                f"{root_real!r} (a symlinked component); nothing was created")
        return {"root": root_expanded, "checks": checks, "ok": False}

    try:
        os.makedirs(doctor_dir, exist_ok=False)
        _record("mkdir", True, doctor_dir)
    except OSError as e:
        _record("mkdir", False, str(e))
        return {"root": root_expanded, "checks": checks, "ok": False}

    lock_path = os.path.join(doctor_dir, "doctor.lock")
    child: subprocess.Popen | None = None
    try:
        fcntl = _flock_module()
        child = subprocess.Popen(
            [sys.executable, "-c", _DOCTOR_LOCK_CHILD_SRC, lock_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        # Bounded: at the deadline the watchdog kills the child, which ends the read at EOF.
        watchdog = threading.Timer(_DOCTOR_READY_TIMEOUT_S, child.kill)
        watchdog.start()
        try:
            ready = child.stdout.readline()
        finally:
            watchdog.cancel()
        if ready.strip() != "locked":
            raise RuntimeError(
                f"doctor lock-holder child did not report ready within "
                f"{_DOCTOR_READY_TIMEOUT_S:g} s (got {ready!r}); a blocking flock may hang on "
                "this filesystem")
        lock_fh = open(lock_path, "a")
        try:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                # Only contention shows the child's lock excluding the parent. Anything else
                # (ENOLCK, ENOTSUP, EIO...) is a locking failure: the flock_probe failure below.
                if e.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                _record("flock_exclusion", True,
                        "parent was correctly blocked (OSError on LOCK_NB) while the child "
                        "held LOCK_EX")
            else:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                _record("flock_exclusion", False,
                        "parent acquired LOCK_EX|LOCK_NB while the child held the lock -- "
                        "no exclusion")
        finally:
            lock_fh.close()
        child.stdin.close()
        child.wait(timeout=10)
        lock_fh2 = open(lock_path, "a")
        try:
            fcntl.flock(lock_fh2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_fh2.fileno(), fcntl.LOCK_UN)
            _record("flock_release", True, "parent acquired the lock after the child exited")
        finally:
            lock_fh2.close()
    except Exception as e:  # noqa: BLE001 -- a doctor probe failing must never raise
        _record("flock_probe", False, f"lock probe failed: {e}")
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    try:
        probe_path = os.path.join(doctor_dir, "probe.parquet")
        frame = pl.DataFrame({
            "target": ["A", "B"], "feature": ["g0", "g1"],
            "log2_fold_change": [1.0, -1.0], "p_adj": [0.01, 0.5],
        })
        write_de_parquet([frame], probe_path)
        meta = read_de_parquet_metadata(probe_path)
        if meta["codecs"] != ["ZSTD"]:
            raise ValueError(f"expected codecs ['ZSTD'], got {meta['codecs']!r}")
        roundtrip = pl.read_parquet(probe_path)
        if not roundtrip.equals(frame):
            raise ValueError("round-tripped frame does not equal the original")
        _record("parquet_zstd_roundtrip", True, probe_path)
    except Exception as e:  # noqa: BLE001
        _record("parquet_zstd_roundtrip", False, str(e))

    try:
        src = os.path.join(doctor_dir, "rename-src")
        dst = os.path.join(doctor_dir, "rename-dst")
        os.makedirs(src)
        with open(os.path.join(src, "marker"), "w") as fh:
            fh.write("ok")
        os.rename(src, dst)
        if not os.path.isfile(os.path.join(dst, "marker")):
            raise ValueError("renamed directory is missing its marker file")
        _record("atomic_rename", True, dst)
    except Exception as e:  # noqa: BLE001
        _record("atomic_rename", False, str(e))

    try:
        shutil.rmtree(doctor_dir)
        _record("cleanup", True, doctor_dir)
    except OSError as e:
        _record("cleanup", False, str(e))

    return {"root": root_expanded, "checks": checks, "ok": all(c["ok"] for c in checks)}


def _fault_hook(stage: str) -> None:
    """Test seam: if ``CELL_EVAL2_GTC_FAULT`` is set to ``stage``, hard-kill this process
    (``os._exit(3)``, bypassing normal cleanup) to simulate a crash at that exact point in
    ``GroundTruthDEGCache.get_or_compute``. A test may instead monkeypatch this module
    attribute (``cell_eval2.ground_truth_cache.store._fault_hook = ...`` -- note the
    ``store`` submodule, where ``get_or_compute`` actually looks it up by bare name, not the
    ``cell_eval2.ground_truth_cache`` package) to inject a fault without a subprocess. Calls
    are by bare module-level name (not ``self._fault_hook``) so a monkeypatch of the module
    attribute is honoured everywhere it's called from."""
    if os.environ.get("CELL_EVAL2_GTC_FAULT") == stage:
        os._exit(3)


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """Best-effort directory fsync: some filesystems refuse to fsync a directory file
    descriptor at all (EINVAL/ENOTSUP) -- that is tolerated, everything else is not (failing
    to open the directory included)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
    finally:
        os.close(fd)


def _atomic_write_json_line(path: str, obj: Any) -> None:
    """Write ``canonical_json_bytes(obj)`` plus a trailing newline to ``path`` via a
    same-directory tmp file + ``os.replace`` (mirrors ``cache._atomic_write``'s pattern;
    kept local here so this module does not import ``cache.py``'s private helper)."""
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid4().hex}")
    success = False
    try:
        with open(tmp, "wb") as fh:
            fh.write(canonical_json_bytes(obj))
            fh.write(b"\n")
        os.replace(tmp, path)
        success = True
    finally:
        if not success:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _producer_versions_best_effort() -> dict[str, str]:
    """Manifest-provenance version snapshot. Unlike ``identity.producer_versions`` (a hard
    error for the semantic document, which cannot key an entry without them), a missing
    distribution here is recorded as ``"unknown"`` rather than raised: the manifest's
    ``producer`` block is diagnostic only -- the semantic document already carries the
    versions that are part of the cache key."""
    # Keyed by the IMPORT name (``cell_eval2``), not the distribution name (``cell-eval2``), so
    # the manifest's producer block and the run sidecar's ``producer`` block spell it the same way.
    versions: dict[str, str] = {}
    for key, dist in (("cell_eval2", "cell-eval2"), ("gpudge", "gpudge"),
                      ("cellstream", "cellstream"), ("polars", "polars"), ("pyarrow", "pyarrow")):
        try:
            versions[key] = _importlib_metadata.version(dist)
            continue
        except _importlib_metadata.PackageNotFoundError:
            pass
        if dist == "cellstream":
            try:
                versions[key] = _importlib_metadata.version("shardad")
                continue
            except _importlib_metadata.PackageNotFoundError:
                pass
        versions[key] = "unknown"
    return versions


class GroundTruthDEGCache:
    """Flock-guarded, content-addressed store for ``de_real.parquet`` cache entries.

    An entry lives at ``root/entry_relpath(cache_id)`` (``ground-truth-deg/v{V}/<2-hex>/
    <cache_id>/{manifest.json, de_real.parquet}``). Every write is staged in a sibling
    ``.tmp-<cache_id>-<pid>-<uuid>`` directory, validated in place, then published with one
    ``os.rename`` -- so any concurrent reader only ever sees either nothing or a fully
    valid, already-validated-by-its-writer entry.

    ``self.root`` and ``self.policy`` are the ONLY source of truth for where this cache
    lives and what it may do -- see the module docstring.
    """

    def __init__(
        self, root: str, *, policy: str, stale_staging_age_s: float = STALE_STAGING_AGE_S
    ) -> None:
        if policy not in CACHE_POLICIES:
            raise GroundTruthContractError(f"policy must be one of {CACHE_POLICIES}, got {policy!r}")
        expanded = os.path.expanduser(root)
        if not os.path.isabs(expanded):
            raise GroundTruthContractError(
                f"root must be an absolute path after os.path.expanduser, got {root!r} -> "
                f"{expanded!r}"
            )
        # Canonical before anything walks it: makedirs on a raw ".../missing/../cache" would create
        # "missing" on the way, a path score_cellstream's realpath-based alias check never saw.
        expanded = os.path.realpath(expanded)
        self.policy = policy
        self.stale_staging_age_s = float(stale_staging_age_s)
        versioned = os.path.join(expanded, "ground-truth-deg", f"v{CACHE_FORMAT_VERSION}")
        if policy == "readonly":
            # NEVER creates directories, lock files or staging dirs -- a readonly cache is a
            # pure reader, even of a root that doesn't exist yet.
            if not os.path.isdir(expanded):
                raise GroundTruthCacheFilesystemError(
                    f"readonly ground-truth cache root does not exist: {expanded!r}. The "
                    "readonly policy never creates directories -- populate the cache once "
                    "with policy='readwrite', or point at an existing cache root/mount."
                )
        else:
            # Containment BEFORE creating anything: a symlinked <root>/ground-truth-deg (or v1)
            # pointing elsewhere must not make this constructor write outside the root.
            versioned_real = os.path.realpath(versioned)
            if not _is_within(versioned_real, os.path.realpath(expanded)):
                raise GroundTruthCacheFilesystemError(
                    f"ground-truth cache namespace {versioned!r} resolves to {versioned_real!r}, "
                    f"outside the cache root {os.path.realpath(expanded)!r} (a symlinked "
                    "component); nothing was created")
            try:
                os.makedirs(versioned, exist_ok=True)  # umask-derived permissions
            except OSError as e:
                raise GroundTruthCacheFilesystemError(
                    f"could not create ground-truth cache root {versioned!r}: {e}"
                ) from e
        self.root = os.path.realpath(expanded)

    # ── paths ────────────────────────────────────────────────────────────────────────

    def entry_dir(self, cache_id_hex: str) -> str:
        entry = os.path.join(self.root, entry_relpath(cache_id_hex))  # validates cache_id_hex
        parent = os.path.dirname(entry)
        parent_real = os.path.realpath(parent)
        if not _is_within(parent_real, self.root):
            raise GroundTruthCacheFilesystemError(
                f"cache entry parent directory for {cache_id_hex} escapes the cache root "
                f"via a symlink: {parent!r} resolves to {parent_real!r}, which is not under "
                f"{self.root!r}"
            )
        return entry

    def lock_path(self, cache_id_hex: str) -> str:
        entry = self.entry_dir(cache_id_hex)
        return os.path.join(os.path.dirname(entry), f".{cache_id_hex}.lock")

    def _staging_dir(self, cache_id_hex: str) -> str:
        entry = self.entry_dir(cache_id_hex)
        return os.path.join(
            os.path.dirname(entry), f".tmp-{cache_id_hex}-{os.getpid()}-{uuid4().hex}"
        )

    # ── validation ───────────────────────────────────────────────────────────────────

    def validate_entry(self, entry_dir: str, request: GroundTruthDEGCacheRequest) -> dict:
        """Run the full check sequence against ``entry_dir`` (a committed entry OR a
        not-yet-published staging directory) for ``request``. Returns the manifest dict on
        success; raises ``GroundTruthCacheCorruptError`` naming the failed check on the
        first violation."""
        return _validate_entry_files(entry_dir, root=self.root, cache_id=request.cache_id,
                                     request=request)

    # ── stale staging cleanup ────────────────────────────────────────────────────────

    def _cleanup_stale_staging(self, parent: str, cache_id_hex: str) -> None:
        prefix = f".tmp-{cache_id_hex}-"
        try:
            names = os.listdir(parent)
        except OSError:
            return
        now = time.time()
        for name in names:
            if not name.startswith(prefix):
                continue
            path = os.path.join(parent, name)
            try:
                age = now - os.lstat(path).st_mtime
            except OSError:
                continue
            if age < self.stale_staging_age_s:
                continue
            logger.warning(
                "ground_truth_cache: removing stale staging dir %r (age=%.0fs, "
                "threshold=%.0fs) for cache_id=%s -- a previous writer likely crashed "
                "before publishing", path, age, self.stale_staging_age_s, cache_id_hex,
            )
            shutil.rmtree(path, ignore_errors=True)  # never let a stuck removal block a writer

    # ── lookup / populate ────────────────────────────────────────────────────────────

    def lookup(self, request: GroundTruthDEGCacheRequest) -> GroundTruthDEGCacheResult | None:
        """Read-only: does a valid, committed entry already exist for ``request``? Safe
        under any policy -- never creates anything, never touches ``fcntl``. Returns
        ``None`` on a plain miss (nothing at the entry path); raises
        ``GroundTruthCacheCorruptError`` if an entry directory exists but fails validation, or
        if something other than a directory sits at the entry path (a corrupt entry is never
        silently treated as a miss)."""
        entry = self.entry_dir(request.cache_id)
        if not os.path.lexists(entry):
            return None
        # Anything else at the entry path goes to validation, whose first check refuses all but a
        # real directory: never a miss, so readonly reports no false miss and readwrite computes
        # no table it could not publish.
        t0 = time.monotonic()
        manifest = self.validate_entry(entry, request)
        validate_seconds = time.monotonic() - t0
        artifact = manifest["artifact"]
        return GroundTruthDEGCacheResult(
            cache_id=request.cache_id,
            status="hit",
            policy=self.policy,
            entry_dir=entry,
            table_path=os.path.join(entry, ENTRY_TABLE_NAME),
            manifest=manifest,
            de_real_sha256=artifact["sha256"],
            de_real_rows=artifact["row_count"],
            ground_truth_fingerprint_sha256=request.fingerprint.sha256,
            de_semantics_sha256=request.semantics.sha256,
            validate_seconds=validate_seconds,
        )

    def _log_done(self, result: GroundTruthDEGCacheResult) -> None:
        size_bytes = (result.manifest.get("artifact", {}).get("size_bytes")
                      if result.manifest else None)
        logger.info(
            "ground_truth_cache: done cache_id=%s status=%s lock_wait=%.3fs compute=%s "
            "write=%s validate=%s rows=%s bytes=%s",
            result.cache_id, result.status, result.lock_wait_seconds,
            "n/a" if result.compute_seconds is None else f"{result.compute_seconds:.3f}s",
            "n/a" if result.write_seconds is None else f"{result.write_seconds:.3f}s",
            "n/a" if result.validate_seconds is None else f"{result.validate_seconds:.3f}s",
            result.de_real_rows, size_bytes,
            extra={"cache_id": result.cache_id, "status": result.status},
        )

    def get_or_compute(
        self, request: GroundTruthDEGCacheRequest, compute: Callable[[], pl.DataFrame]
    ) -> GroundTruthDEGCacheResult:
        """Look up ``request``; on a miss under ``self.policy == "readwrite"``, call
        ``compute()`` exactly once (under an exclusive flock on this cache-id's lock file),
        publish it, and return the result. Under ``self.policy == "readonly"``, a miss
        raises ``GroundTruthCacheMissError`` instead of ever calling ``compute`` -- the
        POLICY THAT DECIDES THIS IS ALWAYS ``self.policy`` (the cache instance this method
        is called on), never anything the request carries."""
        cid = request.cache_id
        logger.info("ground_truth_cache: resolve cache_id=%s… policy=%s", cid[:12], self.policy,
                    extra={"cache_id": cid, "status": "start"})

        hit = self.lookup(request)
        if hit is not None:
            self._log_done(hit)
            return hit

        entry = self.entry_dir(cid)
        if self.policy == "readonly":
            raise GroundTruthCacheMissError(
                f"ground-truth cache miss for {cid} under readonly policy (entry: {entry}). "
                "Run once with ground_truth_cache_policy='readwrite' to populate this entry, or "
                "check that the cache root/mount is correct."
            )

        parent = os.path.dirname(entry)
        lock_file_path = self.lock_path(cid)
        # Resolve fcntl BEFORE creating any lock file / staging directory: an unsupported
        # platform must fail with a clear, targeted error and touch nothing on disk.
        fcntl = _flock_module()
        try:
            # The 2-hex-prefix directory (parent of the entry) may not exist yet -- this is
            # the FIRST write for this prefix. `readwrite`'s __init__ only creates the
            # versioned root, not every prefix subdirectory, so create it lazily here.
            os.makedirs(parent, exist_ok=True)
            lock_fh = open(lock_file_path, "a")  # never truncate: advisory ownership, not content
        except OSError as e:
            raise GroundTruthCacheFilesystemError(
                f"could not open lock file {lock_file_path!r} for cache id {cid}: {e}"
            ) from e

        staging: str | None = None
        lock_wait_seconds = 0.0
        compute_failed = False
        try:
            t_lock0 = time.monotonic()
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            except OSError as e:
                raise GroundTruthCacheFilesystemError(
                    f"could not acquire an exclusive advisory lock on {lock_file_path!r} for "
                    f"cache id {cid}: {e}. This filesystem must support POSIX advisory locks "
                    "(flock) for the ground-truth cache to write a shared entry safely -- "
                    "writing an unlocked shared entry is refused rather than silently "
                    "degrading. Run `cell-eval2 cache doctor <root>` to diagnose the mount."
                ) from e
            lock_wait_seconds = time.monotonic() - t_lock0

            # Recheck under the lock: another writer may have published while we waited.
            hit = self.lookup(request)
            if hit is not None:
                result = replace(hit, status="hit_after_wait", lock_wait_seconds=lock_wait_seconds)
                self._log_done(result)
                return result

            self._cleanup_stale_staging(parent, cid)

            staging = self._staging_dir(cid)
            try:
                os.makedirs(staging)
            except OSError as e:
                raise GroundTruthCacheFilesystemError(
                    f"could not create staging directory {staging!r} for cache id {cid}: {e}"
                ) from e

            t_compute0 = time.monotonic()
            try:
                df = compute()
            except BaseException:
                # The DE kernel's own failure (including an OSError reading the archive) is
                # NOT a cache filesystem error -- let it propagate unwrapped so the caller
                # sees the real cause. The `except OSError` below only re-labels OUR I/O.
                compute_failed = True
                raise
            compute_seconds = time.monotonic() - t_compute0
            if not isinstance(df, pl.DataFrame):
                raise GroundTruthContractError(
                    f"compute() for cache id {cid} must return a polars.DataFrame, got "
                    f"{type(df).__name__}"
                )
            # Computed BEFORE releasing df: this is the manifest's perturbation-universe
            # digest, and df is not touched again after the parquet write below.
            sorted_unique_targets = df["target"].unique().sort().to_list()

            table_path = os.path.join(staging, ENTRY_TABLE_NAME)
            t_write0 = time.monotonic()
            report = write_de_parquet([df], table_path)
            _fsync_file(table_path)
            write_seconds = time.monotonic() - t_write0

            _fault_hook("after_parquet")

            manifest = {
                "cache_format": CACHE_FORMAT,
                "cache_format_version": CACHE_FORMAT_VERSION,
                "cache_id": cid,
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ground_truth_fingerprint": dict(request.fingerprint.document),
                "ground_truth_fingerprint_sha256": request.fingerprint.sha256,
                "de_semantics": dict(request.semantics.document),
                "de_semantics_sha256": request.semantics.sha256,
                "artifact": {
                    "path": ENTRY_TABLE_NAME,
                    "sha256": report.sha256,
                    "size_bytes": report.size_bytes,
                    "row_count": report.rows,
                    "columns": [dict(c) for c in report.columns],
                    "compression": "zstd",
                    "compression_level": DE_PARQUET_COMPRESSION_LEVEL,
                    "target_row_group_rows": DE_PARQUET_ROW_GROUP_ROWS,
                },
                "perturbations": {
                    "count": len(sorted_unique_targets),
                    "ordered_values_sha256": sequence_sha256(sorted_unique_targets),
                },
                "producer": _producer_versions_best_effort(),
            }
            manifest_path = os.path.join(staging, ENTRY_MANIFEST_NAME)
            _atomic_write_json_line(manifest_path, manifest)
            _fsync_file(manifest_path)

            _fault_hook("after_manifest")

            t_validate0 = time.monotonic()
            try:
                self.validate_entry(staging, request)
            except GroundTruthCacheCorruptError as e:
                raise GroundTruthCacheFilesystemError(
                    f"ground-truth cache write for {cid} produced an entry that failed its "
                    f"own validation immediately after writing (a writer bug, not external "
                    f"corruption): {e}"
                ) from e
            validate_seconds = time.monotonic() - t_validate0

            _fsync_dir(staging)
            _fault_hook("before_rename")
            try:
                os.rename(staging, entry)
            except OSError as e:
                if os.path.isdir(entry):
                    # Lost a publish race to another writer. Our staged copy is redundant --
                    # discard it and report the winner's entry as a hit.
                    shutil.rmtree(staging, ignore_errors=True)
                    staging = None
                    hit = self.lookup(request)
                    result = replace(hit, status="hit_after_wait",
                                     lock_wait_seconds=lock_wait_seconds,
                                     compute_seconds=compute_seconds)
                    self._log_done(result)
                    return result
                raise GroundTruthCacheFilesystemError(
                    f"could not publish cache entry for {cid} ({staging!r} -> {entry!r}): {e}"
                ) from e
            staging = None  # published: no longer ours to clean up on a later failure
            _fsync_dir(parent)
            _fault_hook("after_rename")

            result = GroundTruthDEGCacheResult(
                cache_id=cid,
                status="miss_computed",
                policy=self.policy,
                entry_dir=entry,
                table_path=os.path.join(entry, ENTRY_TABLE_NAME),
                manifest=manifest,
                de_real_sha256=report.sha256,
                de_real_rows=report.rows,
                ground_truth_fingerprint_sha256=request.fingerprint.sha256,
                de_semantics_sha256=request.semantics.sha256,
                lock_wait_seconds=lock_wait_seconds,
                validate_seconds=validate_seconds,
                compute_seconds=compute_seconds,
                write_seconds=write_seconds,
            )
            self._log_done(result)
            return result
        except OSError as e:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            if compute_failed:
                raise
            raise GroundTruthCacheFilesystemError(
                f"filesystem error while populating ground-truth cache entry {cid} at "
                f"{entry!r}: {e}"
            ) from e
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt mid-compute or mid-write must not
            # leave an unpublished staging directory behind -- hits never sweep stale staging.
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_fh.close()
