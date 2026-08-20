from __future__ import annotations

import hashlib
try:
    import fcntl  # POSIX advisory locking for the shared cache manifest
except ImportError:  # non-POSIX (e.g. Windows): re-read+merge runs lock-free
    fcntl = None
import io as _io
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import polars as pl
from scipy.sparse import issparse

from .catalog import derived_policy
from .moments import GroupMoments

CACHE_FORMAT_VERSION = 2


def _hash_obj(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _hash_str_array(values) -> str:
    arr = np.asarray(values, dtype=str)
    h = hashlib.sha256()
    h.update(arr.size.to_bytes(8, "little"))
    for v in arr:
        b = str(v).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def _hash_value_counts(labels) -> str:
    arr = np.asarray(labels, dtype=str)
    uniq, counts = np.unique(arr, return_counts=True)  # uniq is sorted -> deterministic
    h = hashlib.sha256()
    for u, c in zip(uniq, counts):
        b = str(u).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
        h.update(int(c).to_bytes(8, "little"))
    return h.hexdigest()


def _hash_labels(labels) -> str:
    """Hash the per-cell label assignment, not just the multiset: pseudobulk groups rows by
    label, so a permutation that preserves counts but changes which cell has which label must
    change the fingerprint (a multiset-only hash would collide — and even strict mode, which
    hashes X, wouldn't catch a pure label permutation). Cheap: the sorted label set + per-cell
    codes (hashed as an int buffer)."""
    arr = np.asarray(labels, dtype=str)
    uniq, inv = np.unique(arr, return_inverse=True)  # uniq sorted; inv = per-cell code (obs order)
    h = hashlib.sha256()
    h.update(_hash_str_array(uniq).encode("ascii"))
    h.update(np.ascontiguousarray(inv.astype(np.int64, copy=False)))
    return h.hexdigest()


def _hash_X(X) -> str:
    h = hashlib.sha256()
    if issparse(X):
        # Canonical, deterministic across formats: COO has no indices/indptr and CSR indices
        # may be unsorted. tocsr() returns the same object for already-CSR input (no copy); we
        # copy only when indices need sorting, so we never mutate the caller's adata.X.
        xc = X.tocsr()  # returns self if already CSR (no copy); converts other formats
        if not xc.has_sorted_indices:
            xc = xc.copy()  # copy ONLY when we must sort, so we never mutate the caller's matrix
            xc.sort_indices()
        # buffer-protocol reads (no full .tobytes() copy). indices/indptr are normalized to
        # int64 because their width (int32 vs int64) varies by platform/size/slicing and would
        # otherwise make a logically-identical matrix hash differently (strict false-miss).
        h.update(np.ascontiguousarray(xc.data))
        h.update(np.ascontiguousarray(xc.indices.astype(np.int64, copy=False)))
        h.update(np.ascontiguousarray(xc.indptr.astype(np.int64, copy=False)))
    else:
        h.update(np.ascontiguousarray(np.asarray(X)))  # avoid a second full-size copy (RAM)
    return h.hexdigest()


def fingerprint_adata(adata, *, pert_col: str, strict: bool = False) -> str:
    """Metadata-only structural fingerprint (load-mode agnostic: identical for an
    in-memory or a backed read of the same file). strict adds an X content hash."""
    n_obs, n_vars = adata.shape
    parts = [
        "adata", int(n_obs), int(n_vars), str(adata.X.dtype),
        _hash_str_array(adata.var.index.values),
        _hash_labels(adata.obs[pert_col].to_numpy()),  # per-cell assignment, not just the multiset
    ]
    if strict:
        src = adata.to_memory() if getattr(adata, "isbacked", False) else adata
        parts.append(_hash_X(src.X))
    return _hash_obj(*parts)


def fingerprint_de_table(df: pl.DataFrame, *, strict: bool = False) -> str:
    parts = [
        "de", df.height, _hash_obj(*sorted(df.columns)),
        _hash_value_counts(df["target"].to_numpy()) if "target" in df.columns else "no-target",
        # hash the feature values (not just n_unique): a changed feature SET with the same
        # unique-count must change the fingerprint, else a stale DE rank could false-hit.
        _hash_value_counts(df["feature"].to_numpy()) if "feature" in df.columns else "no-feature",
    ]
    if strict:
        buf = _io.BytesIO()
        df.write_parquet(buf)
        parts.append(hashlib.sha256(buf.getvalue()).hexdigest())
    return _hash_obj(*parts)


logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"


class _Miss:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISS"


MISS = _Miss()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: str, write_fn) -> None:
    d = os.path.dirname(path) or "."
    # uuid4 so concurrent writers never share a temp name — even across threads and across
    # containers with namespaced (colliding) PIDs sharing a cache volume. pid kept for debugging.
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    success = False
    try:
        write_fn(tmp)
        os.replace(tmp, path)
        success = True
    finally:
        if not success:  # on success os.replace already moved tmp; only clean up on failure
            try:
                os.remove(tmp)
            except OSError:
                pass  # never let cleanup mask the original error (or race another worker)


def _dump_npz(value, tmp: str) -> None:
    perts, means = value
    with open(tmp, "wb") as fh:
        np.savez(fh, perts=np.asarray(perts, dtype=str), means=np.asarray(means))


def _load_npz(path: str):
    with np.load(path, allow_pickle=False) as z:
        perts, means = z["perts"], z["means"]  # KeyError if a key is absent -> caught as MISS
    # shape/schema sanity (spec §7): a parseable-but-misshapen file recomputes, not misaligns
    if perts.ndim != 1 or means.ndim != 2 or means.shape[0] != perts.shape[0]:
        raise ValueError(f"npz shape mismatch: perts {perts.shape}, means {means.shape}")
    return (perts, means)


def _dump_npz_moments(value, tmp: str) -> None:
    """Persist a bulk and its moments as ONE artifact.

    The moments' own labels are not stored, so the two label vectors must already be equal --
    otherwise a permuted-but-correctly-labelled GroupMoments would work cold and come back
    silently MISLABELLED after a cache round-trip. Assert rather than store a second copy:
    every producer in this repo builds both from the same ``perts`` array, so an inequality is
    a bug, not a case to support. ``CacheStore.put`` only swallows OSError, so this propagates.
    """
    (perts, means), moments = value
    perts = np.asarray(perts, dtype=str)
    mom_perts = np.asarray(moments.perts, dtype=str)
    if not np.array_equal(perts, mom_perts):
        raise ValueError(
            "refusing to cache moments whose labels differ from their bulk's: the artifact "
            f"stores one label vector. bulk={perts.tolist()[:5]}... "
            f"moments={mom_perts.tolist()[:5]}..."
        )
    jk = moments.jk
    with open(tmp, "wb") as fh:
        np.savez(fh, perts=perts, means=np.asarray(means),
                 counts=np.asarray(moments.counts, dtype=np.float64),
                 sumsq=np.asarray(moments.sumsq, dtype=np.float64),
                 # np.savez cannot store None, and a zero-length jk would round-trip as an
                 # EMPTY correction that correction_for would index into. Absence gets its own
                 # key; exactly one of the two is ever written.
                 **({"jk_absent": np.array(True)} if jk is None
                    else {"jk": np.asarray(jk, dtype=np.float64)}))


def _load_npz_moments(path: str):
    """A self-contained pseudobulk-plus-moments artifact (issue #198).

    The four base arrays plus exactly one of ``jk``/``jk_absent`` are REQUIRED, so a hit stays
    atomic -- there is no state in which means are fresh and moments stale. A pre-change cache
    has no ``*_moments_*`` key at all, so it can never be reused as if it carried moments.
    """
    with np.load(path, allow_pickle=False) as z:
        # KeyError if any key is absent -> caught as MISS by CacheStore.get
        perts, means = z["perts"], z["means"]
        counts, sumsq = z["counts"], z["sumsq"]
        has_jk, has_absent = "jk" in z.files, "jk_absent" in z.files
        if has_jk == has_absent:
            raise ValueError(
                f"npz_moments must carry exactly one of 'jk'/'jk_absent'; got "
                f"jk={has_jk}, jk_absent={has_absent}")
        if has_absent and not bool(z["jk_absent"]):
            raise ValueError("npz_moments jk_absent is present but not true")
        jk = None if has_absent else z["jk"]
    if (perts.ndim != 1 or means.ndim != 2 or means.shape[0] != perts.shape[0]
            or counts.shape != perts.shape or sumsq.shape != perts.shape
            or (jk is not None and jk.shape != perts.shape)):
        raise ValueError(
            f"npz_moments shape mismatch: perts {perts.shape}, means {means.shape}, "
            f"counts {counts.shape}, sumsq {sumsq.shape}, "
            f"jk {None if jk is None else jk.shape}"
        )
    return (perts, means), GroupMoments(perts=perts, counts=counts, sumsq=sumsq, jk=jk)


def _dump_parquet(value: pl.DataFrame, tmp: str) -> None:
    value.write_parquet(tmp)


def _load_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def _dump_json(value, tmp: str) -> None:
    if not isinstance(value, dict):
        # The store cannot tell a dict from an arbitrary object once it is on disk, and a
        # silently-coerced value comes back as something the caller did not put in.
        raise TypeError(f"kind='json' stores a dict, got {type(value).__name__}")
    with open(tmp, "w") as fh:
        json.dump(value, fh, sort_keys=True)


def _load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


_HANDLERS = {
    "npz": (".npz", _dump_npz, _load_npz),
    # `.moments.npz`, NOT `.npz`: _artifact_filename hashes (key, fingerprint, params) but NOT
    # the kind, and _entry_valid does not check kind either. Production keys already differ, so
    # this is belt-and-braces -- but with a shared extension a caller passing the wrong `kind`
    # for the same key could have a plain loader silently read a moments file and drop its extra
    # arrays. A distinct extension makes that structurally impossible, and unlike hashing the
    # kind into the digest it invalidates no existing cache entry.
    "npz_moments": (".moments.npz", _dump_npz_moments, _load_npz_moments),
    "parquet": (".parquet", _dump_parquet, _load_parquet),
    # #276: the replicate anchor is THREE objects under one key (the aggregate, the
    # per-split frame and the sidecar), and the store keeps one value per key. npz cannot
    # hold a dict and parquet cannot hold two differently-shaped frames without encoding one
    # as an opaque blob, so the round-trip requirement (spec 4.5) needs a structured kind.
    # Adding a handler invalidates nothing: `_artifact_filename` hashes
    # (key, fingerprint, params) and every existing key is unchanged.
    "json": (".json", _dump_json, _load_json),
}


def _handler(kind: str):
    try:
        return _HANDLERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown cache kind {kind!r}; expected one of {sorted(_HANDLERS)}"
        ) from None


def _safe_key(key: str) -> str:
    """Reject keys that aren't a bare name, so `key + ext` can't escape the cache folder
    (path-traversal defense; CacheStore is public though current keys are internal literals)."""
    if not key or key in (".", "..") or key != os.path.basename(key):
        raise ValueError(f"cache key must be a bare name (no path separators), got {key!r}")
    return key


def _artifact_filename(key: str, fingerprint: str, params: dict, ext: str) -> str:
    """Content-addressed artifact filename: distinct (fingerprint, params) -> distinct file (the
    key stays a readable prefix, a short content digest disambiguates). A fingerprint-INDEPENDENT
    filename (the old `key+ext`) let two processes putting different content under one key race:
    each _atomic_write hits the same path (last writer's bytes win) while the manifest merge can
    leave that key's entry pointing at the OTHER writer's fingerprint -> get() returns the wrong
    content (F9.2). Embedding the digest makes every distinct entry its own immutable file, so the
    manifest entry and file content can never disagree. The digest is order-independent in params
    (sort_keys) to match _entry_valid's params == comparison."""
    digest = _hash_obj(fingerprint, json.dumps(params, sort_keys=True, default=str))[:16]
    return f"{_safe_key(key)}-{digest}{ext}"


class CacheStore:
    """Disk-backed artifact cache for one folder. Every hit is gated by a stored
    (fingerprint, params); any mismatch, missing file, or read failure is a miss
    that triggers recompute. Writes are atomic (temp file + os.replace)."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._manifest = self._load_manifest()
        # Baseline snapshot for dirty-tracking: on save we merge only the entries THIS
        # store actually changed, so we never revert keys other processes updated.
        self._baseline = dict(self._manifest.get("artifacts", {}))

    def _manifest_path(self) -> str:
        return os.path.join(self.root, _MANIFEST)

    def _fresh_manifest(self) -> dict:
        return {"cache_format_version": CACHE_FORMAT_VERSION, "artifacts": {}}

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if not os.path.exists(path):
            return self._fresh_manifest()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):  # valid JSON but not an object -> ignore (no AttributeError)
                raise ValueError("manifest is not a dict")
            if data.get("cache_format_version") != CACHE_FORMAT_VERSION:
                logger.warning("cache %s: format-version mismatch; ignoring cache", self.root)
                return self._fresh_manifest()
            if not isinstance(data.get("artifacts"), dict):
                raise ValueError("bad manifest shape")
            return data
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("cache %s: manifest unreadable (%s); ignoring cache", self.root, e)
            return self._fresh_manifest()

    def _save_manifest(self) -> None:
        # Re-read + merge the on-disk manifest under an exclusive lock before the atomic
        # replace. Without this, two CacheStores sharing a dir each write back their
        # construction-time snapshot, dropping entries the other added (-> orphaned files
        # + silent recompute). The lock is best-effort: where flock is unavailable we still
        # re-read+merge, which closes most of the race window.
        own = self._manifest.get("artifacts", {})
        # Only OUR modified/added entries should win on merge. Merging all of `own`
        # would revert keys another process updated since our snapshot (lost update).
        dirty = {k: v for k, v in own.items() if self._baseline.get(k) != v}
        lock_fh = None
        try:
            if fcntl is not None:
                try:
                    lock_fh = open(self._manifest_path() + ".lock", "a")  # never truncate the lockfile
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                except OSError:  # lock unsupported on this FS -> proceed lock-free
                    if lock_fh is not None:
                        lock_fh.close()
                    lock_fh = None
            disk = self._load_manifest()  # fresh read (under the lock if acquired)
            merged = disk.get("artifacts", {})
            merged.update(dirty)  # only our changed entries win; preserve everything else on disk
            new_manifest = {"cache_format_version": CACHE_FORMAT_VERSION, "artifacts": merged}

            def _w(tmp: str) -> None:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(new_manifest, fh, indent=2, sort_keys=True)
            _atomic_write(self._manifest_path(), _w)
            # Commit in-memory state ONLY after the write succeeds: a transient write
            # failure then leaves the dirty keys still dirty, so the next save retries them.
            self._manifest = new_manifest
            self._baseline = dict(merged)
        finally:
            if lock_fh is not None:
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_fh.close()

    def _entry_valid(self, key: str, fingerprint: str, params: dict, kind: str | None = None) -> bool:
        e = self._manifest["artifacts"].get(key)
        if not isinstance(e, dict):  # missing or malformed (non-dict) entry -> miss
            return False
        if e.get("fingerprint") != fingerprint or e.get("params") != params:
            return False
        # An entry written by a DIFFERENT handler must not be decoded by this one: the loaders
        # are not interchangeable and a plain npz loader would silently drop a moments file's
        # extra arrays. Entries predating this field carry no "kind" and stay accepted.
        if kind is not None and e.get("kind") is not None and e["kind"] != kind:
            return False
        fn = e.get("filename")  # missing/edited/old-schema entry -> miss, never KeyError
        if not fn or fn != os.path.basename(fn):  # reject a tampered traversal filename -> miss
            return False
        return os.path.exists(os.path.join(self.root, fn))

    def get(self, key: str, *, fingerprint: str, params: dict, kind: str):
        _ext, _dump, load = _handler(kind)  # validate kind up front (clear ValueError)
        if not self._entry_valid(key, fingerprint, params, kind):
            return MISS
        path = os.path.join(self.root, self._manifest["artifacts"][key]["filename"])
        try:
            return load(path)
        except Exception as e:  # noqa: BLE001 - any read failure is a cache miss -> recompute
            logger.warning("cache %s: artifact %r unreadable (%s); recomputing",
                           self.root, key, e)
            return MISS

    def put(self, key: str, value, *, fingerprint: str, params: dict, kind: str) -> None:
        ext, dump, _load = _handler(kind)
        filename = _artifact_filename(key, fingerprint, params, ext)  # content-addressed (F9.2)
        prev = self._manifest["artifacts"].get(key)
        prev_filename = prev.get("filename") if isinstance(prev, dict) else None
        # A write failure (full disk / read-only fs / permissions) must not fail the run — the
        # computed result is valid, we just couldn't cache it. (Spec §7: cache problems never raise.)
        try:
            _atomic_write(os.path.join(self.root, filename), lambda tmp: dump(value, tmp))
            self._manifest["artifacts"][key] = {
                "fingerprint": fingerprint, "params": params, "kind": kind,
                "filename": filename, "timestamp": _now_iso(),
            }
            self._save_manifest()
        except OSError as e:
            logger.warning("cache %s: failed to write %r (%s); result not cached", self.root, key, e)
            return
        # GC the file this put superseded for `key` (content-addressed names would otherwise leave
        # one file per distinct content forever). Only delete when the just-merged manifest no longer
        # references it -- a superseded content-addressed file embeds this key, so it can only be
        # referenced by this key's entry; a lost GC race merely wastes disk, never affects correctness.
        if prev_filename and prev_filename != filename:
            referenced = {e.get("filename") for e in self._manifest["artifacts"].values()
                          if isinstance(e, dict)}
            # Traversal guard mirroring _entry_valid: only remove a plain basename within root, so a
            # tampered/legacy manifest filename ("../x") can never delete outside the cache dir.
            if prev_filename not in referenced and prev_filename == os.path.basename(prev_filename):
                try:
                    os.remove(os.path.join(self.root, prev_filename))
                except OSError:  # best-effort; a leftover file only wastes disk
                    pass

    def get_or_compute(self, key: str, *, fingerprint: str, params: dict, kind: str, compute):
        hit = self.get(key, fingerprint=fingerprint, params=params, kind=kind)
        if hit is not MISS:
            return hit
        value = compute()
        self.put(key, value, fingerprint=fingerprint, params=params, kind=kind)
        return value


def config_hash(cfg_dict: dict) -> str:
    """Digest of the numerics-affecting config (drops cache dirs, outdir, num_threads,
    gather_threads, and metrics — metrics enter the result key as the resolved name list)."""
    skip = {"cache_real", "cache_pred", "cache_strict", "outdir", "num_threads",
            "gather_threads", "metrics"}
    kept = {k: v for k, v in cfg_dict.items() if k not in skip}
    return _hash_obj("config", json.dumps(kept, sort_keys=True, default=str))


def result_fingerprint(*, real_fp, pred_fp, de_fps, config_digest, metric_names) -> str:
    # metric_names is the resolved list IN ORDER (not sorted): the tidy result's row order
    # follows metric order, so a same-set/different-order request must miss and recompute
    # rather than return rows in a previously-cached order.
    # `derived_policy` additionally binds each derived metric to the COMPONENTS it divides,
    # which the names alone do not carry (#257).
    return _hash_obj("result", real_fp, pred_fp, *de_fps, config_digest, *metric_names,
                     json.dumps(derived_policy(metric_names), sort_keys=True))
