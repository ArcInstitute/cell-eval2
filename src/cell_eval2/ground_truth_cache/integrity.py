"""Integrity validation for the ground-truth DEG cache (see docs/ground_truth_cache.md).

Entry-file checks shared by ``store.GroundTruthDEGCache.validate_entry`` (bound to a
``GroundTruthDEGCacheRequest``) and ``inspect_entry`` (the ``cache inspect`` CLI / ad-hoc
auditing, no request). ``validate_entry`` / ``inspect_entry`` share one set of file-level
check functions (``_check_*``), each raising ``GroundTruthCacheCorruptError`` naming the
cache id, the check, and the exact path (see ``identity.py``'s module docstring: this cache
never papers over a corrupt entry).

Split into two halves (review of PR #389, finding P1 "`cache inspect` can declare an
identity-invalid entry valid"):

* (a) **self-consistency** (``_check_identity_self_consistency``): unconditional, always run
  by both callers. Confirms the manifest's own ``ground_truth_fingerprint``/``de_semantics``
  documents are present, schema-valid, hash to their own recorded digests, and that
  ``cache_id(fingerprint_doc, semantics_doc)`` reproduces both the manifest's own
  ``cache_id`` field and the id the entry is being checked under. This is what makes
  ``inspect_entry`` (which has no request to compare against) still a real identity check,
  not just a file/byte check.
* (b) **request equality** (``_check_request_equality``): optional, only run when a caller
  supplies a ``GroundTruthDEGCacheRequest`` (``GroundTruthDEGCache.validate_entry`` always
  does; ``inspect_entry`` never does). Confirms the manifest's identity documents equal the
  SPECIFIC request's own fingerprint/semantics, not just that they're internally consistent.

``polars`` is imported at module scope (a base dependency of ``cell_eval2``); ``pyarrow`` is
never imported here directly -- Parquet metadata reads go through ``de_export``'s lazy
imports, and ``fcntl`` is never needed for reads (only ``store.py``'s writer needs it).
"""
from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping

import polars as pl

from ..de import REQUIRED_COLS
from ..de_export import read_de_parquet_metadata, sha256_file
from .identity import (
    ENTRY_MANIFEST_NAME,
    ENTRY_TABLE_NAME,
    CACHE_FORMAT,
    CACHE_FORMAT_VERSION,
    GroundTruthCacheCorruptError,
    GroundTruthContractError,
    GroundTruthDEGCacheRequest,
    cache_id as _compute_cache_id,
    document_sha256,
    entry_relpath,
    sequence_sha256,
    validate_fingerprint_document,
    validate_semantic_document,
    _HEX64,
)

#: Arrow type families the required DE columns may have, whatever the manifest says.
_STRING_TYPES = frozenset({"string", "large_string", "string_view"})
_FLOAT_TYPES = frozenset({"halffloat", "float", "double"})
_REQUIRED_COL_TYPES = {"target": _STRING_TYPES, "feature": _STRING_TYPES,
                       "log2_fold_change": _FLOAT_TYPES, "p_adj": _FLOAT_TYPES}


def _corrupt(cache_id: str, check: str, path: str, detail: str) -> GroundTruthCacheCorruptError:
    return GroundTruthCacheCorruptError(
        f"ground-truth cache entry {cache_id} failed check {check!r} at {path!r}: {detail}"
    )


def _is_within(path: str, root: str) -> bool:
    """``path`` is ``root`` or lies under it, by path components: a root of ``/`` works (a
    ``root + os.sep`` string prefix would be ``//``), and ``/a/bc`` is not under ``/a/b``."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:   # different drives (Windows), or one path relative
        return False


def _is_int(value, expected: int) -> bool:
    """``value`` is the int ``expected``, not merely equal to it: True == 1, and a JSON ``true``
    in an integer field is a malformed manifest."""
    return type(value) is int and value == expected


def _strip_trailing(path: str) -> str:
    """``path`` without trailing separators or terminal ``.`` components (``a/b/``, ``a/b/./`` ->
    ``a/b``), lexically -- no symlink is resolved, and a root stays itself: ``/``, and on Windows
    a drive or UNC root (``C:/``, not ``C:``, which is the current directory on drive C)."""
    seps = os.sep + (os.altsep or "")   # Windows separates on "/" too
    drive, rest = os.path.splitdrive(path)
    head = rest
    while True:
        trimmed = head.rstrip(seps)
        if not trimmed:
            return drive + rest[:1]   # the root's own separator, or nothing for a bare drive
        if os.path.basename(trimmed) == "." and os.path.dirname(trimmed):
            head = os.path.dirname(trimmed)
            continue
        return drive + trimmed


def _check_entry_type(entry_dir: str, cache_id: str) -> None:
    """Only real directories, published by rename, are entries. A symlink at the entry path --
    even to a valid copy of the entry inside the root -- would alias storage the cache does not
    own, and a regular file or a dangling link is no entry at all. The leaf is taken lexically:
    ``lstat("link/")`` would follow the link."""
    leaf = _strip_trailing(entry_dir)
    if os.path.islink(leaf) or not os.path.isdir(leaf):
        raise _corrupt(cache_id, "entry_type", entry_dir,
                        "the entry path is not a real directory (a symlink, a regular file or a "
                        "dangling link)")


def _check_containment(entry_dir: str, root: str, cache_id: str) -> str:
    real = os.path.realpath(entry_dir)
    if not _is_within(real, root):
        raise _corrupt(
            cache_id, "containment", entry_dir,
            f"resolves to {real!r}, which is not under cache root {root!r}",
        )
    return real


def _check_regular_files(entry_dir: str, cache_id: str) -> tuple[str, str]:
    manifest_path = os.path.join(entry_dir, ENTRY_MANIFEST_NAME)
    table_path = os.path.join(entry_dir, ENTRY_TABLE_NAME)
    for label, path in (("manifest", manifest_path), ("artifact", table_path)):
        try:
            st = os.lstat(path)
        except OSError as e:
            raise _corrupt(cache_id, "regular_files", path,
                            f"{label} is missing or unreadable: {e}") from e
        if not stat.S_ISREG(st.st_mode):
            raise _corrupt(cache_id, "regular_files", path,
                            f"{label} is not a regular file (mode={oct(st.st_mode)}; a "
                            "symlink or other special file counts as corruption)")
    return manifest_path, table_path


def _load_manifest_json(manifest_path: str, cache_id: str) -> dict:
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.loads(fh.read())
    except (OSError, UnicodeDecodeError, ValueError) as e:
        # json.JSONDecodeError is itself a ValueError; UnicodeDecodeError (invalid UTF-8
        # bytes, raised by the `.read()` inside the `with` above) is listed explicitly so a
        # reader never mistakes it for something this function forgot to handle. Both, like
        # OSError, are ordinary entry corruption -- never a traceback out of `inspect_entry`.
        raise _corrupt(cache_id, "manifest_json", manifest_path,
                        f"could not read/parse manifest: {e}") from e
    if not isinstance(manifest, dict):
        raise _corrupt(cache_id, "manifest_json", manifest_path,
                        f"manifest is not a JSON object (got {type(manifest).__name__})")
    return manifest


def _check_format_and_id(manifest: dict, manifest_path: str, cache_id: str) -> None:
    if manifest.get("cache_format") != CACHE_FORMAT:
        raise _corrupt(cache_id, "cache_format", manifest_path,
                        f"manifest cache_format={manifest.get('cache_format')!r} != {CACHE_FORMAT!r}")
    if not _is_int(manifest.get("cache_format_version"), CACHE_FORMAT_VERSION):
        raise _corrupt(cache_id, "cache_format_version", manifest_path,
                        f"manifest cache_format_version={manifest.get('cache_format_version')!r} "
                        f"!= {CACHE_FORMAT_VERSION!r}")
    if manifest.get("cache_id") != cache_id:
        raise _corrupt(cache_id, "cache_id", manifest_path,
                        f"manifest cache_id={manifest.get('cache_id')!r} != {cache_id!r}")


def _check_identity_self_consistency(manifest: dict, manifest_path: str, cache_id: str) -> None:
    """(a) Unconditional manifest self-consistency -- run by BOTH ``inspect_entry`` and
    ``GroundTruthDEGCache.validate_entry``, never skipped and never conditioned on a caller
    supplying a request (see module docstring). Checks, in order:

    1. ``ground_truth_fingerprint`` and ``de_semantics`` are present ``Mapping`` objects.
    2. Each passes its own schema validator (``validate_fingerprint_document`` /
       ``validate_semantic_document``).
    3. Each document's ``document_sha256`` equals its recorded ``*_sha256`` field.
    4. ``cache_id(fingerprint_doc, semantics_doc)`` reproduces the manifest's own recorded
       ``cache_id`` field AND the cache id this entry is being checked under (``cache_id``,
       the directory's own id / the request's derived id -- whichever the caller supplied to
       ``_validate_entry_files``).
    """
    fp_doc = manifest.get("ground_truth_fingerprint")
    if not isinstance(fp_doc, Mapping):
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        "manifest has no 'ground_truth_fingerprint' object")
    sem_doc = manifest.get("de_semantics")
    if not isinstance(sem_doc, Mapping):
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        "manifest has no 'de_semantics' object")

    try:
        validate_fingerprint_document(fp_doc)
        validate_semantic_document(sem_doc)
    except GroundTruthContractError as e:
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        f"identity document failed schema validation: {e}") from e

    fp_sha = document_sha256(fp_doc)
    if fp_sha != manifest.get("ground_truth_fingerprint_sha256"):
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        f"document_sha256(ground_truth_fingerprint)={fp_sha!r} != manifest "
                        f"ground_truth_fingerprint_sha256="
                        f"{manifest.get('ground_truth_fingerprint_sha256')!r}")

    sem_sha = document_sha256(sem_doc)
    if sem_sha != manifest.get("de_semantics_sha256"):
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        f"document_sha256(de_semantics)={sem_sha!r} != manifest "
                        f"de_semantics_sha256={manifest.get('de_semantics_sha256')!r}")

    recomputed = _compute_cache_id(fp_doc, sem_doc)
    if recomputed != manifest.get("cache_id"):
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        f"cache_id(ground_truth_fingerprint, de_semantics)={recomputed!r} != "
                        f"manifest cache_id={manifest.get('cache_id')!r}")
    if recomputed != cache_id:
        raise _corrupt(cache_id, "identity_self_consistency", manifest_path,
                        f"cache_id(ground_truth_fingerprint, de_semantics)={recomputed!r} != "
                        f"the cache id under check ({cache_id!r})")


def _check_request_equality(
    manifest: dict, request: GroundTruthDEGCacheRequest, manifest_path: str, cache_id: str
) -> None:
    """(b) Optional: does this entry's identity equal the CALLER's specific request? Only
    invoked when a request is given -- ``GroundTruthDEGCache.validate_entry`` always passes
    one; ``inspect_entry`` never does (there is no specific request to compare against; (a)
    above is what makes inspection meaningful without one). Assumes (a) has already run, so
    ``manifest["ground_truth_fingerprint"]``/``["de_semantics"]`` are known-present Mappings.
    """
    fp_doc = manifest["ground_truth_fingerprint"]
    if dict(fp_doc) != dict(request.fingerprint.document):
        raise _corrupt(cache_id, "fingerprint_mismatch", manifest_path,
                        "manifest ground_truth_fingerprint does not match the request's fingerprint")

    sem_doc = manifest["de_semantics"]
    if dict(sem_doc) != dict(request.semantics.document):
        raise _corrupt(cache_id, "semantics_mismatch", manifest_path,
                        "manifest de_semantics does not match the request's semantics")


def _check_artifact_bytes(manifest: dict, table_path: str, cache_id: str) -> dict:
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise _corrupt(cache_id, "artifact_manifest", table_path, "manifest has no 'artifact' object")
    # Validation always reads ENTRY_TABLE_NAME; a manifest naming another path contradicts it.
    if artifact.get("path") != ENTRY_TABLE_NAME:
        raise _corrupt(cache_id, "artifact_path", table_path,
                        f"manifest artifact.path {artifact.get('path')!r} != {ENTRY_TABLE_NAME!r}")
    try:
        actual_size = os.path.getsize(table_path)
    except OSError as e:
        raise _corrupt(cache_id, "artifact_size", table_path, f"could not stat artifact: {e}") from e
    if actual_size != artifact.get("size_bytes"):
        raise _corrupt(cache_id, "artifact_size", table_path,
                        f"on-disk size {actual_size} != manifest artifact.size_bytes "
                        f"{artifact.get('size_bytes')!r}")
    actual_sha = sha256_file(table_path)
    if actual_sha != artifact.get("sha256"):
        raise _corrupt(cache_id, "artifact_sha256", table_path,
                        f"on-disk sha256 {actual_sha!r} != manifest artifact.sha256 "
                        f"{artifact.get('sha256')!r}")
    return dict(artifact)


def _check_parquet_metadata(artifact: dict, table_path: str, cache_id: str) -> None:
    try:
        meta = read_de_parquet_metadata(table_path)
    except Exception as e:  # noqa: BLE001 -- an unreadable parquet file is corruption, not a crash
        raise _corrupt(cache_id, "parquet_metadata", table_path,
                        f"could not read parquet metadata: {e}") from e
    if meta["codecs"] != ["ZSTD"]:
        raise _corrupt(cache_id, "parquet_codec", table_path,
                        f"codecs={meta['codecs']!r} != ['ZSTD']")
    if not _is_int(artifact.get("row_count"), meta["rows"]):
        raise _corrupt(cache_id, "parquet_row_count", table_path,
                        f"parquet rows={meta['rows']} != manifest artifact.row_count="
                        f"{artifact.get('row_count')!r}")
    columns = artifact.get("columns")
    if not isinstance(columns, list) or not all(
            isinstance(c, Mapping) and isinstance(c.get("name"), str)
            and isinstance(c.get("arrow_type"), str) for c in columns):
        raise _corrupt(cache_id, "artifact_columns", table_path,
                        "manifest artifact.columns is not a list of {name, arrow_type} objects: "
                        f"{columns!r}")
    actual_cols = [c["name"] for c in meta["columns"]]
    expected_cols = [c["name"] for c in columns]
    if actual_cols != expected_cols:
        raise _corrupt(cache_id, "parquet_columns", table_path,
                        f"parquet columns {actual_cols!r} != manifest artifact.columns "
                        f"{expected_cols!r}")
    actual_types = [c["arrow_type"] for c in meta["columns"]]
    expected_types = [c["arrow_type"] for c in columns]
    if actual_types != expected_types:
        raise _corrupt(cache_id, "parquet_column_types", table_path,
                        f"parquet arrow types {actual_types!r} != manifest artifact.columns "
                        f"types {expected_types!r}")
    missing = [c for c in REQUIRED_COLS if c not in actual_cols]
    if missing:
        raise _corrupt(cache_id, "parquet_required_columns", table_path,
                        f"missing required DE columns {missing!r} (have {actual_cols!r})")
    # The manifest's types come from the very frame being cached, so matching them proves nothing
    # about the frame itself: check the required columns' type families on their own.
    types = dict(zip(actual_cols, actual_types))
    wrong = {c: types.get(c) for c, allowed in _REQUIRED_COL_TYPES.items()
             if types.get(c) not in allowed}
    if wrong:
        raise _corrupt(cache_id, "parquet_required_column_types", table_path,
                        f"required DE columns with non-DE arrow types: {wrong!r} (target/feature "
                        "must be strings, log2_fold_change/p_adj floating point)")


def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """``lf.collect`` on whichever streaming spelling the installed polars supports."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        pass
    try:
        return lf.collect(streaming=True)  # type: ignore[call-arg]
    except TypeError:
        return lf.collect()


def _check_target_feature_content(table_path: str, cache_id: str) -> None:
    lf = pl.scan_parquet(table_path).select(["target", "feature"])
    nulls = _collect_streaming(
        lf.select(
            pl.col("target").null_count().alias("target_nulls"),
            pl.col("feature").null_count().alias("feature_nulls"),
        )
    ).row(0, named=True)
    if nulls["target_nulls"] or nulls["feature_nulls"]:
        raise _corrupt(cache_id, "target_feature_nulls", table_path,
                        f"target_nulls={nulls['target_nulls']}, feature_nulls={nulls['feature_nulls']}")
    dup = _collect_streaming(lf.group_by(["target", "feature"]).len().filter(pl.col("len") > 1).head(1))
    if dup.height > 0:
        row = dup.row(0, named=True)
        raise _corrupt(cache_id, "duplicate_target_feature", table_path,
                        f"duplicate (target, feature) row: target={row['target']!r} "
                        f"feature={row['feature']!r} count={row['len']}")


def _check_perturbation_universe(manifest: dict, table_path: str, cache_id: str) -> None:
    perturbations = manifest.get("perturbations")
    if not isinstance(perturbations, Mapping):
        raise _corrupt(cache_id, "perturbations_manifest", table_path,
                        "manifest has no 'perturbations' object")
    targets = sorted(_collect_streaming(pl.scan_parquet(table_path).select("target").unique())["target"].to_list())
    if not _is_int(perturbations.get("count"), len(targets)):
        raise _corrupt(cache_id, "perturbation_count", table_path,
                        f"distinct target count {len(targets)} != manifest perturbations.count "
                        f"{perturbations.get('count')!r}")
    digest = sequence_sha256(targets)
    if digest != perturbations.get("ordered_values_sha256"):
        raise _corrupt(cache_id, "perturbation_universe", table_path,
                        f"sequence_sha256(sorted targets)={digest!r} != manifest "
                        f"perturbations.ordered_values_sha256={perturbations.get('ordered_values_sha256')!r}")


def _validate_entry_files(
    entry_dir: str, *, root: str, cache_id: str, request: GroundTruthDEGCacheRequest | None
) -> dict:
    """The full check sequence (spec §11 / the storage contract), shared by
    ``GroundTruthDEGCache.validate_entry`` (bound to a request) and ``inspect_entry``
    (``request=None``: runs (a) but not (b) -- see the module docstring for the split).
    ``entry_dir`` may be a committed entry OR a staging directory being validated before
    publish -- both live under ``root``."""
    _check_entry_type(entry_dir, cache_id)
    _check_containment(entry_dir, root, cache_id)
    manifest_path, table_path = _check_regular_files(entry_dir, cache_id)
    manifest = _load_manifest_json(manifest_path, cache_id)
    _check_format_and_id(manifest, manifest_path, cache_id)
    _check_identity_self_consistency(manifest, manifest_path, cache_id)
    if request is not None:
        _check_request_equality(manifest, request, manifest_path, cache_id)
    artifact = _check_artifact_bytes(manifest, table_path, cache_id)
    _check_parquet_metadata(artifact, table_path, cache_id)
    _check_target_feature_content(table_path, cache_id)
    _check_perturbation_universe(manifest, table_path, cache_id)
    return manifest


def inspect_entry(entry_or_id: str, *, root: str) -> dict:
    """Validate one cache entry without a ``GroundTruthDEGCacheRequest`` -- for the `cache
    inspect` CLI and ad-hoc auditing. ``entry_or_id`` is either a 64-hex cache id (resolved
    under ``root`` via ``entry_relpath``) or an entry directory path directly.

    Runs every check that does not need a request, INCLUDING full identity self-consistency
    (regular files, format/version, identity-document schema + digest + cache-id
    recomputation, artifact size/sha, parquet codec/rows/columns, content nulls/duplicates,
    perturbation-universe digest) -- see ``_check_identity_self_consistency``. It does not,
    and cannot, check equality against any SPECIFIC caller's request (there is none here).

    Never raises for a failed check, and never for a malformed/corrupt entry more broadly --
    ``problems`` carries the message(s) and ``valid`` is ``False``. This includes manifests
    that are not valid JSON, not valid UTF-8, not a JSON object, or fail any contract/type
    check on the way to being validated: every one of those is exactly the kind of ambient
    corruption this function exists to report, not to raise.
    """
    root_real = os.path.realpath(os.path.expanduser(root))
    if isinstance(entry_or_id, str) and _HEX64.match(entry_or_id):
        cache_id_hex = entry_or_id
        entry_dir = os.path.join(root_real, entry_relpath(cache_id_hex))
    else:
        entry_dir = entry_or_id
        cache_id_hex = os.path.basename(os.path.normpath(entry_dir))
    problems: list[str] = []
    manifest: dict | None = None
    try:
        manifest = _validate_entry_files(entry_dir, root=root_real, cache_id=cache_id_hex, request=None)
    except Exception as e:  # noqa: BLE001
        # Broad on purpose: `inspect_entry`'s documented contract is "never raises for a
        # failed check", so any of the ordinary ways a hand-corrupted, truncated or unreadable
        # entry can misbehave (not just our own `GroundTruthCacheCorruptError`; an OSError
        # reading the artifact too) becomes a problem entry, not a traceback.
        problems.append(str(e))
    manifest_summary = None
    if manifest is not None:
        manifest_summary = {k: v for k, v in manifest.items()
                             if k not in ("ground_truth_fingerprint", "de_semantics")}
    return {
        "entry_dir": entry_dir,
        "cache_id": cache_id_hex,
        "valid": manifest is not None and not problems,
        "problems": problems,
        "manifest": manifest_summary,
    }
