"""Identity contracts for the ground-truth DEG cache (see docs/ground_truth_cache.md).

The ground-truth DEG cache is a content-addressed, immutable cache of the unthresholded
real (ground-truth) differential-expression table computed on the Cellstream ``.shad``
scoring path (``score_cellstream``). This module implements the IDENTITY half of that
cache: canonical serialization, the ground-truth fingerprint of a cell archive, the
real-DE semantic document, cache-ID construction, the on-disk entry path, the typed
errors, and the dataclasses every consumer builds against.

Identity = a strong **fingerprint** of the source archive x the resolved **real-DE
semantics** that produced the table:

* The fingerprint (``GroundTruthFingerprint``, docs/ground_truth_cache.md §4.3 / spec §8.4) is derived
  from the archive's ``payload_sha256`` (a strong content checksum, written only with
  ``payload_checksum=True``), its ordered feature axis (``var_names``), and its ordered
  perturbation labels (``obs[pert_col]`` in storage/row order). There is no structural
  fallback: an archive without a strong ``payload_sha256`` is not cache-eligible (see
  ``cell_source.cell_fingerprint`` for the OLD structural-fallback behavior this cache
  deliberately does not replicate).
* The semantics (``RealDESemantics``, §9.1) are the resolved subset of the real-DE
  configuration that can change ``de_real.parquet``'s rows or values -- an explicit
  allowlist built by ``build_real_de_semantics``, never the raw ``EvalConfig``.

The cache ID (§9.3) is the SHA-256 of the canonical JSON of ``{fingerprint, semantics}``,
and the entry lives at ``entry_relpath(cache_id)``.

**Identity vs. policy/root.** ``GroundTruthDEGCacheRequest`` carries ONLY the two things
that determine a cache id: ``fingerprint`` and ``semantics``. It does not carry ``root`` or
``policy`` -- those are owned exclusively by ``GroundTruthDEGCache`` (see ``store.py``), so
there is exactly one place that decides whether an entry may be computed/written and exactly
one place that decides where the cache lives. A caller cannot construct a "readwrite" request
against a "readonly" cache instance and have it honoured; ``get_or_compute`` always enforces
``self.policy`` (the cache instance's own policy), never anything carried on the request.

**Cross-repository oracle.** ``tests/fixtures/ground_truth_cache/gtc_reference.py`` is
the stdlib-only Phase-0 reference implementation of this contract, with golden vectors in
``golden_vectors.json``. Every function here must reproduce it byte for byte -- same
domain tag, same canonical-JSON rules, same digest algorithm, same document keys/order,
same cache ID. An archive producer that records the fingerprint must reproduce the same
oracle independently; this module and that producer are two implementations of one contract,
not a client/server pair.

**Import weight.** This module is stdlib + numpy only (the archive opener,
``cell_eval2._cell_archive.open_cell_store``, is imported lazily inside
``fingerprint_ground_truth_cell_archive``): no ``polars``, no ``pyarrow``, no ``fcntl``. Both
``cell_eval2.__init__`` (which re-exports ``fingerprint_ground_truth_cell_archive``) and any
caller that only wants to fingerprint an archive or resolve semantics can import this module
(directly, or via the ``cell_eval2.ground_truth_cache`` package) on a platform without POSIX
advisory locks, e.g. Windows -- see ``store.py`` for where ``fcntl`` actually gets imported
(lazily, only when a write/lock is attempted).

Integrity validation (entry-file checks, Parquet-metadata checks, ``inspect_entry``) is
``integrity.py``. Storage (the ``GroundTruthDEGCache`` class -- flock, staging, atomic
publish) is ``store.py``. Both build against the identity contract defined here.
"""
from __future__ import annotations

import importlib.metadata as _importlib_metadata
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────────────

CACHE_FORMAT = "cell-eval2-ground-truth-deg"
CACHE_FORMAT_VERSION = 1
REAL_DE_SEMANTICS_VERSION = 1

#: Domain tag for the length-prefixed ordered-string digest (§8.3).
DOMAIN_SEQUENCE = b"cell-eval2-string-sequence-v1"
_NUL = bytes([0])

FINGERPRINT_ALGORITHM = "cell-eval2-ground-truth-cellstream-v1"
ARCHIVE_FORMAT = "cellstream-cell-archive"
CACHE_KEY_ALGORITHM = "cell-eval2-real-deg-cache-key-v1"

#: Exact key set (and order) of the §9.1 semantic document, version 1. A builder must
#: emit precisely these keys -- tests enforce equality, not subset.
SEMANTIC_DOCUMENT_KEYS_V1: tuple[str, ...] = (
    "cache_format_version",
    "real_de_semantics_version",
    "cell_eval2_version",
    "gpudge_version",
    "cellstream_decoder_version",
    "de_backend",
    "config_version",
    "input_type",
    "target_sum",
    "pert_col",
    "control",
    "mean_calc",
    "epsilon",
    "clip_value",
    "fdr_scope",
    "filter_gene_min_cpm_cell",
    "validate_input",
    "max_counts_per_cell",
    "allow_fractional_real",
    "native_gpu_normalize",
    "device",
    "precision",
)

#: Exact key set (and order) of the §8.4 fingerprint document, version 1.
FINGERPRINT_DOCUMENT_KEYS_V1: tuple[str, ...] = (
    "fingerprint_algorithm",
    "archive_format",
    "payload_sha256",
    "n_obs",
    "n_vars",
    "feature_axis_sha256",
    "perturbation_axis_sha256",
    "pert_col",
)

CACHE_POLICIES: tuple[str, ...] = ("readwrite", "readonly")
CACHE_STATUSES: tuple[str, ...] = ("hit", "hit_after_wait", "miss_computed")

#: A staging directory older than this (seconds) is considered abandoned by a dead
#: writer and eligible for cleanup by the storage half (``store.py``).
STALE_STAGING_AGE_S = 86_400

ENTRY_TABLE_NAME = "de_real.parquet"
ENTRY_MANIFEST_NAME = "manifest.json"

# \A...\Z, not ^...$: `$` also matches just before a trailing newline.
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


# ── Errors ───────────────────────────────────────────────────────────────────────────


class GroundTruthCacheError(Exception):
    """Base for every ground-truth cache failure."""


class GroundTruthContractError(GroundTruthCacheError, ValueError):
    """Canonical-JSON / document contract violation (= ``gtc_reference.ContractError``)."""


class GroundTruthFingerprintError(GroundTruthCacheError):
    """Archive lacks a strong identity (no ``payload_sha256``) or has unusable axes."""


class GroundTruthCacheMissError(GroundTruthCacheError):
    """``readonly`` policy miss: no committed entry, and this caller may not compute one."""


class GroundTruthCacheCorruptError(GroundTruthCacheError):
    """A committed entry failed validation on read."""


class GroundTruthCacheFilesystemError(GroundTruthCacheError):
    """Lock / atomic-rename / permission failure while reading or writing an entry."""


# ── Canonical JSON + digests (§8.3 / §8.4) ──────────────────────────────────────────


def _check_json_value(value: Any, path: str = "$") -> None:
    """Reject anything ``json.dumps`` would coerce ambiguously: non-str keys, non-finite
    floats, and non-JSON types (this module never uses ``json.dumps(default=...)``)."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GroundTruthContractError(f"non-finite number at {path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise GroundTruthContractError(f"non-string key at {path}: {k!r}")
            _check_json_value(v, f"{path}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_json_value(v, f"{path}[{i}]")
        return
    raise GroundTruthContractError(f"unsupported JSON type at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON (§8.4): sorted keys, compact separators, Unicode preserved, no
    NaN/inf. Floats serialize via Python's shortest round-trip repr (``1e6 ->
    1000000.0``), so callers must normalise numeric *types* (int vs float) before
    hashing -- this function does not."""
    _check_json_value(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def document_sha256(value: Any) -> str:
    """Lowercase-hex SHA-256 of ``canonical_json_bytes(value)``."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sequence_sha256(values: Sequence[str], *, domain: bytes = DOMAIN_SEQUENCE) -> str:
    """Length-prefixed ordered-string digest (§8.3):
    ``sha256(domain || 0x00 || u64be(count) || (u64be(len(utf8)) || utf8)*)``.
    Only ``str`` elements are accepted -- no None/bytes/int coercion."""
    items = list(values)
    h = hashlib.sha256()
    h.update(domain)
    h.update(_NUL)
    h.update(len(items).to_bytes(8, "big"))
    for i, v in enumerate(items):
        if not isinstance(v, str):
            raise GroundTruthContractError(
                f"sequence element {i} is {type(v).__name__}, not str; identifiers must be "
                "strings (no None/bytes/int coercion)"
            )
        b = v.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def validate_fingerprint_document(doc: Mapping) -> None:
    """Exact-key-set + type check for a §8.4 fingerprint document (version 1)."""
    if sorted(doc.keys()) != sorted(FINGERPRINT_DOCUMENT_KEYS_V1):
        raise GroundTruthContractError(
            f"fingerprint document keys {sorted(doc.keys())} != "
            f"{sorted(FINGERPRINT_DOCUMENT_KEYS_V1)}"
        )
    if (doc["fingerprint_algorithm"] != FINGERPRINT_ALGORITHM
            or doc["archive_format"] != ARCHIVE_FORMAT):
        raise GroundTruthContractError("fingerprint algorithm/format tags do not match version 1")
    for k in ("payload_sha256", "feature_axis_sha256", "perturbation_axis_sha256"):
        if not isinstance(doc[k], str) or not _HEX64.match(doc[k]):
            raise GroundTruthContractError(f"{k} must be 64 lowercase hex chars")
    for k in ("n_obs", "n_vars"):
        if isinstance(doc[k], bool) or not isinstance(doc[k], int) or doc[k] < 0:
            raise GroundTruthContractError(f"{k} must be a non-negative int")
    if not isinstance(doc["pert_col"], str) or not doc["pert_col"]:
        raise GroundTruthContractError("pert_col must be a non-empty str")


def validate_semantic_document(doc: Mapping) -> None:
    """Exact-key-set + type check for a §9.1 semantic document (version 1)."""
    keys = tuple(doc.keys())
    if set(keys) != set(SEMANTIC_DOCUMENT_KEYS_V1):
        missing = sorted(set(SEMANTIC_DOCUMENT_KEYS_V1) - set(keys))
        extra = sorted(set(keys) - set(SEMANTIC_DOCUMENT_KEYS_V1))
        raise GroundTruthContractError(
            f"semantic document key mismatch: missing={missing} extra={extra}"
        )
    # type() is int, not ==: True == 1, and a JSON `true` is no version.
    if type(doc["cache_format_version"]) is not int or doc["cache_format_version"] != CACHE_FORMAT_VERSION:
        raise GroundTruthContractError("cache_format_version must be 1")
    if (type(doc["real_de_semantics_version"]) is not int
            or doc["real_de_semantics_version"] != REAL_DE_SEMANTICS_VERSION):
        raise GroundTruthContractError("real_de_semantics_version must be 1")
    for k in ("cell_eval2_version", "gpudge_version", "cellstream_decoder_version", "de_backend",
              "config_version", "input_type", "pert_col", "control", "mean_calc", "fdr_scope",
              "device", "precision"):
        if not isinstance(doc[k], str) or not doc[k]:
            raise GroundTruthContractError(f"{k} must be a non-empty str, got {doc[k]!r}")
    for k in ("validate_input", "allow_fractional_real", "native_gpu_normalize"):
        if not isinstance(doc[k], bool):
            raise GroundTruthContractError(f"{k} must be bool, got {doc[k]!r}")
    for k in ("epsilon", "max_counts_per_cell"):
        v = doc[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise GroundTruthContractError(f"{k} must be a finite number, got {v!r}")
    for k in ("target_sum", "clip_value", "filter_gene_min_cpm_cell"):
        v = doc[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v)):
            raise GroundTruthContractError(f"{k} must be a finite number or null, got {v!r}")


def cache_key_document(fingerprint_doc: Mapping, semantics_doc: Mapping) -> dict:
    """The §9.3 key document. Both sub-documents are validated first."""
    validate_fingerprint_document(fingerprint_doc)
    validate_semantic_document(semantics_doc)
    return {
        "cache_key_algorithm": CACHE_KEY_ALGORITHM,
        "ground_truth_fingerprint": dict(fingerprint_doc),
        "de_semantics": dict(semantics_doc),
    }


def cache_id(fingerprint_doc: Mapping, semantics_doc: Mapping) -> str:
    """Lowercase-hex SHA-256 cache ID (§9.3)."""
    return document_sha256(cache_key_document(fingerprint_doc, semantics_doc))


def entry_relpath(cache_id_hex: str) -> str:
    """§10.1 layout: ``ground-truth-deg/v1/<2-char prefix>/<cache-id>``. Rejects anything
    that is not exactly 64 lowercase hex characters (path-separator / traversal safety)."""
    if not isinstance(cache_id_hex, str) or not _HEX64.match(cache_id_hex):
        raise GroundTruthContractError(f"cache id must be 64 lowercase hex chars, got {cache_id_hex!r}")
    return f"ground-truth-deg/v{CACHE_FORMAT_VERSION}/{cache_id_hex[:2]}/{cache_id_hex}"


def _check_document_digest(doc: Mapping, sha256: Any, kind: str) -> None:
    """A digest dataclass holds a document plus ITS digest: refuse any other ``sha256``, which
    would otherwise reach the result and the sidecar while lookups keyed on the document."""
    if sha256 != document_sha256(doc):
        raise GroundTruthContractError(
            f"{kind} digest {sha256!r} is not the SHA-256 of its document "
            f"({document_sha256(doc)})")


# ── Dataclasses ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GroundTruthFingerprint:
    """A validated §8.4 fingerprint document plus its digest."""

    document: Mapping[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        doc = dict(self.document)
        validate_fingerprint_document(doc)
        _check_document_digest(doc, self.sha256, "fingerprint")
        object.__setattr__(self, "document", doc)

    @classmethod
    def from_axes(
        cls,
        *,
        payload_sha256: str,
        var_names: Sequence[str],
        pert_labels: Sequence[str],
        pert_col: str,
    ) -> "GroundTruthFingerprint":
        """Build the fingerprint document from an archive's already-extracted axes.

        ``n_obs``/``n_vars`` are derived from the axes, never passed separately, so they
        cannot disagree with the digests. Mirrors
        ``gtc_reference.ground_truth_fingerprint_document`` exactly.
        """
        if not isinstance(payload_sha256, str) or not _HEX64.match(payload_sha256):
            raise GroundTruthContractError(
                "payload_sha256 must be 64 lowercase hex characters (a strong Cellstream "
                f"payload checksum written with payload_checksum=True); got {payload_sha256!r}"
            )
        if not isinstance(pert_col, str) or not pert_col:
            raise GroundTruthContractError(f"pert_col must be a non-empty str, got {pert_col!r}")
        var_names = list(var_names)
        pert_labels = list(pert_labels)
        document = {
            "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "archive_format": ARCHIVE_FORMAT,
            "payload_sha256": payload_sha256,
            "n_obs": len(pert_labels),
            "n_vars": len(var_names),
            "feature_axis_sha256": sequence_sha256(var_names),
            "perturbation_axis_sha256": sequence_sha256(pert_labels),
            "pert_col": pert_col,
        }
        return cls(document=document, sha256=document_sha256(document))


@dataclass(frozen=True)
class RealDESemantics:
    """A validated §9.1 semantic document plus its digest."""

    document: Mapping[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        doc = dict(self.document)
        validate_semantic_document(doc)
        _check_document_digest(doc, self.sha256, "semantic")
        object.__setattr__(self, "document", doc)


@dataclass(frozen=True)
class GroundTruthDEGCacheRequest:
    """Everything needed to IDENTIFY one cache entry -- not how to read or write it.

    Identity-only: ``fingerprint`` x ``semantics`` determine the ``cache_id`` (and
    therefore ``entry_relpath``). The cache ROOT and the read/write POLICY are owned
    exclusively by ``GroundTruthDEGCache`` (``store.py``) -- a request never carries its
    own opinion about where the cache lives or whether this call may write. (Earlier
    revisions duplicated ``root``/``policy`` here, which let a caller build a
    ``policy="readwrite"`` request against a ``readonly`` cache instance; only the cache
    instance's own policy is ever consulted now.)
    """

    fingerprint: GroundTruthFingerprint
    semantics: RealDESemantics

    @property
    def cache_id(self) -> str:
        return cache_id(self.fingerprint.document, self.semantics.document)

    @property
    def entry_relpath(self) -> str:
        return entry_relpath(self.cache_id)


@dataclass(frozen=True)
class GroundTruthDEGCacheResult:
    """Outcome of one cache lookup/populate call, and the provenance needed to audit it."""

    cache_id: str | None
    status: str
    policy: str
    entry_dir: str | None
    table_path: str | None
    manifest: Mapping[str, Any] | None
    de_real_sha256: str | None
    de_real_rows: int | None
    ground_truth_fingerprint_sha256: str | None
    de_semantics_sha256: str | None
    fingerprint_seconds: float | None = None
    lock_wait_seconds: float = 0.0
    validate_seconds: float | None = None
    compute_seconds: float | None = None
    write_seconds: float | None = None

    def to_sidecar_dict(self, *, producer: Mapping[str, Any]) -> dict:
        """The §12.5 sidecar schema written next to a scored artifact."""
        return {
            "schema_version": 1,
            "status": self.status,
            "policy": self.policy,
            "cache_id": self.cache_id,
            "cache_entry": self.entry_dir,
            "ground_truth_fingerprint_sha256": self.ground_truth_fingerprint_sha256,
            "de_semantics_sha256": self.de_semantics_sha256,
            "de_real_sha256": self.de_real_sha256,
            "de_real_rows": self.de_real_rows,
            "lock_wait_seconds": self.lock_wait_seconds,
            "compute_seconds": self.compute_seconds,
            "producer": dict(producer),
        }


# ── Fingerprinting a cell archive ───────────────────────────────────────────────────


def _archive_label(store_or_path, store) -> str:
    if isinstance(store_or_path, (str, os.PathLike)):
        return os.fspath(store_or_path)
    return str(getattr(store, "path", repr(store)))


def fingerprint_ground_truth_cell_archive(store_or_path, *, pert_col: str) -> GroundTruthFingerprint:
    """Fingerprint a Cellstream cell-layout archive for the ground-truth cache.

    ``store_or_path`` is either an already-open cell store (duck-typed: it needs only
    ``.manifest``, ``.var``, ``.obs``) or a ``str | os.PathLike`` path, in which case this
    function opens it via ``_cell_archive.open_cell_store`` and closes it again --
    including on error -- before returning. An already-open store is left open and
    untouched: the caller keeps ownership.

    Requires a strong content checksum: ``store.manifest["payload_sha256"]`` must be a
    64-lowercase-hex string, written only with ``payload_checksum=True`` (cellstream /
    shardad writers). The ground-truth cache NEVER falls back to a structural
    fingerprint (contrast ``cell_source.cell_fingerprint``'s old structural fallback) --
    an archive without one raises ``GroundTruthFingerprintError``.

    ``var_names`` come from ``store.var.index`` and must all already be ``str`` (no
    None/NaN, no silent coercion). ``pert_labels`` come from ``store.obs[pert_col]`` in
    the archive's storage (row) order and must all already be non-null ``str``. Both
    violations raise ``GroundTruthContractError``; a missing ``pert_col`` column raises
    ``GroundTruthFingerprintError`` listing the columns that ARE present.
    """
    opened_here = isinstance(store_or_path, (str, os.PathLike))
    if opened_here:
        from .._cell_archive import open_cell_store

        store = open_cell_store(store_or_path)
    else:
        store = store_or_path
    try:
        label = _archive_label(store_or_path, store)

        payload_sha256 = store.manifest.get("payload_sha256")
        if not isinstance(payload_sha256, str) or not _HEX64.match(payload_sha256):
            raise GroundTruthFingerprintError(
                f"{label}: archive has no strong payload checksum "
                f"(payload_sha256={payload_sha256!r}). The ground-truth cache never falls "
                "back to a structural fingerprint -- regenerate or concatenate the archive "
                "with payload_checksum=True (cellstream/shardad writers)."
            )

        var_names: list[str] = []
        for v in store.var.index:
            if not isinstance(v, str):
                raise GroundTruthContractError(
                    f"{label}: feature axis (var.index) contains {v!r} "
                    f"({type(v).__name__}), not a str; every feature name must already be a "
                    "string -- the ground-truth fingerprint does not coerce it"
                )
            var_names.append(v)

        try:
            raw_labels = list(store.obs[pert_col])
        except KeyError:
            raise GroundTruthFingerprintError(
                f"{label}: pert_col {pert_col!r} not found in archive obs; available "
                f"columns: {list(getattr(store.obs, 'columns', []))}"
            ) from None
        pert_labels: list[str] = []
        for v in raw_labels:
            if not isinstance(v, str):
                raise GroundTruthContractError(
                    f"{label}: obs[{pert_col!r}] contains a null or non-string value "
                    f"{v!r} ({type(v).__name__}); perturbation labels must all be "
                    "non-null strings, taken in storage order as-is"
                )
            pert_labels.append(v)

        return GroundTruthFingerprint.from_axes(
            payload_sha256=payload_sha256,
            var_names=var_names,
            pert_labels=pert_labels,
            pert_col=pert_col,
        )
    finally:
        if opened_here:
            store.close()


# ── Precision resolution ────────────────────────────────────────────────────────────


def resolve_de_precision(
    x_dtype, *, input_type: str, native_gpu_normalize: bool, target_sum
) -> str:
    """The dtype in which the DE backend consumes expression values (approved 2026-09-16).

    Mirrors ``de_compute._to_linear`` (module lines ~194-235) and the gpudge
    native-normalize branch of ``de_compute.compute_de`` (~lines 725-760):

    * ``input_type="counts"``: when ``native_gpu_normalize`` is set, or when
      ``target_sum == 1e6`` (gpudge's native-CPM path, ``compute_de``'s
      ``native_cpm = is_counts and target_sum == 1e6`` short-circuit), gpudge is handed
      the archive's raw matrix untouched, so the resolved precision is the archive's own
      dtype (``np.dtype(x_dtype).name``). Otherwise ``_to_linear``'s counts branch does
      ``out.X.astype(float)``, which is float64.
    * ``input_type="lognorm"``: ``_to_linear``'s lognorm branch preserves float32 input
      (``dt = np.float32 if X.dtype == np.float32 else np.float64``) rather than
      upcasting; anything else becomes float64.

    On the ``score_cellstream`` path (float32 cell-layout archives,
    ``native_gpu_normalize=True``), this resolves to ``"float32"``.
    """
    dt = np.dtype(x_dtype)
    if input_type == "counts":
        if native_gpu_normalize or target_sum == 1e6:
            return dt.name
        return "float64"
    if input_type == "lognorm":
        return "float32" if dt == np.float32 else "float64"
    raise ValueError(f"input_type must be 'counts' or 'lognorm', got {input_type!r}")


# ── Producer versions ────────────────────────────────────────────────────────────────


def producer_versions(
    *, require: Sequence[str] = ("cell-eval2", "gpudge", "cellstream")
) -> dict[str, str]:
    """Installed-distribution versions for the semantic document's provenance keys.

    Uses ``importlib.metadata.version`` (the same lookup pattern as
    ``baseline._package_provenance``). For ``"cellstream"``, falls back to the
    ``"shardad"`` distribution when ``cellstream`` itself is not installed (pre-rename
    images) -- but the version string is always recorded under the ``"cellstream"`` key,
    since that is what ``build_real_de_semantics`` reads for
    ``cellstream_decoder_version``. A missing required distribution is a hard error: the
    cache cannot key an entry without knowing what produced it.

    ``"polars"`` and ``"pyarrow"`` are additionally recorded, when installed, for
    manifest provenance (the storage half writes Parquet); their absence is not an
    error.
    """
    versions: dict[str, str] = {}
    for name in require:
        try:
            versions[name] = _importlib_metadata.version(name)
            continue
        except _importlib_metadata.PackageNotFoundError:
            pass
        if name == "cellstream":
            try:
                versions[name] = _importlib_metadata.version("shardad")
                continue
            except _importlib_metadata.PackageNotFoundError:
                pass
        raise GroundTruthContractError(
            f"the ground-truth cache cannot key an entry without the installed {name!r} "
            "distribution's version, and it is not installed"
            + (" (nor is 'shardad', its pre-rename name)" if name == "cellstream" else "")
        )
    for optional in ("polars", "pyarrow"):
        try:
            versions[optional] = _importlib_metadata.version(optional)
        except _importlib_metadata.PackageNotFoundError:
            pass
    return versions


def _resolve_producer_versions() -> dict[str, str]:
    """Indirection used by ``build_real_de_semantics`` when the caller passes no explicit
    ``versions=`` -- looked up through the PACKAGE namespace (``cell_eval2.ground_truth_cache``,
    not this submodule) at call time, so a test's
    ``monkeypatch.setattr(cell_eval2.ground_truth_cache, "producer_versions", ...)`` (the
    supported patch point, unchanged since before the identity/integrity/store split) is
    honoured here even though ``producer_versions`` is defined in this submodule."""
    from cell_eval2 import ground_truth_cache as _pkg

    return _pkg.producer_versions()


# ── Semantic document builder ───────────────────────────────────────────────────────


def build_real_de_semantics(
    cfg,
    *,
    control: str,
    input_type: str,
    target_sum,
    device: str,
    de_backend: str,
    precision: str,
    native_gpu_normalize: bool,
    versions: Mapping[str, str] | None = None,
) -> RealDESemantics:
    """Build the §9.1 semantic document -- the ONLY place semantic keys are produced.

    An explicit allowlist (docs/ground_truth_cache.md §5.2,
    ``tests/fixtures/ground_truth_cache/eval_config_classification.yaml``), not
    ``cfg.to_dict()``: most ``EvalConfig`` fields cannot change ``de_real.parquet`` and
    must never enter the cache key.

    ``cfg`` is the run's ``EvalConfig``, but its ``input_type``/``target_sum`` fields may
    NOT yet be the resolved ones the real control block actually used (e.g. v1's
    autodetection, or a data-derived median target sum) -- that is why ``input_type``,
    ``target_sum``, ``control``, ``device`` and ``de_backend`` are passed explicitly,
    already resolved by the caller. ``pert_col`` is read from ``cfg.pert_col`` directly:
    by convention the caller has already set it to the resolved column name (e.g.
    ``score_cellstream``'s ``"target"`` -> ``"perturbation"`` remap).
    """
    if cfg.de.replicate_col is not None:
        raise GroundTruthContractError(
            "de.replicate_col must be None for the ground-truth cache: it is deseq2-only "
            "(pseudobulk replicates), and the deseq2 backend never runs on the cellstream "
            f"real-DE path; got {cfg.de.replicate_col!r}"
        )
    v = dict(versions) if versions is not None else _resolve_producer_versions()
    resolved_target_sum = float(target_sum) if target_sum is not None else None
    resolved_clip_value = float(cfg.de.clip_value) if cfg.de.clip_value is not None else None
    resolved_filter_cpm = (
        float(cfg.filter.filter_gene_min_cpm_cell)
        if cfg.filter.filter_gene_min_cpm_cell is not None else None
    )
    document = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "real_de_semantics_version": REAL_DE_SEMANTICS_VERSION,
        "cell_eval2_version": str(v["cell-eval2"]),
        "gpudge_version": str(v["gpudge"]),
        "cellstream_decoder_version": str(v["cellstream"]),
        "de_backend": str(de_backend),
        "config_version": cfg.version,
        "input_type": str(input_type),
        "target_sum": resolved_target_sum,
        "pert_col": cfg.pert_col,
        "control": str(control),
        "mean_calc": cfg.de.mean_calc,
        "epsilon": float(cfg.de.epsilon),
        "clip_value": resolved_clip_value,
        "fdr_scope": cfg.de.fdr_scope,
        "filter_gene_min_cpm_cell": resolved_filter_cpm,
        "validate_input": bool(cfg.validate_input),
        "max_counts_per_cell": float(cfg.max_counts_per_cell),
        "allow_fractional_real": False,
        "native_gpu_normalize": bool(native_gpu_normalize),
        "device": str(device),
        "precision": str(precision),
    }
    validate_semantic_document(document)
    return RealDESemantics(document=document, sha256=document_sha256(document))
