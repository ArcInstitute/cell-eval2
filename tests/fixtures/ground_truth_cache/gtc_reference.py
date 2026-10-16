"""Reference implementation of the ground-truth DEG cache identity contract.

This is the Phase 0 contract kit for the ground-truth cache (docs/ground_truth_cache.md;
the section numbers here are the original design spec's: §8.3 canonical sequence digest,
§8.4 fingerprint document, §9.1 semantic document, §9.3 cache ID). It is deliberately
**stdlib-only** so every implementation can vendor or import it and check the same golden
vectors (``golden_vectors.json`` next to this file).

``cell_eval2.ground_truth_cache`` implements these vectors, and any producer that records
the fingerprint must reproduce them byte for byte.

Contract summary
----------------
* ``canonical_json_bytes``: UTF-8 bytes of JSON with keys sorted by code point,
  no insignificant whitespace, Unicode preserved (``ensure_ascii=False``), and
  non-finite numbers rejected. Dict keys must be ``str``. Floats serialize with
  Python's shortest round-trip repr (``1e6 -> 1000000.0``, ``1e-9 -> 1e-09``),
  so a producer must normalise numeric *types* (int vs float) before hashing;
  the semantic builder is responsible for that, not this function.
* ``sequence_sha256``: SHA-256 over ``DOMAIN_SEQUENCE`` + one NUL byte, then the
  element count as u64 big-endian, then per element the UTF-8 byte length as
  u64 big-endian followed by the bytes. Only ``str`` elements are accepted.
* ``ground_truth_fingerprint_document`` / ``document_sha256``: the §8.4
  document and its digest (SHA-256 of the canonical JSON bytes).
* ``cache_key_document`` / ``cache_id``: the §9.3 key document and the
  lowercase-hex SHA-256 cache ID.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence

DOMAIN_SEQUENCE = b"cell-eval2-string-sequence-v1"
NUL = bytes([0])
FINGERPRINT_ALGORITHM = "cell-eval2-ground-truth-cellstream-v1"
ARCHIVE_FORMAT = "cellstream-cell-archive"
CACHE_KEY_ALGORITHM = "cell-eval2-real-deg-cache-key-v1"
CACHE_FORMAT_VERSION = 1
REAL_DE_SEMANTICS_VERSION = 1

# \A...\Z, not ^...$: `$` also matches just before a trailing newline.
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Exact key set of the §9.1 semantic document, version 1. A builder must emit
#: precisely these keys (tests enforce equality, not subset).
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

#: Exact key set of the §8.4 fingerprint document, version 1.
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


class ContractError(ValueError):
    """A value violates the identity contract (ambiguous type, non-finite number, ...)."""


def _check_json_value(value, path: str = "$") -> None:
    """Reject anything json.dumps would coerce ambiguously: non-str keys, non-finite floats,
    and non-JSON types (json.dumps(default=...) is never used here)."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite number at {path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ContractError(f"non-string key at {path}: {k!r}")
            _check_json_value(v, f"{path}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_json_value(v, f"{path}[{i}]")
        return
    raise ContractError(f"unsupported JSON type at {path}: {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    """Canonical JSON (§8.4): sorted keys, compact separators, Unicode preserved, no NaN/inf."""
    _check_json_value(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def document_sha256(value) -> str:
    """Lowercase-hex SHA-256 of ``canonical_json_bytes(value)``."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sequence_sha256(values: Iterable[str], *, domain: bytes = DOMAIN_SEQUENCE) -> str:
    """Length-prefixed ordered-string digest (§8.3)."""
    items = list(values)
    h = hashlib.sha256()
    h.update(domain)
    h.update(NUL)
    h.update(len(items).to_bytes(8, "big"))
    for i, v in enumerate(items):
        if not isinstance(v, str):
            raise ContractError(
                f"sequence element {i} is {type(v).__name__}, not str; identifiers must be "
                "strings (no None/bytes/int coercion)"
            )
        b = v.encode("utf-8")  # raises UnicodeEncodeError on lone surrogates
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def ground_truth_fingerprint_document(
    *,
    payload_sha256: str,
    var_names: Sequence[str],
    pert_labels: Sequence[str],
    pert_col: str,
) -> dict:
    """The §8.4 fingerprint document for a cell archive.

    ``var_names`` is the archive's ordered feature axis; ``pert_labels`` is
    ``obs[pert_col]`` in the archive's **storage (row) order**. ``n_obs`` /
    ``n_vars`` are derived from the axes, never passed separately, so they
    cannot disagree with the digests.
    """
    if not isinstance(payload_sha256, str) or not _HEX64.match(payload_sha256):
        raise ContractError(
            "payload_sha256 must be 64 lowercase hex characters (a strong Cellstream payload "
            f"checksum written with payload_checksum=True); got {payload_sha256!r}"
        )
    if not isinstance(pert_col, str) or not pert_col:
        raise ContractError(f"pert_col must be a non-empty str, got {pert_col!r}")
    var_names = list(var_names)
    pert_labels = list(pert_labels)
    return {
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "archive_format": ARCHIVE_FORMAT,
        "payload_sha256": payload_sha256,
        "n_obs": len(pert_labels),
        "n_vars": len(var_names),
        "feature_axis_sha256": sequence_sha256(var_names),
        "perturbation_axis_sha256": sequence_sha256(pert_labels),
        "pert_col": pert_col,
    }


def validate_semantic_document(doc: Mapping) -> None:
    """Exact-key-set + type check for a §9.1 document (version 1)."""
    keys = tuple(doc.keys())
    if set(keys) != set(SEMANTIC_DOCUMENT_KEYS_V1):
        missing = sorted(set(SEMANTIC_DOCUMENT_KEYS_V1) - set(keys))
        extra = sorted(set(keys) - set(SEMANTIC_DOCUMENT_KEYS_V1))
        raise ContractError(f"semantic document key mismatch: missing={missing} extra={extra}")
    # type() is int, not ==: True == 1, and a JSON `true` is no version.
    if type(doc["cache_format_version"]) is not int or doc["cache_format_version"] != CACHE_FORMAT_VERSION:
        raise ContractError("cache_format_version must be 1")
    if (type(doc["real_de_semantics_version"]) is not int
            or doc["real_de_semantics_version"] != REAL_DE_SEMANTICS_VERSION):
        raise ContractError("real_de_semantics_version must be 1")
    for k in ("cell_eval2_version", "gpudge_version", "cellstream_decoder_version", "de_backend",
              "config_version", "input_type", "pert_col", "control", "mean_calc", "fdr_scope",
              "device", "precision"):
        if not isinstance(doc[k], str) or not doc[k]:
            raise ContractError(f"{k} must be a non-empty str, got {doc[k]!r}")
    for k in ("validate_input", "allow_fractional_real", "native_gpu_normalize"):
        if not isinstance(doc[k], bool):
            raise ContractError(f"{k} must be bool, got {doc[k]!r}")
    for k in ("epsilon", "max_counts_per_cell"):
        v = doc[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ContractError(f"{k} must be a finite number, got {v!r}")
    for k in ("target_sum", "clip_value", "filter_gene_min_cpm_cell"):
        v = doc[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise ContractError(f"{k} must be a finite number or null, got {v!r}")


def validate_fingerprint_document(doc: Mapping) -> None:
    if sorted(doc.keys()) != sorted(FINGERPRINT_DOCUMENT_KEYS_V1):
        raise ContractError(
            f"fingerprint document keys {sorted(doc.keys())} != {sorted(FINGERPRINT_DOCUMENT_KEYS_V1)}"
        )
    if doc["fingerprint_algorithm"] != FINGERPRINT_ALGORITHM or doc["archive_format"] != ARCHIVE_FORMAT:
        raise ContractError("fingerprint algorithm/format tags do not match version 1")
    for k in ("payload_sha256", "feature_axis_sha256", "perturbation_axis_sha256"):
        if not isinstance(doc[k], str) or not _HEX64.match(doc[k]):
            raise ContractError(f"{k} must be 64 lowercase hex chars")
    for k in ("n_obs", "n_vars"):
        if isinstance(doc[k], bool) or not isinstance(doc[k], int) or doc[k] < 0:
            raise ContractError(f"{k} must be a non-negative int")
    if not isinstance(doc["pert_col"], str) or not doc["pert_col"]:
        raise ContractError("pert_col must be a non-empty str")


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
        raise ContractError(f"cache id must be 64 lowercase hex chars, got {cache_id_hex!r}")
    return f"ground-truth-deg/v{CACHE_FORMAT_VERSION}/{cache_id_hex[:2]}/{cache_id_hex}"
