#!/usr/bin/env python
"""Regenerate ``golden_vectors.json`` from ``gtc_reference.py`` (+ ``fixture_manifest.json``).

Stdlib only. The vectors pin: canonical JSON bytes, the length-prefixed sequence digest,
ground-truth fingerprint documents/digests (synthetic + the frozen fixture archives), an
illustrative resolved semantic document/digest, and the resulting cache ID + entry path.
Any implementation must reproduce every ``expected`` value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gtc_reference as gtc  # noqa: E402


def canonical_json_cases() -> list[dict]:
    inputs = [
        {"b": 1, "a": [1, 2, {"z": None, "y": True}]},
        {"ünï": "cødé ∆ 細胞", "n": 1e6, "e": 1e-9, "i": 10000, "f": 10000.0, "neg": -0.0},
        {"a": {}, "b": [], "c": "", "d": False},
        {"nested": {"k2": {"k1": [3, 2, 1]}, "k1": "x"}, "": "empty-key"},
        {"quote": 'a"b\\c' + "\n\t/"},
    ]
    out = []
    for obj in inputs:
        b = gtc.canonical_json_bytes(obj)
        out.append({"input": obj, "expected_utf8": b.decode("utf-8"),
                    "expected_sha256": gtc.document_sha256(obj)})
    return out


def sequence_cases() -> list[dict]:
    seqs = [
        [],
        ["a"],
        ["a", "b"],
        ["b", "a"],
        ["ab"],
        ["a" + chr(0) + "b"],
        ["a,b"],
        ["", ""],
        ["non-targeting", "TG01", "TG01", "TG02"],
        ["gène", "β-actin", "ENSG00000000003"],
    ]
    return [{"input": s, "expected_sha256": gtc.sequence_sha256(s)} for s in seqs]


def synthetic_fingerprint() -> dict:
    sha = "0123456789abcdef" * 4
    doc = gtc.ground_truth_fingerprint_document(
        payload_sha256=sha,
        var_names=["g0", "g1", "g2"],
        pert_labels=["ctrl", "ctrl", "A", "B"],
        pert_col="perturbation",
    )
    return {"input": {"payload_sha256": sha, "var_names": ["g0", "g1", "g2"],
                      "pert_labels": ["ctrl", "ctrl", "A", "B"], "pert_col": "perturbation"},
            "expected_document": doc, "expected_sha256": gtc.document_sha256(doc)}


def illustrative_semantics() -> dict:
    """A fully resolved §9.1 document for the production configuration as of Phase 0:
    preset cell-eval-0.7.6 (version=v1), gpudge backend on CUDA, cellstream path
    (native_gpu_normalize is True there), lognorm real archive (target_sum resolves to null)."""
    return {
        "cache_format_version": 1,
        "real_de_semantics_version": 1,
        "cell_eval2_version": "0.17.0",
        "gpudge_version": "0.9.1",
        "cellstream_decoder_version": "0.10.0",
        "de_backend": "gpudge",
        "config_version": "v1",
        "input_type": "lognorm",
        "target_sum": None,
        "pert_col": "perturbation",
        "control": "non-targeting",
        "mean_calc": "geometric",
        "epsilon": 1e-09,
        "clip_value": None,
        "fdr_scope": "per_pert",
        "filter_gene_min_cpm_cell": 5.0,
        "validate_input": True,
        "max_counts_per_cell": 1000000000.0,
        "allow_fractional_real": False,
        "native_gpu_normalize": True,
        "device": "cuda",
        "precision": "float32",
    }


def main() -> None:
    fixture = json.loads((HERE / "fixture_manifest.json").read_text(encoding="utf-8"))
    sem = illustrative_semantics()
    gtc.validate_semantic_document(sem)
    syn = synthetic_fingerprint()
    cid = gtc.cache_id(syn["expected_document"], sem)

    # A second semantics document differing in ONE included field must change the cache ID.
    sem_cpm0 = dict(sem, filter_gene_min_cpm_cell=None)
    cid_cpm0 = gtc.cache_id(syn["expected_document"], sem_cpm0)
    assert cid_cpm0 != cid

    fixture_fps = []
    for rec in fixture["archives"]:
        entry = {
            "file": rec["file"],
            "payload_sha256": rec["archive_manifest"]["payload_sha256"],
            "feature_axis_sha256": rec["feature_axis_sha256"],
            "perturbation_axis_sha256": rec["perturbation_axis_sha256"],
            "expected_document": rec["ground_truth_fingerprint"],
            "expected_sha256": rec["ground_truth_fingerprint_sha256"],
        }
        if rec["ground_truth_fingerprint"] is not None:
            c = gtc.cache_id(rec["ground_truth_fingerprint"], sem)
            entry["expected_cache_id_with_illustrative_semantics"] = c
            entry["expected_entry_relpath"] = gtc.entry_relpath(c)
        fixture_fps.append(entry)

    vectors = {
        "contract": {
            "domain_sequence": gtc.DOMAIN_SEQUENCE.decode("ascii"),
            "fingerprint_algorithm": gtc.FINGERPRINT_ALGORITHM,
            "archive_format": gtc.ARCHIVE_FORMAT,
            "cache_key_algorithm": gtc.CACHE_KEY_ALGORITHM,
            "cache_format_version": gtc.CACHE_FORMAT_VERSION,
            "real_de_semantics_version": gtc.REAL_DE_SEMANTICS_VERSION,
            "semantic_document_keys_v1": list(gtc.SEMANTIC_DOCUMENT_KEYS_V1),
            "fingerprint_document_keys_v1": list(gtc.FINGERPRINT_DOCUMENT_KEYS_V1),
            "canonical_json": ("keys sorted by code point; separators ',' and ':'; ensure_ascii=False; "
                               "allow_nan=False; floats via Python shortest repr"),
            "sequence_digest": "sha256(domain || 0x00 || u64be(count) || (u64be(len(utf8)) || utf8)*)",
        },
        "canonical_json": canonical_json_cases(),
        "sequence_sha256": sequence_cases(),
        "rejections": {
            "canonical_json": ["float('nan')", "float('inf')", "{1: 'non-str key'}", "bytes", "set"],
            "sequence_sha256": ["[None]", "[1]", "[b'x']", "lone surrogate str"],
            "fingerprint": ["payload_sha256=None", "payload_sha256=''", "uppercase hex", "63/65-char hex"],
        },
        "ground_truth_fingerprint": {"synthetic": syn, "fixture_archives": fixture_fps},
        "de_semantics": {
            "illustrative_production_v1": {"document": sem, "expected_sha256": gtc.document_sha256(sem)},
            "mutation_filter_gene_min_cpm_cell_null": {"document": sem_cpm0,
                                                       "expected_sha256": gtc.document_sha256(sem_cpm0)},
        },
        "cache_id": {
            "synthetic_fingerprint_x_illustrative_semantics": {
                "expected_key_document": gtc.cache_key_document(syn["expected_document"], sem),
                "expected_cache_id": cid,
                "expected_entry_relpath": gtc.entry_relpath(cid),
            },
            "synthetic_fingerprint_x_mutated_semantics": {"expected_cache_id": cid_cpm0},
        },
    }
    (HERE / "golden_vectors.json").write_text(json.dumps(vectors, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    print("wrote", HERE / "golden_vectors.json")
    print("cache_id (synthetic x illustrative):", cid)


if __name__ == "__main__":
    main()
