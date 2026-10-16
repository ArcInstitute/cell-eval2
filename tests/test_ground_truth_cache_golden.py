"""Golden-vector + frozen-fixture tests for the ground-truth DEG cache identity contract.

The cache's Phase 0 contract kit. Pins the reference implementation in
tests/fixtures/ground_truth_cache/gtc_reference.py to golden_vectors.json, and the frozen
archives to fixture_manifest.json. Everything here is stdlib + pytest; the checks that need
to open a cell archive skip when cellstream is not importable (it is not in the CI test group).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent / "fixtures" / "ground_truth_cache"


def _load_reference():
    spec = importlib.util.spec_from_file_location("gtc_reference", KIT / "gtc_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gtc = _load_reference()
VECTORS = json.loads((KIT / "golden_vectors.json").read_text(encoding="utf-8"))
FIXTURE = json.loads((KIT / "fixture_manifest.json").read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── canonical JSON ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", VECTORS["canonical_json"], ids=lambda c: c["expected_sha256"][:8])
def test_canonical_json_golden(case):
    b = gtc.canonical_json_bytes(case["input"])
    assert b.decode("utf-8") == case["expected_utf8"]
    assert gtc.document_sha256(case["input"]) == case["expected_sha256"]


def test_canonical_json_is_order_insensitive_and_compact():
    a = {"b": 1, "a": {"d": 2, "c": 3}}
    b = {"a": {"c": 3, "d": 2}, "b": 1}
    assert gtc.canonical_json_bytes(a) == gtc.canonical_json_bytes(b) == b'{"a":{"c":3,"d":2},"b":1}'


@pytest.mark.parametrize("bad", [
    {"x": float("nan")}, {"x": float("inf")}, {"x": -math.inf}, {1: "non-str key"},
    {"x": b"bytes"}, {"x": {1, 2}}, {"x": [1, {"y": float("nan")}]},
])
def test_canonical_json_rejections(bad):
    with pytest.raises(gtc.ContractError):
        gtc.canonical_json_bytes(bad)


def test_canonical_json_float_repr_contract():
    # Floats hash by Python's shortest round-trip repr; ints and floats are DISTINCT.
    assert gtc.canonical_json_bytes({"t": 1e6}) == b'{"t":1000000.0}'
    assert gtc.canonical_json_bytes({"t": 1000000}) == b'{"t":1000000}'
    assert gtc.canonical_json_bytes({"e": 1e-9}) == b'{"e":1e-09}'
    assert gtc.document_sha256({"t": 1e6}) != gtc.document_sha256({"t": 1000000})


# ── ordered-string digest ──────────────────────────────────────────────────────

@pytest.mark.parametrize("case", VECTORS["sequence_sha256"], ids=lambda c: c["expected_sha256"][:8])
def test_sequence_digest_golden(case):
    assert gtc.sequence_sha256(case["input"]) == case["expected_sha256"]


def test_sequence_digest_is_order_and_boundary_sensitive():
    d = {tuple(c["input"]): c["expected_sha256"] for c in VECTORS["sequence_sha256"]}
    assert d[("a", "b")] != d[("b", "a")]
    assert d[("a", "b")] != d[("ab",)]
    assert d[("a", "b")] != d[("a" + chr(0) + "b",)]
    assert d[("a", "b")] != d[("a,b",)]
    assert d[()] != d[("", "")]
    assert len(set(d.values())) == len(d)


@pytest.mark.parametrize("bad", [[None], [1], [b"x"], ["ok", None]])
def test_sequence_digest_rejects_non_str(bad):
    with pytest.raises(gtc.ContractError):
        gtc.sequence_sha256(bad)


def test_sequence_digest_domain_tag_matches_contract():
    assert gtc.DOMAIN_SEQUENCE.decode("ascii") == VECTORS["contract"]["domain_sequence"]
    # Manual construction of the ["a"] vector from the documented algorithm.
    h = hashlib.sha256()
    h.update(b"cell-eval2-string-sequence-v1" + bytes([0]))
    h.update((1).to_bytes(8, "big"))
    h.update((1).to_bytes(8, "big") + b"a")
    expected = next(c for c in VECTORS["sequence_sha256"] if c["input"] == ["a"])["expected_sha256"]
    assert h.hexdigest() == expected


# ── fingerprint document ───────────────────────────────────────────────────────

def test_synthetic_fingerprint_golden():
    syn = VECTORS["ground_truth_fingerprint"]["synthetic"]
    doc = gtc.ground_truth_fingerprint_document(**syn["input"])
    assert doc == syn["expected_document"]
    assert list(doc.keys()) == list(gtc.FINGERPRINT_DOCUMENT_KEYS_V1)
    assert gtc.document_sha256(doc) == syn["expected_sha256"]
    gtc.validate_fingerprint_document(doc)


@pytest.mark.parametrize("sha", [None, "", "0123456789ABCDEF" * 4, "0" * 63, "0" * 65, "g" * 64,
                                 "0" * 64 + "\n"])
def test_fingerprint_requires_strong_payload_checksum(sha):
    with pytest.raises(gtc.ContractError):
        gtc.ground_truth_fingerprint_document(payload_sha256=sha, var_names=["g"], pert_labels=["c"],
                                              pert_col="perturbation")


def test_fingerprint_excludes_location_and_changes_with_axes():
    base = dict(payload_sha256="ab" * 32, var_names=["g0", "g1"], pert_labels=["c", "A"],
                pert_col="perturbation")
    ref = gtc.document_sha256(gtc.ground_truth_fingerprint_document(**base))
    # Reordered features, reordered labels, renamed feature, changed assignment, other pert_col.
    for mut in (dict(var_names=["g1", "g0"]), dict(pert_labels=["A", "c"]), dict(var_names=["g0", "gX"]),
                dict(pert_labels=["c", "B"]), dict(pert_col="target"), dict(payload_sha256="cd" * 32)):
        assert gtc.document_sha256(gtc.ground_truth_fingerprint_document(**{**base, **mut})) != ref


# ── frozen fixture archives ────────────────────────────────────────────────────

ARCHIVES = {a["file"]: a for a in FIXTURE["archives"]}


@pytest.mark.parametrize("name", sorted(ARCHIVES))
def test_fixture_archive_bytes_are_frozen(name):
    rec = ARCHIVES[name]
    path = KIT / "archives" / name
    assert path.is_file(), f"missing frozen archive {path}"
    assert path.stat().st_size == rec["size_bytes"]
    assert _sha256_file(path) == rec["file_sha256"]


@pytest.mark.parametrize("name", sorted(ARCHIVES))
def test_fixture_axes_reproduce_recorded_fingerprint(name):
    rec = ARCHIVES[name]
    assert gtc.sequence_sha256(rec["var_names"]) == rec["feature_axis_sha256"]
    assert gtc.sequence_sha256(rec["pert_labels_storage_order"]) == rec["perturbation_axis_sha256"]
    sha = rec["archive_manifest"]["payload_sha256"]
    if sha is None:
        assert rec["ground_truth_fingerprint"] is None and rec["cache_eligible"] is False
        with pytest.raises(gtc.ContractError):
            gtc.ground_truth_fingerprint_document(payload_sha256=sha, var_names=rec["var_names"],
                                                  pert_labels=rec["pert_labels_storage_order"],
                                                  pert_col=rec["pert_col"])
        return
    doc = gtc.ground_truth_fingerprint_document(payload_sha256=sha, var_names=rec["var_names"],
                                                pert_labels=rec["pert_labels_storage_order"],
                                                pert_col=rec["pert_col"])
    assert doc == rec["ground_truth_fingerprint"]
    assert gtc.document_sha256(doc) == rec["ground_truth_fingerprint_sha256"]
    golden = next(g for g in VECTORS["ground_truth_fingerprint"]["fixture_archives"] if g["file"] == name)
    assert golden["expected_sha256"] == rec["ground_truth_fingerprint_sha256"]


def test_fixture_design_invariants():
    real, nochk, onev = ARCHIVES["real.shad"], ARCHIVES["real_nochecksum.shad"], ARCHIVES["real_onevalue.shad"]
    # Same cells, same axes -> same axis digests; the checksum alone decides eligibility.
    assert real["feature_axis_sha256"] == nochk["feature_axis_sha256"] == onev["feature_axis_sha256"]
    assert real["perturbation_axis_sha256"] == nochk["perturbation_axis_sha256"] == onev["perturbation_axis_sha256"]
    assert real["cache_eligible"] and not nochk["cache_eligible"]
    # One expression value changed -> different payload -> different fingerprint (shape unchanged).
    assert real["archive_manifest"]["payload_sha256"] != onev["archive_manifest"]["payload_sha256"]
    assert real["ground_truth_fingerprint_sha256"] != onev["ground_truth_fingerprint_sha256"]
    assert real["ground_truth_fingerprint"]["n_obs"] == onev["ground_truth_fingerprint"]["n_obs"] == 320
    assert real["ground_truth_fingerprint"]["n_vars"] == 150
    # Storage order groups the reference first (cellstream group_by/reference contract).
    labels = real["pert_labels_storage_order"]
    assert labels[:80] == ["non-targeting"] * 80
    assert set(labels) == set(FIXTURE["design"]["perturbations"]) | {"non-targeting"}


# ── semantic document + cache ID ───────────────────────────────────────────────

def test_illustrative_semantics_golden():
    sem = VECTORS["de_semantics"]["illustrative_production_v1"]
    gtc.validate_semantic_document(sem["document"])
    assert set(sem["document"]) == set(gtc.SEMANTIC_DOCUMENT_KEYS_V1)
    assert gtc.document_sha256(sem["document"]) == sem["expected_sha256"]
    mut = VECTORS["de_semantics"]["mutation_filter_gene_min_cpm_cell_null"]
    assert gtc.document_sha256(mut["document"]) == mut["expected_sha256"] != sem["expected_sha256"]


@pytest.mark.parametrize("mutation", [
    {"extra_key": 1}, {"cache_format_version": 2}, {"real_de_semantics_version": 2},
    {"epsilon": float("nan")}, {"target_sum": "10000"}, {"validate_input": "true"}, {"device": ""},
    {"cache_format_version": True}, {"real_de_semantics_version": True},   # True == 1 (PR #399)
])
def test_semantic_document_validation(mutation):
    sem = dict(VECTORS["de_semantics"]["illustrative_production_v1"]["document"])
    sem.update(mutation)
    with pytest.raises(gtc.ContractError):
        gtc.validate_semantic_document(sem)


def test_semantic_document_missing_key_rejected():
    sem = dict(VECTORS["de_semantics"]["illustrative_production_v1"]["document"])
    sem.pop("precision")
    with pytest.raises(gtc.ContractError):
        gtc.validate_semantic_document(sem)


def test_cache_id_golden():
    syn = VECTORS["ground_truth_fingerprint"]["synthetic"]["expected_document"]
    sem = VECTORS["de_semantics"]["illustrative_production_v1"]["document"]
    golden = VECTORS["cache_id"]["synthetic_fingerprint_x_illustrative_semantics"]
    assert gtc.cache_key_document(syn, sem) == golden["expected_key_document"]
    cid = gtc.cache_id(syn, sem)
    assert cid == golden["expected_cache_id"]
    assert gtc.entry_relpath(cid) == golden["expected_entry_relpath"] == f"ground-truth-deg/v1/{cid[:2]}/{cid}"
    mut = dict(sem, filter_gene_min_cpm_cell=None)
    assert gtc.cache_id(syn, mut) == VECTORS["cache_id"]["synthetic_fingerprint_x_mutated_semantics"]["expected_cache_id"]
    assert gtc.cache_id(syn, mut) != cid


def test_cache_id_for_fixture_archives_golden():
    sem = VECTORS["de_semantics"]["illustrative_production_v1"]["document"]
    for g in VECTORS["ground_truth_fingerprint"]["fixture_archives"]:
        if g["expected_document"] is None:
            assert "expected_cache_id_with_illustrative_semantics" not in g
            continue
        assert gtc.cache_id(g["expected_document"], sem) == g["expected_cache_id_with_illustrative_semantics"]


@pytest.mark.parametrize("bad", ["", "abc", "0" * 63, "0" * 65, "../" + "0" * 61, "0" * 32 + "/" + "0" * 31, "A" * 64,
                                 "0" * 64 + "\n"])
def test_entry_relpath_rejects_unsafe_ids(bad):
    with pytest.raises(gtc.ContractError):
        gtc.entry_relpath(bad)


# ── cellstream-gated: the recorded axes are what the archives actually contain ─

@pytest.mark.parametrize("name", sorted(ARCHIVES))
def test_fixture_axes_match_archive_contents(name):
    cellstream_cell = pytest.importorskip("cellstream.cell", reason="cellstream is not in the CI test group")
    rec = ARCHIVES[name]
    store = cellstream_cell.open_cell(str(KIT / "archives" / name))
    try:
        m = store.manifest
        assert m.get("payload_sha256") == rec["archive_manifest"]["payload_sha256"]
        assert [str(v) for v in store.var.index.tolist()] == rec["var_names"]
        assert [str(v) for v in store.obs[rec["pert_col"]].tolist()] == rec["pert_labels_storage_order"]
        if m.get("payload_sha256"):
            assert store.verify_payload()
    finally:
        store.close()
