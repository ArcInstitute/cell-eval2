"""Production-code tests for the ground-truth DEG cache identity contract.

Pins ``cell_eval2.ground_truth_cache`` (the production implementation) against the same
golden vectors / fixtures / classification the Phase-0 oracle uses
(``tests/test_ground_truth_cache_golden.py`` pins ``gtc_reference.py`` itself). Every
production function here must reproduce the oracle byte for byte.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import os
import random
import string
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from cell_eval2 import ground_truth_cache as gtcache
from cell_eval2.config import EvalConfig, FilterParams

KIT = Path(__file__).resolve().parent / "fixtures" / "ground_truth_cache"


def _load_reference():
    spec = importlib.util.spec_from_file_location("gtc_reference", KIT / "gtc_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gtc_ref = _load_reference()
VECTORS = json.loads((KIT / "golden_vectors.json").read_text(encoding="utf-8"))
FIXTURE = json.loads((KIT / "fixture_manifest.json").read_text(encoding="utf-8"))
ARCHIVES = {a["file"]: a for a in FIXTURE["archives"]}
CLASSIFICATION = yaml.safe_load((KIT / "eval_config_classification.yaml").read_text(encoding="utf-8"))
FIELDS = CLASSIFICATION["fields"]


# ── 1. Golden vectors reproduced byte-for-byte ─────────────────────────────────────


@pytest.mark.parametrize("case", VECTORS["canonical_json"], ids=lambda c: c["expected_sha256"][:8])
def test_canonical_json_golden(case):
    b = gtcache.canonical_json_bytes(case["input"])
    assert b.decode("utf-8") == case["expected_utf8"]
    assert gtcache.document_sha256(case["input"]) == case["expected_sha256"]


@pytest.mark.parametrize("case", VECTORS["sequence_sha256"], ids=lambda c: c["expected_sha256"][:8])
def test_sequence_sha256_golden(case):
    assert gtcache.sequence_sha256(case["input"]) == case["expected_sha256"]


@pytest.mark.parametrize("bad", [
    {"x": float("nan")}, {"x": float("inf")}, {"x": -math.inf}, {1: "non-str key"},
    {"x": b"bytes"}, {"x": {1, 2}}, {"x": [1, {"y": float("nan")}]},
])
def test_canonical_json_rejections_raise_contract_error(bad):
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.canonical_json_bytes(bad)


@pytest.mark.parametrize("bad", [[None], [1], [b"x"], ["ok", None]])
def test_sequence_sha256_rejections_raise_contract_error(bad):
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.sequence_sha256(bad)


# ── 2. Randomized parity against the oracle ────────────────────────────────────────


def _random_json_value(depth: int, rnd: random.Random):
    if depth <= 0:
        choice = rnd.randrange(5)
    else:
        choice = rnd.randrange(7)
    if choice == 0:
        return None
    if choice == 1:
        return rnd.choice([True, False])
    if choice == 2:
        return rnd.randint(-10_000, 10_000)
    if choice == 3:
        return rnd.choice([1e6, 1e-9, 0.0, -0.0, 3.14159, rnd.uniform(-1e3, 1e3)])
    if choice == 4:
        alphabet = string.ascii_letters + string.digits + "ünï∆細胞 \t\"\\/"
        return "".join(rnd.choice(alphabet) for _ in range(rnd.randrange(0, 12)))
    if choice == 5:
        return [_random_json_value(depth - 1, rnd) for _ in range(rnd.randrange(0, 4))]
    return {
        f"k{i}_{_random_json_value(0, rnd)!r}"[:20]: _random_json_value(depth - 1, rnd)
        for i in range(rnd.randrange(0, 4))
    }


@pytest.mark.parametrize("seed", range(200))
def test_canonical_json_matches_oracle_randomized(seed):
    rnd = random.Random(seed)
    value = _random_json_value(3, rnd)
    try:
        expected = gtc_ref.canonical_json_bytes(value)
    except gtc_ref.ContractError:
        with pytest.raises(gtcache.GroundTruthContractError):
            gtcache.canonical_json_bytes(value)
        return
    assert gtcache.canonical_json_bytes(value) == expected
    assert gtcache.document_sha256(value) == gtc_ref.document_sha256(value)


@pytest.mark.parametrize("seed", range(200))
def test_sequence_sha256_matches_oracle_randomized(seed):
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + "ünï∆細胞 \t/,"
    values = ["".join(rnd.choice(alphabet) for _ in range(rnd.randrange(0, 10)))
              for _ in range(rnd.randrange(0, 8))]
    assert gtcache.sequence_sha256(values) == gtc_ref.sequence_sha256(values)


# ── 3. Fingerprint document construction ───────────────────────────────────────────


def test_from_axes_reproduces_synthetic_golden():
    syn = VECTORS["ground_truth_fingerprint"]["synthetic"]
    fp = gtcache.GroundTruthFingerprint.from_axes(**syn["input"])
    assert fp.document == syn["expected_document"]
    assert list(fp.document.keys()) == list(gtcache.FINGERPRINT_DOCUMENT_KEYS_V1)
    assert fp.sha256 == syn["expected_sha256"]


@pytest.mark.parametrize("name", sorted(n for n, a in ARCHIVES.items() if a["cache_eligible"]))
def test_from_axes_reproduces_fixture_archive_golden(name):
    rec = ARCHIVES[name]
    fp = gtcache.GroundTruthFingerprint.from_axes(
        payload_sha256=rec["archive_manifest"]["payload_sha256"],
        var_names=rec["var_names"],
        pert_labels=rec["pert_labels_storage_order"],
        pert_col=rec["pert_col"],
    )
    assert fp.document == rec["ground_truth_fingerprint"]
    assert fp.sha256 == rec["ground_truth_fingerprint_sha256"]
    golden = next(g for g in VECTORS["ground_truth_fingerprint"]["fixture_archives"] if g["file"] == name)
    assert fp.sha256 == golden["expected_sha256"]


@pytest.mark.parametrize("sha", [None, "", "0123456789ABCDEF" * 4, "0" * 63, "0" * 65, "g" * 64,
                                 "0" * 64 + "\n"])
def test_from_axes_requires_strong_payload_checksum(sha):
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.GroundTruthFingerprint.from_axes(
            payload_sha256=sha, var_names=["g"], pert_labels=["c"], pert_col="perturbation"
        )


@pytest.mark.parametrize("bad", ["0" * 63, "A" * 64, "../" + "0" * 61, "0" * 64 + "\n"])
def test_entry_relpath_rejects_unsafe_ids(bad):
    """The production twin of the oracle's check: exactly 64 lowercase hex characters. A
    trailing newline would otherwise slip past ``$`` and into the entry path (Copilot, PR #399)."""
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.entry_relpath(bad)


# ── 4. Cellstream-gated archive fingerprinting ─────────────────────────────────────


def test_fingerprint_ground_truth_cell_archive_by_path():
    cellstream = pytest.importorskip("cellstream", reason="cellstream not installed")
    del cellstream
    fp = gtcache.fingerprint_ground_truth_cell_archive(
        str(KIT / "archives" / "real.shad"), pert_col="perturbation"
    )
    golden = next(g for g in VECTORS["ground_truth_fingerprint"]["fixture_archives"] if g["file"] == "real.shad")
    assert fp.sha256 == golden["expected_sha256"]


def test_fingerprint_ground_truth_cell_archive_by_open_store_stays_usable():
    pytest.importorskip("cellstream", reason="cellstream not installed")
    from cell_eval2._cell_archive import open_cell_store

    store = open_cell_store(str(KIT / "archives" / "real.shad"))
    try:
        fp = gtcache.fingerprint_ground_truth_cell_archive(store, pert_col="perturbation")
        golden = next(g for g in VECTORS["ground_truth_fingerprint"]["fixture_archives"]
                      if g["file"] == "real.shad")
        assert fp.sha256 == golden["expected_sha256"]
        # We did not close it -- the store must still be usable.
        assert store.n_obs == 320
        assert list(store.var.index)[:1] == ["TG01"]
    finally:
        store.close()


def test_fingerprint_ground_truth_cell_archive_rejects_missing_checksum():
    pytest.importorskip("cellstream", reason="cellstream not installed")
    path = str(KIT / "archives" / "real_nochecksum.shad")
    with pytest.raises(gtcache.GroundTruthFingerprintError) as exc_info:
        gtcache.fingerprint_ground_truth_cell_archive(path, pert_col="perturbation")
    msg = str(exc_info.value)
    assert "payload_checksum=True" in msg
    assert path in msg


def test_fingerprint_ground_truth_cell_archive_onevalue_differs():
    pytest.importorskip("cellstream", reason="cellstream not installed")
    real = gtcache.fingerprint_ground_truth_cell_archive(
        str(KIT / "archives" / "real.shad"), pert_col="perturbation"
    )
    onevalue = gtcache.fingerprint_ground_truth_cell_archive(
        str(KIT / "archives" / "real_onevalue.shad"), pert_col="perturbation"
    )
    assert real.sha256 != onevalue.sha256


def test_fingerprint_ground_truth_cell_archive_wrong_pert_col():
    pytest.importorskip("cellstream", reason="cellstream not installed")
    with pytest.raises(gtcache.GroundTruthFingerprintError):
        gtcache.fingerprint_ground_truth_cell_archive(
            str(KIT / "archives" / "real.shad"), pert_col="not_a_column"
        )


# ── 5. Duck-typed fake store ────────────────────────────────────────────────────────


def _fake_store(*, payload_sha256, var_names, pert_labels, pert_col="perturbation"):
    var = pd.DataFrame(index=pd.Index(var_names))
    obs = pd.DataFrame({pert_col: pert_labels})
    return SimpleNamespace(
        manifest={"payload_sha256": payload_sha256},
        var=var,
        obs=obs,
        path=f"<fake:{payload_sha256}>",
    )


_GOOD_PAYLOAD = "ab" * 32


def test_fake_store_null_label_raises_contract_error():
    store = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0", "g1"], pert_labels=["c", None])
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.fingerprint_ground_truth_cell_archive(store, pert_col="perturbation")


def test_fake_store_non_str_var_name_raises_contract_error():
    store = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0", 1], pert_labels=["c", "A"])
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.fingerprint_ground_truth_cell_archive(store, pert_col="perturbation")


def test_fake_store_label_order_changes_sha():
    a = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0", "g1"], pert_labels=["c", "A"])
    b = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0", "g1"], pert_labels=["A", "c"])
    fp_a = gtcache.fingerprint_ground_truth_cell_archive(a, pert_col="perturbation")
    fp_b = gtcache.fingerprint_ground_truth_cell_archive(b, pert_col="perturbation")
    assert fp_a.sha256 != fp_b.sha256


def test_fake_store_feature_rename_changes_sha():
    a = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0", "g1"], pert_labels=["c", "A"])
    b = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0", "gX"], pert_labels=["c", "A"])
    fp_a = gtcache.fingerprint_ground_truth_cell_archive(a, pert_col="perturbation")
    fp_b = gtcache.fingerprint_ground_truth_cell_archive(b, pert_col="perturbation")
    assert fp_a.sha256 != fp_b.sha256


def test_fake_store_missing_payload_raises_fingerprint_error():
    store = _fake_store(payload_sha256=None, var_names=["g0"], pert_labels=["c"])
    with pytest.raises(gtcache.GroundTruthFingerprintError):
        gtcache.fingerprint_ground_truth_cell_archive(store, pert_col="perturbation")


def test_fake_store_63_char_hex_raises_fingerprint_error():
    store = _fake_store(payload_sha256="a" * 63, var_names=["g0"], pert_labels=["c"])
    with pytest.raises(gtcache.GroundTruthFingerprintError):
        gtcache.fingerprint_ground_truth_cell_archive(store, pert_col="perturbation")


def test_fake_store_wrong_pert_col_lists_available_columns():
    store = _fake_store(payload_sha256=_GOOD_PAYLOAD, var_names=["g0"], pert_labels=["c"])
    with pytest.raises(gtcache.GroundTruthFingerprintError) as exc_info:
        gtcache.fingerprint_ground_truth_cell_archive(store, pert_col="nope")
    assert "perturbation" in str(exc_info.value)


# ── 6. Semantic document builder + cache ID golden reproduction ───────────────────


_GOLDEN_VERSIONS = {"cell-eval2": "0.17.0", "gpudge": "0.9.1", "cellstream": "0.10.0"}


def _illustrative_cfg():
    cfg = EvalConfig.v1()
    cfg.pert_col = "perturbation"
    cfg.filter = FilterParams(filter_gene_min_cpm_cell=5.0)
    cfg.de = dataclasses.replace(cfg.de, epsilon=1e-9, clip_value=None, fdr_scope="per_pert")
    return cfg


_ILLUSTRATIVE_RESOLVED = dict(
    control="non-targeting",
    input_type="lognorm",
    target_sum=None,
    device="cuda",
    de_backend="gpudge",
    precision="float32",
    native_gpu_normalize=True,
    versions=_GOLDEN_VERSIONS,
)


def test_build_real_de_semantics_reproduces_illustrative_golden():
    sem = gtcache.build_real_de_semantics(_illustrative_cfg(), **_ILLUSTRATIVE_RESOLVED)
    golden = VECTORS["de_semantics"]["illustrative_production_v1"]
    assert sem.document == golden["document"]
    assert sem.sha256 == golden["expected_sha256"]


def test_build_real_de_semantics_reproduces_null_cpm_mutation_golden():
    cfg = EvalConfig.v1()
    cfg.pert_col = "perturbation"
    cfg.de = dataclasses.replace(cfg.de, epsilon=1e-9, clip_value=None, fdr_scope="per_pert")
    sem = gtcache.build_real_de_semantics(cfg, **_ILLUSTRATIVE_RESOLVED)
    golden = VECTORS["de_semantics"]["mutation_filter_gene_min_cpm_cell_null"]
    assert sem.document == golden["document"]
    assert sem.sha256 == golden["expected_sha256"]


def test_cache_id_and_entry_relpath_reproduce_golden():
    syn_doc = VECTORS["ground_truth_fingerprint"]["synthetic"]["expected_document"]
    sem = gtcache.build_real_de_semantics(_illustrative_cfg(), **_ILLUSTRATIVE_RESOLVED)
    golden = VECTORS["cache_id"]["synthetic_fingerprint_x_illustrative_semantics"]
    assert gtcache.cache_key_document(syn_doc, sem.document) == golden["expected_key_document"]
    cid = gtcache.cache_id(syn_doc, sem.document)
    assert cid == golden["expected_cache_id"]
    assert gtcache.entry_relpath(cid) == golden["expected_entry_relpath"]


def test_de_replicate_col_is_prohibited():
    cfg = _illustrative_cfg()
    cfg.de = dataclasses.replace(cfg.de, replicate_col="rep")
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.build_real_de_semantics(cfg, **_ILLUSTRATIVE_RESOLVED)


# ── 7. Classification-driven mutation coverage ─────────────────────────────────────


#: EvalConfig fields whose real-DE-facing value is read from an explicit resolved
#: builder argument rather than the raw EvalConfig field (§9.2 "resolved: true", except
#: pert_col which the builder reads straight off cfg.pert_col by convention).
_RESOLVED_ARG_FOR_FIELD = {
    "control": "control",
    "input_type": "input_type",
    "target_sum": "target_sum",
    "device": "device",
    "de.backend": "de_backend",
}

_BASE_RESOLVED = dict(
    control="non-targeting",
    input_type="counts",
    target_sum=10_000.0,
    device="cuda",
    de_backend="gpudge",
    precision="float32",
    native_gpu_normalize=True,
    versions=_GOLDEN_VERSIONS,
)

#: One alternative, individually valid value per classified (non-container) field.
#: Parametrized over eval_config_classification.yaml below -- a newly classified field
#: with no entry here fails the coverage assertion loudly rather than being skipped.
MUTATIONS: dict[str, object] = {
    # included
    "pert_col": "other_pert_col",
    "control": "other-control",
    "version": "v1",
    "input_type": "lognorm",
    "max_counts_per_cell": 2_000_000.0,
    "target_sum": 20_000.0,
    "validate_input": False,
    "filter.filter_gene_min_cpm_cell": 10.0,
    "device": "cpu",
    "de.backend": "pdex",
    "de.mean_calc": "geometric",
    "de.epsilon": 5e-8,
    "de.clip_value": 15.0,
    "de.fdr_scope": "global",
    # folded
    "allow_discrete": True,
    "autodetect_input_type": True,
    "de.method": "deseq2",  # degenerate: DEParams derives method from backend, which is
                            # left at "auto" here, so __post_init__ re-derives it back to
                            # "wilcoxon" -- the field cannot actually be moved independently
                            # of de.backend, which is exactly what "folded into de_backend"
                            # means. The id-unchanged assertion still holds trivially.
    # prohibited
    "de.replicate_col": "rep",
    # excluded
    "metrics": ["de_spearman_lfc"],
    "control_source": "pred",
    "allow_fractional_counts": True,
    "num_threads": 2,
    "pert_chunk": 128,
    "outdir": "/tmp/x",
    "cache_real": "/tmp/c",
    "cache_pred": "/tmp/p",
    "cache_strict": True,
    "gather_threads": 4,
    "target_gene_map": {"a": "b"},
    "bulk_target_sum": 20_000.0,
    "discrimination.distance": "l1",
    "discrimination.rank_denominator": "n",
    "discrimination.exclude_target_gene": False,
    "discrimination.embed_key": "X_pca",
    "discrimination.tie_policy": "position",
    "discrimination.exclusion_scope": "row",
    "de.p_adj_threshold": 0.1,
    "de.min_abs_log2fc": 1.0,
    "de.sort_by": "p_adj",
    "de.nan_lfc_policy": "keep",
    "de.auc_pval_floor": "clip",
    "de.auc_pval_floor_value": 0.5,
}

_CLASSIFIED_NON_CONTAINER = {name for name, e in FIELDS.items() if e["class"] != "container"}


def test_mutations_cover_every_classified_non_container_field():
    missing = _CLASSIFIED_NON_CONTAINER - set(MUTATIONS)
    assert not missing, f"no MUTATIONS entry for classified field(s): {sorted(missing)}"
    stale = set(MUTATIONS) - _CLASSIFIED_NON_CONTAINER
    assert not stale, f"MUTATIONS has entries for unclassified/removed field(s): {sorted(stale)}"


def _base_cfg() -> EvalConfig:
    cfg = EvalConfig.v2()
    cfg.pert_col = "perturbation"
    return cfg


def _with_field(cfg: EvalConfig, field_path: str, value) -> EvalConfig:
    cfg2 = copy.deepcopy(cfg)
    if "." not in field_path:
        setattr(cfg2, field_path, value)
        return cfg2
    parent_name, child_name = field_path.split(".", 1)
    parent = getattr(cfg2, parent_name)
    params = getattr(type(parent), "__dataclass_params__", None)
    if params is not None and params.frozen:
        setattr(cfg2, parent_name, dataclasses.replace(parent, **{child_name: value}))
    else:
        setattr(parent, child_name, value)
    return cfg2


_FIXED_FINGERPRINT_DOC = VECTORS["ground_truth_fingerprint"]["synthetic"]["expected_document"]


def _semantics_cache_id(cfg: EvalConfig, resolved_overrides: dict | None = None) -> str:
    kwargs = dict(_BASE_RESOLVED)
    if resolved_overrides:
        kwargs.update(resolved_overrides)
    sem = gtcache.build_real_de_semantics(cfg, **kwargs)
    return gtcache.cache_id(_FIXED_FINGERPRINT_DOC, sem.document)


_BASE_CFG = _base_cfg()
_BASE_ID = _semantics_cache_id(_BASE_CFG)


@pytest.mark.parametrize("field", sorted(n for n, e in FIELDS.items() if e["class"] == "included"))
def test_included_field_mutation_changes_cache_id(field):
    value = MUTATIONS[field]
    if field in _RESOLVED_ARG_FOR_FIELD:
        new_id = _semantics_cache_id(_BASE_CFG, {_RESOLVED_ARG_FOR_FIELD[field]: value})
    else:
        new_id = _semantics_cache_id(_with_field(_BASE_CFG, field, value))
    assert new_id != _BASE_ID, f"included field {field!r} did not change the cache id"


@pytest.mark.parametrize("field", sorted(n for n, e in FIELDS.items() if e["class"] in ("excluded", "folded")))
def test_excluded_or_folded_field_mutation_leaves_cache_id_unchanged(field):
    value = MUTATIONS[field]
    new_id = _semantics_cache_id(_with_field(_BASE_CFG, field, value))
    assert new_id == _BASE_ID, f"{FIELDS[field]['class']} field {field!r} changed the cache id"


@pytest.mark.parametrize("field", sorted(n for n, e in FIELDS.items() if e["class"] == "prohibited"))
def test_prohibited_field_raises_contract_error(field):
    value = MUTATIONS[field]
    cfg2 = _with_field(_BASE_CFG, field, value)
    with pytest.raises(gtcache.GroundTruthContractError):
        _semantics_cache_id(cfg2)


# ── 8. resolve_de_precision truth table ────────────────────────────────────────────


@pytest.mark.parametrize("x_dtype,input_type,native,target_sum,expected", [
    ("float32", "counts", True, None, "float32"),
    ("float32", "counts", True, 1e6, "float32"),
    ("float32", "counts", True, 5e4, "float32"),
    ("float64", "counts", True, None, "float64"),
    ("float32", "counts", False, 1e6, "float32"),
    ("float32", "counts", False, 5e4, "float64"),
    ("float32", "lognorm", True, None, "float32"),
    ("float32", "lognorm", False, 1e6, "float32"),
    ("float64", "lognorm", True, None, "float64"),
    ("float64", "lognorm", False, None, "float64"),
    ("int16", "counts", True, None, "int16"),
])
def test_resolve_de_precision_truth_table(x_dtype, input_type, native, target_sum, expected):
    got = gtcache.resolve_de_precision(
        x_dtype, input_type=input_type, native_gpu_normalize=native, target_sum=target_sum
    )
    assert got == expected


def test_resolve_de_precision_bad_input_type_raises_value_error():
    with pytest.raises(ValueError):
        gtcache.resolve_de_precision(
            "float32", input_type="bogus", native_gpu_normalize=True, target_sum=None
        )


# ── 9. validate_semantic_document / request / result contracts ────────────────────


@pytest.mark.parametrize("mutation", [
    {"epsilon": float("nan")},
    {"epsilon": float("inf")},
    {"extra_key": 1},
    {"cache_format_version": 2},
    {"cache_format_version": True},        # True == 1 (Copilot, PR #399)
    {"real_de_semantics_version": True},
])
def test_validate_semantic_document_rejects_bad_values(mutation):
    doc = dict(VECTORS["de_semantics"]["illustrative_production_v1"]["document"])
    doc.update(mutation)
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.validate_semantic_document(doc)


def test_validate_semantic_document_rejects_missing_key():
    doc = dict(VECTORS["de_semantics"]["illustrative_production_v1"]["document"])
    doc.pop("precision")
    with pytest.raises(gtcache.GroundTruthContractError):
        gtcache.validate_semantic_document(doc)


def test_digest_classes_refuse_a_digest_that_is_not_their_documents():
    """GroundTruthFingerprint and RealDESemantics are "a validated document plus ITS digest". A
    caller-supplied digest that is not the document's would let a request hit by document while
    the result and the sidecar report the bogus digest (Copilot, PR #399)."""
    fp = gtcache.GroundTruthFingerprint.from_axes(
        payload_sha256="ab" * 32, var_names=["g"], pert_labels=["c"], pert_col="perturbation")
    sem = gtcache.build_real_de_semantics(_illustrative_cfg(), **_ILLUSTRATIVE_RESOLVED)
    assert gtcache.GroundTruthFingerprint(document=fp.document, sha256=fp.sha256) == fp
    assert gtcache.RealDESemantics(document=sem.document, sha256=sem.sha256) == sem
    with pytest.raises(gtcache.GroundTruthContractError, match="digest"):
        gtcache.GroundTruthFingerprint(document=fp.document, sha256="0" * 64)
    with pytest.raises(gtcache.GroundTruthContractError, match="digest"):
        gtcache.RealDESemantics(document=sem.document, sha256="0" * 64)


def _sample_request() -> gtcache.GroundTruthDEGCacheRequest:
    syn = VECTORS["ground_truth_fingerprint"]["synthetic"]
    fp = gtcache.GroundTruthFingerprint.from_axes(**syn["input"])
    sem = gtcache.build_real_de_semantics(_illustrative_cfg(), **_ILLUSTRATIVE_RESOLVED)
    return gtcache.GroundTruthDEGCacheRequest(fingerprint=fp, semantics=sem)


def test_cache_request_is_identity_only_and_derives_cache_id():
    """P1 fix: `GroundTruthDEGCacheRequest` carries only `fingerprint`/`semantics` now --
    `root`/`policy` belong exclusively to `GroundTruthDEGCache` (see
    test_ground_truth_cache_store.py's readonly-enforcement tests)."""
    req = _sample_request()
    golden = VECTORS["cache_id"]["synthetic_fingerprint_x_illustrative_semantics"]
    assert req.cache_id == golden["expected_cache_id"]
    assert req.entry_relpath == golden["expected_entry_relpath"]
    assert {f.name for f in dataclasses.fields(req)} == {"fingerprint", "semantics"}


def test_cache_request_rejects_root_and_policy_kwargs():
    """Reproduces the exact PR #389 review finding: a caller could previously build a
    `policy="readwrite"` request against a `readonly` `GroundTruthDEGCache` instance and
    have `get_or_compute` honour the REQUEST's policy on a miss (computing + locking
    despite the cache being readonly). Removing the fields makes that call a `TypeError`
    at construction time -- there is no longer any way to ask."""
    syn = VECTORS["ground_truth_fingerprint"]["synthetic"]
    fp = gtcache.GroundTruthFingerprint.from_axes(**syn["input"])
    sem = gtcache.build_real_de_semantics(_illustrative_cfg(), **_ILLUSTRATIVE_RESOLVED)
    with pytest.raises(TypeError):
        gtcache.GroundTruthDEGCacheRequest(
            root="/tmp/gtc", policy="readwrite", fingerprint=fp, semantics=sem
        )
    with pytest.raises(TypeError):
        gtcache.GroundTruthDEGCacheRequest(policy="readwrite", fingerprint=fp, semantics=sem)
    with pytest.raises(TypeError):
        gtcache.GroundTruthDEGCacheRequest(root="/tmp/gtc", fingerprint=fp, semantics=sem)


def test_cache_result_to_sidecar_dict_has_exactly_twelve_keys():
    result = gtcache.GroundTruthDEGCacheResult(
        cache_id="a" * 64,
        status="hit",
        policy="readwrite",
        entry_dir="/cache/ground-truth-deg/v1/aa/" + "a" * 64,
        table_path="/cache/.../de_real.parquet",
        manifest={"schema_version": 1},
        de_real_sha256="b" * 64,
        de_real_rows=1234,
        ground_truth_fingerprint_sha256="c" * 64,
        de_semantics_sha256="d" * 64,
        lock_wait_seconds=0.01,
        compute_seconds=1.5,
    )
    sidecar = result.to_sidecar_dict(producer={"name": "cell_eval2", "version": "0.17.0"})
    expected_keys = {
        "schema_version", "status", "policy", "cache_id", "cache_entry",
        "ground_truth_fingerprint_sha256", "de_semantics_sha256", "de_real_sha256",
        "de_real_rows", "lock_wait_seconds", "compute_seconds", "producer",
    }
    assert set(sidecar) == expected_keys


# ── 10. import robustness without POSIX fcntl (P1 fix) ─────────────────────────────
#
# Before this fix, `ground_truth_cache.py` imported `fcntl` unconditionally at module
# import time, so `import cell_eval2` failed outright on any platform without POSIX
# advisory locks (e.g. Windows). `fcntl` is now imported LAZILY, only inside
# `store.py`'s write/doctor paths (`store._flock_module`) -- these tests run in
# SUBPROCESSES (not an in-process `monkeypatch.setitem(sys.modules, "fcntl", None)` +
# `importlib.reload`) so a `cell_eval2`/`ground_truth_cache.store` module already
# imported and cached by an earlier test in this session cannot mask the bug.

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


#: Every subprocess bound in this module only has to catch a HANG: a child imports cell_eval2
#: cold, which alone has taken 10-14 s on a loaded node, so a tight bound fails a correct run.
_CHILD_TIMEOUT_S = 120


def _env_with_src() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _SRC_DIR if not existing else f"{_SRC_DIR}{os.pathsep}{existing}"
    return env


_IMPORT_WITHOUT_FCNTL_SRC = """
import sys
sys.modules["fcntl"] = None  # hide fcntl before cell_eval2 is ever imported
import cell_eval2
import cell_eval2.ground_truth_cache
import cell_eval2.ground_truth_cache.identity
from cell_eval2 import fingerprint_ground_truth_cell_archive
print("import-ok", flush=True)
"""


def test_import_cell_eval2_and_identity_succeed_without_fcntl():
    """``import cell_eval2``, ``import cell_eval2.ground_truth_cache``, and
    ``import cell_eval2.ground_truth_cache.identity`` must all succeed even when
    ``fcntl`` cannot be imported at all -- fingerprinting and ordinary scoring never
    need POSIX advisory locks."""
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_FCNTL_SRC],
        capture_output=True, text=True, env=_env_with_src(), timeout=_CHILD_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stderr
    assert "import-ok" in proc.stdout


_GTC_FCNTL_HIDDEN_SRC = """
import sys
sys.path.insert(0, sys.argv[1])
root, mode = sys.argv[2], sys.argv[3]

from cell_eval2 import ground_truth_cache as gtcache
from cell_eval2.config import EvalConfig

# A distinct fingerprint per mode -- "hidden-miss-write" must be a genuine MISS, never
# accidentally colliding with the entry "populate" published.
payload = "ab" * 32 if mode != "hidden-miss-write" else "cd" * 32
fp = gtcache.GroundTruthFingerprint.from_axes(
    payload_sha256=payload, var_names=["g0"], pert_labels=["A", "non-targeting"],
    pert_col="perturbation",
)
cfg = EvalConfig.v1()
cfg.pert_col = "perturbation"
sem = gtcache.build_real_de_semantics(
    cfg, control="non-targeting", input_type="counts", target_sum=10000.0, device="cuda",
    de_backend="gpudge", precision="float32", native_gpu_normalize=True,
    versions={"cell-eval2": "0.17.0", "gpudge": "0.9.1", "cellstream": "0.10.0"},
)
req = gtcache.GroundTruthDEGCacheRequest(fingerprint=fp, semantics=sem)

if mode == "populate":
    # Normal write, fcntl available: populates the entry `hidden-lookup` will later read.
    import polars as pl
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    cache.get_or_compute(req, lambda: pl.DataFrame(
        {"target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0], "p_adj": [0.01]}
    ))
    print("populated", flush=True)
elif mode == "hidden-lookup":
    sys.modules["fcntl"] = None
    cache = gtcache.GroundTruthDEGCache(root, policy="readonly")
    hit = cache.lookup(req)
    assert hit is not None and hit.status == "hit", hit
    print("lookup-ok", flush=True)
elif mode == "hidden-miss-write":
    sys.modules["fcntl"] = None
    import polars as pl
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    try:
        cache.get_or_compute(req, lambda: pl.DataFrame(
            {"target": ["B"], "feature": ["g0"], "log2_fold_change": [1.0], "p_adj": [0.01]}
        ))
    except gtcache.GroundTruthCacheFilesystemError as e:
        assert "fcntl" in str(e), str(e)
        print("miss-write-raised-ok", flush=True)
    else:
        raise AssertionError("expected GroundTruthCacheFilesystemError")
else:
    raise AssertionError(f"unknown mode {mode!r}")
"""


def test_readonly_lookup_works_and_readwrite_miss_raises_with_fcntl_hidden(tmp_path):
    """``GroundTruthDEGCache(root, policy="readwrite").get_or_compute(...)`` on a MISS
    with ``fcntl`` hidden raises ``GroundTruthCacheFilesystemError`` mentioning
    ``fcntl``; ``lookup`` of an ALREADY-PUBLISHED entry still works with ``fcntl``
    hidden (a readonly lookup never locks)."""
    pytest.importorskip("pyarrow")
    root = str(tmp_path / "cache")
    env = _env_with_src()

    def _run(mode: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-c", _GTC_FCNTL_HIDDEN_SRC, _SRC_DIR, root, mode],
            capture_output=True, text=True, env=env, timeout=_CHILD_TIMEOUT_S,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    assert "populated" in _run("populate")
    assert "lookup-ok" in _run("hidden-lookup")
    assert "miss-write-raised-ok" in _run("hidden-miss-write")
