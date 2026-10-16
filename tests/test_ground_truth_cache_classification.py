"""Every cell_eval2 EvalConfig field must be classified for the ground-truth DEG cache key.

docs/ground_truth_cache.md §5.2 requires the semantic-key builder to be an explicit allowlist and
tests to enumerate every EvalConfig field. This file checks the classification's internal
consistency and that it covers exactly the current dataclass fields -- so adding an EvalConfig
field fails here until the field is classified in eval_config_classification.yaml.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

KIT = Path(__file__).resolve().parent / "fixtures" / "ground_truth_cache"
CLASSES = {"included", "folded", "excluded", "prohibited", "container"}


def _load_reference():
    spec = importlib.util.spec_from_file_location("gtc_reference", KIT / "gtc_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gtc = _load_reference()
DOC = yaml.safe_load((KIT / "eval_config_classification.yaml").read_text(encoding="utf-8"))
FIELDS: dict[str, dict] = DOC["fields"]


def test_every_entry_has_a_valid_class_and_reason():
    for name, entry in FIELDS.items():
        assert entry["class"] in CLASSES, name
        if entry["class"] != "container":
            assert entry.get("reason"), f"{name} has no reason"


def test_included_fields_map_onto_distinct_semantic_keys():
    keys = {}
    for name, entry in FIELDS.items():
        if entry["class"] != "included":
            continue
        assert "resolved" in entry and isinstance(entry["resolved"], bool), name
        key = entry["semantic_key"]
        assert key in gtc.SEMANTIC_DOCUMENT_KEYS_V1, f"{name} -> unknown semantic key {key!r}"
        assert key not in keys, f"{name} and {keys[key]} both claim semantic key {key!r}"
        keys[key] = name


def test_folded_fields_name_an_included_semantic_key():
    included_keys = {e["semantic_key"] for e in FIELDS.values() if e["class"] == "included"}
    for name, entry in FIELDS.items():
        if entry["class"] == "folded":
            assert entry["into"] in included_keys, f"{name} folds into {entry['into']!r}, not an included key"


def test_prohibited_fields_state_the_required_value():
    for name, entry in FIELDS.items():
        if entry["class"] == "prohibited":
            assert "required_value" in entry, name


def test_every_semantic_key_has_a_producer():
    produced = {e["semantic_key"] for e in FIELDS.values() if e["class"] == "included"}
    produced |= set(DOC["non_config_inputs"])
    missing = set(gtc.SEMANTIC_DOCUMENT_KEYS_V1) - produced
    assert not missing, f"semantic keys with no producer: {sorted(missing)}"
    extra = produced - set(gtc.SEMANTIC_DOCUMENT_KEYS_V1)
    assert not extra, f"producers for keys outside the v1 document: {sorted(extra)}"


def test_containers_have_children_and_children_have_containers():
    containers = {n for n, e in FIELDS.items() if e["class"] == "container"}
    for name in FIELDS:
        if "." in name:
            parent = name.rsplit(".", 1)[0]
            assert parent in containers, f"{name} has no container entry {parent!r}"
    for c in containers:
        assert any(n.startswith(c + ".") for n in FIELDS), f"container {c} has no classified children"


def test_classification_covers_exactly_the_current_eval_config_fields():
    from dataclasses import fields as dc_fields

    from cell_eval2.config import (
        DEParams,
        DiscriminationParams,
        EvalConfig,
        FilterParams,
    )

    nested = {"filter": FilterParams, "discrimination": DiscriminationParams, "de": DEParams}
    expected = set()
    for f in dc_fields(EvalConfig):
        expected.add(f.name)
        if f.name in nested:
            expected |= {f"{f.name}.{g.name}" for g in dc_fields(nested[f.name])}
    classified = set(FIELDS)
    assert classified == expected, (
        f"unclassified: {sorted(expected - classified)}; stale: {sorted(classified - expected)}"
    )
    for name in nested:
        assert FIELDS[name]["class"] == "container"
