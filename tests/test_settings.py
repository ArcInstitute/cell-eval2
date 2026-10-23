import dataclasses
import json
from decimal import Decimal

import numpy as np
import pytest

from cell_eval2 import settings
from cell_eval2.baseline import DIGEST_EXEMPT_FIELDS, config_digest
from cell_eval2.config import DEParams, EvalConfig, FilterParams
from cell_eval2.run import RunFacts, _compute_metrics_with_facts
from cell_eval2.settings import (FACT_PATHS, NO_NEUTRAL, NOT_RECORDED, SETTINGS,
                                 build_settings, check_recordable, engine_record)
from _edge_fixtures import config, scenario

KEYS_0_20_0 = {
    "pert_col", "control", "version", "control_source", "input_type_real", "input_type_pred",
    "max_counts_per_cell", "target_sum", "allow_discrete", "autodetect_input_type",
    "validate_input", "filter.filter_gene_min_cpm_cell", "device", "pert_chunk",
    *(f"discrimination.{k}" for k in ("distance", "rank_denominator", "exclude_target_gene",
                                      "embed_key", "tie_policy", "exclusion_scope")),
    *(f"de.{k}" for k in ("p_adj_threshold", "min_abs_log2fc", "sort_by", "method",
                          "nan_lfc_policy", "backend", "replicate_col", "mean_calc", "epsilon",
                          "clip_value", "fdr_scope", "auc_pval_floor", "auc_pval_floor_value",
                          "engine")),
    "target_gene_map", "bulk_target_sum", "comparator", "scoring_path",
}


def _dotted(path):
    return ".".join(path)


def _leaves(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and k in ("filter", "discrimination", "de"):
            yield from _leaves(v, prefix + (k,))
        else:
            yield prefix + (k,)


def _facts(**kw):
    base = dict(config=EvalConfig(), input_type_real="counts", input_type_pred="counts",
                comparator="bulk_lognorm", device="cpu", de_backend="pdex",
                de_backend_used=True, scoring_path="in_memory", labels=("A", "B"))
    return RunFacts(**{**base, **kw})


def test_every_config_field_is_a_setting_or_not_recorded():
    fields = {k.field for k in SETTINGS if k.field is not None}
    for leaf in _leaves(EvalConfig().to_dict()):
        assert leaf in fields or leaf[0] in NOT_RECORDED, f"unlisted EvalConfig field {leaf}"
    assert all(f in set(_leaves(EvalConfig().to_dict())) for f in fields)
    assert NOT_RECORDED == {"metrics"} | DIGEST_EXEMPT_FIELDS


def test_the_registry_is_the_0_20_0_key_set_and_later_keys_declare_neutral_values():
    paths = {_dotted(k.path) for k in SETTINGS}
    assert paths >= KEYS_0_20_0 and len(paths) == len(SETTINGS)
    for k in SETTINGS:
        assert (k.neutral is NO_NEUTRAL) == (_dotted(k.path) in KEYS_0_20_0), k.path
    assert FACT_PATHS == ("pert_col", "control", "input_type_real", "input_type_pred",
                          "comparator", "de.engine", "scoring_path")


def test_settings_hold_exactly_the_registry_keys():
    s = build_settings(_facts())
    assert {_dotted(p) for p in _leaves(s)} == {_dotted(k.path) for k in SETTINGS}
    assert not ({"input_type", "metrics"} | DIGEST_EXEMPT_FIELDS) & set(s)
    assert s["target_gene_map"] is None and s["scoring_path"] == "in_memory"
    assert s["de"]["engine"] == engine_record("pdex") and s["de"]["backend"] == "pdex"
    assert build_settings(_facts(de_backend_used=False, de_backend="auto"))["de"]["engine"] is None


def test_numbers_are_written_as_the_config_holds_them():
    s = build_settings(_facts(config=EvalConfig(target_sum=1_000_000)))
    assert type(s["target_sum"]) is int and type(s["de"]["epsilon"]) is float


def test_supplied_real_de_preserves_configured_backend_and_resolved_engine():
    facts = _facts(config=EvalConfig(de=DEParams(backend="auto")), de_real_computed=False)
    s = build_settings(facts)
    assert s["de"]["backend"] == "auto"
    assert s["de"]["engine"] == engine_record("pdex")


@pytest.mark.parametrize("field,value", [("pert_chunk", np.int64(32)),
                                         ("target_sum", np.float32(1000000))])
def test_non_native_scalars_are_refused_because_their_digest_would_change(field, value):
    cfg = dataclasses.replace(config("de"), **{field: value})
    native = dataclasses.replace(cfg, **{field: value.item()})
    assert config_digest(cfg, comparator="bulk_lognorm") != config_digest(
        native, comparator="bulk_lognorm")
    with pytest.raises(ValueError, match=field + ".*native Python value"):
        check_recordable(cfg)


def test_natively_serializable_numpy_float_is_recordable():
    cfg = dataclasses.replace(config("de"), target_sum=np.float64(1000000))
    check_recordable(cfg)
    native = dataclasses.replace(cfg, target_sum=1000000.0)
    assert config_digest(cfg, comparator="bulk_lognorm") == config_digest(
        native, comparator="bulk_lognorm")


def test_recordability_names_all_offending_settings_including_nested_values():
    cfg = EvalConfig(pert_chunk=np.int64(32), target_sum=np.float32(1000000),
                     target_gene_map={"G0": [np.int64(1)]})
    with pytest.raises(ValueError) as exc:
        check_recordable(cfg)
    for name in ("pert_chunk", "target_sum", "target_gene_map", "native Python value"):
        assert name in str(exc.value)


def test_equal_final_settings_reached_two_ways_compare_equal():
    a = build_settings(_facts(config=EvalConfig.from_preset("v2")))
    b = build_settings(_facts(config=EvalConfig(target_sum=1_000_000, de=DEParams(epsilon=1e-9))))
    assert a == b                              # numerically, as the contract's §12 compares


def test_nulls_and_plain_numbers_are_recordable():
    check_recordable(EvalConfig())                               # v2's de.clip_value is None
    check_recordable(EvalConfig(target_sum=None))
    check_recordable(EvalConfig(filter=FilterParams(filter_gene_min_cpm_cell=None)))
    check_recordable(EvalConfig(de=DEParams(p_adj_threshold=1)))


@pytest.mark.parametrize("value", [Decimal("0.05"), np.longdouble("0.05")])
def test_a_float_setting_json_cannot_hold_is_refused(value):
    with pytest.raises(ValueError, match="de.p_adj_threshold"):
        check_recordable(EvalConfig(de=DEParams(p_adj_threshold=value)))


def test_a_resolved_target_sum_equals_the_same_number_asked_for():
    pred, real, _ = scenario("random")
    _, f1 = _compute_metrics_with_facts(pred, real, config=dataclasses.replace(config("de"), target_sum=None))
    _, f2 = _compute_metrics_with_facts(pred, real, config=dataclasses.replace(
        config("de"), target_sum=f1.config.target_sum))
    assert build_settings(f1) == build_settings(f2)


def test_device_engine_version_and_provenance_each_change_the_settings(monkeypatch):
    base = build_settings(_facts())
    assert build_settings(_facts(device="cuda")) != base
    monkeypatch.setattr(settings, "_package_provenance",
                        lambda name: {"version": "0.0.0", "provenance": "release"})
    v = build_settings(_facts())
    monkeypatch.setattr(settings, "_package_provenance",
                        lambda name: {"version": "0.0.0", "provenance": "local-editable"})
    assert v != base and build_settings(_facts()) != v


def test_two_commits_of_one_engine_version_are_different_settings(monkeypatch):
    monkeypatch.setattr(settings, "_package_provenance",
                        lambda name: {"version": "0.0.0", "provenance": "git"})
    def git(commit):
        return json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": commit}})
    monkeypatch.setattr(settings, "_direct_url", lambda dist: git("a" * 40))
    a = build_settings(_facts())
    monkeypatch.setattr(settings, "_direct_url", lambda dist: git("b" * 40))
    b = build_settings(_facts())
    assert a["de"]["engine"]["commit"] == "a" * 40 and a != b       # the fact check tells them apart
    monkeypatch.setattr(settings, "_direct_url",
                        lambda dist: json.dumps({"url": "x", "dir_info": {"editable": True}}))
    assert build_settings(_facts())["de"]["engine"]["commit"] is None   # editable: not identifiable


def test_engine_record_names_the_distribution_never_a_path():
    rec = engine_record("pdex")
    assert set(rec) == {"name", "version", "provenance", "commit"} and rec["name"] == "pdex"
    assert "/" not in json.dumps(rec)


def test_a_non_string_target_gene_label_is_refused():
    check_recordable(EvalConfig(target_gene_map={"123": "G1"}))
    with pytest.raises(ValueError, match="123"):
        check_recordable(EvalConfig(target_gene_map={123: "G1"}))
