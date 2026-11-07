import dataclasses
import json
from decimal import Decimal
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest

from cell_eval2 import baseline, settings
from cell_eval2.baseline import DIGEST_EXEMPT_FIELDS, _environment_record, config_digest
from cell_eval2.config import DEParams, EvalConfig, FilterParams
from cell_eval2.run import RunFacts, _compute_metrics_with_facts
from cell_eval2.settings import (FACT_PATHS, NO_NEUTRAL, NOT_RECORDED, SETTINGS,
                                 SettingsMismatch, build_settings, check_facts,
                                 check_recordable, engine_record, facts_checker,
                                 recorded_facts, vcs_commit)
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
    assert s["de"]["engine"] == "pdex" and s["de"]["backend"] == "pdex"
    assert build_settings(_facts(de_backend_used=False, de_backend="auto"))["de"]["engine"] is None


def test_numbers_are_written_as_the_config_holds_them():
    s = build_settings(_facts(config=EvalConfig(target_sum=1_000_000)))
    assert type(s["target_sum"]) is int and type(s["de"]["epsilon"]) is float


def test_supplied_real_de_preserves_configured_backend_and_resolved_engine():
    facts = _facts(config=EvalConfig(de=DEParams(backend="auto")), de_real_computed=False)
    s = build_settings(facts)
    assert s["de"]["backend"] == "auto"
    assert s["de"]["engine"] == "pdex"


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


class _FakeDist:
    """The slice of ``importlib.metadata.Distribution`` that settings and provenance read."""

    def __init__(self, version, direct_url=None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name):
        return None if self._direct_url is None else json.dumps(self._direct_url)


def _git(commit):
    return {"url": "x", "vcs_info": {"vcs": "git", "commit_id": commit}}


_ENGINE_VARIANTS = [
    ("0.0.1", None, "release", None),
    ("9.9.9", None, "release", None),
    ("0.0.1", {"url": "x", "dir_info": {"editable": True}}, "local-editable", None),
    ("0.0.1", _git("a" * 40), "git", "a" * 40),
    ("0.0.1", _git("b" * 40), "git", "b" * 40),
]


def _differing_paths(a, b):
    return {_dotted(p) for p in _leaves(a) if settings._get(a, p) != settings._get(b, p)}


def test_an_engine_version_provenance_or_commit_does_not_change_the_settings(monkeypatch):
    base = build_settings(_facts())
    for version, direct_url, provenance, commit in _ENGINE_VARIANTS:
        monkeypatch.setattr(baseline.Distribution, "from_name", staticmethod(
            lambda name, v=version, d=direct_url: _FakeDist(v, d)))
        # The environment record sees the variant, so the equality below is not vacuous.
        assert _environment_record("pdex")["pdex"] == {"version": version,
                                                       "provenance": provenance}
        assert vcs_commit(settings._direct_url("pdex")) == commit
        assert build_settings(_facts()) == base


@pytest.mark.parametrize("error", [RuntimeError("unreadable"), PackageNotFoundError("pdex")])
def test_the_engine_name_is_recorded_without_any_metadata_lookup(monkeypatch, error):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(baseline.Distribution, "from_name", staticmethod(boom))
    monkeypatch.setattr(settings, "_direct_url", boom)
    assert build_settings(_facts())["de"]["engine"] == "pdex"


def test_a_different_engine_name_or_device_changes_the_settings():
    base = build_settings(_facts(de_real_computed=False))
    other = build_settings(_facts(de_backend="scanpy", de_real_computed=False))
    assert other["de"]["engine"] == "scanpy"
    assert _differing_paths(base, other) == {"de.engine"}
    assert build_settings(_facts(device="cuda")) != build_settings(_facts())


@pytest.mark.parametrize("backend, distribution", [("gpudge", "gpudge"), ("pdex", "pdex"),
                                                   ("scanpy", "scanpy"), ("deseq2", "deseq2_gpu")])
def test_engine_record_is_the_distribution_name(backend, distribution):
    assert engine_record(backend) == distribution


_OLD_ENGINE = {"name": "pdex", "version": "0.0.1", "provenance": "git", "commit": "a" * 40}


def _expected_facts(engine):
    expected = recorded_facts(build_settings(_facts()))
    expected["de.engine"] = engine
    return expected


@pytest.mark.parametrize("engine", [
    "pdex",
    _OLD_ENGINE,
    {**_OLD_ENGINE, "version": "9.9.9", "provenance": "release", "commit": None},
    {"name": "pdex"},
])
def test_replay_accepts_an_engine_of_the_same_name_in_either_shape(engine):
    check_facts(_expected_facts(engine), _facts())


@pytest.mark.parametrize("engine", [
    "scanpy",
    {**_OLD_ENGINE, "name": "scanpy"},
    None,
    {"version": "0.0.1", "provenance": "release"},
    {"name": 1},
])
def test_replay_refuses_an_engine_of_another_name(engine):
    with pytest.raises(SettingsMismatch, match="de.engine"):
        check_facts(_expected_facts(engine), _facts())


def test_replay_refuses_an_engine_where_none_ran_and_the_reverse():
    none_ran = _facts(de_backend_used=False, de_backend="auto")
    with pytest.raises(SettingsMismatch, match="de.engine: expected 'pdex', got None"):
        check_facts(_expected_facts("pdex"), none_ran)
    with pytest.raises(SettingsMismatch, match="de.engine: expected None, got 'pdex'"):
        check_facts(_expected_facts(None), _facts())
    check_facts(_expected_facts(None), none_ran)


def test_the_refusal_prints_the_engine_name_of_an_old_expected_object():
    with pytest.raises(SettingsMismatch) as exc:
        check_facts(_expected_facts({**_OLD_ENGINE, "name": "scanpy"}), _facts())
    assert "de.engine: expected 'scanpy', got 'pdex'" in str(exc.value)


def test_facts_checker_replays_a_document_that_holds_the_old_engine_object():
    doc = {"suite": {"settings": build_settings(_facts())}}
    doc["suite"]["settings"]["de"]["engine"] = _OLD_ENGINE
    facts_checker(doc)(_facts())
    with pytest.raises(SettingsMismatch, match="de.engine"):
        facts_checker(doc)(_facts(de_backend="scanpy"))


def test_a_non_string_target_gene_label_is_refused():
    check_recordable(EvalConfig(target_gene_map={"123": "G1"}))
    with pytest.raises(ValueError, match="123"):
        check_recordable(EvalConfig(target_gene_map={123: "G1"}))
