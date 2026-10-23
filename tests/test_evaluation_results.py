"""Pin the document contract and CSV agreement (#412, design §6, contract §13–14)."""
import json
import errno
import builtins
from copy import deepcopy
from pathlib import Path
from stat import S_IMODE

import jsonschema
import polars as pl
import pytest

from cell_eval2 import evaluation_results
from cell_eval2.catalog_text import metric_text
from cell_eval2.config import DEParams, EvalConfig
from cell_eval2.evaluation_results import (
    FILENAME, build_document, check_document, load_document, no_document_reason,
    remove_stale_document, suite_commit, write_document,
)
from cell_eval2.run import RunFacts, _compute_metrics_with_facts
from cell_eval2.settings import build_settings
from _edge_fixtures import config, scenario

SCHEMA = json.loads((Path(__file__).parent / "fixtures/arcbench/evaluation_results_v1.schema.json").read_text())


def _validate(doc):
    jsonschema.Draft202012Validator(SCHEMA).validate(doc)


def _frame(rows):
    return pl.DataFrame(rows, schema={"perturbation": pl.Utf8, "metric": pl.Utf8,
                                     "value": pl.Float64}, orient="row")


def _facts(**kw):
    base = dict(config=EvalConfig(), input_type_real="counts", input_type_pred="counts",
                comparator="bulk_lognorm", device="cpu", de_backend="pdex",
                de_backend_used=True, scoring_path="in_memory", labels=("A", "B", "C"))
    return RunFacts(**{**base, **kw})


def test_values_follow_the_labels_and_nan_or_missing_rows_become_null():
    df = _frame([("B", "pds_l1", 0.5), ("A", "pds_l1", float("nan"))])          # C: no row
    doc = build_document(df, _facts(), metrics=["pds_l1"])
    m = doc["metrics"]["pds_l1"]
    assert doc["axes"]["perturbation"]["labels"] == ["A", "B", "C"]
    assert m["axis"] == "perturbation" and m["values"] == [None, 0.5, None]
    assert m["statistics"] == {"count": 3, "null_count": 2, "mean": 0.5, "std": None,
                               "min": 0.5, "max": 0.5, "median": 0.5}
    assert m["headline"] == {"statistic": "mean", "direction": "higher", "min": 0.0, "max": 1.0}
    _validate(doc)


def test_a_scalar_metric_has_one_value_and_no_axis():
    df = _frame([(p, "de_wilcoxon_nsig_spearman", 0.25) for p in "ABC"])
    m = build_document(df, _facts(), metrics=["de_wilcoxon_nsig_spearman"])["metrics"][
        "de_wilcoxon_nsig_spearman"]
    assert m["statistics"] == {"value": 0.25} and m["headline"]["statistic"] == "value"
    assert "axis" not in m and "values" not in m


def test_only_scalar_metrics_leave_axes_empty():                 # Review Focus 3
    df = _frame([(p, "de_wilcoxon_nsig_spearman", float("nan")) for p in "ABC"])
    doc = build_document(df, _facts(), metrics=["de_wilcoxon_nsig_spearman"])
    assert doc["axes"] == {} and doc["metrics"]["de_wilcoxon_nsig_spearman"]["statistics"] == {"value": None}
    _validate(doc)


def test_disagreeing_scalar_rows_raise():
    df = _frame([("A", "de_wilcoxon_nsig_spearman", 0.1), ("B", "de_wilcoxon_nsig_spearman", 0.2)])
    with pytest.raises(ValueError, match="nsig_spearman"):
        build_document(df, _facts(), metrics=["de_wilcoxon_nsig_spearman"])


def test_the_derived_metric_is_its_ratio_of_sums():
    rows = [(p, "expr_mse_unbiased_capped", v) for p, v in zip("ABC", (1.0, 2.0, 3.0))]
    rows += [(p, "expr_distance_unbiased", v) for p, v in zip("ABC", (2.0, 2.0, 8.0))]
    doc = build_document(_frame(rows), _facts(), metrics=["expr_mse_unbiased_capped_norm"])
    m = doc["metrics"]["expr_mse_unbiased_capped_norm"]
    assert m["statistics"] == {"ratio_of_sums": 0.5} and "axis" not in m
    assert "expr_distance_unbiased" in doc["metrics"]          # components ride along


@pytest.mark.parametrize("rows, labels, match", [
    ([("A", "pds_l1", float("inf"))], ("A",), "finite"),
    ([("Z", "pds_l1", 0.5)], ("A",), "label"),
    ([("A", "pds_l1", 0.5), ("A", "pds_l1", 0.6)], ("A",), "duplicate"),
    ([("A", "not_a_metric", 0.5)], ("A",), "not_a_metric"),
    ([], (), "label"),
])
def test_build_refuses_bad_frames(rows, labels, match):
    with pytest.raises(ValueError, match=match):
        build_document(_frame(rows), _facts(labels=labels), metrics=["pds_l1"])


def test_v1_and_deseq2_configs_write_no_document():
    for cfg in (EvalConfig(version="v1"), EvalConfig(de=DEParams(backend="deseq2"))):
        assert "writes no evaluation_results.json" in no_document_reason(cfg)
        with pytest.raises(ValueError, match="writes no evaluation_results.json"):
            build_document(_frame([("A", "pds_l1", 0.5)]), _facts(config=cfg, labels=("A",)),
                           metrics=["pds_l1"])
    assert no_document_reason(EvalConfig()) is None


def test_metric_entry_fields():
    doc = build_document(_frame([("A", "pds_l1", 0.5), ("A", "de_wilcoxon_lfc_nmae", 0.3),
                                 ("A", "de_wilcoxon_sig_jaccard", 0.2)]),
                         _facts(labels=("A",)),
                         metrics=["pds_l1", "de_wilcoxon_lfc_nmae", "de_wilcoxon_sig_jaccard",
                                  "expr_real_mass_ratio"])
    pds, nmae, jac, mass = (doc["metrics"][k] for k in ("pds_l1", "de_wilcoxon_lfc_nmae",
                                                         "de_wilcoxon_sig_jaccard",
                                                         "expr_real_mass_ratio"))
    assert pds["aliases"] == ["discrimination_score_l1"] and "aliases" not in jac
    assert pds["facets"] == {"family": "pds", "kind": "anndata"}
    assert nmae["facets"] == {"family": "lfc", "kind": "de", "de_method": "wilcoxon"}
    assert nmae["headline"] == {"statistic": "mean", "direction": "lower", "min": 0.0}
    assert mass["headline"]["direction"] is None and mass["statistics"]["count"] == 1
    assert pds["version"] == 1 and pds["display_name"] == metric_text("pds_l1").display_name


def test_suite_block():
    import cell_eval2
    doc = build_document(_frame([("A", "pds_l1", 0.5)]), _facts(labels=("A",)), metrics=["pds_l1"])
    assert doc["schema_version"] == 1
    assert doc["suite"] == {"name": "cell_eval2", "version": cell_eval2.__version__,
                            "commit": suite_commit(), "settings": build_settings(_facts(labels=("A",)))}


@pytest.mark.parametrize("raw, want", [
    (json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}), "a" * 40),
    (json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": "abc"}}), None),
    (json.dumps({"url": "file:///x", "dir_info": {"editable": True}}), None),
    ("not json", None),
    (None, None),
])
def test_suite_commit(monkeypatch, raw, want):
    monkeypatch.setattr(evaluation_results, "_direct_url", lambda: raw)
    assert suite_commit() == want


def test_write_then_load_is_strict_json_and_schema_valid(tmp_path):
    doc = build_document(_frame([("A", "pds_l1", float("nan"))]), _facts(labels=("A",)), metrics=["pds_l1"])
    write_document(doc, tmp_path / FILENAME)
    assert load_document(tmp_path / FILENAME) == doc
    _validate(load_document(tmp_path / FILENAME))


def test_written_document_uses_normal_file_permissions(tmp_path, valid_document):
    """Readers need the same access as other published outputs (#412, design §6.1)."""
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / FILENAME
    write_document(valid_document, destination)
    assert S_IMODE(destination.stat().st_mode) == S_IMODE(ordinary.stat().st_mode)


@pytest.mark.parametrize("existing", [False, True])
def test_partial_write_leaves_destination_intact_and_no_temporary_sibling(
        tmp_path, monkeypatch, existing):
    """A failed publication must preserve the last document (#412, design §6.1)."""
    doc = build_document(_frame([("A", "pds_l1", 0.5)]), _facts(labels=("A",)),
                         metrics=["pds_l1"])
    destination = tmp_path / FILENAME
    if existing:
        destination.write_text("previous document\n")
    original_open = builtins.open
    writes = []

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, value):
            writes.append(self.stream.write(value[:20]))
            self.stream.flush()
            raise OSError(errno.ENOSPC, "simulated full disk")

    def failing_open(*args, **kwargs):
        stream = original_open(*args, **kwargs)
        return FailingWriter(stream) if "x" in stream.mode else stream

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError, match="simulated full disk"):
        write_document(doc, destination)
    assert writes == [20]
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])
    if existing:
        assert destination.read_text() == "previous document\n"


@pytest.mark.parametrize("text, match", [('{"a": NaN}', "NaN"), ('{"a": 1, "a": 2}', "repeated")])
def test_load_document_refuses_non_strict_json(tmp_path, text, match):
    (tmp_path / "d.json").write_text(text)
    with pytest.raises(ValueError, match=match):
        load_document(tmp_path / "d.json")


def test_a_real_in_memory_run():
    from cell_eval2.run import aggregate_metrics_wide, metric_output_names
    pred, real, _ = scenario("real_zero_delta")
    cfg = config("full")
    df, facts = _compute_metrics_with_facts(pred, real, config=cfg)
    doc = build_document(df, facts, metrics=metric_output_names(cfg))
    _validate(doc)
    wide = aggregate_metrics_wide(df, metrics=metric_output_names(cfg))
    stat = {s: i for i, s in enumerate(wide["statistic"].to_list())}
    assert sorted(doc["metrics"]) == wide.columns[1:]
    for m, e in doc["metrics"].items():
        if "axis" in e:
            for s in ("mean", "std", "min", "max", "median"):
                w = wide[m][stat[s]]
                assert e["statistics"][s] == (None if w != w else w), (m, s)   # bit for bit
    assert doc["metrics"]["de_wilcoxon_lfc_nmae"]["statistics"]["null_count"] == 1
    derived = "expr_mse_unbiased_capped_norm"
    assert doc["metrics"][derived]["statistics"]["ratio_of_sums"] == wide[derived][stat["mean"]]


@pytest.fixture
def valid_document():
    rows = [("A", "pds_l1", 0.5), ("A", "de_wilcoxon_lfc_nmae", 0.3),
            ("A", "de_wilcoxon_nsig_spearman", 0.25),
            ("A", "expr_mse_unbiased_capped", 1.0),
            ("A", "expr_distance_unbiased", 2.0)]
    return build_document(_frame(rows), _facts(),
                          metrics=[r[1] for r in rows] + ["expr_mse_unbiased_capped_norm"])


@pytest.mark.parametrize("mutation, match", [
    (lambda d: d["metrics"].__setitem__("Bad", d["metrics"].pop("pds_l1")), "slug"),
    (lambda d: d["metrics"]["pds_l1"]["statistics"].__setitem__("Mean", 0.5), "slug"),
    (lambda d: d["metrics"]["pds_l1"].__setitem__("aliases", ["1x"]), "label"),
    (lambda d: d["metrics"]["de_wilcoxon_lfc_nmae"].__setitem__(
        "aliases", d["metrics"]["pds_l1"]["aliases"]), "alias"),
    (lambda d: d["metrics"]["pds_l1"].__setitem__("aliases", ["pds_l1"]), "alias"),
    (lambda d: d["metrics"]["pds_l1"]["headline"].__setitem__("min", 1.0), "min"),
    (lambda d: d["metrics"]["pds_l1"]["values"].pop(), "values"),
    (lambda d: d["axes"].__setitem__("unused", deepcopy(d["axes"]["perturbation"])), "axis"),
    (lambda d: d["metrics"]["pds_l1"]["facets"].__setitem__("unknown", "x"), "facet"),
    (lambda d: d["metrics"]["pds_l1"]["statistics"].pop("mean"), "statistic"),
    (lambda d: d["metrics"]["pds_l1"]["statistics"].__setitem__("unknown", 0.5), "statistic"),
    (lambda d: d["metrics"]["pds_l1"]["statistics"].__setitem__("mean", 1.5), "bound"),
    (lambda d: d["metrics"]["pds_l1"]["values"].__setitem__(0, True), "number"),
    (lambda d: d["metrics"]["pds_l1"]["statistics"].__setitem__("mean", float("nan")), "finite"),
    (lambda d: d["metrics"]["pds_l1"].__setitem__("version", 0), "version"),
    (lambda d: d["suite"].__setitem__("commit", "abc"), "commit"),
    (lambda d: d["axes"]["perturbation"]["labels"].__setitem__(1, "A"), "label"),
    (lambda d: d["axes"]["perturbation"]["labels"].__setitem__(0, ""), "label"),
])
def test_check_document_catches_each_rule(valid_document, mutation, match):
    doc = deepcopy(valid_document)
    mutation(doc)
    with pytest.raises(ValueError, match=match):
        check_document(doc)


def test_remove_stale_document(tmp_path, caplog):
    assert not remove_stale_document(tmp_path)
    path = tmp_path / FILENAME
    path.write_text("old run", encoding="utf-8")
    with caplog.at_level("INFO", logger=evaluation_results.__name__):
        assert remove_stale_document(tmp_path)
    assert not path.exists() and FILENAME in caplog.text
    assert not remove_stale_document(tmp_path)


def _replace(doc, path, value):
    for part in path[:-1]:
        doc = doc[part]
    doc[path[-1]] = value


@pytest.mark.parametrize("path, value", [
    (("schema_version",), True),
    (("schema_version",), 2),
    (("suite", "name"), "Bad"),
    (("suite", "version"), ""),
    (("suite", "commit"), "A" * 40),
    (("suite", "commit"), "a" * 40 + "\n"),
    (("suite", "settings"), []),
    (("suite", "settings", "nested"), {1: "bad key"}),
    (("suite", "settings", "nested"), [float("inf")]),
    (("suite", "settings", "nested"), object()),
    (("statistics",), {}),
    (("statistics", "mean", "combine"), "unknown"),
    (("statistics", "mean", "description"), ""),
    (("axes", "perturbation", "labels"), []),
    (("axes", "perturbation", "labels"), [1]),
    (("axes", "perturbation", "labels"), "ABC"),
    (("metrics",), {}),
    (("metrics", "pds_l1", "display_name"), ""),
    (("metrics", "pds_l1", "description"), None),
    (("metrics", "pds_l1", "version"), True),
    (("metrics", "pds_l1", "version"), 1.0),
    (("metrics", "pds_l1", "headline", "direction"), "up"),
    (("metrics", "pds_l1", "headline", "statistic"), "mean\n"),
    (("metrics", "pds_l1", "headline", "statistic"), "unknown"),
    (("metrics", "pds_l1", "headline", "min"), None),
    (("metrics", "pds_l1", "headline", "max"), float("inf")),
    (("metrics", "pds_l1", "statistics", "mean"), float("-inf")),
    (("metrics", "pds_l1", "statistics", "mean"), "0.5"),
    (("metrics", "pds_l1", "statistics", "count"), True),
    (("metrics", "pds_l1", "statistics", "mean"), -0.1),
    (("metrics", "pds_l1", "aliases"), ["same", "same"]),
    (("metrics", "pds_l1", "aliases"), ["label\n"]),
    (("metrics", "pds_l1", "aliases"), "label"),
    (("metrics", "pds_l1", "facets", "family"), ""),
    (("metrics", "pds_l1", "facets", "Bad"), "pds"),
    (("metrics", "pds_l1", "axis"), "unknown"),
    (("metrics", "pds_l1", "values"), "ABC"),
    (("metrics", "pds_l1", "values"), [float("-inf"), None, None]),
])
def test_check_document_reports_invalid_field_path(valid_document, path, value):
    import re
    doc = deepcopy(valid_document)
    _replace(doc, path, value)
    with pytest.raises(ValueError, match=re.escape(".".join(path))):
        check_document(doc)


@pytest.mark.parametrize("group", ["axes", "facets", "statistics"])
def test_declaration_names_must_be_slugs(valid_document, group):
    valid_document[group]["Bad"] = next(iter(valid_document[group].values()))
    with pytest.raises(ValueError, match=rf"{group}\.Bad.*slug"):
        check_document(valid_document)


@pytest.mark.parametrize("field", ["axis", "values"])
def test_axis_and_values_are_required_together(valid_document, field):
    valid_document["metrics"]["pds_l1"].pop(field)
    with pytest.raises(ValueError, match="metrics.pds_l1.*axis.*values"):
        check_document(valid_document)


@pytest.mark.parametrize("path", [(), ("suite",), ("statistics", "mean"),
                                   ("axes", "perturbation"), ("facets", "family"),
                                   ("metrics", "pds_l1"), ("metrics", "pds_l1", "headline")])
def test_required_and_extra_fields_match_schema(valid_document, path):
    import re
    for operation in ("missing", "extra"):
        doc = deepcopy(valid_document)
        entry = doc
        for part in path:
            entry = entry[part]
        key = next(iter(entry)) if operation == "missing" else "unexpected"
        if operation == "missing":
            entry.pop(key)
        else:
            entry[key] = None
        with pytest.raises(ValueError, match=re.escape(".".join((*path, key)))):
            check_document(doc)
        with pytest.raises(jsonschema.ValidationError):
            _validate(doc)


def test_check_document_accepts_nested_mappings(valid_document):
    from collections import UserDict

    def mappings(value):
        if isinstance(value, dict):
            return UserDict({k: mappings(v) for k, v in value.items()})
        if isinstance(value, list):
            return [mappings(v) for v in value]
        return value

    check_document(mappings(valid_document))


def test_key_orders_preserve_the_run_selection():
    rows = [("B", "pds_l1", 0.5), ("A", "pds_l2", 0.25), ("A", "expr_mae", 1.0),
            ("A", "delta_mae", 2.0)]
    doc = build_document(_frame(rows), _facts(), metrics=["pds_l2", "pds_l1"])
    assert list(doc) == ["schema_version", "suite", "statistics", "axes", "facets", "metrics"]
    assert list(doc["suite"]) == ["name", "version", "commit", "settings"]
    assert list(doc["metrics"]) == ["pds_l2", "pds_l1", "delta_mae", "expr_mae"]
    assert list(doc["statistics"]) == ["count", "null_count", "mean", "std", "min", "max",
                                       "median", "value", "ratio_of_sums"]
    assert list(doc["facets"]) == ["family", "kind", "de_method"]
    entry = doc["metrics"]["pds_l1"]
    assert list(entry) == ["version", "headline", "display_name", "description", "aliases",
                           "facets", "statistics", "axis", "values"]
    assert list(entry["headline"]) == ["statistic", "direction", "min", "max"]
    assert list(entry["statistics"]) == ["count", "null_count", "mean", "std", "min", "max", "median"]


@pytest.mark.parametrize("values", [[None, float("nan"), None],
                                    [1e16, -1e16, 1.0], [0.2, None, 0.3]])
def test_statistics_match_csv_with_nulls_and_frame_order(values):
    from cell_eval2.run import aggregate_metrics_wide
    df = _frame([(p, "expr_mse_unbiased", v) for p, v in zip("CAB", values)])
    doc = build_document(df, _facts(), metrics=["expr_mse_unbiased"])
    entry = doc["metrics"]["expr_mse_unbiased"]
    wide = aggregate_metrics_wide(df, metrics=["expr_mse_unbiased"])
    for stat, value in wide.iter_rows():
        if stat not in ("count", "null_count"):
            assert entry["statistics"][stat] == (None if value != value else value)
    assert entry["statistics"]["null_count"] == sum(v is None or v != v for v in values)
    _validate(doc)


@pytest.mark.parametrize("rows", [[], [("A", "de_wilcoxon_nsig_spearman", None)],
                                  [("A", "de_wilcoxon_nsig_spearman", float("nan"))]])
def test_absent_or_undefined_scalar_is_null(rows):
    name = "de_wilcoxon_nsig_spearman"
    doc = build_document(_frame(rows), _facts(), metrics=[name])
    assert doc["metrics"][name]["statistics"] == {"value": None}
    assert doc["axes"] == {}
    _validate(doc)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
@pytest.mark.parametrize("name", ["pds_l1", "de_wilcoxon_nsig_spearman", "expr_distance_unbiased"])
def test_infinite_rows_are_never_dropped(value, name):
    with pytest.raises(ValueError, match="finite"):
        build_document(_frame([("A", name, value)]), _facts(), metrics=[name])


def test_derived_requires_components():
    with pytest.raises(ValueError, match="expr_mse_unbiased_capped"):
        build_document(_frame([]), _facts(), metrics=["expr_mse_unbiased_capped_norm"])


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_refuses_each_nonfinite_token(tmp_path, token):
    path = tmp_path / FILENAME
    path.write_text('{"nested": [' + token + ']}', encoding="utf-8")
    with pytest.raises(ValueError, match=token):
        load_document(path)


def test_load_reports_nested_duplicate_key(tmp_path):
    path = tmp_path / FILENAME
    path.write_text('{"nested": {"duplicate": 1, "duplicate": 2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeated.*duplicate"):
        load_document(path)


def test_write_validates_before_replacing_and_uses_exact_format(tmp_path, valid_document):
    path = tmp_path / FILENAME
    write_document(valid_document, path)
    original = (json.dumps(valid_document, allow_nan=False, indent=2) + "\n").encode("utf-8")
    assert path.read_bytes() == original
    loaded = load_document(path)
    check_document(loaded)
    loaded["metrics"]["pds_l1"]["version"] = 0
    with pytest.raises(ValueError, match="version"):
        write_document(loaded, path)
    assert path.read_bytes() == original


def test_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    assert SCHEMA["$id"] == "urn:arcbench:evaluation-results:1"
