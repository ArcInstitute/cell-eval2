"""Replay preserves recorded identity and refuses drift (#412, design §9)."""
import dataclasses
import json
import pathlib
import types

import pytest

import cell_eval2.cli as cli
import cell_eval2.run as run
import cell_eval2.settings as settings
from cell_eval2.catalog import CATALOG, resolve_metrics
from cell_eval2.evaluation_results import FILENAME, load_document
from cell_eval2.settings import SETTINGS, settings_to_config
from _edge_fixtures import config, scenario


@pytest.fixture(scope="module")
def first(tmp_path_factory):
    """Score once so replays exercise a real document (#412, design §9)."""
    root = tmp_path_factory.mktemp("settings_from")
    pred, real, _ = scenario("random")
    result = types.SimpleNamespace(pred=str(root / "pred.h5ad"),
                                   real=str(root / "real.h5ad"),
                                   doc=str(root / "first" / FILENAME))
    pred.write_h5ad(result.pred)
    real.write_h5ad(result.real)
    cli.main(["run", "-ap", result.pred, "-ar", result.real, "-o", str(root / "first"),
              "--profile", "full", "--set", "de.backend=pdex", "--set", "device=cpu"])
    return result


_replay = None


@pytest.fixture(autouse=True)
def _bind_replay(first, monkeypatch):
    """Bind the plan's replay shorthand per test (#412, design §9)."""
    def replay(out, *extra):
        cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(out),
                  "--settings-from", first.doc, *extra])
    monkeypatch.setitem(globals(), "_replay", replay)


def _get(value, path):
    for part in path.split(".") if isinstance(path, str) else path:
        value = value[part]
    return value


def _set_dotted(value, path, new_value):
    parts = path.split(".") if isinstance(path, str) else path
    _get(value, parts[:-1])[parts[-1]] = new_value


def test_replay_reproduces_settings_and_values(first, tmp_path):
    _replay(tmp_path / "o")
    a, b = load_document(first.doc), load_document(tmp_path / "o" / FILENAME)
    assert a["suite"]["settings"] == b["suite"]["settings"]
    meta = [json.loads((d / "run_meta.json").read_text())
            for d in (pathlib.Path(first.doc).parent, tmp_path / "o")]
    assert meta[0]["config_digest"] == meta[1]["config_digest"]      # same run identity
    _assert_same_values(a, b)


def test_supplied_real_de_replays_auto_backend_with_the_same_digest(first, tmp_path):
    _, real, _ = scenario("random")
    table = run._compute_de_side(real, cfg=config("de"), fp=None, store=None, side="real")
    de_path = tmp_path / "de_real.parquet"
    table.write_parquet(de_path)
    src, out = tmp_path / "src", tmp_path / "replay"
    args = ["run", "-ap", first.pred, "-ar", first.real, "--de-real", str(de_path)]
    cli.main([*args, "-o", str(src), "--profile", "de",
              "--set", "de.backend=auto", "--set", "device=cpu"])
    cli.main([*args, "-o", str(out), "--settings-from", str(src / FILENAME)])
    meta = [json.loads((d / "run_meta.json").read_text()) for d in (src, out)]
    assert meta[0]["config_digest"] == meta[1]["config_digest"]
    a, b = load_document(src / FILENAME), load_document(out / FILENAME)
    assert a["suite"]["settings"] == b["suite"]["settings"]
    _assert_same_values(a, b)


def _assert_same_values(a, b):
    for k, e in a["metrics"].items():
        tol = CATALOG[k].tolerance
        for x, y in zip(e.get("values", []) + list(e["statistics"].values()),
                        b["metrics"][k].get("values", []) + list(b["metrics"][k]["statistics"].values())):
            if x is None or y is None:
                assert x is y, k
            elif tol == "exact":
                assert x == y, k
            else:
                assert abs(x - y) <= 1e-8 + 1e-8 * abs(x), k


def test_an_int_valued_setting_replays_with_its_run_identity(first, tmp_path):
    # `--set` parses with yaml.safe_load, so `=1` is an int, and config_digest tells 1 from 1.0
    src, out = tmp_path / "src", tmp_path / "o"
    cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(src), "--profile", "full",
              "--set", "de.backend=pdex", "--set", "device=cpu", "--set", "de.min_abs_log2fc=1"])
    assert type(load_document(src / FILENAME)["suite"]["settings"]["de"]["min_abs_log2fc"]) is int
    cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(out),
              "--settings-from", str(src / FILENAME)])
    meta = [json.loads((d / "run_meta.json").read_text()) for d in (src, out)]
    assert meta[0]["config_digest"] == meta[1]["config_digest"]
    with pytest.raises(SystemExit, match="de.min_abs_log2fc"):    # 1 -> 1.0 moves the digest
        cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(tmp_path / "o2"),
                  "--settings-from", str(src / FILENAME), "--set", "de.min_abs_log2fc=1.0"])


def test_settings_to_config_replays_every_recorded_setting(first):
    doc = load_document(first.doc)
    cfg = settings_to_config(doc)
    s = doc["suite"]["settings"]
    for key in SETTINGS:
        if key.field is not None:
            assert _get(cfg.to_dict(), key.field) == _get(s, key.path), key.path
    assert cfg.metrics == list(doc["metrics"])                 # document order (Task 6)


def test_an_explicit_selection_replays_in_its_own_order(first, tmp_path):
    # resolve_metrics keeps an explicit order and puts a derived metric before its components;
    # config_digest hashes that order, and CATALOG order would reverse both
    sel = ["pds_l2", "pds_l1", "expr_mse_unbiased_capped_norm"]
    src, out = tmp_path / "src", tmp_path / "o"
    cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(src), "--set", "device=cpu",
              "--set", f"metrics=[{', '.join(sel)}]"])
    assert list(load_document(src / FILENAME)["metrics"]) == resolve_metrics(sel)[0]
    cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(out),
              "--settings-from", str(src / FILENAME)])
    meta = [json.loads((d / "run_meta.json").read_text()) for d in (src, out)]
    assert meta[0]["config_digest"] == meta[1]["config_digest"]


def test_a_resolved_target_sum_replays_with_equal_settings_and_values(first, tmp_path):
    # the one exception to run identity (Task 9, design §9): the document records the RESOLVED
    # median, config_digest hashes the REQUESTED None
    src, out = tmp_path / "src", tmp_path / "o"
    cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(src), "--profile", "full",
              "--set", "de.backend=pdex", "--set", "device=cpu", "--set", "target_sum=null"])
    a = load_document(src / FILENAME)
    assert a["suite"]["settings"]["target_sum"] is not None              # recorded resolved
    cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(out),
              "--settings-from", str(src / FILENAME)])
    b = load_document(out / FILENAME)
    assert a["suite"]["settings"] == b["suite"]["settings"]
    _assert_same_values(a, b)
    meta = [json.loads((d / "run_meta.json").read_text()) for d in (src, out)]
    assert meta[0]["config_digest"] != meta[1]["config_digest"]       # by design


@pytest.mark.parametrize("extra, match", [
    (["--preset", "v2"], "mutually exclusive"),
    (["--config", "x.yaml"], "mutually exclusive"),
    (["--profile", "de"], "--profile"),
    (["--set", "de.p_adj_threshold=0.01"], "de.p_adj_threshold"),
    (["--pert-col", "other"], "pert_col"),
    (["--no-target-genes"], "target_gene_map"),
])
def test_a_flag_that_changes_a_recorded_setting_is_refused(first, tmp_path, extra, match):
    with pytest.raises(SystemExit, match=match):
        _replay(tmp_path / "o", *extra)


def test_settings_from_without_a_prediction_is_refused_before_any_scoring(first, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "compute_ceiling", lambda *a, **k: pytest.fail("the ceiling ran"))
    with pytest.raises(SystemExit, match="-ap"):
        cli.main(["run", "-ar", first.real, "--ceiling", "-o", str(tmp_path / "o"),
                  "--settings-from", first.doc])


def test_exempt_overrides_are_allowed(first, tmp_path):
    _replay(tmp_path / "o", "--set", "num_threads=2", "--cache-pred", str(tmp_path / "c"))
    assert load_document(tmp_path / "o" / FILENAME)["suite"]["settings"] == \
        load_document(first.doc)["suite"]["settings"]


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.__setitem__("schema_version", 2), "schema_version"),
    (lambda d: d["suite"].__setitem__("name", "other_suite"), "other_suite"),
    (lambda d: d["metrics"]["pds_l1"].__setitem__("version", 2), "pds_l1"),
    (lambda d: d["metrics"].__setitem__("not_a_metric", d["metrics"]["pds_l1"]), "not_a_metric"),
    (lambda d: d["suite"]["settings"]["de"].pop("epsilon"), "de.epsilon"),
    (lambda d: d["suite"]["settings"].__setitem__("future_knob", 1), "future_knob"),   # RF 5
])
def test_document_level_refusals(first, tmp_path, mutate, match):
    doc = load_document(first.doc)
    mutate(doc)
    path = tmp_path / "d.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match=match):
        cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(tmp_path / "o"),
                  "--settings-from", str(path)])


@pytest.mark.parametrize("text, kind", [("[]", "list"), ("null", "NoneType")])
def test_a_non_object_document_exits_cleanly(first, tmp_path, text, kind):
    """Valid JSON that is not an object is refused as data, not a traceback (Copilot, #418)."""
    (tmp_path / "d.json").write_text(text)
    with pytest.raises(SystemExit, match=f"expected a JSON object, got {kind}"):
        cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(tmp_path / "o"),
                  "--settings-from", str(tmp_path / "d.json")])


def test_a_missing_document_exits_cleanly(first, tmp_path):
    """An unreadable --settings-from file is a clean refusal, not a traceback (Gemini, #418)."""
    with pytest.raises(SystemExit, match="--settings-from: .*No such file"):
        cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(tmp_path / "o"),
                  "--settings-from", str(tmp_path / "missing.json")])


@pytest.mark.parametrize("path, value", [
    ("scoring_path", "streaming"),
    ("input_type_pred", "lognorm"),
    ("comparator", "lognorm"),
    ("de.engine", {"name": "pdex", "version": "0.0.0", "provenance": "release"}),
])
def test_a_differing_fact_is_refused_before_any_metric_runs(first, tmp_path, monkeypatch, path, value):
    doc = load_document(first.doc)
    _set_dotted(doc["suite"]["settings"], path, value)
    (tmp_path / "d.json").write_text(json.dumps(doc))
    for name in ("dispatch_de_metrics", "dispatch_anndata_metrics"):
        monkeypatch.setattr(run, name, lambda *a, **k: pytest.fail("a metric ran"))
    with pytest.raises(SystemExit, match=path):
        cli.main(["run", "-ap", first.pred, "-ar", first.real, "-o", str(tmp_path / "o"),
                  "--settings-from", str(tmp_path / "d.json")])
    assert not (tmp_path / "o" / "results.csv").exists()


def test_a_missing_key_takes_its_declared_neutral_value(monkeypatch, first):
    doc = load_document(first.doc)
    del doc["suite"]["settings"]["target_sum"]
    keys = [dataclasses.replace(k, neutral=12345.0) if k.path == ("target_sum",) else k
            for k in settings.SETTINGS]
    monkeypatch.setattr(settings, "SETTINGS", tuple(keys))
    assert settings_to_config(doc).target_sum == 12345.0      # the neutral value, not the default
    monkeypatch.setattr(settings, "SETTINGS", tuple(
        dataclasses.replace(k, neutral=settings.NO_NEUTRAL) if k.path == ("target_sum",) else k
        for k in keys))
    with pytest.raises(ValueError, match="target_sum"):
        settings_to_config(doc)


@pytest.mark.parametrize("path, value", [
    ("de.future_knob", 1),
    ("discrimination.future_knob", {}),
    ("de.epsilon", -1),
])
def test_settings_reject_unknown_nested_keys_and_invalid_config(first, path, value):
    """Nested keys and config validation must not escape replay (#412, design §9)."""
    doc = load_document(first.doc)
    _set_dotted(doc["suite"]["settings"], path, value)
    with pytest.raises(ValueError, match=path.split(".")[-1]):
        settings_to_config(doc)


def test_missing_fact_settings_are_required_even_without_a_config_field(first):
    """A fact-only key still belongs to the required registry (#412, design §9)."""
    doc = load_document(first.doc)
    del doc["suite"]["settings"]["input_type_pred"]
    with pytest.raises(ValueError, match="input_type_pred"):
        settings_to_config(doc)


def test_changed_fields_compares_json_leaves_and_metrics_but_ignores_all_exempt_fields(first):
    """Overrides must preserve numeric spelling and selection order (#412, design §9)."""
    base = settings_to_config(load_document(first.doc))
    data = base.to_dict()
    data.update(outdir="other", cache_real="real-cache", cache_pred="pred-cache",
                cache_strict=not base.cache_strict, num_threads=2, gather_threads=3,
                allow_fractional_counts=not base.allow_fractional_counts)
    assert settings.changed_fields(base, settings.EvalConfig.from_dict(data)) == []
    data["metrics"] = list(reversed(data["metrics"]))
    data["de"]["min_abs_log2fc"] = int(data["de"]["min_abs_log2fc"])
    assert settings.changed_fields(base, settings.EvalConfig.from_dict(data)) == [
        "de.min_abs_log2fc", "metrics"]
    base = dataclasses.replace(base, target_gene_map={"GENE_A": "GENE_A"})
    cfg = dataclasses.replace(base, target_gene_map={"GENE_A": None})
    assert settings.changed_fields(base, cfg) == ["target_gene_map.GENE_A"]


def test_check_facts_names_every_difference_and_both_values(first):
    """One refusal must explain every resolved mismatch (#412, design §9)."""
    doc = load_document(first.doc)
    s = doc["suite"]["settings"]
    facts = run.RunFacts(config=settings_to_config(doc),
                         input_type_real=s["input_type_real"],
                         input_type_pred=s["input_type_pred"], comparator=s["comparator"],
                         device=s["device"], de_backend=s["de"]["backend"],
                         de_backend_used=True, scoring_path=s["scoring_path"], labels=())
    actual = settings.recorded_facts(s)
    assert actual == {path: _get(s, path) for path in settings.FACT_PATHS}
    settings.check_facts(actual, facts)
    expected = {path: f"different_{path}" for path in settings.FACT_PATHS}
    with pytest.raises(settings.SettingsMismatch) as exc:
        settings.check_facts(expected, facts)
    for path in settings.FACT_PATHS:
        assert f"{path}: expected {expected[path]!r}, got {actual[path]!r}" in str(exc.value)


def test_settings_from_refuses_a_metric_override_before_scoring(first, tmp_path, monkeypatch):
    """Metric selection is identity even though it is outside settings (#412, design §9)."""
    monkeypatch.setattr(cli, "_compute_metrics_with_facts",
                        lambda *a, **k: pytest.fail("a metric ran"))
    with pytest.raises(SystemExit, match="--settings-from:.*metrics"):
        _replay(tmp_path / "o", "--set", "metrics=[pds_l1]")
