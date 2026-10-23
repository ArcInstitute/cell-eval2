"""CLI document publication and failure ordering (#412, design §6.6 and §8)."""
import json
import logging
from pathlib import Path

import jsonschema
import polars as pl
import pytest

import cell_eval2.cli as cli
import cell_eval2.run as run
from cell_eval2.evaluation_results import FILENAME, load_document
from cell_eval2.run import RunFacts
from _edge_fixtures import scenario


SCHEMA = json.loads((Path(__file__).parent / "fixtures" / "arcbench" /
                     "evaluation_results_v1.schema.json").read_text())


def _validate(doc):
    jsonschema.Draft202012Validator(SCHEMA).validate(doc)


@pytest.fixture(scope="module")
def h5ads(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation_results_cli")
    pred, real, _ = scenario("real_zero_delta")
    paths = {"pred": str(root / "pred.h5ad"), "real": str(root / "real.h5ad")}
    pred.write_h5ad(paths["pred"])
    real.write_h5ad(paths["real"])
    return paths


def _run(h5ads, out, *extra):
    cli.main(["run", "-ap", h5ads["pred"], "-ar", h5ads["real"], "-o", str(out), "--profile",
              "full", "--set", "de.backend=pdex", "--set", "device=cpu", *extra])


def _fake_compute(p, r, *, config, **kw):
    frame = pl.DataFrame(schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64})
    facts = RunFacts(config=config, input_type_real="counts", input_type_pred="counts",
                     comparator="bulk_lognorm", device="cpu", de_backend="pdex",
                     de_backend_used=True, scoring_path="in_memory", labels=("G0",))
    return frame, facts


def test_run_writes_a_valid_document_beside_agg_results(h5ads, tmp_path):
    _run(h5ads, tmp_path / "o")
    doc = load_document(tmp_path / "o" / FILENAME)
    _validate(doc)
    agg = pl.read_csv(tmp_path / "o" / "agg_results.csv")
    assert sorted(doc["metrics"]) == agg.columns[1:]
    for m, e in doc["metrics"].items():
        if "axis" in e:
            col = dict(zip(agg["statistic"].to_list(), agg[m].to_list()))
            for s in ("mean", "std", "min", "max", "median"):
                assert e["statistics"][s] == (None if col[s] != col[s] else col[s]), (m, s)
    assert doc["suite"]["settings"]["scoring_path"] == "in_memory"


def test_the_document_is_written_last(h5ads, tmp_path, monkeypatch):
    seen = {}
    real_write = cli.write_document

    def spy(doc, path):
        seen.update({n: (tmp_path / "o" / n).exists()
                     for n in ("results.csv", "agg_results.csv", "metric_aggregation.csv", "run_meta.json")})
        real_write(doc, path)
    monkeypatch.setattr(cli, "write_document", spy)
    _run(h5ads, tmp_path / "o")
    assert all(seen.values()) and len(seen) == 4


def test_a_result_cache_hit_writes_the_same_document(h5ads, tmp_path):
    _run(h5ads, tmp_path / "a", "--cache-pred", str(tmp_path / "c"))
    _run(h5ads, tmp_path / "b", "--cache-pred", str(tmp_path / "c"))       # served from the cache
    assert (tmp_path / "a" / FILENAME).read_bytes() == (tmp_path / "b" / FILENAME).read_bytes()


# `minimal` carries no derived metric: the fake's empty frame cannot build one (aggregation
# would rightly raise), and the decision under test is made from the config alone.
@pytest.mark.parametrize("extra", [["--profile", "minimal", "--set", "version=v1"],
                                   ["--profile", "minimal", "--set", "de.backend=deseq2"]])
def test_v1_and_deseq2_write_none_warn_once_and_drop_a_stale_document(
        h5ads, tmp_path, monkeypatch, caplog, extra):                  # Review Focus 1
    out = tmp_path / "o"
    out.mkdir()
    (out / FILENAME).write_text("stale")
    monkeypatch.setattr(cli, "_compute_metrics_with_facts", _fake_compute)  # no DESeq2/v1 run needed
    with caplog.at_level(logging.WARNING):
        _run(h5ads, out, *extra)
    assert not (out / FILENAME).exists()
    assert (out / "agg_results.csv").exists()
    assert sum("writes no evaluation_results.json" in r.getMessage() for r in caplog.records) == 1


def test_real_data_only_mode_writes_none(h5ads, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "compute_ceiling", lambda *a, **k: (pl.DataFrame(), pl.DataFrame()))
    cli.main(["run", "-ar", str(h5ads["real"]), "--ceiling", "-o", str(tmp_path / "o")])
    assert not (tmp_path / "o" / FILENAME).exists()


def test_a_document_that_fails_to_build_leaves_the_aggregate_and_meta_alone(
        h5ads, tmp_path, monkeypatch):                                  # Review Focus 2
    out = tmp_path / "o"
    _run(h5ads, out)
    before = {n: (out / n).read_bytes() for n in ("agg_results.csv", "run_meta.json")}

    def refuse(*a, **k):
        raise ValueError("bad doc")
    monkeypatch.setattr(cli, "build_document", refuse)
    with pytest.raises(ValueError, match="bad doc"):
        _run(h5ads, out)
    assert {n: (out / n).read_bytes() for n in before} == before
    assert not (out / FILENAME).exists()


def test_a_failure_after_the_de_tables_are_written_leaves_no_document(h5ads, tmp_path, monkeypatch):
    out = tmp_path / "o"
    _run(h5ads, out)                                   # an earlier run's document is in place

    def boom(*a, **k):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(run, "dispatch_anndata_metrics", boom)   # runs after _write_de_tables
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _run(h5ads, out, "--write-degenes")
    assert (out / "de_real.parquet").exists() and not (out / FILENAME).exists()


def test_a_non_string_target_gene_label_is_refused_before_scoring(h5ads, tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("metrics: full\ntarget_gene_map:\n  123: G1\n")    # Review Focus 4
    monkeypatch.setattr(cli, "_compute_metrics_with_facts", lambda *a, **k: pytest.fail("scored"))
    with pytest.raises(SystemExit, match="123"):
        cli.main(["run", "-ap", str(h5ads["pred"]), "-ar", str(h5ads["real"]),
                  "--config", str(cfg), "-o", str(tmp_path / "o")])
