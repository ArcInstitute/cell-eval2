import dataclasses

import pytest
from polars.testing import assert_frame_equal

from cell_eval2 import norm, run
from _edge_fixtures import CONTROL, PERTS, config, scenario


def test_in_memory_facts_and_an_unchanged_frame():
    pred, real, _ = scenario("random")
    cfg = config("full")
    df, facts = run._compute_metrics_with_facts(pred, real, config=cfg)
    # Row order, not content: de_*_lfc_nmae rows come back in varying order (#415).
    assert_frame_equal(df, run.compute_metrics(pred, real, config=cfg), check_row_order=False)
    assert facts.scoring_path == "in_memory" and facts.labels == tuple(PERTS)
    assert (facts.input_type_real, facts.input_type_pred) == ("counts", "counts")
    assert facts.comparator == "bulk_lognorm"
    assert (facts.device, facts.de_backend, facts.de_backend_used) == ("cpu", "pdex", True)
    assert facts.config.target_sum == 1e6 and facts.config.pert_col == "target"


def test_target_sum_none_is_recorded_resolved():
    pred, real, _ = scenario("random")
    cfg = dataclasses.replace(config("de"), target_sum=None)
    _, facts = run._compute_metrics_with_facts(pred, real, config=cfg)
    ctrl = real[real.obs["target"] == CONTROL]
    assert facts.config.target_sum == norm.resolve_target_sum(ctrl, input_type="counts",
                                                              target_sum=None)


def test_supplied_tables_name_no_engine():
    pred, real, _ = scenario("random")
    cfg = config("de")
    de_real = run._compute_de_side(real, cfg=cfg, fp=None, store=None, side="real")
    de_pred = run._compute_de_side(pred, cfg=cfg, fp=None, store=None, side="pred")
    _, facts = run._compute_metrics_with_facts(pred, real, config=dataclasses.replace(
        cfg, de=dataclasses.replace(cfg.de, backend="auto")), de_pred=de_pred, de_real=de_real)
    assert facts.de_backend_used is False and facts.de_backend == "auto"


def test_a_result_cache_hit_reports_the_same_facts(tmp_path):
    pred, real, _ = scenario("random")
    cfg = dataclasses.replace(config("minimal"), cache_pred=str(tmp_path / "p"))
    _, cold = run._compute_metrics_with_facts(pred, real, config=cfg)
    seen = []
    _, warm = run._compute_metrics_with_facts(pred, real, config=cfg, check_facts=seen.append)
    assert warm == cold and seen == [warm]


def test_check_facts_runs_before_any_metric(monkeypatch):
    pred, real, _ = scenario("random")
    for name in ("dispatch_de_metrics", "dispatch_anndata_metrics", "dispatch_cell_metrics"):
        monkeypatch.setattr(run, name, lambda *a, **k: pytest.fail("a metric ran"))

    def refuse(facts):
        raise RuntimeError("refused")
    with pytest.raises(RuntimeError, match="refused"):
        run._compute_metrics_with_facts(pred, real, config=config("full"), check_facts=refuse)
