"""Issue #248: `cfg.target_gene_map` must reach the discrimination metrics.

The unit-level fix is inert unless the dispatch threads the map. On `main` both branches
of `run.dispatch_anndata_metrics` passed `genes` and `exclude_target_gene` but never the
map, so nothing downstream could repair the construct-ID lookup -- while the eleven
chance-corrected DE metrics in the SAME run resolved through it. These tests pin the
wiring on both branches, and the agreement between the two families.
"""

import numpy as np
import pytest
from cell_eval2.config import EvalConfig
from cell_eval2.run import dispatch_anndata_metrics

# Guide-level panel: labels are construct IDs, genes are bare symbols. A-1's predicted
# effect nails its own transcript (gene A) but its remaining biology looks like B-1's,
# so its rank flips the moment the exclusion genuinely fires. See
# tests/test_discrimination.py::_guide_panel for the by-hand l1 arithmetic.
_PERTS = np.array(["ctrl", "A-1", "B-1"])
_REAL = np.array([[0.0, 0.0, 0.0], [-9.0, 1.0, 0.0], [0.0, 5.0, 5.0]])
_PRED = np.array([[0.0, 0.0, 0.0], [-9.0, 4.0, 4.0], [0.0, 5.0, 5.0]])
_GENES = np.asarray(["A", "B", "C"], dtype=str)
_MAP = {"A-1": "A", "B-1": "B"}


def _bulks():
    return {"lognorm": (_PERTS, _PRED)}, {"lognorm": (_PERTS, _REAL)}


def _cfg(**kw):
    return EvalConfig(metrics=["pds_l1"], version="v2", device="cpu", control="ctrl",
                      control_source="pred", **kw)


def _scores(cfg):
    pred_bulks, real_bulks = _bulks()
    rows = dispatch_anndata_metrics(
        ["pds_l1"], pred_bulks, real_bulks, _GENES, cfg, comparator="lognorm")
    return {r["perturbation"]: r["value"] for r in rows if r["metric"] == "pds_l1"}


def test_dispatch_threads_target_gene_map_into_discrimination():
    # THE WIRING TEST. Identical data and identical exclude_target_gene=True; the only
    # difference is cfg.target_gene_map. Before #248 this had no effect whatsoever.
    cfg = _cfg(target_gene_map=_MAP)
    cfg.discrimination.rank_denominator = "n"
    assert _scores(cfg) == pytest.approx({"A-1": 0.5, "B-1": 1.0})


def test_dispatch_without_the_map_raises_rather_than_scoring_the_leak():
    # The silent no-op is what #248 is about: this used to return {"A-1": 1.0, ...},
    # a plausible number produced with nothing excluded.
    cfg = _cfg()
    cfg.discrimination.rank_denominator = "n"
    assert cfg.target_gene_map is None
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        _scores(cfg)


def test_dispatch_map_is_inert_when_exclusion_is_disabled():
    # exclude_target_gene=False never indexes the panel, so the map must not move the
    # score -- and the un-excluded score is the inflated one (A-1 rescued by its own
    # transcript), which is precisely the value the leak was returning.
    cfg = _cfg(target_gene_map=_MAP)
    cfg.discrimination.exclude_target_gene = False
    cfg.discrimination.rank_denominator = "n"
    with_map = _scores(cfg)

    cfg_no_map = _cfg()
    cfg_no_map.discrimination.exclude_target_gene = False
    cfg_no_map.discrimination.rank_denominator = "n"
    assert with_map == pytest.approx(_scores(cfg_no_map))
    assert with_map == pytest.approx({"A-1": 1.0, "B-1": 1.0})


def test_target_gene_map_does_not_leak_into_other_anndata_metrics():
    # The map enters the dispatch through a signature-filtered kwargs dict, so a metric
    # that does not declare it must be unaffected (and must not raise on an unexpected
    # keyword). delta_pearson has no target-gene semantics at all.
    pred_bulks, real_bulks = _bulks()
    cfg_map = EvalConfig(metrics=["delta_pearson"], version="v2", device="cpu",
                         control="ctrl", control_source="pred", target_gene_map=_MAP)
    cfg_bare = EvalConfig(metrics=["delta_pearson"], version="v2", device="cpu",
                          control="ctrl", control_source="pred")
    got = dispatch_anndata_metrics(
        ["delta_pearson"], pred_bulks, real_bulks, _GENES, cfg_map,
        comparator="lognorm")
    exp = dispatch_anndata_metrics(
        ["delta_pearson"], pred_bulks, real_bulks, _GENES, cfg_bare,
        comparator="lognorm")
    assert [r["value"] for r in got] == pytest.approx([r["value"] for r in exp])


def test_target_gene_map_separates_result_cache_keys():
    # NEWLY LOAD-BEARING for pds_*: before #248 the map moved only the DE metrics, so a
    # pds-only run could not be affected by it. Now it can, which makes the map's
    # presence in the result-cache digest a correctness requirement rather than
    # bookkeeping -- two maps that score differently must not share a cache key.
    from cell_eval2.run import _result_config_digest

    a = _result_config_digest(
        _cfg(target_gene_map=_MAP), de_backend_used=False, comparator="lognorm",
    )
    b = _result_config_digest(_cfg(target_gene_map={"A-1": "C", "B-1": "B"}),
                              de_backend_used=False, comparator="lognorm")
    assert a != b

    # ...and an absent map still drops out of the digest, so warm caches from before
    # this change are not invalidated for runs that never supply one.
    assert (_result_config_digest(_cfg(), de_backend_used=False, comparator="lognorm")
            != a)
