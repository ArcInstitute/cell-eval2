"""Issue #248, checkpoint-2 review findings: the zero-resolve GATE and the result CACHE.

Two defects in the first cut of the #248 fix, both found by cross-provider review of the PR:

1. The gate was judged on the labels the kernel was HANDED. `scale.py` restricts the pred
   bulks to a shard and `partition_inmem.py` passes one piece at a time, while both hand the
   REAL bulks over whole -- so a panel resolving some of its targets scored fine as a whole
   and hard-failed on the shard that happened to hold only unresolved ones. The raise
   depended on how the data was chunked rather than on the data.

2. The result-cache key is (inputs + config + metric names) and carried nothing about the
   exclusion's MEANING, so a pre-#248 run that excluded nothing kept being served from cache
   under a key the fixed code reproduces exactly.

Both are regressions-in-waiting rather than cosmetics: (1) turns ordinary biology into a
crash, (2) silently preserves the very score the fix exists to correct.
"""

import numpy as np
import pytest
from cell_eval2.config import EvalConfig
from cell_eval2.distances import resolve_exclusion_columns
from cell_eval2.metrics.discrimination import discrimination_score
from cell_eval2.run import _discrimination_exclusion_used, _result_config_digest

# A PARTIAL panel: A-1 and B-1 resolve, ZZZ-1 does not (its gene is not measured).
_GENES = np.asarray(["A", "B", "C"], dtype=str)
_PERTS = np.array(["ctrl", "A-1", "B-1", "ZZZ-1"])
_MAP = {"A-1": "A", "B-1": "B", "ZZZ-1": "ZZZ_NOT_MEASURED"}


# --------------------------------------------------------------------------- the gate

def test_the_gate_is_judged_on_the_whole_panel_not_on_the_slice():
    """A shard holding only unresolved perturbations must not raise.

    This is the exact shape `scale.py::_restrict` and `partition_inmem.py` produce. Gating on
    the slice made a legitimate partial panel fail on some chunkings and not others.
    """
    whole = resolve_exclusion_columns(_PERTS, _GENES, target_gene_map=_MAP)
    assert whole == {1: 0, 2: 1}          # A-1 -> col A, B-1 -> col B; ctrl and ZZZ-1 omitted

    shard = np.array(["ZZZ-1"])           # the piece with nothing resolvable in it
    assert resolve_exclusion_columns(
        shard, _GENES, target_gene_map=_MAP, gate_labels=_PERTS
    ) == {}


def test_the_gate_still_fires_on_a_genuine_construct_id_mismatch():
    """The fix must not defang the raise -- that raise is the entire point of #248.

    Whole panel, no map: nothing resolves, so this IS the construct-ID-vs-symbol mismatch and
    scoring on would silently exclude nothing.
    """
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        resolve_exclusion_columns(_PERTS, _GENES, target_gene_map=None, gate_labels=_PERTS)


def test_a_slice_cannot_mask_a_mismatch_the_whole_panel_has():
    """The mirror of the first test: gating globally must not let a resolvable SHARD hide a
    panel that resolves nothing. Here the gate set resolves nothing, so it raises even though
    the slice handed in would have resolved on its own."""
    unresolvable = np.array(["ZZZ-1", "YYY-1"])
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        resolve_exclusion_columns(np.array(["A"]), _GENES, gate_labels=unresolvable)


def test_the_kernel_gates_on_the_real_keys_not_the_pred_keys():
    """End to end through the CPU kernel: pred restricted to the unresolved perturbation,
    real whole. Before the fix this raised.

    The shard carries the control row because that is what `scale.py` produces --
    `_restrict(pred_bulks, chosen | {cfg.control})` always keeps it -- so this is the real
    shape, not a contrived one. The control resolves to no gene either, which is exactly why
    a control-plus-unresolved shard was the tripwire.
    """
    real = np.array([[0.0, 0.0, 0.0], [-9.0, 1.0, 0.0], [0.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
    pred_keys = np.array(["ctrl", "ZZZ-1"])
    pred = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = discrimination_score(
        pred_bulk=(pred_keys, pred), real_bulk=(_PERTS, real), genes=_GENES,
        distance="l1", control="ctrl", exclude_target_gene=True, target_gene_map=_MAP,
    )
    assert set(out) == {"ZZZ-1"}
    assert np.isfinite(list(out.values())[0])


# ------------------------------------------------------------------- map precedence

def test_the_map_is_authoritative_even_when_the_raw_label_is_a_real_gene():
    """`target_gene_map` wins over a raw label that happens to name a measured gene.

    Without this, a raw-first-with-map-fallback implementation would pass every other test in
    the suite while silently violating the documented contract -- the map exists precisely to
    override what the label looks like.
    """
    cols = resolve_exclusion_columns(np.array(["A"]), _GENES, target_gene_map={"A": "B"})
    assert cols == {0: 1}, "the map must select gene B, not the raw label's own gene A"


def test_an_authoritative_map_pointing_at_an_unmeasured_gene_is_unresolved():
    """...and a map entry naming a gene absent from the panel resolves to NOTHING rather than
    silently falling back to the raw label. There is no column to drop, and counting it as
    resolved would reopen the hole for a map that is present but wrong."""
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        resolve_exclusion_columns(np.array(["A"]), _GENES, target_gene_map={"A": "NOPE"})


# --------------------------------------------------------------------------- the cache

def _cfg(**kw):
    return EvalConfig(metrics=["pds_l1"], version="v2", device="cpu", control="ctrl",
                      control_source="pred", **kw)


def test_the_exclusion_semantics_term_is_scoped_to_exclusion_enabled_pds_runs():
    """The scoping is by (metric family, exclusion flag) -- NOT by whether this particular
    panel was actually affected. A gene-level panel that scored identically before and after
    #248 still takes a new key and recomputes once; see `_discrimination_exclusion_used` for
    why that cannot be decided at digest time."""
    cfg = _cfg()
    assert cfg.discrimination.exclude_target_gene is True
    assert _discrimination_exclusion_used(cfg, ["pds_l1"]) is True
    assert _discrimination_exclusion_used(cfg, ["pds_cosine", "expr_mae"]) is True
    # No discrimination metric -> the #248 hole could not have touched this run.
    assert _discrimination_exclusion_used(cfg, ["expr_mae", "expr_mse"]) is False
    off = _cfg()
    off.discrimination.exclude_target_gene = False
    assert _discrimination_exclusion_used(off, ["pds_l1"]) is False


def test_a_pre_248_warm_cache_cannot_be_served_to_the_fixed_code():
    """THE REGRESSION TEST. Identical config and identical inputs; the only difference is
    whether the run is one the exclusion hole could have poisoned. The digests must differ,
    or a warm cache keeps returning the inflated pre-#248 score."""
    cfg = _cfg(target_gene_map={"A-1": "A"})
    poisonable = _result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm", pds_exclusion_used=True,
    )
    pre_248 = _result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm", pds_exclusion_used=False,
    )
    assert poisonable != pre_248


def test_a_run_with_no_discrimination_metric_keeps_its_warm_cache():
    """The scoping half: a run with no `pds_*` metric could not have been touched by the #248
    hole under any panel, so it must not be invalidated. Bumping the global key instead would
    be safe but would cold-start every cache in the project; the file's existing terms
    (`replicate_col`, `de_rank_cache_semantics`) are all scoped this way."""
    cfg = _cfg()
    unaffected = ["expr_mae"]
    assert _result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm",
        pds_exclusion_used=_discrimination_exclusion_used(cfg, unaffected),
    ) == _result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm", pds_exclusion_used=False,
    )
