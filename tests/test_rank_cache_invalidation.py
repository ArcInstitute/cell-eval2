"""The de_<method>_rank cache must not serve a stale rank matrix (2026-07-25 ultrareview).

The rank artifact was keyed on 4 rank knobs plus a VALUE-BLIND table fingerprint (height +
sorted column NAMES + target/feature value-counts -- no log2FC, no p-values), while the DE table
itself is keyed on ~20 content knobs. So changing a DE-content knob recomputed the table
correctly and then served the PREVIOUS run's ranks: every rank-derived de_* metric silently
wrong, while de_wilcoxon_nsig_* in the same frame was freshly correct.

Tested DIRECTLY on _prepare_de_cached rather than end-to-end through compute_metrics: the
DE-content knobs that are in the table key but not the rank key (mean_calc, epsilon, ...) move
log2FC without moving p-values, so the SIGNIFICANT SET is unchanged, and de_wilcoxon_overlap at
k=None computes intersect1d(real[:k], pred[:k]) / k over that whole set (metrics/de.py:169-179)
-- within-set reordering is invisible and the metric stays 1.0. An end-to-end test would
therefore pass with the bug present.
"""
import json
from dataclasses import replace

import numpy as np
import polars as pl

from cell_eval2 import EvalConfig
from cell_eval2.cache import CacheStore, fingerprint_de_table
from cell_eval2.de import prep_de_side
from cell_eval2.run import _prepare_de_cached, _result_config_digest

N_TARGETS, N_GENES = 3, 6


def _cfg():
    """Defaults are enough: sort_by='abs_log2_fold_change', p_adj_threshold=0.05,
    nan_lfc_policy='mask', min_abs_log2fc=0.0, method='wilcoxon', control='non-targeting',
    cache_strict=False -- the last one being what makes the OLD fingerprint non-strict."""
    return EvalConfig()


def _de_table(*, reverse: bool) -> pl.DataFrame:
    """A minimal gpudge-shaped DE table, in two variants.

    They share HEIGHT, COLUMN NAMES and (target, feature) value-counts -- i.e. everything the
    non-strict fingerprint looks at -- and differ ONLY in log2_fold_change, which fully determines
    the abs-LFC ranking. Every row is significant (p_adj=0 << the 0.05 threshold), so the
    significant SET is identical too and the only difference between the two rank matrices is
    gene ORDER within each target. That is precisely the collision the value-blind fingerprint
    allowed, reduced to its essentials.
    """
    lfc_row = np.arange(1.0, N_GENES + 1.0)
    if reverse:
        lfc_row = lfc_row[::-1]
    n = N_TARGETS * N_GENES
    return pl.DataFrame({
        "target": [f"GENE{i}" for i in range(N_TARGETS) for _ in range(N_GENES)],
        "feature": [f"g{j}" for _ in range(N_TARGETS) for j in range(N_GENES)],
        "log2_fold_change": np.tile(lfc_row, N_TARGETS),
        "p_value": np.full(n, 1e-8),
        "p_adj": np.zeros(n),
    })


def _prepare(table, store, cfg):
    """real side cached, pred side NOT (pred_store=None) -- so within one call pred_rank is the
    freshly computed oracle for what real_rank must equal."""
    return _prepare_de_cached(table, table, cfg=cfg, real_store=store, pred_store=None,
                              de_real_supplied=False, de_pred_supplied=False)


def test_a_changed_de_table_is_not_served_from_the_rank_cache(tmp_path):
    cfg = _cfg()
    store = CacheStore(str(tmp_path / "real"))
    a, b = _de_table(reverse=False), _de_table(reverse=True)

    first = _prepare(a, store, cfg)
    assert first.real_rank.equals(first.pred_rank), "sanity: both sides rank table A identically"

    second = _prepare(b, store, cfg)
    assert not second.pred_rank.equals(first.pred_rank), (
        "the two fixtures rank identically; the assertion below would be vacuous"
    )
    assert second.real_rank.equals(second.pred_rank), (
        "the real side was served table A's STALE rank matrix from the cache"
    )


def test_the_stored_rank_fingerprint_tracks_de_table_content(tmp_path):
    """Key-level companion: asserts on the manifest, so it cannot be satisfied by luck."""
    cfg = _cfg()
    root = tmp_path / "real"
    store = CacheStore(str(root))
    key = f"de_{cfg.de.method}_rank"

    def stored_fp():
        manifest = json.loads((root / "manifest.json").read_text())
        return manifest["artifacts"][key]["fingerprint"]

    _prepare(_de_table(reverse=False), store, cfg)
    fp_a = stored_fp()
    _prepare(_de_table(reverse=True), store, cfg)
    fp_b = stored_fp()

    # With the bug, the second call HITS, never calls store.put, and the manifest still holds A's
    # fingerprint -- so fp_a == fp_b.
    assert fp_a != fp_b, "the rank fingerprint is blind to DE-table content"


def test_the_two_fixtures_collide_under_the_old_value_blind_fingerprint():
    """Premise guard. Characterizes cache.py (which this PR does NOT change), so it passes before
    AND after the fix. Its job is to fail loudly if a future edit makes the two fixtures differ in
    height, columns or (target, feature) value-counts -- at which point their non-strict
    fingerprints would differ on their own and the two tests above would pass even with the bug
    restored."""
    cfg = _cfg()
    prepped = [
        prep_de_side(t, name="real", sort_by=cfg.de.sort_by,
                     nan_lfc_policy=cfg.de.nan_lfc_policy,
                     min_abs_log2fc=cfg.de.min_abs_log2fc)[0]
        for t in (_de_table(reverse=False), _de_table(reverse=True))
    ]
    assert fingerprint_de_table(prepped[0], strict=False) == \
        fingerprint_de_table(prepped[1], strict=False), "fixtures no longer collide"
    assert fingerprint_de_table(prepped[0], strict=True) != \
        fingerprint_de_table(prepped[1], strict=True), "strict hashing cannot tell them apart"


def test_result_digest_is_versioned_when_the_de_backend_enters_the_key(monkeypatch):
    """A result cached under the OLD rank semantics must not be served after the fix.

    compute_metrics consults the result cache BEFORE it calls _prepare_de_cached -- so a run that
    was already poisoned by a colliding rank would still be served from the result cache, and the
    newly strict rank key would never be reached. The digest therefore carries a semantics marker,
    added only when de_backend_used is True.
    """
    import cell_eval2.run as run_mod

    cfg = replace(_cfg(), de=replace(_cfg().de, backend="scanpy"))

    before_used = _result_config_digest(cfg, de_backend_used=True, comparator="lognorm")
    before_unused = _result_config_digest(cfg, de_backend_used=False, comparator="lognorm")

    monkeypatch.setattr(run_mod, "_DE_RESULT_SEMANTICS", run_mod._DE_RESULT_SEMANTICS + 1)
    after_used = _result_config_digest(cfg, de_backend_used=True, comparator="lognorm")
    after_unused = _result_config_digest(cfg, de_backend_used=False, comparator="lognorm")

    assert before_used != after_used, "bumping the marker must invalidate DE result entries"
    assert before_unused == after_unused, (
        "a run that never invokes the DE engine must keep its result cache"
    )
