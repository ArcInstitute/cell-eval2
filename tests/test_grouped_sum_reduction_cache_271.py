"""#271: every cache that can serve a pre-fix group sum must key on the reduction semantics.

`prep._grouped_sums` reduces WIDE as of #271, so the `bulk_lognorm` pseudobulk (and the deseq2
backend's per-replicate pseudobulk) moved. ⚠️ **Neither the version string nor the competition
digest can see that.** The result cache keys on (inputs + config) with `cell_eval2_version`
deliberately absent, and `competition_digest()` does not move -- the competition RULE is unchanged;
what moved is the pseudobulk the members read. So without the terms these tests pin, a pre-#271 run
at the SAME version reproduces every key exactly and its cached bulk / moments / deseq2 DE table /
final score is served in preference to recomputing.

Same shape as `test_discrimination_exclusion_gate_and_cache.py` (#248) and
`test_partition.py::test_the_purity_floor_moves_the_partial_semantics_payload` (the purity floor):
mutate the term, assert the identity moves, and assert the SCOPING -- a run that could not have
been affected must keep its warm cache.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_eval2 import EvalConfig


def _cfg(**kw):
    from cell_eval2.run import _resolve_config
    base = dict(metrics=["expr_mae"], version="v2", device="cpu", control="ctrl",
                pert_col="target", input_type="counts", validate_input=False)
    return _resolve_config(EvalConfig(**{**base, **kw}), {})


# --------------------------------------------------------------------------- the predicate

def test_the_271_term_is_scoped_to_bulk_lognorm_and_to_deseq2():
    """By PATH, not by data. `prep._grouped_sums` has exactly two families of caller: the
    `bulk_lognorm` pseudobulk, and `deseq2_de._pseudobulk`. A `lognorm` run reduces through
    `_grouped_means` instead and must keep its warm cache; so must any other DE backend, whose
    tables come from cells rather than from a pseudobulk.

    ⚠️ It deliberately over-invalidates within those paths: an integer-count submission below
    float32's 2**24 reduces identically either way, but the key is computed before the bulks are
    built, so the values are not in hand -- see `_grouped_sum_reduction_used`."""
    from cell_eval2.run import _grouped_sum_reduction_used as used

    assert used(comparator="bulk_lognorm", de_backend=None) is True
    assert used(comparator="bulk_lognorm", de_backend="gpudge") is True
    assert used(comparator="lognorm", de_backend="deseq2") is True
    assert used(comparator=None, de_backend="deseq2") is True
    # the two cases that keep their cache
    assert used(comparator="lognorm", de_backend=None) is False
    assert used(comparator="lognorm", de_backend="pdex") is False


# --------------------------------------------------------------------------- the result cache

def test_a_pre_271_warm_RESULT_cache_cannot_be_served_to_the_fixed_code(monkeypatch):
    """THE REGRESSION TEST. Identical config, identical inputs; the only difference is the
    reduction semantics. The digests must differ, or a warm cache keeps returning a score
    computed from group sums this build would not produce."""
    from cell_eval2 import run
    from cell_eval2.run import _result_config_digest

    cfg = _cfg()
    now = _result_config_digest(cfg, de_backend_used=False, comparator="bulk_lognorm")
    monkeypatch.setattr(run, "_GROUPED_SUM_REDUCTION_SEMANTICS", 0)
    pre_271 = _result_config_digest(cfg, de_backend_used=False, comparator="bulk_lognorm")
    assert now != pre_271, (
        "the result digest does not carry the #271 reduction semantics, so a pre-fix cached "
        "score would be served for an identical config and identical inputs")


def test_a_LOGNORM_run_keeps_its_warm_result_cache_across_271(monkeypatch):
    """The other half of the same statement, and the reason the term is scoped: a run that never
    reaches `_grouped_sums` must not be invalidated. Its digest is INSENSITIVE to the term."""
    from cell_eval2 import run
    from cell_eval2.run import _result_config_digest

    cfg = _cfg()
    now = _result_config_digest(cfg, de_backend_used=False, comparator="lognorm")
    monkeypatch.setattr(run, "_GROUPED_SUM_REDUCTION_SEMANTICS", 0)
    assert _result_config_digest(cfg, de_backend_used=False, comparator="lognorm") == now


# --------------------------------------------------------------------------- artifact caches

def _captured_params(monkeypatch):
    """Record every `CacheStore.get_or_compute` params dict, keyed by artifact key."""
    from cell_eval2 import cache as cache_mod

    seen: dict = {}
    real = cache_mod.CacheStore.get_or_compute

    def spy(self, key, *, fingerprint, params, kind, compute):
        seen[key] = dict(params)
        return real(self, key, fingerprint=fingerprint, params=params, kind=kind,
                    compute=compute)

    monkeypatch.setattr(cache_mod.CacheStore, "get_or_compute", spy)
    return seen


def _counts_adata(n=6, g=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["ctrl"] * (n // 2) + ["A"] * (n - n // 2))
    X = sp.csr_matrix(rng.integers(1, 50, size=(n, g)).astype(np.float32))
    import anndata as ad_mod
    return ad_mod.AnnData(X=X, obs=pd.DataFrame({"target": labels},
                                                index=[f"c{i}" for i in range(n)]),
                          var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))


@pytest.mark.parametrize("with_moments", [False, True])
@pytest.mark.parametrize("target,want", [("bulk_lognorm", True), ("lognorm", False)])
def test_the_PSEUDOBULK_ARTIFACT_key_carries_the_271_term_only_for_bulk_lognorm(
        tmp_path, monkeypatch, target, want, with_moments):
    """The artifact cache is the one that stores the moved object itself. A
    `pseudobulk_bulk_lognorm` npz holds `bulk_lognorm_means` of the very group sums #271 re-rounded --
    the transformed means (`perts` + `means`), not the raw sums; the `npz_moments` variant
    additionally stores `counts`, `sumsq` and `jk` -- and none of its other params can see that the
    reduction dtype changed.

    Asserted on the params the cache is actually keyed with, not on a digest, so the failure
    message names the missing term."""
    from cell_eval2.cache import CacheStore
    from cell_eval2.run import _side_bulks

    seen = _captured_params(monkeypatch)
    adata = _counts_adata()
    cfg = _cfg()
    store = CacheStore(str(tmp_path))
    _side_bulks(adata, fp="fp-for-this-test", store=store, norms=[target], cfg=cfg,
                side="real", effective_input_type="counts",
                moment_norms={target} if with_moments else None)
    # the moments artifact is a SEPARATE key with its own params dict (cache.py keeps one manifest
    # entry per key), so it needs its own coverage rather than being assumed to follow (codex r3)
    key = f"pseudobulk_moments_{target}" if with_moments else f"pseudobulk_{target}"
    assert key in seen, f"nothing was cached under {key!r}; captured {sorted(seen)}"
    assert ("grouped_sum_reduction_semantics" in seen[key]) is want, (
        f"{key} params: expected the #271 term present={want}, got {sorted(seen[key])}")


@pytest.mark.parametrize("backend,want", [("deseq2", True), ("pdex", False)])
def test_the_DE_TABLE_key_carries_the_271_term_ONLY_for_deseq2(backend, want):
    """deseq2 is the third caller of `_grouped_sums`, so its DE table moved. Every other backend
    computes from cells and keeps its warm cache -- which is why the term sits inside the existing
    `backend == "deseq2"` block rather than beside it.

    Driven through `run._compute_de_side` with a store whose `get_or_compute` captures `params` and
    never calls `compute`, so no DE engine (private or otherwise) runs. No `_cache_backend` patch is
    needed -- it returns an EXPLICIT backend verbatim and only resolves `"auto"` against the host
    (codex round 3 removed the patch that was hiding that), so the captured `backend` is asserted
    instead.

    The first cut of this test read the SOURCE TEXT with `inspect.getsource` and split on string
    literals: it did fail against the pre-fix commit, so it was not vacuous, but reformatting could
    break it and a commented-out assignment could satisfy it (codex round 2)."""
    import anndata as ad_mod

    from cell_eval2 import run

    captured = {}

    class _CapturingStore:
        def get_or_compute(self, key, *, fingerprint, params, kind, compute):
            captured[key] = dict(params)
            raise _Stop                      # never execute the engine

    class _Stop(Exception):
        pass

    cfg = _cfg(metrics=["de_wilcoxon_overlap"], de={"backend": backend,
                                                    "replicate_col": "guide"})
    adata = _counts_adata(n=8, g=4)
    adata.obs["guide"] = ["c1", "c1", "c2", "c2", "A1", "A1", "A2", "A2"]
    assert isinstance(adata, ad_mod.AnnData)

    with pytest.raises(_Stop):
        run._compute_de_side(adata, cfg=cfg, fp="fp-for-this-test", store=_CapturingStore(),
                             side="real")
    key = f"de_{cfg.de.method}_table"
    assert key in captured, f"nothing was keyed under {key!r}; captured {sorted(captured)}"
    assert captured[key]["backend"] == backend, (
        f"the explicit backend did not reach the key verbatim: {captured[key]['backend']!r}")
    assert ("grouped_sum_reduction_semantics" in captured[key]) is want, (
        f"{key} params for backend={backend!r}: expected the #271 term present={want}, got "
        f"{sorted(captured[key])}")


# --------------------------------------------------------------------------- partials, bundles

def test_the_271_term_moves_the_PARTIAL_SEMANTICS_payload(monkeypatch):
    """A partial directory straddling this change mixes pieces whose bulks were rounded
    differently under one metric name -- #246's exact failure. Unconditional here: a partial is a
    transient scale-run intermediate, so the stricter rule costs no warm cache.

    A PAYLOAD-mutation test, like its purity-floor sibling in `test_partition.py`: the refusal
    machinery is already covered generically, so what is unproven without this is only whether the
    term is IN the payload those tests compare."""
    from cell_eval2 import partition, run
    from cell_eval2.catalog import resolve_metrics

    names, _ = resolve_metrics("vcc2026", version="v2")
    before = partition.result_semantics(names, comparator="bulk_lognorm")
    monkeypatch.setattr(run, "_GROUPED_SUM_REDUCTION_SEMANTICS", 0)
    after = partition.result_semantics(names, comparator="bulk_lognorm")
    assert before != after, (
        "result_semantics does not carry the #271 reduction semantics, so a partial directory "
        "straddling the change would aggregate silently")
    assert partition._semantics_diff(before, after) == ["grouped_sum_reduction_semantics"]


def test_a_pre_271_REFERENCE_BUNDLE_is_refused_rather_than_consumed(tmp_path):
    """`_check_bundle_semantics` compares KEY SETS first, so a bundle recorded before the term
    existed fails loudly instead of having its pseudobulks stacked with this build's. A reference
    bundle is a rebuildable cache, so "rebuild it" is a real remedy."""
    from cell_eval2.partition_inmem import (BUNDLE_SEMANTICS_KEY, _bundle_semantics,
                                            _check_bundle_semantics)

    cfg = _cfg()
    mine = _bundle_semantics(cfg)
    assert "grouped_sum_reduction_semantics" in mine, (
        "the in-mem reference bundle's semantic subset does not record the #271 reduction, so a "
        "pre-fix bundle's pseudobulks would be consumed as if compatible")

    pre_271 = {k: v for k, v in mine.items() if k != "grouped_sum_reduction_semantics"}
    with pytest.raises(ValueError, match="Rebuild the bundle"):
        _check_bundle_semantics(str(tmp_path), cfg, {BUNDLE_SEMANTICS_KEY: pre_271},
                                caller="test")
    # a bundle from THIS build still passes -- the guard is not simply always-raise
    _check_bundle_semantics(str(tmp_path), cfg, {BUNDLE_SEMANTICS_KEY: mine}, caller="test")


def test_the_ANCHOR_identity_carries_the_271_term_under_bulk_lognorm(monkeypatch):
    """An anchor is a FROZEN artifact: one built before the reduction widened carries a replicate
    computed from differently-rounded group sums, for members `vcc2026` scores. Nothing else in
    `anchor_semantic_params` can see it -- the reduction dtype is not a config knob, and the
    `comparator` key records which comparator was resolved, not what a group sum under it means."""
    from cell_eval2 import run
    from cell_eval2.anchor import anchor_semantic_params
    from cell_eval2.catalog import resolve_metrics

    cfg = _cfg(metrics=["expr_mae"], input_type="counts")
    real = _counts_adata(n=8, g=5)
    names = list(resolve_metrics(cfg.metrics, version=cfg.version)[0])
    params = anchor_semantic_params(cfg, real, names)
    assert params["comparator"] == "bulk_lognorm", (
        "fixture no longer resolves to bulk_lognorm, so this test would prove nothing")
    assert "grouped_sum_reduction_semantics" in params
    monkeypatch.setattr(run, "_GROUPED_SUM_REDUCTION_SEMANTICS", 0)
    moved = anchor_semantic_params(cfg, real, names)
    assert moved["grouped_sum_reduction_semantics"] != \
        params["grouped_sum_reduction_semantics"]
