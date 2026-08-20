"""CPU-only behaviour of _RefBundle. score_piece itself needs a GPU (gpudge DE), so everything
testable without one is pinned here rather than in the GPU-gated score_piece module."""
import json
import os
from dataclasses import replace

import anndata as ad
import numpy as np
import polars as pl
import pytest
from cell_eval2 import partition_inmem
from cell_eval2.config import EvalConfig
from _helpers import full_minus_moments, resolved_comparator


@pytest.fixture(autouse=True)
def _gpudge_resolvable(monkeypatch):
    """_RefBundle.__init__ calls _require_partition_config, which rejects any backend that does
    not RESOLVE to gpudge -- and on a CPU node backend='auto' resolves to pdex, so every test in
    this module would die at construction. Patch the module-level binding it actually calls
    (partition_inmem.py:43). Setting backend='gpudge' explicitly does NOT work: resolution still
    probes for a CUDA device. Same seam as test_cellstream.py:418."""
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda backend: "gpudge")


def _cfg(**kw):
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    return replace(EvalConfig.v2(), pert_col="target_gene",
                   metrics=full_minus_moments(), **kw)


def _write_bundle(d, *, control_format="h5ad", with_pred_control=False):
    """A minimal on-disk reference bundle: exactly the artifacts score_piece reads."""
    os.makedirs(d, exist_ok=True)
    genes = [f"g{i}" for i in range(4)]
    with open(os.path.join(d, "reference.json"), "w", encoding="utf-8") as fh:
        json.dump({"perturbation_universe": ["A", "B"], "var_index": genes,
                   "control_format": control_format, "real_ref_fingerprint": "fp"}, fh)
    np.savez(os.path.join(d, "real_pseudobulk_counts.npz"),
             perts=np.array(["A", "B"], dtype=str), means=np.ones((2, 4)))
    pl.DataFrame({"target": ["A", "B"], "feature": ["B", "A"],
                  "log2fc": [1.0, 2.0]}).write_parquet(
        os.path.join(d, "real_de.parquet"))
    ctrl = ad.AnnData(X=np.ones((3, 4), dtype=np.float32),
                      obs={"target_gene": ["non-targeting"] * 3}, var={"gene": genes})
    ctrl.write_h5ad(os.path.join(d, "real_control.h5ad"))
    if with_pred_control:
        ctrl.write_h5ad(os.path.join(d, "pred_control.h5ad"))
        np.savez(os.path.join(d, "pred_pseudobulk_counts.npz"),
                 perts=np.array(["non-targeting"], dtype=str), means=np.ones((1, 4)))
    return d


def test_bundle_reads_nothing_on_construction(tmp_path):
    # Stronger than spying on one loader: point it at a directory that does not exist. Any eager
    # read of the manifest, an npz, the parquet or the control would raise here.
    partition_inmem._RefBundle(str(tmp_path / "does-not-exist"), _cfg())


def test_bundle_reads_control_once_across_repeated_access(tmp_path, monkeypatch):
    _write_bundle(str(tmp_path))
    real = partition_inmem.load_anndata
    calls = []

    def spy(*a, **k):
        calls.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(partition_inmem, "load_anndata", spy)
    b = partition_inmem._RefBundle(str(tmp_path), _cfg())
    first, second = b.control_ad, b.control_ad
    assert first is second
    assert len(calls) == 1


def test_bundle_reads_parquet_and_npz_once(tmp_path, monkeypatch):
    _write_bundle(str(tmp_path))
    pq_reads, npz_reads = [], []
    real_pq, real_np_load = pl.read_parquet, np.load
    monkeypatch.setattr(pl, "read_parquet", lambda p, *a, **k: pq_reads.append(p) or real_pq(p, *a, **k))
    monkeypatch.setattr(np, "load", lambda p, *a, **k: npz_reads.append(p) or real_np_load(p, *a, **k))

    b = partition_inmem._RefBundle(str(tmp_path), _cfg())
    assert b.real_de.height == 2 and b.real_de.height == 2
    assert len(pq_reads) == 1

    first = b.real_bulks(["counts"])["counts"]
    second = b.real_bulks(["counts"])["counts"]
    assert first[0] is second[0]      # memoized object, not a re-read
    assert len(npz_reads) == 1


def test_bundle_target_resolution_uses_whole_real_de_and_is_memoized(tmp_path, monkeypatch):
    """The H1-shaped rows resolve only against the whole table's feature union."""
    _write_bundle(str(tmp_path))
    reads = []
    real_read_parquet = pl.read_parquet

    def spy(path, *args, **kwargs):
        reads.append(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", spy)
    cfg = _cfg()
    bundle = partition_inmem._RefBundle(str(tmp_path), cfg)
    first = bundle.target_resolution(cfg)
    second = bundle.target_resolution(cfg)

    assert first.mapping == {"A": "A", "B": "B"}
    assert first is second
    assert len(reads) == 1


def test_bundle_target_resolution_honours_config_map(tmp_path):
    _write_bundle(str(tmp_path))
    cfg = _cfg(target_gene_map={"A": "GENEA"})
    resolution = partition_inmem._RefBundle(str(tmp_path), cfg).target_resolution(cfg)
    assert resolution.mapping == {"A": "GENEA", "B": "B"}


def test_bundle_reads_the_manifest_once(tmp_path):
    d = _write_bundle(str(tmp_path))
    b = partition_inmem._RefBundle(d, _cfg())
    first = b.manifest
    os.remove(os.path.join(d, "reference.json"))   # a second read would now raise
    assert b.manifest is first


def test_bundle_pred_control_bulks_reads_once_and_returns_the_row(tmp_path, monkeypatch):
    d = _write_bundle(str(tmp_path), with_pred_control=True)
    reads = []
    real_np_load = np.load
    monkeypatch.setattr(np, "load", lambda p, *a, **k: reads.append(p) or real_np_load(p, *a, **k))
    b = partition_inmem._RefBundle(d, _cfg(control_source="pred"))
    perts, means = b.pred_control_bulks(["counts"])["counts"]
    assert list(perts) == ["non-targeting"] and means.shape == (1, 4)
    b.pred_control_bulks(["counts"])
    assert len([p for p in reads if "pred_pseudobulk" in str(p)]) == 1


def test_bundle_pred_control_missing_raises_from_the_accessor(tmp_path):
    _write_bundle(str(tmp_path), with_pred_control=False)
    b = partition_inmem._RefBundle(str(tmp_path), _cfg(control_source="pred"))
    with pytest.raises(FileNotFoundError, match="build_pred_control_reference"):
        _ = b.control_ad


def test_bundle_pred_control_bulks_missing_raises(tmp_path):
    _write_bundle(str(tmp_path), with_pred_control=False)
    b = partition_inmem._RefBundle(str(tmp_path), _cfg(control_source="pred"))
    with pytest.raises(FileNotFoundError, match="build_pred_control_reference"):
        b.pred_control_bulks(["counts"])


def test_bundle_records_cache_dir_and_config_hash(tmp_path):
    from cell_eval2.cache import config_hash
    _write_bundle(str(tmp_path))
    cfg = _cfg()
    b = partition_inmem._RefBundle(str(tmp_path), cfg)
    assert b.cache_dir == str(tmp_path)
    # resolved exactly as _RefBundle.__init__ resolves it, so the Task 4 guard compares like for like
    resolved = partition_inmem._require_partition_config(partition_inmem._resolve_config(cfg, {}))
    # #185: the identity digest is config_hash MINUS target_sum, which is verified against the
    # bundle manifest instead. Asserted against the helper AND against the un-excluded digest, so
    # a future change that quietly puts target_sum back fails here rather than in a driver.
    assert b.config_hash == partition_inmem._bundle_identity_hash(resolved)
    assert b.config_hash != config_hash(resolved.to_dict())


def _piece():
    return ad.AnnData(X=np.ones((2, 4), dtype=np.float32),
                      obs={"target_gene": ["A", "B"]},
                      var={"gene": [f"g{i}" for i in range(4)]})


@pytest.fixture
def _no_reads_allowed(monkeypatch):
    """Both guard tests must fail if the guards ever move BELOW the piece load or the manifest
    read. Make the reads explode, so only a guard raised before them can produce the expected
    ValueError. `open` is the one that matters most: _RefBundle.manifest uses the builtin, and a
    module-level `open` attribute shadows it (raising=False because there is none to replace).
    NOT covered: the sharded-control ShardedArchive path, which these fixtures never take."""
    def boom(*a, **k):
        raise AssertionError("an artifact read happened before the bundle guards")

    monkeypatch.setattr(partition_inmem, "open", boom, raising=False)
    monkeypatch.setattr(partition_inmem, "load_anndata", boom)
    monkeypatch.setattr(pl, "read_parquet", boom)
    monkeypatch.setattr(np, "load", boom)


def test_score_piece_rejects_a_bundle_from_another_cache_dir(tmp_path, _no_reads_allowed):
    # The target cache_dir does not even exist: nothing but a guard can raise ValueError here.
    cfg = _cfg()
    bundle = partition_inmem._RefBundle(_write_bundle(str(tmp_path / "a")), cfg)
    with pytest.raises(ValueError, match="cache_dir"):
        partition_inmem.score_piece(_piece(), str(tmp_path / "nonexistent"),
                                    config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_rejects_a_bundle_built_for_another_config(tmp_path, _no_reads_allowed):
    d = _write_bundle(str(tmp_path))
    bundle = partition_inmem._RefBundle(d, _cfg())
    cfg = _cfg(control_source="pred")
    with pytest.raises(ValueError, match="config"):
        partition_inmem.score_piece(
            _piece(), d, config=cfg, bundle=bundle,
            comparator=resolved_comparator(cfg))


# --- #185: the guard must compare post-adoption equivalence, not pre-adoption identity --------

def _write_bundle_with_target(d, target_sum, **kw):
    """_write_bundle plus the manifest keys score_piece adopts/verifies AFTER the identity guard:
    `normalize_target_sum` (#155) and the comparator pair (#264). Needed by any test that must get
    PAST the guard -- _bundle_comparator runs first and would otherwise raise for a missing
    'comparator' before _bundle_target_sum is ever reached."""
    _write_bundle(d, **kw)
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    man["normalize_target_sum"] = target_sum
    man["comparator"] = resolved_comparator(_cfg())
    man["bulk_target_sum"] = _cfg().bulk_target_sum
    # #181: the semantic subset, or _check_bundle_semantics refuses the bundle as unverifiable
    # before anything downstream runs.
    man[partition_inmem.BUNDLE_SEMANTICS_KEY] = partition_inmem._bundle_semantics(_cfg())
    # A real bundle records both, and both are now load-bearing: `effective_input_type` is what
    # _check_control_space compares the piece against under control_source='real', and
    # (norms, effective_input_type) is what decides whether `device` is semantic at all.
    man["effective_input_type"] = "counts"
    man.setdefault("norms", ["counts"])
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    return d


@pytest.mark.parametrize("bundle_ts,call_ts", [
    (None, 54611.0),      # bundle built pre-adoption, consumed post-adoption
    (54611.0, None),      # ...and the reverse
    (1000000, 1000000.0),  # equal but differently REPRESENTED -- these hash differently
    (1000000.0, np.float32(1e6)),
])
def test_score_piece_accepts_equivalent_pre_and_post_adoption_configs(
        tmp_path, monkeypatch, bundle_ts, call_ts):
    """#185's acceptance. The guard ran BEFORE _bundle_target_sum's adopt-or-verify step, so it
    compared PRE-adoption configs -- config identity rather than post-adoption equivalence -- and
    rejected callers that mean exactly the same thing.

    The narrower rows are worse than they look: _apply_bundle_target_sum's closing comment exists
    to stop 1000000 / 1000000.0 / np.float32(1e6) diverging downstream, and the hash guard ran
    first, so that canonicalization never got the chance.

    ⚠️ STRENGTHENED after codex-review. The first version used `_no_reads_allowed`, which explodes
    on the FIRST `bundle.manifest` open -- so it proved only that `_bundle_identity_hash` did not
    reject, and never reached `_bundle_target_sum`'s adopt-or-verify step, which is the half of #185
    that actually has to accept these pairs. Now the manifest read is permitted and the run is
    stopped at `load_anndata`, which sits AFTER the comparator check, the target adoption, the bulk
    target check and the semantic verification -- so reaching that sentinel means all five accepted
    the pair. The sentinel is REQUIRED, so a guard that started rejecting would fail here.
    """
    d = _write_bundle_with_target(str(tmp_path), 54611.0 if bundle_ts is None else bundle_ts)
    bundle = partition_inmem._RefBundle(d, _cfg(target_sum=bundle_ts))
    cfg = _cfg(target_sum=call_ts)

    class _ReachedThePiece(Exception):
        pass

    monkeypatch.setattr(partition_inmem, "load_anndata",
                        lambda *a, **k: (_ for _ in ()).throw(_ReachedThePiece()))
    with pytest.raises(_ReachedThePiece):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_still_rejects_a_bundle_built_at_a_DIFFERENT_target(tmp_path):
    """The protection #185 must not remove: a bundle genuinely built at one target and consumed
    at another. It now comes from the manifest comparison rather than the hash, and the message
    names BOTH values -- which the hash guard never could."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    cfg_call = _cfg(target_sum=1e4)
    bundle = partition_inmem._RefBundle(d, cfg_call)     # same cfg -> identity guard passes
    with pytest.raises(ValueError, match=r"target_sum=10000.0 disagrees.*normalize_target_sum"):
        partition_inmem.score_piece(_piece(), d, config=cfg_call, bundle=bundle,
                                    comparator=resolved_comparator(cfg_call))


def test_score_piece_still_rejects_a_bundle_built_for_a_genuinely_different_config(
        tmp_path, _no_reads_allowed):
    """Excluding target_sum must not weaken the rest of the digest."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    bundle = partition_inmem._RefBundle(d, _cfg(target_sum=1e6))
    cfg = _cfg(target_sum=1e6, control_source="pred")
    with pytest.raises(ValueError, match="different config"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_bundle_identity_hash_excludes_only_target_sum(tmp_path):
    """Pinned field by field, because a digest that quietly stopped covering a numerics field
    would be invisible: every driver builds the bundle from the object it then passes."""
    base = _cfg(target_sum=1e6)
    h = partition_inmem._bundle_identity_hash
    assert h(base) == h(replace(base, target_sum=1e4)), "target_sum must be excluded"
    assert h(base) == h(replace(base, target_sum=None))
    for field, value in (("bulk_target_sum", 28_000.0), ("control_source", "pred"),
                         ("input_type", "lognorm"), ("version", "v1"),
                         ("allow_discrete", True), ("pert_col", "other")):
        assert h(base) != h(replace(base, **{field: value})), f"{field} must stay in the digest"


def test_score_piece_reads_reference_json_exactly_once_across_repeated_calls(
        tmp_path, monkeypatch):
    """#185's test gap 1. test_partition_inmem_refbundle's own read-once test exercises
    `bundle.manifest` DIRECTLY, and test_partition_inmem_score_piece counts CONTROL loads -- so
    deleting `manifest=manifest` from score_piece's _bundle_target_sum call would restore a
    per-batch reference.json read with both tests still green. This counts opens of that file
    across repeated score_piece calls and requires exactly one.

    CPU-stubbed: the guards and _bundle_target_sum run for real, then load_anndata raises to stop
    before the GPU work.
    """
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    cfg = _cfg(target_sum=1e6)
    bundle = partition_inmem._RefBundle(d, cfg)

    opens = []
    real_open = open

    def counting_open(path, *a, **k):
        if str(path).endswith("reference.json"):
            opens.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(partition_inmem, "open", counting_open, raising=False)

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(partition_inmem, "load_anndata", stop)
    for _ in range(3):
        with pytest.raises(_Stop):
            partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                        comparator=resolved_comparator(cfg))
    assert len(opens) == 1, (
        f"reference.json was opened {len(opens)} times across 3 score_piece calls; the bundle "
        "exists so it is read once (#153/#185)")


# --- #181: verify a named semantic subset against the bundle, not just normalize_target_sum ----

def test_bundle_semantics_records_the_named_subset_and_nothing_else():
    """Pinned as an exact set. The subset is a DELIBERATE choice per #155 spec 4.3 -- a blanket
    config_hash comparison would reject the three drivers, which all rebind `control` and/or
    `input_type` between building a bundle and consuming it -- so silently growing or shrinking it
    is the mistake to catch."""
    cfg = _cfg()
    sem = partition_inmem._bundle_semantics(cfg)
    assert set(sem) == set(partition_inmem.BUNDLE_SEMANTIC_FIELDS)
    from cell_eval2.run import _cache_device, _GROUPED_SUM_REDUCTION_SEMANTICS
    assert sem == {"de.mean_calc": cfg.de.mean_calc, "de.epsilon": cfg.de.epsilon,
                   "de.clip_value": cfg.de.clip_value, "control_source": cfg.control_source,
                   "filter.filter_gene_min_cpm_cell": cfg.filter.filter_gene_min_cpm_cell,
                   # RESOLVED, not the raw field: "auto" means different things on a GPU and a CPU
                   # host, and it is the resolved value that decides fp32 vs fp64 accumulation.
                   "device": _cache_device(cfg),
                   # #271: NOT a config path -- a code-semantics counter, special-cased in
                   # `_bundle_semantics` the way `device` is. A bundle built before
                   # `prep._grouped_sums` reduced wide holds pseudobulks rounded the other way, and
                   # every other field here compares equal across that change.
                   "grouped_sum_reduction_semantics": _GROUPED_SUM_REDUCTION_SEMANTICS}
    # The two fields the issue explicitly EXCLUDES must not have crept in.
    assert "control" not in sem and "input_type" not in sem and "pert_col" not in sem


@pytest.mark.parametrize("field,value", [
    ("mean_calc", "geometric"),
    ("epsilon", 1e-3),
    ("clip_value", 20.0),
])
def test_score_piece_refuses_a_bundle_built_under_different_de_semantics(tmp_path, field, value):
    """#181's acceptance: "a bundle built with mean_calc='arithmetic' and consumed with
    'geometric' raises, naming the offending field". Each of these changes the cached REAL DE
    table, and aggregate_partials cannot see it -- every partial records the same caller-derived
    config_hash, so its guard observes agreement and passes."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)     # semantics recorded from _cfg()
    cfg = _cfg(target_sum=1e6, de=replace(_cfg().de, **{field: value}))
    bundle = partition_inmem._RefBundle(d, cfg)          # same cfg -> identity guard passes
    with pytest.raises(ValueError, match=rf"different scoring semantics.*de\.{field}"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_refuses_a_bundle_built_under_a_different_control_source(tmp_path):
    """control_source decides which control pool the pred DE runs against AND whether
    pred_control.* artifacts are required at all, so it belongs in the subset."""
    d = _write_bundle_with_target(str(tmp_path), 1e6, with_pred_control=True)
    cfg = _cfg(target_sum=1e6, control_source="pred")
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(ValueError, match="different scoring semantics.*control_source"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_refuses_a_bundle_with_no_recorded_semantics(tmp_path, monkeypatch):
    """The legacy policy, and it follows _apply_bundle_target_sum's pre-#155 precedent: a
    reference bundle is a rebuildable cache (the three drivers build one per context into a temp
    dir), so refusing an unverifiable one has a real remedy and accepting it does not."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    del man[partition_inmem.BUNDLE_SEMANTICS_KEY]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    cfg = _cfg(target_sum=1e6)
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(ValueError, match=r"has no 'semantic_fields'.*Rebuild the bundle"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_accepts_a_bundle_whose_semantics_match(tmp_path, monkeypatch):
    """The guard must not reject the ordinary case. Stops at the piece load, which is past it."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    cfg = _cfg(target_sum=1e6)
    bundle = partition_inmem._RefBundle(d, cfg)

    class _Stop(Exception):
        pass

    monkeypatch.setattr(partition_inmem, "load_anndata",
                        lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_the_drivers_per_context_control_rebind_still_passes_the_semantic_check(tmp_path,
                                                                               monkeypatch):
    """The reason the check is a NAMED SUBSET and not config_hash.

    ⚠️ CLAIM CORRECTED after codex-review. The streaming builders rebind `control` BEFORE writing
    the manifest (`_build_reference_streaming_core` does `_replace(cfg, control=source.control)`),
    so for those three drivers the manifest's own `control` already equals the consumer's and a
    blanket comparison would in fact pass. What a blanket comparison really refuses is the PUBLIC
    non-streaming `build_reference`, which does NOT rebind -- and #181 excludes `control` by name
    for that reason ("varies per context by design"). This test pins the weaker, true property: a
    consumer differing from the bundle's cfg in `control`/`input_type` alone is accepted."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    build_cfg = _cfg(target_sum=1e6)
    # exactly what the drivers do: replace(cfg, control=<per-context>, input_type=<peeked>)
    piece_cfg = replace(build_cfg, control="ctx-specific-control", input_type="lognorm")
    manifest = partition_inmem._RefBundle(d, build_cfg).manifest
    partition_inmem._check_bundle_semantics(d, piece_cfg, manifest, caller="test")  # must not raise

def test_score_piece_refuses_a_bundle_built_under_a_different_CPM_gate(tmp_path):
    """The omission codex-review caught, and the most consequential one: the gate is passed into
    BOTH the cached real DE and every per-piece pred DE, so a mismatch compares two DE tables over
    different gene universes. Measured on CCL_2: cutoff 0 inverts three of four DE metrics."""
    from dataclasses import replace as _replace
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    base = _cfg()
    cfg = _cfg(target_sum=1e6, filter=_replace(base.filter, filter_gene_min_cpm_cell=0.0))
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(ValueError,
                       match=r"different scoring semantics.*filter\.filter_gene_min_cpm_cell"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_refuses_a_manifest_whose_semantic_KEY_SET_differs(tmp_path):
    """`recorded.get(k) != mine[k]` accepted a manifest that OMITTED a field whenever the expected
    value happened to be None -- and `de.clip_value` IS None under v2, so a manifest missing it
    compared equal and went unverified (codex-review). Key sets are compared first."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    assert man[partition_inmem.BUNDLE_SEMANTICS_KEY].pop("de.clip_value") is None, \
        "fixture must drop a field whose expected value is None, or this test proves nothing"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    cfg = _cfg(target_sum=1e6)
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(ValueError, match="records semantic fields"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_refuses_a_bundle_whose_REAL_side_is_in_another_SPACE(tmp_path, monkeypatch):
    """codex-review round 2, and this one is a defect in code this branch did not add.

    `score_piece` resolves `piece_eff` from the PREDICTION and hands that single type to
    `compute_de` together with the bundle's cached REAL control -- so on a mixed pair the control's
    raw counts get the prediction's space applied to them. The whole-prediction driver does not have
    this bug: `compute_metrics` converts the real control into the pred's scale FIRST (run.py:819)
    precisely so `compute_de` normalizes both sides identically. #181's own text flags
    `effective_input_type` as "written, never verified" -- it is now verified.

    REFUSED, not converted: the conversion is the right end state but MOVES NUMBERS for every mixed
    partitioned run, so it belongs in its own issue. A matched pair -- the only kind any in-tree
    driver produces -- is unaffected, which the neighbouring accept tests cover.
    """
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    man["effective_input_type"] = "lognorm"       # a lognorm-real bundle...
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    # A DE metric must be selected, or the guarded block never runs. _cfg() already pins
    # metrics=full_minus_moments(), so override via replace rather than through _cfg's **kw.
    cfg = replace(_cfg(target_sum=1e6), metrics=["de_wilcoxon_overlap"])
    bundle = partition_inmem._RefBundle(d, cfg)
    # ⚠️ `_piece()` puts the gene names in a var COLUMN, so its var_names are a RangeIndex and the
    # gene-axis check rejects it before the DE block. No other test in this module reaches that far,
    # which is why it never mattered. Build a piece whose var INDEX matches the bundle's var_index.
    import pandas as pd
    piece = ad.AnnData(X=np.ones((2, 4), dtype=np.float32),
                       obs={"target_gene": ["A", "B"]},
                       var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))
    with pytest.raises(NotImplementedError, match="hands ONE input_type to compute_de"):
        partition_inmem.score_piece(piece, d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_score_piece_refuses_a_bundle_built_on_another_DEVICE(tmp_path):
    """The resolved device decides fp32 (GPU) vs fp64 (CPU) pseudobulk means, and
    `run._side_bulks` keys its own cache on it for exactly that reason -- so a bundle whose real
    pseudobulk was accumulated in one and a piece scored in the other are not comparable. Missing
    from #181's first subset (codex-review); the issue does not exclude it."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)     # semantics recorded from _cfg() (cpu-ish)
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    man[partition_inmem.BUNDLE_SEMANTICS_KEY]["device"] = "cuda"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    cfg = _cfg(target_sum=1e6, device="cpu")
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(ValueError, match="different scoring semantics.*device"):
        partition_inmem.score_piece(_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


# --- codex-review round 3: the control artifact `control_source` actually selects ---------------

def _de_piece():
    """A piece whose var INDEX matches _write_bundle's var_index, so the gene-axis check passes and
    the DE path is reachable. `_piece()` puts the genes in a var COLUMN, so its var_names are a
    RangeIndex -- no other test in this module gets far enough for that to matter."""
    import pandas as pd
    return ad.AnnData(X=np.ones((2, 4), dtype=np.float32),
                      obs={"target_gene": ["A", "B"]},
                      var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))


def _set_manifest(d, **kw):
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    man.update(kw)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    return man


def test_under_control_source_pred_the_PRED_controls_space_is_what_is_checked(tmp_path):
    """⚠️ codex-review round 3, and it cuts both ways. Under `control_source="pred"` the bundle hands
    compute_de `pred_control.h5ad`, NOT the real control -- so my round-2 guard, which always
    compared against the real side's `effective_input_type`, would:

      * REFUSE a supported mixed pair whose pred control matches its pieces (`score_cellstream`
        produces exactly that, resolving both sides independently); and
      * ACCEPT a stale pred control in a different space whenever the real side happened to match.

    Both directions are pinned here.
    """
    d = _write_bundle_with_target(str(tmp_path), 1e6, with_pred_control=True)
    cfg = replace(_cfg(target_sum=1e6, control_source="pred"), metrics=["de_wilcoxon_overlap"])
    # The bundle's recorded #181 semantics must come from a control_source='pred' cfg, or that check
    # fires first -- correctly -- and masks what this test is about.
    _set_manifest(d, **{partition_inmem.BUNDLE_SEMANTICS_KEY:
                        partition_inmem._bundle_semantics(cfg)})

    # (a) real side lognorm, pred control counts, counts piece -> the real side is IRRELEVANT here,
    #     so this must NOT be refused for it. Stops at the DE engine, which is past the guard.
    _set_manifest(d, effective_input_type="lognorm",
                  **{partition_inmem.PRED_CONTROL_TYPE_KEY: "counts"})
    bundle = partition_inmem._RefBundle(d, cfg)
    # A UNIQUE sentinel at compute_de, not "any exception that isn't the guard's": the weaker form
    # passes for a run that dies of something unrelated before ever reaching DE (codex-review r4).
    partition_inmem._check_control_space(d, cfg, bundle.manifest, piece_eff="counts")  # no raise

    # (b) pred control lognorm, counts piece -> refused, whatever the real side says.
    _set_manifest(d, effective_input_type="counts",
                  **{partition_inmem.PRED_CONTROL_TYPE_KEY: "lognorm"})
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(NotImplementedError, match=r"pred-control artifact.*is 'lognorm'"):
        partition_inmem.score_piece(_de_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_a_bundle_missing_the_control_space_key_is_REFUSED_not_skipped(tmp_path):
    """`manifest.get(...)` returning None and skipping the comparison is the same fail-open that
    round 2's semantics loop shipped. A bundle predating the key is refused with a rebuild
    instruction, like every other guard in this family."""
    d = _write_bundle_with_target(str(tmp_path), 1e6, with_pred_control=True)
    cfg = replace(_cfg(target_sum=1e6, control_source="pred"), metrics=["de_wilcoxon_overlap"])
    _set_manifest(d, effective_input_type="counts",     # no PRED_CONTROL_TYPE_KEY
                  **{partition_inmem.BUNDLE_SEMANTICS_KEY:
                     partition_inmem._bundle_semantics(cfg)})
    bundle = partition_inmem._RefBundle(d, cfg)
    with pytest.raises(ValueError, match=r"records pred_control_effective_input_type=None"):
        partition_inmem.score_piece(_de_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


@pytest.mark.parametrize("manifest_kw,shape", [
    ({"norms": [], "effective_input_type": "counts"}, "DE-only (no real pseudobulk)"),
    ({"norms": ["lognorm"], "effective_input_type": "lognorm"}, "non-counts real side"),
])
def test_device_IS_compared_even_where_it_looks_inert(tmp_path, manifest_kw, shape):
    """⚠️ This pins a DELIBERATE REVERSAL, and the reasoning matters more than the assertion.

    codex-review round 3 observed that `device` can only move a cached artifact when the GPU
    accumulator actually ran -- so comparing it for the two shapes above is a false rejection. I
    gated it on `manifest["norms"]` and the real side's type. Round 4 showed that gate opened a
    WORSE hole: `norms` describes the REAL pseudobulks, while `_build_pred_control_reference_core`
    resolves its own selection and writes its own, so the two can differ. With a lognorm real side
    and a counts pred control the gate dropped `device` even though the cached pred-control
    pseudobulk could be CUDA/fp32 while later pieces are CPU/fp64 -- their TYPES match, so
    `_check_control_space` passes, and `_augment_pred_control` then stacks incompatible rows.

    So the comparison is unconditional again: a false rejection is loud and costs one rebuild, a
    missed guard silently mixes fp32 and fp64. The proper fix -- record the pred-control's own norms
    and accumulation mode, compare the two artifact sets independently -- is reported, not guessed.
    """
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    man = _set_manifest(d, **manifest_kw)
    man[partition_inmem.BUNDLE_SEMANTICS_KEY]["device"] = "cuda"
    _set_manifest(d, **{partition_inmem.BUNDLE_SEMANTICS_KEY:
                        man[partition_inmem.BUNDLE_SEMANTICS_KEY]})
    cfg = _cfg(target_sum=1e6, device="cpu")
    manifest = partition_inmem._RefBundle(d, cfg).manifest
    with pytest.raises(ValueError, match="different scoring semantics.*device"):
        partition_inmem._check_bundle_semantics(d, cfg, manifest, caller="test")


def test_an_anndata_only_selection_is_not_refused_for_a_controls_SPACE(tmp_path, monkeypatch):
    """codex-review round 4: `compute_de` is the only consumer of a control AnnData, so an
    anndata-only selection never touches one -- and refusing such a run for a control's space would
    be a false rejection whose message describes an operation that will not happen."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    _set_manifest(d, effective_input_type="lognorm")     # deliberately mismatched vs a counts piece
    cfg = replace(_cfg(target_sum=1e6), metrics=["expr_mae"])
    bundle = partition_inmem._RefBundle(d, cfg)

    class _GuardRan(Exception):
        pass

    # Pinned at the guard itself: it must not be CALLED. Asserting on a downstream outcome would
    # also pass if the guard ran and happened to accept.
    monkeypatch.setattr(partition_inmem, "_check_control_space",
                        lambda *a, **k: (_ for _ in ()).throw(_GuardRan()))
    # The anndata path then fails on this fixture's missing bulk_lognorm npz, which is fine -- the
    # assertion is that it got there without the control-space guard firing.
    with pytest.raises(FileNotFoundError, match="real_pseudobulk_bulk_lognorm"):
        partition_inmem.score_piece(_de_piece(), d, config=cfg, bundle=bundle,
                                    comparator=resolved_comparator(cfg))


def test_rebuilding_the_real_bundle_ORPHANS_pred_artifacts_and_an_anndata_run_refuses_them(tmp_path):
    """⚠️ codex-review round 5: the hole opened by making the type comparison DE-only.

    `_augment_pred_control` consumes `pred_pseudobulk_*.npz` on the ANNDATA path under
    `control_source='pred'`. `_write_reference_bundle` builds a FRESH manifest dict, so rebuilding
    the real bundle DROPS `pred_control_effective_input_type` -- while nothing removes those npz
    files. An anndata-only `delta_*`/`pds_*` run would then silently subtract a stale pred control.

    ⚠️ STRENGTHENED after codex-review round 6, which found the first version proved something
    weaker than its own name. Two defects, both in the fixture:

      * it never STAMPED the key, so `pop(..., None)` was a no-op -- it tested "absent key ->
        refuse", and would NOT have caught a future `_write_reference_bundle` that carried the key
        forward while still orphaning the artifacts, which is the whole mechanism; and
      * it watched `pred_pseudobulk_counts.npz` while `expr_mae` under a v2 counts config resolves
        `bulk_lognorm` -- so the file it asserted survived was not the file at risk.

    Now the sequence is the real one, end to end: stamp the key as the builder does, prove the run
    SCORES in that correctly-ordered state, then rebuild through `_write_reference_bundle` itself and
    require the refusal. The drop is the writer's doing, not an edit to the json -- so the mechanism
    is what is pinned, and the positive-control call is what makes the refusal attributable to the
    rebuild rather than to anything else this fixture happens to lack.
    """
    d = _write_bundle_with_target(str(tmp_path), 1e6, with_pred_control=True)
    cfg = replace(_cfg(target_sum=1e6, control_source="pred"), metrics=["expr_mae"])
    comparator = resolved_comparator(cfg)
    _set_manifest(d, **{partition_inmem.BUNDLE_SEMANTICS_KEY:
                        partition_inmem._bundle_semantics(cfg)})

    # The normalizations this selection ACTUALLY reads -- not "counts". Both sides, because
    # _augment_pred_control looks the pred control up per norm.
    norms = partition_inmem._needed_normalizations(["expr_mae"], comparator=comparator)
    assert norms, "the selection must read at least one pseudobulk, or nothing is at risk"
    for norm in norms:
        np.savez(os.path.join(d, f"real_pseudobulk_{norm}.npz"),
                 perts=np.array(["A", "B"], dtype=str), means=np.ones((2, 4)))
        np.savez(os.path.join(d, f"pred_pseudobulk_{norm}.npz"),
                 perts=np.array(["non-targeting"], dtype=str), means=np.ones((1, 4)))

    # Stamp it, the way build_pred_control_reference does...
    _set_manifest(d, **{partition_inmem.PRED_CONTROL_TYPE_KEY: "counts"})
    # ...and REQUIRE that the run scores in that state. Without this half the refusal below could
    # come from any other thing missing here, and the test would pass for the wrong reason.
    #
    # What this positive control does and does NOT prove (codex-review round 7): it DOES reach
    # `_augment_pred_control`, which loads and stacks the pred pseudobulk -- so a missing or
    # malformed artifact would stop it here, which is the fixture-completeness half. It does NOT
    # prove numerical dependence: `expr_mae` skips the control ROW, so its emitted values do not
    # depend on that row's contents. That is enough for a guard regression -- with the guard removed
    # the post-rebuild call succeeds and the refusal below fails -- but a `delta_*` metric would
    # additionally show the stale row MOVING a value, and this test deliberately does not claim that.
    scored = partition_inmem.score_piece(
        _de_piece(), d, config=cfg, bundle=partition_inmem._RefBundle(d, cfg),
        comparator=comparator)
    assert scored.height > 0

    # The rebuild, through the REAL writer whose fresh-dict manifest IS the mechanism.
    partition_inmem._write_reference_bundle(
        d, cfg=cfg,
        bulks={norm: (["A", "B"], np.ones((2, 4))) for norm in norms},
        de_df=pl.read_parquet(os.path.join(d, "real_de.parquet")),
        control_ad=ad.read_h5ad(os.path.join(d, "real_control.h5ad")),
        real_ref_fingerprint="fp", var_index=[f"g{i}" for i in range(4)],
        universe=["A", "B"], control_format="h5ad", comparator=comparator)

    with open(os.path.join(d, "reference.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    # Both halves of the hole, and both are the writer's doing rather than this test's:
    assert partition_inmem.PRED_CONTROL_TYPE_KEY not in man          # the key is gone...
    for norm in norms:                                               # ...the artifacts are not.
        assert os.path.isfile(os.path.join(d, f"pred_pseudobulk_{norm}.npz"))

    with pytest.raises(ValueError, match="Re-run build_pred_control_reference"):
        partition_inmem.score_piece(
            _de_piece(), d, config=cfg, bundle=partition_inmem._RefBundle(d, cfg),
            comparator=comparator)


def test_the_pred_control_builder_stamps_the_key_a_rebuild_drops(tmp_path, monkeypatch):
    """The other half: after `build_pred_control_reference` runs, the key is there -- so the guard
    above cannot fire on a correctly-ordered flow, only on an orphaned one."""
    d = _write_bundle_with_target(str(tmp_path), 1e6)
    cfg0 = replace(_cfg(target_sum=1e6, control_source="pred"), metrics=["expr_mae"])
    # The bundle's recorded #181 semantics must come from a control_source='pred' cfg, or that check
    # fires first (correctly) and this test never reaches the builder.
    _set_manifest(d, **{partition_inmem.BUNDLE_SEMANTICS_KEY:
                        partition_inmem._bundle_semantics(cfg0)})
    p = os.path.join(d, "reference.json")
    with open(p, encoding="utf-8") as fh:
        assert partition_inmem.PRED_CONTROL_TYPE_KEY not in json.load(fh)
    import pandas as pd

    class _CtrlSource:
        """The PertBatchSource surface _build_pred_control_reference_core touches."""
        control = "non-targeting"
        stream_tag = "t"

        def read_control_block(self):
            return ad.AnnData(X=np.ones((3, 4), dtype=np.float32),
                              obs={"target_gene": ["non-targeting"] * 3},
                              var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))

    cfg = replace(_cfg(target_sum=1e6, control_source="pred"), metrics=["expr_mae"])
    partition_inmem._build_pred_control_reference_core(
        _CtrlSource(), config=cfg, cache_dir=d, input_type="counts", comparator="bulk_lognorm")
    with open(p, encoding="utf-8") as fh:
        assert json.load(fh)[partition_inmem.PRED_CONTROL_TYPE_KEY] == "counts"
