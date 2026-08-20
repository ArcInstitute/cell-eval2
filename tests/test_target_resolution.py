import polars as pl
import pytest

from cell_eval2.de import TargetResolution, resolve_target_genes


def _real(rows):
    return pl.DataFrame(
        {"target": [r[0] for r in rows], "feature": [r[1] for r in rows]},
        schema={"target": pl.String, "feature": pl.String},
    )


def test_exact_match_resolves():
    real = _real([("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")])
    res = resolve_target_genes(real, ["A", "B"])
    assert res.mapping == {"A": "A", "B": "B"}
    assert res.n_resolved == 2
    assert res.n_targets == 2


def test_h1_cgs_shape_resolves_against_the_global_union():
    """Each target's own gene is dropped from its OWN rows but is measured elsewhere.

    This is the H1_CGS shape (spec 2.1): a PER-TARGET check would raise on every
    target; the GLOBAL union resolves all of them.
    """
    real = _real([("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")])
    res = resolve_target_genes(real, ["A", "B", "C"])
    assert res.mapping == {"A": "A", "B": "B", "C": "C"}


def test_zero_resolve_does_NOT_raise_here():
    """Resolution never raises -- the gate lives at the eleven metrics' entry
    (metrics.direction._require_resolution), because every DE run builds a PreparedDE
    including legacy-only and v1 runs that have no target-gene semantics."""
    real = _real([("GENEX-1", "GENEX"), ("GENEY-2", "GENEY")])
    res = resolve_target_genes(real, ["GENEX-1", "GENEY-2"])
    assert res.n_resolved == 0
    assert res.n_targets == 2


def test_mapping_is_not_mutable_through_the_dataclass():
    """frozen=True blocks field REASSIGNMENT, not mutation of the dict it holds. The
    memos are identity-keyed on the PreparedDE, so a post-construction mutation would
    silently serve stale exclusions."""
    res = resolve_target_genes(_real([("A", "A")]), ["A"])
    with pytest.raises(TypeError):
        res.mapping["A"] = "B"


def test_resolution_deepcopies_and_serializes():
    """A MappingProxyType stored as a FIELD would pass every test above and then break
    copy.deepcopy and dataclasses.asdict with 'cannot pickle mappingproxy'."""
    import copy
    import dataclasses
    res = resolve_target_genes(_real([("A", "A"), ("B", "B")]), ["A", "B"])
    assert copy.deepcopy(res) == res
    assert dataclasses.asdict(res)["pairs"] == (("A", "A"), ("B", "B"))


def test_global_counts_are_recorded_at_resolution_time():
    res = resolve_target_genes(_real([("A", "A"), ("GENEX-1", "A")]), ["A", "GENEX-1"])
    assert res.n_features == 1
    assert res.unresolved == ("GENEX-1",)


def test_an_unrecorded_feature_count_is_None_not_zero():
    """A hand-built resolution has no global count. Zero would read as a measurement in
    the gate's error message."""
    assert TargetResolution({}, 0).n_features is None


def test_dataclasses_replace_works():
    """`replace` reconstructs via cls(**{every field}), so __init__ must accept `pairs=`
    even though nothing writes it by hand."""
    import dataclasses
    res = resolve_target_genes(_real([("A", "A")]), ["A"])
    bumped = dataclasses.replace(res, n_targets=5)
    assert bumped.n_targets == 5
    assert bumped.pairs == res.pairs


def test_partial_resolution_does_not_raise():
    real = _real([("A", "A"), ("GENEX-1", "A")])
    res = resolve_target_genes(real, ["A", "GENEX-1"])
    assert res.mapping == {"A": "A"}
    assert (res.n_resolved, res.n_targets) == (1, 2)


def test_map_is_authoritative_and_bypasses_index_membership():
    """Spec 2.1: a map entry is NOT re-checked against the index.

    Without this, mapping a target to its correctly-named but deliberately-absent
    gene would still raise, and the escape hatch would be useless.
    """
    real = _real([("GENEX-1", "OTHER")])
    res = resolve_target_genes(real, ["GENEX-1"], target_gene_map={"GENEX-1": "GENEX"})
    assert res.mapping == {"GENEX-1": "GENEX"}


def test_map_overrides_an_exact_match():
    real = _real([("A", "A"), ("A", "B")])
    res = resolve_target_genes(real, ["A"], target_gene_map={"A": "B"})
    assert res.mapping == {"A": "B"}


def test_no_targets_does_not_raise():
    assert resolve_target_genes(_real([]), []).mapping == {}


def _de(rows):
    """(target, feature, lfc, p_adj) -> a minimal DE table."""
    return pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "feature": [r[1] for r in rows],
            "log2_fold_change": [float(r[2]) for r in rows],
            "p_adj": [float(r[3]) for r in rows],
        },
        schema={"target": pl.String, "feature": pl.String,
                "log2_fold_change": pl.Float64, "p_adj": pl.Float64},
    )


def test_prepare_de_derives_resolution_when_omitted():
    from cell_eval2.de import prepare_de
    tbl = _de([("A", "A", 1.0, 0.01), ("A", "B", 1.0, 0.01),
               ("B", "A", 1.0, 0.01), ("B", "B", 1.0, 0.01)])
    prep = prepare_de(tbl, tbl, control="non-targeting")
    assert prep.target_resolution.mapping == {"A": "A", "B": "B"}


def test_explicit_resolution_is_used_verbatim():
    from cell_eval2.de import prepare_de
    tbl = _de([("A", "B", 1.0, 0.01), ("B", "A", 1.0, 0.01)])
    explicit = TargetResolution({"A": "A", "B": "B"}, 2)
    prep = prepare_de(tbl, tbl, control="non-targeting", target_resolution=explicit)
    assert prep.target_resolution is explicit


def test_legacy_shaped_fixture_still_builds_a_prepared_de():
    """tests/test_de.py:105's shape: targets A/B, features g1..g4. Nothing resolves, and
    that must NOT break construction -- every DE run builds a PreparedDE, including runs
    that select only the legacy metrics and every v1 run. The gate lives at the eleven
    metrics' entry instead (metrics.direction._require_resolution, Task 3)."""
    from cell_eval2.de import prepare_de
    tbl = _de([("A", "g1", 1.0, 0.01), ("A", "g2", 1.0, 0.01),
               ("B", "g3", 1.0, 0.01), ("B", "g4", 1.0, 0.01)])
    prep = prepare_de(tbl, tbl, control="non-targeting")
    assert prep.target_resolution.n_resolved == 0


def test_config_field_defaults_to_none_and_is_last():
    import dataclasses
    from cell_eval2.config import EvalConfig
    cfg = EvalConfig()
    assert cfg.target_gene_map is None
    names = [f.name for f in dataclasses.fields(EvalConfig)]
    assert names[-2:] == ["target_gene_map", "bulk_target_sum"], (
        "EvalConfig is a plain (not kw_only) dataclass; a new field must be APPENDED "
        "LAST or positional callers shift (config.py:201-203)"
    )


def test_map_enters_the_result_cache_digest_when_set():
    from dataclasses import replace
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _result_config_digest
    base = EvalConfig()
    mapped = replace(base, target_gene_map={"A": "B"})
    assert _result_config_digest(
        base, de_backend_used=False, comparator="lognorm",
    ) != _result_config_digest(
        mapped, de_backend_used=False, comparator="lognorm",
    )


def test_absent_map_does_not_change_the_digest():
    """A warm cache written before the field existed must still hit (spec 2.7,
    replicate_col precedent at run.py:686-688)."""
    from cell_eval2.config import EvalConfig
    from cell_eval2.cache import config_hash
    from cell_eval2.run import _result_config_digest
    cfg = EvalConfig()
    legacy = cfg.to_dict()
    legacy.pop("target_gene_map")
    # _result_config_digest also drops de.replicate_col when the backend is not deseq2
    # (run.py:686-688) -- omit it here or the hashes differ for that reason instead.
    legacy["de"].pop("replicate_col", None)
    # and discrimination.exclusion_scope on the same inert-field rule (#343): this call requests
    # no pds_* metric, so the scope cannot move its value and is dropped from the key. Omitting
    # it here is what keeps this assertion testing what it says -- that a cache written before
    # the field existed still hits.
    legacy["discrimination"].pop("exclusion_scope", None)
    legacy["device"] = _cache_device_value(cfg)
    legacy["comparator"] = "lognorm"
    assert _result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm",
    ) == config_hash(legacy)


def test_the_BASELINE_digest_drops_the_map_too_when_inert():
    """`baseline.config_digest` must apply the SAME policy as
    `run._result_config_digest` (Copilot, PR #200). It builds its payload from
    `to_dict()` too, so it picked the new field up automatically -- and the two functions
    then disagreed.

    ⚠️ This one is not a silent cache miss. `cli.py:170` raises
    `SystemExit("baseline/user mismatch -- the margins would be meaningless")` unless
    `--allow-config-mismatch` is passed, so a baseline stamped before #195 would have
    HARD-FAILED every run after it.

    The metric list is pinned explicitly to metrics #195 does not touch, which is the
    case where scoring semantics really are unchanged. Under the default `full` profile
    the digest legitimately moves anyway (36 -> 47 resolved metrics), so a default-config
    assertion would pass for the wrong reason and could not detect a regression here.
    """
    from dataclasses import replace

    from cell_eval2.baseline import DIGEST_EXEMPT_FIELDS, config_digest
    from cell_eval2.config import EvalConfig

    cfg = replace(EvalConfig(), metrics=["expr_mae", "delta_pearson"])
    assert cfg.target_gene_map is None
    inert = config_digest(cfg, comparator="lognorm")

    # The field must be dropped CONDITIONALLY, never exempted outright: a supplied map
    # changes which genes are excluded and so must reach the digest.
    assert "target_gene_map" not in DIGEST_EXEMPT_FIELDS
    assert config_digest(
        replace(cfg, target_gene_map={"A": "B"}), comparator="lognorm",
    ) != inert

    # ...and the inert digest is the one a config WITHOUT the field would produce.
    #
    # This reconstructs the whole payload, so it also pins the digest's SCHEMA and must be
    # updated whenever that schema changes deliberately -- as #231 did, adding `metric_agg`,
    # #257 does, adding `metric_derived`, and #264 PR2 does, adding `metric_normalization`
    # (the RESOLVED per-metric space, without which a PR1 baseline and a PR2 run hash the
    # same at one comparator token while computing different numbers).
    # It no longer claims a pre-#195 baseline stamp still matches: a deliberate schema bump
    # ends that guarantee even for metrics whose own semantics did not move. The invariant
    # under test is unchanged -- an INERT `target_gene_map` does not enter the digest.
    import hashlib
    import json

    from cell_eval2.baseline import _cache_backend, _cache_device, _de_backend_used
    from cell_eval2.catalog import CATALOG, derived_policy, resolve_metrics
    from cell_eval2.run import effective_normalization

    legacy = {k: v for k, v in cfg.to_dict().items() if k not in DIGEST_EXEMPT_FIELDS}
    legacy["comparator"] = "lognorm"
    legacy.pop("target_gene_map")
    names = resolve_metrics(cfg.metrics, version=cfg.version)[0]
    legacy["metrics"] = names
    legacy["metric_agg"] = [[n, CATALOG[n].agg] for n in names]
    legacy["metric_derived"] = derived_policy(names)
    legacy["metric_normalization"] = [[n, effective_normalization(CATALOG[n], "lognorm")]
                                      for n in names]
    legacy["device"] = _cache_device(cfg)
    legacy["de"]["backend"] = (_cache_backend(cfg)
                               if _de_backend_used(cfg, names, None) else cfg.de.backend)
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()
    assert inert == expected


def _cache_device_value(cfg):
    from cell_eval2.run import _cache_device
    return _cache_device(cfg)


def test_single_target_slice_does_not_raise_when_resolution_is_dataset_level():
    """The regression test for a shard-local gate (spec 2.7b).

    H1_CGS shape: no target's own gene is among its OWN rows, but every symbol is
    measured somewhere. Resolving from the WHOLE table succeeds; resolving from a
    single-target slice sees only that target's universe and raises.
    """
    whole = _real([("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")])
    dataset_level = resolve_target_genes(whole, ["A", "B", "C"])
    assert dataset_level.n_resolved == 3

    # Resolving from the SLICE finds nothing -- A's own gene is not among A's own rows.
    # resolve_target_genes does not raise (the gate is at the metrics), so the failure
    # this test guards against is silent: the piece would carry an EMPTY resolution,
    # exclude nothing, and then raise at the first direction metric.
    piece = whole.filter(pl.col("target") == "A")
    assert resolve_target_genes(piece, ["A"]).n_resolved == 0
