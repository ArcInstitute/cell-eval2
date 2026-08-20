import json

import polars as pl
import pytest

from cell_eval2 import EvalConfig

PROFILE_KW = dict(metrics="anndata", pert_col="target", input_type="lognorm",
                  validate_input=False)


def build_anchor_dir(real, outdir, **cfg_kw):
    """Local helper -- NOT a new module, and never `from conftest import`."""
    import os

    from cell_eval2.anchor import (_derive_seeds, build_meta, compute_replicate_anchor,
                                   write_anchor)
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.run import _resolve_config, metric_output_names

    os.makedirs(str(outdir), exist_ok=True)
    cfg_in = EvalConfig(**{**PROFILE_KW, **cfg_kw})
    resolved = _resolve_config(cfg_in, {})
    names, _ = resolve_metrics(resolved.metrics, version=resolved.version)
    splits, anchor = compute_replicate_anchor(real, config=cfg_in, base_seed=0, n_splits=2)
    meta = build_meta(real_ad=real, cfg=resolved, names=list(names), base_seed=0,
                      n_splits=2, seeds=_derive_seeds(0, 2),
                      metrics=metric_output_names(resolved))
    write_anchor(str(outdir), splits, anchor, meta=meta)
    return str(outdir)


def _names(resolved):
    from cell_eval2.catalog import resolve_metrics
    return list(resolve_metrics(resolved.metrics, version=resolved.version)[0])


def test_control_source_effective_is_OBSERVED_not_asserted(monkeypatch,
                                                           synthetic_counts_pair):
    """A hardcoded literal is a claim, not an observation: delete the forcing and the artifact
    keeps asserting "pred". This makes the claim falsifiable -- redirect the helper that
    decides the inner config, and the stamp must move with it."""
    from dataclasses import replace

    from cell_eval2 import anchor
    from cell_eval2.config import EvalConfig

    cfg = EvalConfig(control_source="real")
    assert anchor._inner_config(cfg).control_source == "pred"
    assert anchor._inner_config(EvalConfig(control_source="pred")).control_source == "pred"

    _, real = synthetic_counts_pair
    monkeypatch.setattr(anchor, "_inner_config",
                        lambda c: replace(c, control_source="real", cache_real=None,
                                          cache_pred=None, outdir=None))
    meta = anchor.build_meta(real_ad=real, cfg=cfg, names=["pds_cosine"], base_seed=0,
                             n_splits=5, seeds=[1, 2, 3, 4, 5], metrics=["pds_cosine"])
    assert meta["control_source_effective"] == "real", (
        "build_meta does not read _inner_config -- the stamp is still a literal")


def test_real_fingerprint_is_the_STRICT_content_hash(synthetic_pair_with_effect,
                                                     tmp_path):
    """The gate is the content hash and nothing weaker. `real_fingerprint_meta` is stamped
    beside it as provenance, and the two must be DIFFERENT here -- if they ever coincide the
    strictness assertion is vacuous."""
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.cache import fingerprint_adata
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    meta = json.loads(open(f"{outdir}/anchor_meta.json").read())
    resolved = _resolve_config(EvalConfig(**PROFILE_KW), {})

    assert meta["real_fingerprint"] == fingerprint_adata(real, pert_col="target",
                                                         strict=True)
    assert meta["real_fingerprint_meta"] == fingerprint_adata(real, pert_col="target",
                                                              strict=False)
    assert meta["real_fingerprint"] != meta["real_fingerprint_meta"]
    assert meta["semantic_identity"] == semantic_identity(resolved, real, _names(resolved))


def test_a_different_X_with_identical_STRUCTURE_changes_the_fingerprint(
        synthetic_pair_with_effect, tmp_path):
    """The defect the strict hash exists to close, asserted directly: same shape, same
    dtype, same gene names, same per-cell labels, different values. The metadata hash cannot
    tell these apart -- which is why it is never the gate."""
    import json as _json

    from cell_eval2.cache import fingerprint_adata

    _pred, real = synthetic_pair_with_effect
    other = real.copy()
    other.X = other.X + 1.0

    outdir = build_anchor_dir(real, tmp_path)
    meta = _json.loads(open(f"{outdir}/anchor_meta.json").read())
    assert (fingerprint_adata(other, pert_col="target", strict=False)
            == meta["real_fingerprint_meta"]), "structures differ; the point is not made"
    assert (fingerprint_adata(other, pert_col="target", strict=True)
            != meta["real_fingerprint"])
    assert meta["config_hash"]                       # provenance only, never the gate
    assert meta["control_source_requested"] == "real"   # the v2 default the caller carried
    assert meta["control_source_effective"] == "pred"   # what the producer FORCED
    assert meta["base_seed"] == 0
    assert meta["n_splits"] == 2
    assert meta["derived_seeds"] == [2968811710, 3677149159]   # ALL of them, literally
    assert meta["seed_derivation"]
    assert meta["cell_eval2_version"]
    assert meta["bulk_target_sum"]
    assert meta["metric_names"]


def test_semantic_identity_ignores_control_source(synthetic_pair_with_effect):
    """The producer FORCES control_source="pred", so the anchor's value does not depend on
    what the caller asked for -- keying on it would reject a perfectly good anchor and miss
    the cache for no reason."""
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    base = _resolve_config(EvalConfig(**PROFILE_KW), {})
    same = _resolve_config(EvalConfig(**{**PROFILE_KW, "control_source": "pred"}), {})
    assert semantic_identity(base, real, _names(base)) == \
        semantic_identity(same, real, _names(same))


# ONE MUTATION PER VALUE-AFFECTING DEPENDENCY. The tautological form -- "every key
# anchor_semantic_params returns is in the cache params" -- proves the two functions agree
# with each other, not that either covers the anchor's real dependencies. Each entry below
# names a knob that provably moves a SCORED vcc2026 member.
@pytest.mark.parametrize("field,value", [
    # #268: at 1e6 the split-half ceiling is NEGATIVE on 6/6 real lines
    ("bulk_target_sum", 1e6),
    ("target_sum", 5e4),
    ("version", "v1"),
    ("pert_col", "target_gene"),
    ("control", "ntc"),
    # #248: without the map, guide-level labels match no gene and pds_*'s target-gene
    # exclusion silently no-ops -- a trivially-gameable submission BEAT real ones.
    # `anndata` carries pds_l1/pds_l2/pds_cosine, so the discrimination block is active.
    ("target_gene_map", {"ADNP-1": "ADNP"}),
    ("discrimination", {"exclude_target_gene": False}),
    ("discrimination", {"rank_denominator": "n"}),
    # #343: "row" leaves each reference perturbation's own knockdown visible off-diagonal,
    # which a content-free submission can score off -- +0.49..+0.57 of member score on the
    # three official contexts. An anchor frozen under one scope must not enrol against a run
    # scored under the other.
    ("discrimination", {"exclusion_scope": "row"}),
    # #282: on a fully tied row -- what a control-pasting submission produces under cosine --
    # pds_cosine reads {0.5, 0.5} under "midrank" and the target's ALPHABETICAL index under
    # "position". Omitted from the semantic fields until the cross-provider review of #343.
    ("discrimination", {"tie_policy": "position"}),
])
def test_semantic_identity_moves_with_every_value_affecting_knob(
        synthetic_pair_with_effect, field, value):
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    base = _resolve_config(EvalConfig(**PROFILE_KW), {})
    other = _resolve_config(EvalConfig(**{**PROFILE_KW, field: value}), {})
    assert semantic_identity(base, real, _names(base)) != \
        semantic_identity(other, real, _names(other)), (
        f"{field}={value!r} left the semantic identity unchanged; a cache built under one "
        "would be served to the other"
    )


DE_KW = dict(PROFILE_KW, metrics=["expr_mae", "de_wilcoxon_overlap"],
             device="cpu", de={"backend": "pdex"})


def test_the_cpm_filter_moves_a_DE_anchor_and_leaves_an_expression_one_alone(
        synthetic_pair_with_effect):
    """`filter_gene_min_cpm_cell` is read only by DE paths -- run.py:762, de_compute.py,
    partition_inmem, scale.py's DE call -- so it belongs in the DE-gated block.

    NOTE the value: 5.0 is the DEFAULT (config.py:32). An earlier draft mutated to 5.0 and
    the test could never fail. Use 2.0."""
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    de_a = _resolve_config(EvalConfig(**DE_KW), {})
    de_b = _resolve_config(EvalConfig(**{**DE_KW,
                                         "filter": {"filter_gene_min_cpm_cell": 2.0}}), {})
    assert semantic_identity(de_a, real, _names(de_a)) != \
        semantic_identity(de_b, real, _names(de_b))

    expr_a = _resolve_config(EvalConfig(**PROFILE_KW), {})       # "anndata": no DE metric
    expr_b = _resolve_config(EvalConfig(**{**PROFILE_KW,
                                           "filter": {"filter_gene_min_cpm_cell": 2.0}}), {})
    assert semantic_identity(expr_a, real, _names(expr_a)) == \
        semantic_identity(expr_b, real, _names(expr_b))


def test_discrimination_knobs_leave_a_NON_pds_anchor_alone(synthetic_pair_with_effect):
    """Same principle: an anchor over metrics no pds_* member reaches cannot move when the
    discrimination dispatch's parameters change."""
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    kw = dict(PROFILE_KW, metrics=["expr_mae", "expr_mse"])
    a = _resolve_config(EvalConfig(**kw), {})
    b = _resolve_config(EvalConfig(**{**kw,
                                      "discrimination": {"exclude_target_gene": False}}), {})
    assert semantic_identity(a, real, _names(a)) == semantic_identity(b, real, _names(b))


@pytest.mark.parametrize("de_field,value", [
    ("p_adj_threshold", 0.01), ("min_abs_log2fc", 1.0), ("nan_lfc_policy", "keep"),
    ("fdr_scope", "global"), ("mean_calc", "geometric"),
])
def test_de_knobs_enter_the_identity_ONLY_when_a_DE_metric_is_selected(
        synthetic_pair_with_effect, de_field, value):
    """Mirrors `_result_config_digest`'s `de_backend_used` predicate (run.py:880-890).
    Unconditional DE identity would reject an expression-only anchor after a DE-threshold
    change that provably cannot move it -- and, worse, resolving `backend="auto"` for an
    expression-only run RAISES on a CUDA host without gpudge and in a minimal install."""
    from cell_eval2.anchor import anchor_semantic_params, semantic_identity
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    expr_only = _resolve_config(EvalConfig(**PROFILE_KW), {})       # "anndata": no DE metric
    assert not any(k.startswith("de.")
                   for k in anchor_semantic_params(expr_only, real, _names(expr_only)))
    moved = _resolve_config(EvalConfig(**{**PROFILE_KW, "de": {de_field: value}}), {})
    assert semantic_identity(expr_only, real, _names(expr_only)) == \
        semantic_identity(moved, real, _names(moved)), (
        f"de.{de_field} changed an expression-only anchor's identity"
    )

    with_de = dict(PROFILE_KW, metrics=["expr_mae", "de_wilcoxon_overlap"],
                   device="cpu", de={"backend": "pdex"})
    a = _resolve_config(EvalConfig(**with_de), {})
    b = _resolve_config(EvalConfig(**{**with_de,
                                      "de": {"backend": "pdex", de_field: value}}), {})
    assert semantic_identity(a, real, _names(a)) != semantic_identity(b, real, _names(b))


def test_the_resolved_comparator_is_in_the_identity(synthetic_pair_with_effect):
    """The comparator decides which normalization every expr_* metric is computed in
    (#264 moved it, #268 retuned it). It is resolved from the EFFECTIVE input type
    (run.py:1004), which the declared cfg.input_type does not determine under v1 or
    autodetect -- so it cannot be inferred from the fields alone."""
    from cell_eval2.anchor import anchor_semantic_params
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_pair_with_effect
    cfg = _resolve_config(EvalConfig(**PROFILE_KW), {})
    params = anchor_semantic_params(cfg, real, _names(cfg))
    assert "comparator" in params and params["comparator"]
    # BOTH keys -- the pred side is a half, and autodetect is pred-side only
    assert params["input_type_effective_real"] == "lognorm"
    assert params["input_type_effective_pred"] == "lognorm"


@pytest.mark.parametrize("kw", [dict(autodetect_input_type=True), dict(version="v1")])
def test_the_producer_refuses_a_config_whose_HALVES_could_retype(
        synthetic_pair_with_effect, kw):
    """Under autodetect (pred-side only) or v1 (both sides), the half handed to the pred
    side can classify independently of the full matrix, so the stamped comparator would not
    be a property of the dataset. Refuse rather than stamp a comparator the run may not have
    used."""
    from cell_eval2.anchor import compute_replicate_anchor

    _pred, real = synthetic_pair_with_effect
    with pytest.raises(ValueError, match="autodetect_input_type|version=v1"):
        compute_replicate_anchor(real, config=EvalConfig(**{**PROFILE_KW, **kw}),
                                 base_seed=0, n_splits=1)


def test_read_anchor_roundtrips_all_three_parts(synthetic_pair_with_effect, tmp_path):
    from cell_eval2.anchor import compute_replicate_anchor, read_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    got_anchor, got_splits, meta = read_anchor(outdir)
    want_splits, want_anchor = compute_replicate_anchor(
        real, config=EvalConfig(**PROFILE_KW), base_seed=0, n_splits=2)

    assert got_anchor.sort("metric").equals(want_anchor.sort("metric"))
    assert got_splits.sort("split_index", "metric").equals(
        want_splits.sort("split_index", "metric"))
    assert meta["base_seed"] == 0
    # the sidecar path is also accepted, not only the directory
    again, _s, _m = read_anchor(f"{outdir}/anchor_meta.json")
    assert again.equals(got_anchor)


def test_read_anchor_rejects_a_missing_column(synthetic_pair_with_effect, tmp_path):
    """MALFORMED is a caller error and raises, naming the source -- the validation style
    lfc_nmae_ref uses. A frame with the right shape but the wrong columns must not sail
    through into a division."""
    from cell_eval2.anchor import read_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    agg_path = f"{outdir}/anchor_agg.parquet"
    pl.read_parquet(agg_path).drop("replicate").write_parquet(agg_path)
    with pytest.raises(ValueError, match="replicate"):
        read_anchor(outdir)


def test_read_anchor_rejects_a_directory_with_no_sidecar(synthetic_pair_with_effect,
                                                         tmp_path):
    import os

    from cell_eval2.anchor import read_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    os.remove(f"{outdir}/anchor_meta.json")
    with pytest.raises(ValueError, match="anchor_meta.json"):
        read_anchor(outdir)
