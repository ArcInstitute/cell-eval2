import glob
import json
import os

import numpy as np
import polars as pl
import pytest

pytest.importorskip("cellstream")

from cell_eval2 import partition, scale  # noqa: E402
from cell_eval2.config import DiscriminationParams, EvalConfig  # noqa: E402
from cell_eval2.run import compute_metrics  # noqa: E402


class _StopScoring(Exception):
    """Sentinel: abort the run immediately after the value under test is observed, so the
    assertion does not need a GPU to reach it."""


def _make(tmp_path, *, effect=0.0):
    """The shared shard fixture. ``effect=0.0`` (the default) is the original panel and every
    existing caller keeps its numbers bit-for-bit.

    ⚠️ ``effect > 0`` adds an independent Poisson(effect) draw to every gene of every
    non-control cell, so each perturbation is separated from the control by a real, measurable
    shift. (It is NOT targeted at the perturbation's own gene -- this panel's A-E labels name no
    gene at all, which is the whole reason its callers supply a `target_gene_map`.) Only
    `test_a_partial_does_not_move_a_metric_that_ranks_against_the_whole_real_panel` asks for it,
    and it needs it because of issue #172: the two `expr_*` members it scores now drop each
    target's own column, and on a 10-gene Poisson(0.7) panel with no real effect at all the
    surviving `sum_p expr_distance_unbiased` goes NEGATIVE, which the derived metric refuses --
    correctly, and with a message that says to fix the reference panel. A panel that carries a
    measurable aggregate effect is what that member needs, and it is also what makes that test's
    own `len(set(pds)) > 1` guard mean something.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    rng = np.random.default_rng(2)
    n, g = 120, 10
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A", "B", "C", "D", "E"], 20)})
    counts = rng.poisson(0.7, size=(n, g)).astype(np.float32)
    if effect:
        labels = obs["target"].to_numpy()
        for p in "ABCDE":
            n_p = int((labels == p).sum())
            counts[labels == p] += rng.poisson(effect, size=(n_p, g))
    X = sp.csr_matrix(counts)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    h5 = str(tmp_path / "a.h5ad")
    adata.write_h5ad(h5)
    shd = str(tmp_path / "a.shad")
    write_sharded(adata, shd, group_by="target")
    return h5, shd


def _make_ref(tmp_path):
    """Like _make but writes a reference-designated archive (gpudge-streamable):
    group_by='target' + reference='non-targeting'."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    rng = np.random.default_rng(5)
    n, g = 600, 20
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A", "B", "C", "D", "E"], 100)})
    X = sp.csr_matrix(rng.poisson(1.5, size=(n, g)).astype(np.float32))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    h5 = str(tmp_path / "r.h5ad")
    adata.write_h5ad(h5)
    shd = str(tmp_path / "r.shad")
    write_sharded(adata, shd, group_by="target", reference="non-targeting")
    return h5, shd


def _make_lognorm(tmp_path):
    """A .shad whose values are log1p(CP10K) -- the effective-lognorm streaming case, which
    _make/_make_ref (raw Poisson counts) cannot produce."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    rng = np.random.default_rng(11)
    n, g = 240, 12
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A", "B", "C"], 60)})
    X = rng.poisson(1.5, size=(n, g)).astype(np.float32)
    X = np.log1p(X / np.maximum(X.sum(axis=1, keepdims=True), 1.0) * 1e4).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                       var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    shd = str(tmp_path / "lognorm.shad")
    write_sharded(adata, shd, group_by="target", reference="non-targeting")
    return shd


def _cfg():
    # EvalConfig() defaults ARE v2 (control_source="real", input_type="counts", target_sum=1e6);
    # EvalConfig.v2() takes no kwargs, so construct directly.
    return EvalConfig(metrics=["mae"], pert_col="target", control="non-targeting")


def _expr_comparator_sides(tmp_path, lognorm_side):
    _h5, counts = _make_ref(tmp_path)
    lognorm = _make_lognorm(tmp_path)
    return (
        lognorm if lognorm_side in ("pred", "both") else counts,
        lognorm if lognorm_side in ("real", "both") else counts,
    )


def _declare_pds_as_expr_comparator(monkeypatch):
    from dataclasses import replace
    from cell_eval2 import norm
    from cell_eval2.catalog import CATALOG

    monkeypatch.setitem(
        CATALOG, "pds_cosine",
        replace(CATALOG["pds_cosine"], normalization=norm.EXPR_COMPARATOR),
    )


@pytest.mark.parametrize("lognorm_side", ["pred", "real", "both"])
def test_shard_streaming_rejects_declared_lognorm_for_expr_comparator_metrics(
        tmp_path, monkeypatch, lognorm_side):
    _declare_pds_as_expr_comparator(monkeypatch)
    pred, real = _expr_comparator_sides(tmp_path, lognorm_side)
    with pytest.raises(ValueError, match="counts"):
        scale.score_streaming(
            pred, real,
            config=EvalConfig(
                metrics=["pds_cosine"], input_type="lognorm", target_sum=1e6,
            ),
        )


@pytest.mark.parametrize("lognorm_side", ["pred", "real", "both"])
def test_shard_streaming_rejects_autodetected_lognorm_for_expr_comparator_metrics(
        tmp_path, monkeypatch, lognorm_side):
    _declare_pds_as_expr_comparator(monkeypatch)
    pred, real = _expr_comparator_sides(tmp_path, lognorm_side)
    with pytest.raises(ValueError, match="counts"):
        scale.score_streaming(
            pred, real,
            config=EvalConfig(
                metrics=["pds_cosine"], input_type="counts", target_sum=1e6,
            ),
        )


@pytest.mark.parametrize("lognorm_side", ["pred", "real", "both"])
def test_shard_streaming_rejects_lognorm_for_concrete_expression_metrics(
        tmp_path, lognorm_side):
    pred, real = _expr_comparator_sides(tmp_path, lognorm_side)
    with pytest.raises(ValueError, match=r"expr_mse.*pred=.*real="):
        scale.score_streaming(
            pred, real,
            config=EvalConfig(
                metrics=["expr_mse"], input_type="counts", target_sum=28_000.0,
            ),
        )


def test_shard_streaming_writer_stamps_the_resolved_comparator(tmp_path):
    """The stamp must be what the run RESOLVED, not a constant -- so the expectation is derived
    from the resolver rather than written out.

    ⚠️ Was parameterized over ("counts" -> "bulk_lognorm", "lognorm" -> "lognorm"). The lognorm arm
    reached the writer through a stubbed DE-only request, with a comment saying the anndata family
    could not serve stored lognorm values -- i.e. its route WAS the #266 hole: the DE family had no
    non-counts gate. #266 closes it, so a `comparator="lognorm"` partial is no longer reachable
    from `score_streaming` at all (every catalog metric now trips the gate on a non-counts side).
    That branch is covered by
    `test_shard_streaming_refuses_a_lognorm_archive_for_every_metric_kind` below instead, which
    asserts the refusal rather than routing around it.
    """
    _h5, archive = _make_ref(tmp_path)
    parts = tmp_path / "parts"
    cfg = EvalConfig(metrics=["mae"], input_type="counts", target_sum=1e6)
    scale.score_streaming(archive, archive, config=cfg, partial_out=str(parts))
    meta = json.loads((parts / "all.json").read_text())
    from cell_eval2.norm import resolve_comparator
    expected = resolve_comparator(version=cfg.version, pred_input_type="counts",
                                  real_input_type="counts")
    assert expected == "bulk_lognorm", "v2 counts/counts resolves bulk_lognorm since #264"
    assert meta["comparator"] == expected


@pytest.mark.parametrize("metrics,family", [
    (["mae"], "anndata"),
    (["de_wilcoxon_overlap"], "de"),
    (["mae", "de_wilcoxon_overlap"], "both"),
])
def test_shard_streaming_refuses_a_lognorm_archive_for_every_metric_kind(tmp_path, metrics, family):
    """#266/#182. The anndata family was gated before; the DE family was not, and that gap is what
    the old stamping test above used as its route to the writer. The catalog holds only these two
    kinds, so with both gated `score_streaming` is counts-only in practice."""
    archive = _make_lognorm(tmp_path)
    cfg = EvalConfig(metrics=metrics, input_type="counts", target_sum=1e6)
    with pytest.raises(ValueError, match="requires raw counts on BOTH sides"):
        scale.score_streaming(archive, archive, config=cfg,
                              partial_out=str(tmp_path / f"parts_{family}"))
    assert not (tmp_path / f"parts_{family}").exists(), \
        "the gate must fire before any partial is written"


def test_streaming_resolves_target_sum_none_from_the_reference_shard(tmp_path, monkeypatch):
    """#155: target_sum=None must become the real archive's control-pool median before any
    artifact is built -- previously it reached GroupedMeanAccumulator as None and raised
    TypeError from float(None)."""
    from dataclasses import replace

    from cellstream.read import ShardedArchive

    from cell_eval2 import scale
    from cell_eval2.config import EvalConfig
    from cell_eval2.norm import resolve_target_sum

    seen = {}
    # _make_ref returns (h5ad_path, shad_path) -- NOT (real, pred). Passing the .h5ad as the
    # real archive fails in shad_fingerprint long before the capture seam.
    _h5, shd = _make_ref(tmp_path)

    def _capture(*a, **kw):
        seen["target_sum"] = kw.get("target_sum")
        raise _StopScoring()

    monkeypatch.setattr(scale, "streaming_pseudobulk", _capture)
    cfg = replace(EvalConfig.v2(), target_sum=None, metrics=["expr_mae"])
    # One archive on both sides: only the resolved target is under test here, not the scores.
    with pytest.raises(_StopScoring):
        scale.score_streaming(shd, shd, config=cfg)
    expected = resolve_target_sum(ShardedArchive(shd).read_reference(),
                                  input_type="counts", target_sum=None)
    assert seen["target_sum"] == expected > 0


def test_streaming_target_sum_none_without_a_resolvable_control_fails_clearly(tmp_path):
    """read_reference() returns None for an archive written without a designated reference
    (``cellstream.read.ShardedArchive.read_reference``), but read_group RAISES KeyError for an
    unknown label (``...read_group``) -- it NEVER returns None. So the fallback must catch
    KeyError, or the
    caller gets a bare `KeyError: Group label ... not found` instead of the actionable message.
    _make (:15) writes no reference=, so this exercises both halves of the fallback."""
    from dataclasses import replace

    from cell_eval2 import scale
    from cell_eval2.config import EvalConfig

    _h5, shd = _make(tmp_path)
    cfg = replace(EvalConfig.v2(), target_sum=None, metrics=["expr_mae"],
                  control="not-a-real-control")
    with pytest.raises(ValueError, match="not-a-real-control"):
        scale.score_streaming(shd, shd, config=cfg)


@pytest.mark.parametrize("metrics", [["expr_mae"], ["de_wilcoxon_overlap"]])
def test_streaming_lognorm_target_sum_none_raises_for_any_metric(tmp_path, metrics):
    """An unresolved target is NOT inert on the streaming path -- BOTH halves misbehave:

    - anndata: streaming_pseudobulk computes `target_sum / libs` (streaming_bulk.py:128) and
      GroupedMeanAccumulator does float(target_sum) (gpu/bulk.py:56) -> TypeError;
    - DE: compute_de_streaming takes no input_type at all and maps None -> gpudge "median"
      unconditionally (de_compute.py:667), library-size-normalizing already-log1p'd values.
      Silently wrong, which is worse.

    So the guard covers ANY requested metric, and it sits before the DE dispatch, which is what
    makes this runnable on a GPU-less host.

    ⚠️ #266 moved WHERE this is caught, not whether. Its non-counts gate now runs BEFORE the
    median resolution -- so a lognorm archive is refused for the raw-counts contract first, and
    #155's own NotImplementedError (raised from below the resolution) is subsumed. Ordering it that
    way is deliberate: the old order read the control pool for a run that can never succeed, and a
    lognorm archive with a bad control label reported the CONTROL as the problem, which fixing
    would not have helped. The gate carries #155's specific target_sum advice, and this test still
    requires it -- so the wording cannot be silently lost, only relocated.
    """
    from dataclasses import replace

    from cell_eval2 import scale
    from cell_eval2.config import EvalConfig

    shd = _make_lognorm(tmp_path)
    cfg = replace(EvalConfig.v2(), input_type="lognorm", target_sum=None, metrics=metrics,
                  validate_input=False)
    with pytest.raises(ValueError, match="target_sum=None") as ei:
        scale.score_streaming(shd, shd, config=cfg)
    assert "requires raw counts on BOTH sides" in str(ei.value)
    assert "#155" in str(ei.value)


def test_streaming_lognorm_with_a_bad_control_reports_the_INPUT_TYPE_not_the_control(tmp_path):
    """The reason #266's gate runs before the median resolution (codex-review). Previously the
    control read happened first, so an unsupported lognorm archive with a wrong control label
    blamed the control -- and fixing the control would not have made the run possible."""
    from dataclasses import replace

    from cell_eval2 import scale
    from cell_eval2.config import EvalConfig

    shd = _make_lognorm(tmp_path)
    cfg = replace(EvalConfig.v2(), input_type="lognorm", target_sum=None, metrics=["expr_mae"],
                  control="not-a-real-control", validate_input=False)
    with pytest.raises(ValueError, match="requires raw counts on BOTH sides") as ei:
        scale.score_streaming(shd, shd, config=cfg)
    assert "not-a-real-control" not in str(ei.value)


def test_streaming_matches_compute_metrics(tmp_path):
    h5, shd = _make(tmp_path)
    ref = compute_metrics(h5, h5, config=_cfg()).sort(["perturbation", "metric"])
    got = scale.score_streaming(shd, shd, config=_cfg()).sort(["perturbation", "metric"])
    j = ref.join(got, on=["perturbation", "metric"], suffix="_s")
    assert j.height == ref.height and ref.height > 0
    for a, b in zip(j["value"], j["value_s"]):
        assert abs(a - b) <= 1e-4 * abs(a) + 1e-6


def test_subset_partition_is_exact(tmp_path):
    h5, shd = _make(tmp_path)
    whole = scale.score_streaming(shd, shd, config=_cfg()).sort(["perturbation", "metric"])
    out = str(tmp_path / "parts")
    for i in range(2):
        df = scale.score_streaming(shd, shd, config=_cfg(), fraction=2, index=i)
        meta = {
            "real_ref_fingerprint": "rf",
            "config_hash": "cf",
            "comparator": "lognorm",
            "perturbations": sorted(set(df["perturbation"])),
        }
        partition.write_partial(df, out, subset_id=f"s{i}", meta=meta)
    full, _agg = partition.aggregate_partials(out)
    full = full.sort(["perturbation", "metric"])
    assert full.to_dicts() == whole.to_dicts()


def _make_independent_pred(tmp_path):
    """A prediction that is NOT the reference: same labels, an independent draw.

    `_make` scored against itself is a PERFECT prediction, so every `pds_cosine` is exactly
    1.0 at any cohort size -- a constant that cannot show a re-ranking. An independent draw
    makes the ranks non-trivial, which is what the partial has to preserve.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    rng = np.random.default_rng(17)
    n, g = 120, 10
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A", "B", "C", "D", "E"], 20)})
    X = sp.csr_matrix(rng.poisson(0.7, size=(n, g)).astype(np.float32))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    shd = str(tmp_path / "pred.shad")
    write_sharded(adata, shd, group_by="target")
    return shd


def test_a_partial_does_not_move_a_metric_that_ranks_against_the_whole_real_panel(tmp_path):
    """`pds_*` ranks every predicted effect against ALL real effects and takes its denominator
    from the full real count (`discrimination.py:114-150`), so a partial must leave each
    perturbation's value exactly where the whole-panel run put it.

    This is the regression #257's first pass introduced: giving `expr_distance_unbiased` its
    cohort by restricting `real_bulks` before dispatch silently re-ranked PDS on every partial.
    `expr_distance_unbiased` is in the metric list because it is what forced the question -- it
    reads only the real side, so without a cohort it emits the whole panel from every partial
    and `aggregate_partials` rejects the repeated rows. The existing subset test scores `mae`
    alone, which is per-perturbation and could not have caught either half.
    """
    # effect=6.0: the two expr_* members need a panel with a measurable aggregate effect once
    # #172's exclusion has removed each target's own column -- see `_make`.
    _h5, real_shd = _make(tmp_path, effect=6.0)
    pred_shd = _make_independent_pred(tmp_path)
    metrics = ["pds_cosine", "expr_mse_unbiased_capped", "expr_distance_unbiased"]
    # exclude_target_gene=False deliberately, for `pds_cosine`: this fixture's labels are A-E
    # against genes g0-g9, so nothing resolves and the #248 gate would refuse to score at all.
    # ⚠️ The two `expr_*` members do NOT read that flag -- since issue #172 the exclusion is
    # part of what they compute, so switching it off is not available and the map is the only
    # remedy. Same map as `test_target_gene_exclusion_is_live_through_shard_streaming` below;
    # here it exists to make the panel scoreable, not to be the thing under test.
    cfg = EvalConfig(metrics=metrics, pert_col="target", control="non-targeting",
                     target_gene_map={p: f"g{i}" for i, p in enumerate("ABCDE")},
                     discrimination=DiscriminationParams(exclude_target_gene=False))
    whole = scale.score_streaming(pred_shd, real_shd,
                                  config=cfg).sort(["perturbation", "metric"])
    pds = whole.filter(pl.col("metric") == "pds_cosine")["value"].to_list()
    assert len(set(pds)) > 1, (
        f"every pds_cosine value is identical ({pds}); a rank metric that does not vary cannot "
        "detect a changed reference cohort, so this test would assert nothing"
    )

    out = str(tmp_path / "pds-parts")
    for i in range(2):
        df = scale.score_streaming(pred_shd, real_shd, config=cfg, fraction=2, index=i)
        partition.write_partial(df, out, subset_id=f"s{i}", meta={
            "real_ref_fingerprint": "rf", "config_hash": "cf", "comparator": "lognorm",
            "perturbations": sorted(set(df["perturbation"]))})
    full, _agg = partition.aggregate_partials(out)
    assert full.sort(["perturbation", "metric"]).to_dicts() == whole.to_dicts()


def test_target_gene_exclusion_is_live_through_shard_streaming(tmp_path):
    """`exclude_target_gene=True` survives the shard-streaming driver and MOVES the numbers.

    #275 turned the knob OFF in every other shard-streaming pds test on this fixture, because
    its A-E labels resolve to no gene in the g0-g9 panel and #248's gate refuses to score at
    all. That is the honest fix for a parity test -- but it would leave the ENTIRE shard
    layout with no end-to-end proof that exclusion still reaches `resolve_exclusion_columns`
    through `score_streaming`, which is how #248 could silently stop working here and nothing
    would notice (codex flagged exactly this gap on the #275 review).

    So: supply the `target_gene_map` arm of #248's remedy instead of switching exclusion off,
    and pin BOTH halves --

      1. it scores at all (the gate resolves, so the map is genuinely consumed), and
      2. it does not agree with the exclusion-off run.

    (2) is the part that cannot be faked: dropping each target's own transcript re-ranks the
    cohort, so an exclusion that quietly became a no-op would make the two runs identical and
    fail here. `_make_independent_pred` is required for that -- `_make` against itself scores
    a perfect 1.0 everywhere, a constant that cannot show a re-ranking.

    Needs no GPU -- but note it still does NOT run in ordinary CI, and that is a #224 data
    point rather than an oversight: this whole module sits behind
    `pytest.importorskip("cellstream")` (line 9), and `ci.yml` installs base+dev deliberately
    WITHOUT the scale extra. That install's ORIGINAL reason -- resolving `scale` pulled a private
    git URL and failed -- is gone now that cellstream is public; it stays base+dev because that is
    the minimal explicit thing and it keeps the documented skip counts stable, so this module
    still does not run there. So this runs wherever cellstream is installed -- a scale-enabled
    venv, and the H100 release gate -- and skips on the CI runners. There is currently no driver-level
    exclusion assertion that ordinary CI executes.
    """
    _h5, real_shd = _make(tmp_path)
    pred_shd = _make_independent_pred(tmp_path)
    metrics = ["pds_cosine", "pds_l1"]
    common = dict(metrics=metrics, pert_col="target", control="non-targeting")
    # _make/_make_independent_pred label perturbations A-E over var_names g0-g9, so the map is
    # what makes them resolve; without it this config is the #275 failure itself.
    on = EvalConfig(**common, target_gene_map={p: f"g{i}" for i, p in enumerate("ABCDE")},
                    discrimination=DiscriminationParams(exclude_target_gene=True))
    off = EvalConfig(**common,
                     discrimination=DiscriminationParams(exclude_target_gene=False))

    r_on = scale.score_streaming(pred_shd, real_shd, config=on).sort(["perturbation", "metric"])
    r_off = scale.score_streaming(pred_shd, real_shd, config=off).sort(["perturbation", "metric"])
    assert r_on.height == r_off.height > 0

    # Join on the keys, never a positional diff of the two value columns: equal heights do not
    # imply equal key sets. The join height equalling both inputs IS the key-set assertion.
    j = r_on.join(r_off, on=["perturbation", "metric"], how="inner", suffix="_off")
    assert j.height == r_on.height == r_off.height
    # Same null/finiteness guards as the rowstore sibling, and for the same reason: a null
    # propagates through the subtraction and the comparison, and `.sum()` SKIPS nulls -- so a
    # null would quietly lower `moved` instead of failing. pds values are finite here.
    assert j["value"].null_count() == 0 and j["value_off"].null_count() == 0, (
        f"null pds values: {j.filter(pl.col('value').is_null() | pl.col('value_off').is_null())}"
    )
    assert j["value"].is_finite().all() and j["value_off"].is_finite().all(), (
        f"pds values must be finite on both arms; got "
        f"{j.filter(~(pl.col('value').is_finite() & pl.col('value_off').is_finite()))}"
    )
    moved = ((j["value"] - j["value_off"]).abs() > 1e-12).sum()
    assert moved > 0, (
        "exclude_target_gene=True produced numbers identical to exclude_target_gene=False "
        "through score_streaming; the exclusion is a no-op on the shard layout"
    )


def test_derived_ratio_of_sums_is_exact_across_two_streaming_partials(tmp_path, monkeypatch):
    from cell_eval2.catalog import CATALOG, DerivedAgg, MetricSpec
    from cell_eval2.run import aggregate_metrics
    from cell_eval2.scoring import DIAG, Scoring

    numerator = "expr_mse_unbiased_capped"
    denominator = "expr_distance_unbiased"
    derived = "expr_mse_unbiased_capped_norm"

    def _numerator(pred_bulk, control):
        return {str(p): 1.0 for p in pred_bulk[0] if str(p) != control}

    def _denominator(real_bulk, control):
        return {str(p): 2.0 for p in real_bulk[0] if str(p) != control}

    monkeypatch.setitem(CATALOG, numerator, MetricSpec(
        name=numerator, func=_numerator, scoring=DIAG, agg="mean", profiles=("full",),
        kind="anndata", normalization="lognorm"))
    monkeypatch.setitem(CATALOG, denominator, MetricSpec(
        name=denominator, func=_denominator, scoring=DIAG, agg="mean", profiles=("full",),
        kind="anndata", normalization="lognorm"))
    monkeypatch.setitem(CATALOG, derived, MetricSpec(
        name=derived, func=None, agg="ratio_of_sums",
        derived=DerivedAgg(numerator=numerator, denominator=denominator),
        scoring=Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                        clamp_low=None, clamp_high=1.0),
        profiles=("full",), kind="anndata", normalization="lognorm"))

    _h5, shd = _make(tmp_path)
    cfg = EvalConfig(metrics=[numerator, denominator, derived], pert_col="target",
                     control="non-targeting")
    whole = scale.score_streaming(shd, shd, config=cfg)
    whole_value = aggregate_metrics(whole).filter(pl.col("metric") == derived)["mean"][0]

    out = str(tmp_path / "derived-parts")
    for i in range(2):
        scale.score_streaming(shd, shd, config=cfg, fraction=2, index=i, partial_out=out)
    sidecars = sorted(glob.glob(os.path.join(out, "*.json")))
    assert len(sidecars) == 2, f"expected two partial sidecars, found {sidecars}"
    expected_metrics = sorted([numerator, denominator, derived])
    for sidecar in sidecars:
        with open(sidecar, encoding="utf-8") as fh:
            sidecar_metrics = json.load(fh).get("metrics")
        assert sidecar_metrics == expected_metrics, (
            f"sidecar {sidecar} metrics {sidecar_metrics!r} != {expected_metrics!r}"
        )
    _full, partial_agg = partition.aggregate_partials(out)
    partial_value = partial_agg.filter(pl.col("metric") == derived)["mean"][0]
    assert partial_value == pytest.approx(whole_value), (
        f"two-partial derived value {partial_value} != whole-panel value {whole_value}"
    )


def _write_aggregation_partial(out, subset_id, rows, *, metrics=None):
    meta = {
        "real_ref_fingerprint": "rf",
        "config_hash": "cf",
        "comparator": "lognorm",
        "perturbations": sorted({row[0] for row in rows}),
    }
    if metrics is not None:
        meta["metrics"] = metrics
    df = pl.DataFrame(
        rows,
        schema={"perturbation": pl.String, "metric": pl.String, "value": pl.Float64},
        orient="row",
    )
    partition.write_partial(df, out, subset_id=subset_id, meta=meta)


def test_selected_derived_metric_with_no_partial_numerator_rows_raises(tmp_path):
    numerator = "expr_mse_unbiased_capped"
    denominator = "expr_distance_unbiased"
    derived = "expr_mse_unbiased_capped_norm"
    metrics = sorted([derived, numerator, denominator])
    out = str(tmp_path / "denominator-only-parts")
    _write_aggregation_partial(out, "s0", [("A", denominator, 2.0)], metrics=metrics)
    _write_aggregation_partial(out, "s1", [("B", denominator, 4.0)], metrics=metrics)

    with pytest.raises(ValueError) as excinfo:
        partition.aggregate_partials(out, reference_universe=["A", "B"])
    message = str(excinfo.value)
    assert derived in message and f"{numerator} is empty" in message, (
        f"missing-numerator aggregation raised the wrong error: {message!r}"
    )


def test_partial_sidecars_with_different_metric_selections_raise_and_name_both(tmp_path):
    out = str(tmp_path / "mixed-metric-parts")
    first = ["expr_mae"]
    second = ["expr_mse"]
    _write_aggregation_partial(out, "s0", [("A", "expr_mae", 1.0)], metrics=first)
    _write_aggregation_partial(out, "s1", [("B", "expr_mse", 2.0)], metrics=second)

    with pytest.raises(ValueError) as excinfo:
        partition.aggregate_partials(out)
    message = str(excinfo.value)
    assert first[0] in message and second[0] in message, (
        f"metric-selection disagreement did not name both {first!r} and {second!r}: {message!r}"
    )


def test_a_metric_selection_conflict_is_fatal_even_beside_a_sidecar_that_predates_the_key(
        tmp_path):
    """The conflict check must not be gated on EVERY sidecar declaring a selection.

    Two sidecars that declare different selections are incompatible whatever a third, older
    one says -- gating the raise on `all_have_metrics` let exactly that pair through and fall
    back to legacy aggregation (Gemini, PR #262). Only the decision to PASS a selection
    downstream needs unanimity, which the test below pins.
    """
    out = str(tmp_path / "mixed-and-legacy-parts")
    _write_aggregation_partial(out, "s0", [("A", "expr_mae", 1.0)], metrics=["expr_mae"])
    _write_aggregation_partial(out, "s1", [("B", "expr_mse", 2.0)], metrics=["expr_mse"])
    _write_aggregation_partial(out, "s2", [("C", "expr_mae", 3.0)])   # no `metrics` key

    with pytest.raises(ValueError, match="differ in metric selections") as excinfo:
        partition.aggregate_partials(out)
    message = str(excinfo.value)
    assert "expr_mae" in message and "expr_mse" in message, (
        f"the raise did not name both declared selections: {message!r}"
    )


def test_partial_sidecars_without_metrics_keep_legacy_derived_injection(tmp_path):
    from cell_eval2.run import aggregate_metrics

    numerator = "expr_mse_unbiased_capped"
    denominator = "expr_distance_unbiased"
    derived = "expr_mse_unbiased_capped_norm"
    out = str(tmp_path / "legacy-parts")
    _write_aggregation_partial(
        out, "s0", [("A", numerator, 1.0), ("A", denominator, 2.0)]
    )
    _write_aggregation_partial(
        out, "s1", [("B", numerator, 3.0), ("B", denominator, 6.0)]
    )
    sidecar_meta = []
    for sidecar in sorted(glob.glob(os.path.join(out, "*.json"))):
        with open(sidecar, encoding="utf-8") as fh:
            sidecar_meta.append(json.load(fh))
    assert all("metrics" not in meta for meta in sidecar_meta), (
        f"legacy fixture unexpectedly contains metrics metadata: {sidecar_meta!r}"
    )

    full, got = partition.aggregate_partials(out)
    expected = aggregate_metrics(full)
    assert got.to_dicts() == expected.to_dicts(), (
        f"legacy partial aggregate {got.to_dicts()!r} != legacy direct aggregate "
        f"{expected.to_dicts()!r}"
    )
    derived_value = got.filter(pl.col("metric") == derived)["mean"].to_list()
    assert derived_value == pytest.approx([0.5]), (
        f"legacy derived injection produced {derived_value!r}, expected [0.5]"
    )


def test_cache_real_reuse_and_real_fingerprint(tmp_path):
    h5, shd = _make(tmp_path)
    cache = str(tmp_path / "realcache")
    cfg = EvalConfig(
        metrics=["mae"], pert_col="target", control="non-targeting", cache_real=cache
    )
    a = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    assert glob.glob(os.path.join(cache, "stream_pseudobulk_*.npz"))
    b = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    assert a.to_dicts() == b.to_dicts()
    out = str(tmp_path / "parts2")
    scale.score_streaming(shd, shd, config=cfg, partial_out=out)
    meta = json.load(open(glob.glob(os.path.join(out, "*.json"))[0]))
    assert meta["real_ref_fingerprint"].startswith("shad:")


def test_streaming_de_requires_gpudge_backend(tmp_path):
    # DE on the streaming path with an explicit CPU backend errors clearly (no GPU needed):
    # streaming DE is gpudge-only. Replaces the old "raises NotImplementedError (Plan 2)" test.
    _h5, shd = _make_ref(tmp_path)
    cfg = EvalConfig(metrics=["de_wilcoxon_nsig_counts_real"], pert_col="target",
                     control="non-targeting", de={"backend": "scanpy"})
    with pytest.raises(ValueError, match="gpudge"):
        scale.score_streaming(shd, shd, config=cfg)


from cell_eval2.gpu import resolve_device  # noqa: E402

_HAS_GPU = resolve_device("auto") == "cuda"


def _cfg_dev(device, metrics=("mae",)):
    # exclude_target_gene=False deliberately (#275), same reason as the fixture at
    # test_a_partial_does_not_move_a_metric_...: _make's labels are A-E against genes g0-g9,
    # so nothing resolves and #248's gate would refuse to score any pds_* metric at all.
    # These are cpu-vs-cuda parity tests -- both sides share this config, so the equality
    # asserted is unaffected. Safe for the mae-only callers, which never reach discrimination.
    return EvalConfig(metrics=list(metrics), pert_col="target",
                      control="non-targeting", device=device,
                      discrimination=DiscriminationParams(exclude_target_gene=False))


def test_streaming_device_cpu_matches_compute_metrics(tmp_path):
    # device="cpu" streaming reproduces the in-memory reference (the merged CPU path).
    h5, shd = _make(tmp_path)
    ref = compute_metrics(h5, h5, config=_cfg_dev("cpu")).sort(["perturbation", "metric"])
    got = scale.score_streaming(shd, shd, config=_cfg_dev("cpu")).sort(["perturbation", "metric"])
    j = ref.join(got, on=["perturbation", "metric"], suffix="_s")
    assert j.height == ref.height and ref.height > 0
    for a, b in zip(j["value"], j["value_s"]):
        assert abs(a - b) <= 1e-4 * abs(a) + 1e-6


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_streaming_pseudobulk_cuda_dtype_and_parity(tmp_path):
    # Proves the GPU accumulator is actually used on cuda (fp32 means) and matches the
    # fp64 CPU streaming path within tolerance.
    from cell_eval2.streaming_bulk import streaming_pseudobulk

    _h5, shd = _make(tmp_path)
    cpu = streaming_pseudobulk(shd, pert_col="target", norms=["counts", "normalized", "lognorm"],
                               target_sum=1e6, device="cpu")
    gpu = streaming_pseudobulk(shd, pert_col="target", norms=["counts", "normalized", "lognorm"],
                               target_sum=1e6, device="cuda")
    for n in ("counts", "normalized", "lognorm"):
        assert cpu[n][1].dtype == np.float64  # CPU path stays fp64
        assert gpu[n][1].dtype == np.float32  # GPU accumulator -> fp32 (path actually taken)
        assert np.allclose(cpu[n][1], gpu[n][1], rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_streaming_cuda_matches_cpu_full_metrics(tmp_path):
    # End-to-end: cuda vs cpu over pseudobulk + discrimination + delta metrics.
    _h5, shd = _make(tmp_path)
    metrics = ["mae", "pds_l1", "pds_l2", "pds_cosine", "delta_pearson"]
    cpu = scale.score_streaming(shd, shd, config=_cfg_dev("cpu", metrics)).sort(["perturbation", "metric"])
    gpu = scale.score_streaming(shd, shd, config=_cfg_dev("cuda", metrics)).sort(["perturbation", "metric"])
    j = cpu.join(gpu, on=["perturbation", "metric"], suffix="_g")
    assert j.height == cpu.height and cpu.height > 0
    for a, b in zip(j["value"], j["value_g"]):
        assert abs(a - b) <= 1e-4 * abs(a) + 1e-6


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_dispatch_routes_discrimination_to_gpu(tmp_path, monkeypatch):
    # Proves the discrimination metric is routed through the GPU kernel on cuda (not a
    # silent CPU fallback) by spying on discrimination_ranks.
    import cell_eval2.gpu.distances as gd

    _h5, shd = _make(tmp_path)
    calls = {"n": 0, "device": None}
    orig = gd.discrimination_ranks

    def spy(*a, **k):
        calls["n"] += 1
        calls["device"] = k.get("device")
        return orig(*a, **k)

    monkeypatch.setattr(gd, "discrimination_ranks", spy)
    # exclude_target_gene=False deliberately (#275): _make's labels are A-E against genes
    # g0-g9, so #248's gate would raise before the kernel this test spies on is ever reached.
    cfg = EvalConfig(metrics=["pds_l1"], pert_col="target", control="non-targeting",
                     device="cuda",
                     discrimination=DiscriminationParams(exclude_target_gene=False))
    scale.score_streaming(shd, shd, config=cfg)
    assert calls["n"] >= 1
    assert calls["device"] == "cuda"


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_streaming_de_matches_in_memory_gpudge(tmp_path):
    # Streaming gpudge DE metrics match the in-memory gpudge DE metrics on the same data.
    de_metrics = ["de_wilcoxon_overlap", "de_wilcoxon_precision", "de_wilcoxon_nsig_counts_real",
                  "de_wilcoxon_sig_recall", "de_wilcoxon_lfc_spearman", "de_wilcoxon_pr_auc"]
    h5, shd = _make_ref(tmp_path)
    cfg = EvalConfig(metrics=de_metrics, pert_col="target", control="non-targeting",
                     de={"backend": "gpudge"})
    ref = compute_metrics(h5, h5, config=cfg).sort(["perturbation", "metric"])
    got = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    j = ref.join(got, on=["perturbation", "metric"], suffix="_s")
    assert j.height == ref.height > 0
    for a, b in zip(j["value"], j["value_s"]):
        # degenerate perts (no significant genes) give NaN for lfc_spearman/pr_auc on BOTH
        # sides — a match, but abs(a-b) would be NaN, so compare NaN-positions explicitly.
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert abs(a - b) <= 1e-4 * abs(a) + 1e-6


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_streaming_de_subset_partition_is_exact(tmp_path):
    # Per-pert DE metrics partition exactly: the union of fraction subsets equals the whole run.
    h5, shd = _make_ref(tmp_path)
    cfg = EvalConfig(metrics=["de_wilcoxon_overlap", "de_wilcoxon_nsig_counts_real"],
                     pert_col="target", control="non-targeting", de={"backend": "gpudge"})
    whole = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    out = str(tmp_path / "departs")
    for i in range(2):
        df = scale.score_streaming(shd, shd, config=cfg, fraction=2, index=i)
        partition.write_partial(df, out, subset_id=f"s{i}",
                                meta={"real_ref_fingerprint": "rf", "config_hash": "cf",
                                      "comparator": "lognorm",
                                      "perturbations": sorted(set(df["perturbation"]))})
    full, _agg = partition.aggregate_partials(out)
    assert full.sort(["perturbation", "metric"]).to_dicts() == whole.to_dicts()


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_streaming_de_real_cache_reuse(tmp_path):
    # The real-side DE table is cached under cache_real; a second run reuses it (identical result).
    h5, shd = _make_ref(tmp_path)
    cache = str(tmp_path / "realcache")
    cfg = EvalConfig(metrics=["de_wilcoxon_overlap"], pert_col="target",
                     control="non-targeting", de={"backend": "gpudge"}, cache_real=cache)
    a = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    assert glob.glob(os.path.join(cache, "stream_de_*.parquet"))
    b = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    assert a.to_dicts() == b.to_dicts()


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_streaming_full_metrics_coexist(tmp_path):
    # DE (gpudge stream) + anndata (lognorm pseudobulk stream) coexist in one score_streaming
    # call and both match the in-memory reference.
    h5, shd = _make_ref(tmp_path)
    metrics = ["mae", "pds_cosine", "de_wilcoxon_overlap", "de_wilcoxon_nsig_counts_real"]
    # exclude_target_gene=False deliberately (#275): _make_ref's labels are A-E against genes
    # g0-g19, so #248's gate would refuse pds_cosine. Both sides of the streaming-vs-in-memory
    # comparison share this config, so the equality asserted is unaffected.
    cfg = EvalConfig(metrics=metrics, pert_col="target", control="non-targeting",
                     de={"backend": "gpudge"},
                     discrimination=DiscriminationParams(exclude_target_gene=False))
    ref = compute_metrics(h5, h5, config=cfg).sort(["perturbation", "metric"])
    got = scale.score_streaming(shd, shd, config=cfg).sort(["perturbation", "metric"])
    j = ref.join(got, on=["perturbation", "metric"], suffix="_s")
    assert j.height == ref.height > 0
    assert {"expr_mae", "de_wilcoxon_overlap"} <= set(got["metric"])  # both families present
    for a, b in zip(j["value"], j["value_s"]):
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert abs(a - b) <= 1e-4 * abs(a) + 1e-6


def _make_pinned_pred(tmp_path):
    """A prediction whose per-cell scatter is maximal while its group AGGREGATES are controlled.

    #348's shape, built from real cells so the production pseudobulk and the production
    delete-1-cell jackknife are what the test reads -- not injected moments. Within each group the
    cells alternate on a checkerboard between `2 * m` and `0`, so the group mean is exactly `m`
    while the per-cell dispersion (and therefore `jk_pred`) is as large as the depth allows.

    ⚠️ The per-perturbation difference has to be a difference in SHAPE, not in depth. A first
    version gave group `p` the profile `m_p * ones(G)`; under `bulk_lognorm` the pseudobulk divides
    by the group total, so every group normalized to the SAME vector, the budget was 0 in the whole
    panel and in every subset alike, and the partial-vs-whole assertion passed with the plumbing
    removed. Each group therefore boosts one OFF-TARGET gene instead: off-target because #172 drops
    each row's own target column from the budget as well as from the distance, so a boost there
    would be invisible for the same reason.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    n_per, g = 20, 10
    labels = ["non-targeting", "A", "B", "C", "D", "E"]
    obs = pd.DataFrame({"target": np.repeat(labels, n_per)})
    counts = np.zeros((len(labels) * n_per, g), dtype=np.float32)
    for p, label in enumerate(labels):
        m = np.full(g, 5.0)
        if label != "non-targeting":
            m[(p - 1) + 5] += 0.5                   # one OFF-TARGET gene per perturbation: SHAPE
        for i in range(n_per):
            row = p * n_per + i
            keep = (np.arange(g) + i) % 2 == 0
            counts[row, keep] = 2 * m[keep]
    X = sp.csr_matrix(counts)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    shd = str(tmp_path / "pinned.shad")
    write_sharded(adata, shd, group_by="target")
    return shd


def test_348s_budget_is_the_whole_panels_through_shard_streaming(tmp_path):
    """The production path for #348's `pred_bulks_full`, on real cells.

    `score_streaming` restricts the pred bulks to the perturbations a partial emits, and #348's
    correction budget is the one term whose value depends on which OTHER predicted perturbations
    are present. Without the unrestricted dict each fraction would form its own ratio and the
    member would depend on the partitioning; with it, concatenated fractions match a whole-panel
    run exactly.

    ⚠️ The arm has to be in the BINDING regime or this asserts nothing: an independent-draw
    prediction has a budget far above its claim, `r = 1` either way, and the test passes with the
    plumbing removed (measured -- that is why `_make_pinned_pred` exists rather than reusing
    `_make_independent_pred`). The guard below pins that.
    """
    _h5, real_shd = _make(tmp_path, effect=6.0)
    pred_shd = _make_pinned_pred(tmp_path)
    cfg = EvalConfig(metrics=["expr_mse_unbiased_capped", "expr_distance_unbiased"],
                     pert_col="target", control="non-targeting",
                     target_gene_map={p: f"g{i}" for i, p in enumerate("ABCDE")},
                     discrimination=DiscriminationParams(exclude_target_gene=False))
    whole = scale.score_streaming(pred_shd, real_shd, config=cfg).sort(["perturbation", "metric"])

    # the guard: with the correction unbounded the values MUST differ, or `r` is 1 and the
    # partial-vs-whole comparison below cannot distinguish anything
    import cell_eval2.metrics.delta as delta_mod
    unbounded = None
    try:
        saved = delta_mod._across_pert_budget
        delta_mod._across_pert_budget = lambda *_a, **_k: float("inf")
        unbounded = scale.score_streaming(pred_shd, real_shd,
                                          config=cfg).sort(["perturbation", "metric"])
    finally:
        delta_mod._across_pert_budget = saved
    assert unbounded.to_dicts() != whole.to_dicts(), (
        "#348's budget does not bind on this fixture, so partial-vs-whole proves nothing"
    )

    out = str(tmp_path / "mse348-parts")
    for i in range(2):
        df = scale.score_streaming(pred_shd, real_shd, config=cfg, fraction=2, index=i)
        partition.write_partial(df, out, subset_id=f"s{i}", meta={
            "real_ref_fingerprint": "rf", "config_hash": "cf", "comparator": "lognorm",
            "perturbations": sorted(set(df["perturbation"]))})
    full, _agg = partition.aggregate_partials(out)
    assert full.sort(["perturbation", "metric"]).to_dicts() == whole.to_dicts()
