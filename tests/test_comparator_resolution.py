import pytest
from cell_eval2 import norm


def test_the_result_cache_digest_separates_the_two_comparators():
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _result_config_digest

    cfg = EvalConfig()
    assert _result_config_digest(
        cfg, de_backend_used=False, comparator="bulk_lognorm",
    ) != _result_config_digest(
        cfg, de_backend_used=False, comparator="lognorm",
    )


def test_the_baseline_digest_separates_them_too():
    from cell_eval2.baseline import config_digest
    from cell_eval2.config import EvalConfig

    assert config_digest(
        EvalConfig(), comparator="bulk_lognorm",
    ) != config_digest(
        EvalConfig(), comparator="lognorm",
    )


def _meta_pair(tmp_path, *, run_comparator, base_comparator, drop=None):
    from argparse import Namespace
    import json

    common = {
        "cell_eval2_version": "test",
        "config_digest": "same-config",
        "source_fingerprint": "same-source",
        "source_fingerprint_strict": True,
        "resolved_device": "cpu",
        "resolved_de_backend": None,
        "input_type_real_effective": "counts",
        "input_type_pred_effective": "counts",
        "de_real_fingerprint": None,
    }
    run_meta = {**common, "comparator": run_comparator}
    base_meta = {**common, "comparator": base_comparator}
    if drop in ("run", "both"):
        run_meta.pop("comparator")
    if drop in ("base", "both"):
        base_meta.pop("comparator")
    run_path = tmp_path / "run_meta.json"
    base_path = tmp_path / "baseline_meta.json"
    run_path.write_text(json.dumps(run_meta))
    base_path.write_text(json.dumps(base_meta))
    return Namespace(
        user_meta=str(run_path), baseline_meta=str(base_path),
        user_agg=str(tmp_path / "user.csv"), baseline_agg=str(tmp_path / "base.csv"),
        allow_config_mismatch=False,
    )


def test_pairing_a_bulk_lognorm_run_with_a_lognorm_baseline_raises(tmp_path):
    from cell_eval2.cli import _check_baseline_config

    args = _meta_pair(
        tmp_path, run_comparator="bulk_lognorm", base_comparator="lognorm",
    )
    with pytest.raises(SystemExit, match="comparator"):
        _check_baseline_config(args)


def test_the_comparator_gate_is_NOT_waivable(tmp_path):
    from cell_eval2.cli import _check_baseline_config

    args = _meta_pair(
        tmp_path, run_comparator="bulk_lognorm", base_comparator="lognorm",
    )
    args.allow_config_mismatch = True
    with pytest.raises(SystemExit, match="comparator"):
        _check_baseline_config(args)


@pytest.mark.parametrize("drop", ["run", "base", "both"])
def test_a_missing_comparator_key_is_fatal(tmp_path, drop):
    from cell_eval2.cli import _check_baseline_config

    args = _meta_pair(
        tmp_path, run_comparator="bulk_lognorm", base_comparator="bulk_lognorm", drop=drop,
    )
    with pytest.raises(SystemExit, match="comparator"):
        _check_baseline_config(args)


def test_pairing_agrees_when_the_comparators_match(tmp_path):
    from cell_eval2.cli import _check_baseline_config

    args = _meta_pair(
        tmp_path, run_comparator="bulk_lognorm", base_comparator="bulk_lognorm",
    )
    _check_baseline_config(args)


@pytest.mark.parametrize("version,pred,real,expected", [
    ("v2", "counts", "counts", "bulk_lognorm"),
    ("v1", "counts", "counts", "lognorm"),       # v1 never moves
    ("v2", "lognorm", "counts", "lognorm"),      # asymmetric -> the only shared space
    ("v2", "counts", "lognorm", "lognorm"),
    ("v2", "lognorm", "lognorm", "lognorm"),
])
def test_resolve_comparator_truth_table(version, pred, real, expected):
    assert norm.resolve_comparator(
        version=version, pred_input_type=pred, real_input_type=real) == expected


@pytest.mark.parametrize("bad_side", ["pred", "real"])
def test_resolve_comparator_rejects_an_unknown_input_type(bad_side):
    """BOTH sides must be validated. Checking only `pred` is passed by a resolver that
    validates one side and silently accepts anything on the other -- and the whole point
    of A2 is that this decision reads both sides."""
    kwargs = {"pred_input_type": "counts", "real_input_type": "counts"}
    kwargs[f"{bad_side}_input_type"] = "normalized"
    with pytest.raises(ValueError, match=f"{bad_side}_input_type"):
        norm.resolve_comparator(version="v2", **kwargs)


def test_to_normalization_refuses_bulk_lognorm():
    import anndata as ad
    import numpy as np
    import pandas as pd
    a = ad.AnnData(np.ones((4, 3), dtype=np.float32),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
                   var=pd.DataFrame(index=list("abc")))
    with pytest.raises(ValueError, match="group sum"):
        norm.to_normalization(a, "counts", "bulk_lognorm")


def test_build_and_read_agree_on_the_key_exactly():
    from cell_eval2 import run as R

    names = ["pds_cosine", "expr_mse_unbiased", "delta_mse"]
    expected = {
        "bulk_lognorm": {"bulk_lognorm"},
        "lognorm": {"lognorm"},
    }
    for comparator, expected_keys in expected.items():
        assert set(R._needed_normalizations(names, comparator=comparator)) == expected_keys


def test_a_de_metric_resolves_to_no_key():
    from cell_eval2 import run as R
    from cell_eval2.catalog import CATALOG

    assert R.effective_normalization(
        CATALOG["de_wilcoxon_sig_jaccard"], "bulk_lognorm") is None


def test_a_v2_counts_run_actually_resolves_to_bulk_lognorm():
    assert norm.resolve_comparator(
        version="v2", pred_input_type="counts", real_input_type="counts",
    ) == "bulk_lognorm"


@pytest.mark.parametrize("pred_type,real_type,expected", [
    ("counts", "lognorm", "lognorm"),
    ("lognorm", "counts", "lognorm"),
    ("counts", "counts", "bulk_lognorm"),
])
def test_compute_metrics_resolves_from_both_effective_sides(
        monkeypatch, synthetic_counts_pair, pred_type, real_type, expected):
    from cell_eval2 import run
    from cell_eval2.config import EvalConfig

    pred, real = (a.copy() for a in synthetic_counts_pair)
    if pred_type == "lognorm":
        pred = norm.to_normalization(pred, "counts", "lognorm", target_sum=1e4)
    if real_type == "lognorm":
        real = norm.to_normalization(real, "counts", "lognorm", target_sum=1e4)

    seen = []
    original = run.dispatch_anndata_metrics

    def capture(*args, **kwargs):
        seen.append(kwargs["comparator"])
        return original(*args, **kwargs)

    monkeypatch.setattr(run, "dispatch_anndata_metrics", capture)
    run.compute_metrics(
        pred, real,
        config=EvalConfig(
            metrics=["mae"], device="cpu", input_type=real_type,
            autodetect_input_type=True,
        ),
    )
    assert seen == [expected]


@pytest.mark.parametrize("input_type,expected", [
    ("counts", "bulk_lognorm"),
    ("lognorm", "lognorm"),
])
def test_result_digest_call_site_receives_the_resolved_comparator(
        monkeypatch, tmp_path, synthetic_counts_pair, input_type, expected):
    from cell_eval2 import run
    from cell_eval2.config import EvalConfig

    pred, real = (a.copy() for a in synthetic_counts_pair)
    if input_type == "lognorm":
        pred = norm.to_normalization(pred, "counts", "lognorm", target_sum=1e4)
        real = norm.to_normalization(real, "counts", "lognorm", target_sum=1e4)

    seen = []
    original = run._result_config_digest

    def capture(cfg, **kwargs):
        seen.append(kwargs["comparator"])
        return original(cfg, **kwargs)

    monkeypatch.setattr(run, "_result_config_digest", capture)
    run.compute_metrics(
        pred, real,
        config=EvalConfig(
            metrics=["mae"], device="cpu", input_type=input_type,
            cache_pred=str(tmp_path / "pred-cache"),
        ),
    )
    assert seen == [expected]


@pytest.fixture
def stub_gpudge(monkeypatch):
    """The comparator bundle tests do not consume DE; avoid an unrelated GPU dependency."""
    import polars as pl
    from cell_eval2 import partition_inmem

    monkeypatch.setattr(
        partition_inmem, "compute_de",
        lambda *args, **kwargs: pl.DataFrame({"target": []}, schema={"target": pl.Utf8}),
    )
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _name: "gpudge")


def test_a_reference_bundle_records_the_comparator_it_was_given(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    import json
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_reference

    _pred, real = synthetic_counts_pair
    # A DELIBERATELY non-default `bulk_target_sum`: this test is about what the bundle
    # RECORDS, so asserting the shipped default would pass even if `build_reference` wrote a
    # constant instead of the config's value -- and it would break on every future default
    # move, as #268's 1e6 -> 5e4 did.
    assert EvalConfig().bulk_target_sum != 250_000.0, "pick a value that is not the default"
    build_reference(
        real,
        config=EvalConfig(metrics=["pds_cosine"], bulk_target_sum=250_000.0),
        cache_dir=str(tmp_path), control_format="h5ad", comparator="bulk_lognorm",
    )
    meta = json.loads((tmp_path / "reference.json").read_text())
    assert meta["comparator"] == "bulk_lognorm"
    assert meta["bulk_target_sum"] == 250_000.0


def test_score_piece_rejects_a_bulk_target_that_disagrees_with_the_bundle(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_reference, score_piece

    pred, real = synthetic_counts_pair
    built = EvalConfig(metrics=["pds_cosine"], bulk_target_sum=1_000_000.0)
    build_reference(
        real, config=built, cache_dir=str(tmp_path), control_format="h5ad",
        comparator="bulk_lognorm",
    )
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    consumer = EvalConfig(metrics=["pds_cosine"], bulk_target_sum=28_000.0)
    with pytest.raises(ValueError, match=r"bulk_target_sum=28000.*bulk_target_sum=1000000"):
        score_piece(
            piece, str(tmp_path), config=consumer, comparator="bulk_lognorm",
        )


def test_pred_control_builder_rejects_a_bulk_target_that_disagrees_with_the_bundle(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_pred_control_reference, build_reference

    pred, real = synthetic_counts_pair
    built = EvalConfig(metrics=["pds_cosine"], bulk_target_sum=1_000_000.0)
    build_reference(
        real, config=built, cache_dir=str(tmp_path), control_format="h5ad",
        comparator="bulk_lognorm",
    )
    pred_path = tmp_path / "pred.h5ad"
    pred.write_h5ad(pred_path)
    consumer = EvalConfig(metrics=["pds_cosine"], bulk_target_sum=28_000.0)
    with pytest.raises(ValueError, match="bulk_target_sum"):
        build_pred_control_reference(
            str(pred_path), config=consumer, cache_dir=str(tmp_path),
            control="non-targeting", comparator="bulk_lognorm",
        )


def _partition_v1_config():
    from dataclasses import replace
    from cell_eval2.config import EvalConfig

    v1 = EvalConfig.v1()
    return replace(
        v1, metrics=["pds_cosine"], target_sum=1_000_000.0, validate_input=False,
        device="cpu", control_source="real",
        de=replace(v1.de, backend="gpudge", fdr_scope="per_pert"),
    )


def test_partition_one_sided_api_rejects_bulk_lognorm_for_v1(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    from cell_eval2.partition_inmem import build_reference

    _pred, real = synthetic_counts_pair
    with pytest.raises(ValueError, match="requires version='v2'"):
        build_reference(
            real, config=_partition_v1_config(), cache_dir=str(tmp_path),
            control_format="h5ad", comparator="bulk_lognorm",
        )


def test_partition_one_sided_api_rejects_bulk_lognorm_for_a_lognorm_local_side(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_reference

    _pred, real_counts = synthetic_counts_pair
    real = norm.to_normalization(real_counts, "counts", "lognorm", target_sum=1e4)
    with pytest.raises(ValueError, match="requires a counts local side"):
        build_reference(
            real, config=EvalConfig(metrics=["pds_cosine"], input_type="lognorm"),
            cache_dir=str(tmp_path), control_format="h5ad", comparator="bulk_lognorm",
        )


@pytest.mark.parametrize("case", ["v1", "lognorm"])
def test_precompute_cache_rejects_invalid_bulk_lognorm_one_sided_runs(
        tmp_path, synthetic_counts_pair, case):
    from cell_eval2 import run
    from cell_eval2.config import EvalConfig

    _pred, real = synthetic_counts_pair
    if case == "v1":
        cfg = _partition_v1_config()
    else:
        real = norm.to_normalization(real, "counts", "lognorm", target_sum=1e4)
        cfg = EvalConfig(metrics=["pds_cosine"], input_type="lognorm")
    cfg.cache_real = str(tmp_path / "cache")
    with pytest.raises(ValueError, match="bulk_lognorm"):
        run.precompute_cache(real, side="real", config=cfg, comparator="bulk_lognorm")


def test_score_piece_rejects_a_comparator_mismatch(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_reference, score_piece

    pred, real = synthetic_counts_pair
    build_reference(
        real, config=EvalConfig(metrics=["pds_cosine"]), cache_dir=str(tmp_path),
        control_format="h5ad", comparator="bulk_lognorm",
    )
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    with pytest.raises(ValueError, match="comparator") as excinfo:
        score_piece(
            piece, str(tmp_path),
            config=EvalConfig(metrics=["pds_cosine"], input_type="lognorm"),
            comparator="lognorm",
        )
    message = str(excinfo.value)
    assert "lognorm" in message and "bulk_lognorm" in message


@pytest.mark.parametrize("input_type,expected", [
    ("counts", "bulk_lognorm"),
    ("lognorm", "lognorm"),
])
def test_partition_writer_stamps_the_run_comparator(
        tmp_path, synthetic_counts_pair, stub_gpudge, input_type, expected):
    import json
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_reference, score_piece

    pred, real = (a.copy() for a in synthetic_counts_pair)
    if input_type == "lognorm":
        pred = norm.to_normalization(pred, "counts", "lognorm", target_sum=1e4)
        real = norm.to_normalization(real, "counts", "lognorm", target_sum=1e4)
    cfg = EvalConfig(metrics=["mae"], input_type=input_type, device="cpu")
    ref_dir = tmp_path / "ref"
    parts_dir = tmp_path / "parts"
    build_reference(
        real, config=cfg, cache_dir=str(ref_dir), control_format="h5ad",
        comparator=expected,
    )
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    score_piece(
        piece, str(ref_dir), config=cfg, comparator=expected,
        piece_id="p0", partial_out=str(parts_dir),
    )
    meta = json.loads((parts_dir / "p0.json").read_text())
    assert meta["comparator"] == expected


def test_precompute_cache_requires_comparator_for_an_expr_comparator_metric(
        tmp_path, monkeypatch, synthetic_counts_pair):
    from dataclasses import replace
    from cell_eval2 import run
    from cell_eval2.catalog import CATALOG
    from cell_eval2.config import EvalConfig

    monkeypatch.setitem(
        CATALOG, "expr_mae",
        replace(CATALOG["expr_mae"], normalization=norm.EXPR_COMPARATOR),
    )
    _pred, real = synthetic_counts_pair
    with pytest.raises(ValueError, match="requires comparator"):
        run.precompute_cache(
            real, side="real",
            config=EvalConfig(metrics=["mae"], cache_real=str(tmp_path / "cache")),
        )
