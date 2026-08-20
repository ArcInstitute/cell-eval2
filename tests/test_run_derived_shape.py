import polars as pl

from cell_eval2.config import EvalConfig
from cell_eval2.run import aggregate_metrics, aggregate_metrics_wide, compute_metrics

DERIVED = "expr_mse_unbiased_capped_norm"
COMPONENTS = ("expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased")


def test_compute_metrics_emits_no_tidy_rows_for_the_derived_metric(
        synthetic_pair_with_effect):
    """The whole point of the no-per-perturbation-column decision, asserted where it is
    observable: after a real run the tidy frame must NOT carry the derived metric, and the
    aggregate frames must.

    Uses an effect-carrying reference because a null panel has a non-positive aggregate
    denominator, which `run._derived_value` correctly refuses.
    """
    pred, real = synthetic_pair_with_effect
    cfg = EvalConfig(metrics=[*COMPONENTS, DERIVED], pert_col="target",
                     input_type="lognorm", validate_input=False)
    tidy = compute_metrics(pred, real, config=cfg)

    observed = set(tidy["metric"].to_list())
    for component in COMPONENTS:
        assert component in observed, (
            f"{component} missing from the tidy frame -- the assertion below would then be "
            "vacuous, since ANY metric is absent from an empty frame"
        )
    assert DERIVED not in observed, (
        f"{DERIVED} emitted per-perturbation rows; it is derived and must exist only in the "
        "aggregate"
    )

    den_sum = tidy.filter(pl.col("metric") == "expr_distance_unbiased")["value"].sum()
    assert den_sum > 0, (
        f"the fixture's real side carries no aggregate effect (sum={den_sum}); the derived "
        "metric would then refuse it and this test would assert nothing"
    )

    agg = aggregate_metrics(tidy)
    row = agg.filter(pl.col("metric") == DERIVED)
    assert row.height == 1, f"expected exactly one aggregate row, got {row.height}"
    assert row["agg"][0] == "ratio_of_sums", f"derived aggregate kind was {row['agg'][0]!r}"

    wide = aggregate_metrics_wide(tidy)
    col = dict(zip(wide["statistic"].to_list(), wide[DERIVED].to_list()))
    assert col["mean"] == row["mean"][0], (
        f"wide mean {col['mean']} != long derived mean {row['mean'][0]}"
    )
    assert col["median"] != col["median"], "median must be NaN for a derived metric"


def test_the_cli_runs_a_profile_carrying_the_derived_metric(synthetic_pair_with_effect,
                                                            tmp_path):
    """Task 6 moved five CLI tests to `--profile minimal` because every null fixture makes
    `_derived_value` refuse the panel. This is the one that does not: same CLI, a profile
    that carries the derived metric, and a reference with a real effect."""
    from cell_eval2.cli import main

    pred, real = synthetic_pair_with_effect
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "anndata",
          "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm", "-o", str(out)])

    agg = pl.read_csv(out / "agg_results.csv")
    col = agg.columns
    assert DERIVED in col, f"the derived metric is missing from the aggregate: {col}"
    tidy = pl.read_csv(out / "results.csv")
    tidy_metrics = set(tidy["metric"].to_list())
    assert DERIVED not in tidy_metrics, (
        f"{DERIVED} emitted per-perturbation CLI rows: {sorted(tidy_metrics)}"
    )
