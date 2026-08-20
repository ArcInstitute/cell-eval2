import warnings

import polars as pl
import pytest

cell_eval = pytest.importorskip("cell_eval")
pl.enable_string_cache()


def test_compat_agg_anndata_profile_matches_cell_eval(synthetic_pair, tmp_path):
    pred, real = synthetic_pair

    # upstream: skip DE (anndata metrics only), anndata profile
    from cell_eval import MetricsEvaluator as CEEvaluator
    ce = CEEvaluator(adata_pred=pred, adata_real=real, control_pert="non-targeting",
                     pert_col="target", outdir=str(tmp_path / "ce"), skip_de=True)
    _, ce_agg = ce.compute(profile="anndata", write_csv=False)

    from cell_eval2.compat import MetricsEvaluator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ours = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert="non-targeting",
                                pert_col="target", outdir=str(tmp_path / "ours"))
        _, our_agg = ours.compute(profile="anndata", write_csv=False)

    # Assert EVERY shared anndata-profile metric (mae, mse, mae_delta, mse_delta,
    # pearson_delta, discrimination_score_l1/l2/cosine) matches upstream through the
    # full compat pipeline, not just mae. None of the 3 accepted v1 divergences
    # (issue #53) touch the anndata profile, so this is a true parity assertion.
    import math
    common = [c for c in our_agg.columns if c in ce_agg.columns and c != "statistic"]
    assert "mae" in common  # sanity: at least the original metric is still compared
    for m in common:
        ce_v = ce_agg.filter(pl.col("statistic") == "mean").select(m).item()
        our_v = our_agg.filter(pl.col("statistic") == "mean").select(m).item()
        if ce_v is None or (isinstance(ce_v, float) and math.isnan(ce_v)):
            assert our_v is None or (isinstance(our_v, float) and math.isnan(our_v)), m
        else:
            assert our_v == pytest.approx(ce_v, rel=1e-6, abs=1e-9), m
