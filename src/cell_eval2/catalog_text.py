"""Display text, family and headline bounds for evaluation_results.json (#412, design §7).

Both DE families (de_wilcoxon_*, de_deseq2_*) share one row, keyed by `text_key`.
The tables are GENERATED from the text Alex reviewed and approved (2026-10-22) in
internal:docs/internal/superpowers/specs/2026-10-21-evaluation-results-text.md by
internal:tools/catalog_text_sync.py: --emit regenerates them and --check verifies
them; never edit the tables by hand. A bound is declared only where the bounds
test holds it (design §7.1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricText:
    """Display text and declared headline bounds for #412's document.

    `min`/`max` are the headline bounds; `None` means undeclared, so the document
    omits the key rather than writing null (design §6.5). `family` is a display-only
    topic group (design §7).
    """

    display_name: str
    description: str
    family: str
    min: float | None = None
    max: float | None = None


FAMILIES = ("expr", "delta", "pds", "overlap", "significance", "lfc", "direction",
            "agreement", "covariance")


def text_key(name: str) -> str:
    """Share the method-independent DE text between both catalog families (#412, §7)."""
    for method in ("wilcoxon", "deseq2"):
        prefix = f"de_{method}_"
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def metric_text(name: str) -> MetricText:
    """Look up display text, keeping the caller's metric name in failures (#412, §7)."""
    try:
        return METRIC_TEXT[text_key(name)]
    except KeyError:
        raise KeyError(name) from None


def de_method(name: str) -> str | None:
    """Read the prefix registered by `catalog._register_de_family` (#412, design §7)."""
    for method in ("wilcoxon", "deseq2"):
        if name.startswith(f"de_{method}_"):
            return method
    return None


METRIC_TEXT: dict[str, MetricText] = {
    'expr_mae': MetricText(
        display_name='Expression MAE',
        description="Mean absolute difference between the predicted and real pseudobulk expression of the perturbation over all genes, in the comparator's expression space.",
        family='expr',
        min=0.0,
        max=None,
    ),
    'expr_mse': MetricText(
        display_name='Expression MSE',
        description="Mean squared difference between the predicted and real pseudobulk expression of the perturbation over all genes, in the comparator's expression space.",
        family='expr',
        min=0.0,
        max=None,
    ),
    'expr_mse_unbiased': MetricText(
        display_name='Expression MSE, bias-corrected',
        description="Squared error between the predicted and real pseudobulk expression over all genes except the perturbation's own, with both sides' sampling variance subtracted, so it can be negative.",
        family='expr',
        min=None,
        max=None,
    ),
    'expr_mse_unbiased_capped': MetricText(
        display_name='Expression MSE, capped correction',
        description="Bias-corrected squared error in which the prediction's variance correction is capped by the reference's; the numerator of expr_mse_unbiased_capped_norm.",
        family='expr',
        min=None,
        max=None,
    ),
    'expr_distance_unbiased': MetricText(
        display_name='Real effect size, bias-corrected',
        description="Squared distance between the real perturbation's and the real control's pseudobulk expression over all genes except the perturbation's own, with sampling variance subtracted; the denominator of expr_mse_unbiased_capped_norm.",
        family='expr',
        min=None,
        max=None,
    ),
    'expr_real_mass_ratio': MetricText(
        display_name='Real expression mass ratio',
        description="Total expression of the real perturbation's pseudobulk in linear units divided by the comparator's target sum; no value for lognorm input without a numeric target_sum.",
        family='expr',
        min=0.0,
        max=None,
    ),
    'expr_mse_unbiased_capped_norm': MetricText(
        display_name='Normalized expression error',
        description='Sum over the perturbations of expr_mse_unbiased_capped divided by the sum of expr_distance_unbiased, over the perturbations where both are finite; 0 is a perfect prediction and about 1 is predicting the control.',
        family='expr',
        min=None,
        max=None,
    ),
    'delta_mae': MetricText(
        display_name='Effect MAE',
        description='Mean absolute difference between the predicted and real effects of the perturbation (pseudobulk minus control) over all genes.',
        family='delta',
        min=0.0,
        max=None,
    ),
    'delta_mse': MetricText(
        display_name='Effect MSE',
        description='Mean squared difference between the predicted and real effects of the perturbation (pseudobulk minus control) over all genes.',
        family='delta',
        min=0.0,
        max=None,
    ),
    'delta_pearson': MetricText(
        display_name='Effect correlation',
        description='Pearson correlation between the predicted and real effects of the perturbation (pseudobulk minus control) over all genes; no value when the real effect does not vary across genes.',
        family='delta',
        min=-1.0,
        max=1.0,
    ),
    'pds_l1': MetricText(
        display_name='Discrimination (L1)',
        description="One minus the normalized rank of the real perturbation's effect among all real effects by L1 distance to the predicted effect, over all genes except the target genes the discrimination settings exclude (by default, every target gene of the panel); 1 means the prediction is closest to its own perturbation.",
        family='pds',
        min=0.0,
        max=1.0,
    ),
    'pds_l2': MetricText(
        display_name='Discrimination (L2)',
        description="One minus the normalized rank of the real perturbation's effect among all real effects by Euclidean distance to the predicted effect, over all genes except the target genes the discrimination settings exclude (by default, every target gene of the panel); 1 means the prediction is closest to its own perturbation.",
        family='pds',
        min=0.0,
        max=1.0,
    ),
    'pds_cosine': MetricText(
        display_name='Discrimination (cosine)',
        description="One minus the normalized rank of the real perturbation's effect among all real effects by cosine distance to the predicted effect, over all genes except the target genes the discrimination settings exclude (by default, every target gene of the panel); 1 means the prediction is closest to its own perturbation.",
        family='pds',
        min=0.0,
        max=1.0,
    ),
    'overlap': MetricText(
        display_name='DE gene overlap',
        description='Share of the real significant DE genes that are also among the same number of top predicted significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default); 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'overlap_top50': MetricText(
        display_name='DE gene overlap, top 50',
        description='Share of the top 50 real significant DE genes that are also among the top 50 predicted significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of real significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'overlap_top100': MetricText(
        display_name='DE gene overlap, top 100',
        description='Share of the top 100 real significant DE genes that are also among the top 100 predicted significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of real significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'overlap_top200': MetricText(
        display_name='DE gene overlap, top 200',
        description='Share of the top 200 real significant DE genes that are also among the top 200 predicted significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of real significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'overlap_top500': MetricText(
        display_name='DE gene overlap, top 500',
        description='Share of the top 500 real significant DE genes that are also among the top 500 predicted significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of real significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'precision': MetricText(
        display_name='DE gene precision',
        description='Share of the predicted significant DE genes that are also among the same number of top real significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default); 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'precision_top50': MetricText(
        display_name='DE gene precision, top 50',
        description='Share of the top 50 predicted significant DE genes that are also among the top 50 real significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of predicted significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'precision_top100': MetricText(
        display_name='DE gene precision, top 100',
        description='Share of the top 100 predicted significant DE genes that are also among the top 100 real significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of predicted significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'precision_top200': MetricText(
        display_name='DE gene precision, top 200',
        description='Share of the top 200 predicted significant DE genes that are also among the top 200 real significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of predicted significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'precision_top500': MetricText(
        display_name='DE gene precision, top 500',
        description='Share of the top 500 predicted significant DE genes that are also among the top 500 real significant genes, ranked by the de.sort_by setting (absolute log2 fold change by default), with the cutoff capped at the number of predicted significant genes; 0 when either side has no significant gene.',
        family='overlap',
        min=0.0,
        max=1.0,
    ),
    'nsig_counts_real': MetricText(
        display_name='Real significant genes',
        description='Number of genes significant in the real DE table for the perturbation.',
        family='significance',
        min=0.0,
        max=None,
    ),
    'nsig_counts_pred': MetricText(
        display_name='Predicted significant genes',
        description='Number of genes significant in the predicted DE table for the perturbation.',
        family='significance',
        min=0.0,
        max=None,
    ),
    'nsig_spearman': MetricText(
        display_name='Significant-gene count correlation',
        description='Spearman correlation between the real and predicted numbers of significant DE genes across the perturbations with at least one real significant gene, one value for the whole context; no value when fewer than 10 perturbations qualify or their real counts are all equal.',
        family='significance',
        min=-1.0,
        max=1.0,
    ),
    'sig_recall': MetricText(
        display_name='DE gene recall',
        description='Share of the real significant DE genes that are also significant in the prediction; no value when the reference has fewer than 10 significant genes for the perturbation.',
        family='significance',
        min=0.0,
        max=1.0,
    ),
    'sig_jaccard': MetricText(
        display_name='DE gene Jaccard',
        description="Jaccard index of the real and predicted significant DE gene sets, excluding the perturbation's own gene; 1 when both sets are empty.",
        family='significance',
        min=0.0,
        max=1.0,
    ),
    'pr_auc': MetricText(
        display_name='Significance PR AUC',
        description='Average precision of the predicted DE p-values for recovering the real significant genes; no value when the reference has fewer than 10 significant genes or no non-significant gene for the perturbation.',
        family='significance',
        min=0.0,
        max=1.0,
    ),
    'roc_auc': MetricText(
        display_name='Significance ROC AUC',
        description='Area under the ROC curve of the predicted DE p-values for recovering the real significant genes; no value when the reference has fewer than 10 significant genes or no non-significant gene for the perturbation.',
        family='significance',
        min=0.0,
        max=1.0,
    ),
    'lfc_spearman': MetricText(
        display_name='LFC correlation',
        description='Spearman correlation between the real and predicted log2 fold changes over the real significant DE genes; no value when the reference has fewer than 10 such genes with a log2 fold change, or fewer than 2 distinct values.',
        family='lfc',
        min=-1.0,
        max=1.0,
    ),
    'lfc_spearman_pos': MetricText(
        display_name='LFC correlation, up genes',
        description='Spearman correlation between the real and predicted log2 fold changes over the real significant DE genes with a positive real log2 fold change; no value when the reference has fewer than 10 such genes, or fewer than 2 distinct values.',
        family='lfc',
        min=-1.0,
        max=1.0,
    ),
    'lfc_spearman_neg': MetricText(
        display_name='LFC correlation, down genes',
        description='Spearman correlation between the real and predicted log2 fold changes over the real significant DE genes with a negative real log2 fold change; no value when the reference has fewer than 10 such genes, or fewer than 2 distinct values.',
        family='lfc',
        min=-1.0,
        max=1.0,
    ),
    'lfc_nmae': MetricText(
        display_name='LFC normalized MAE',
        description="Mean absolute error between the predicted and real log2 fold changes over the real significant DE genes with a finite real log2 fold change, except the perturbation's own, divided by the mean absolute real log2 fold change; no value when fewer than 10 such genes remain or all their real log2 fold changes are 0.",
        family='lfc',
        min=0.0,
        max=None,
    ),
    'direction_match': MetricText(
        display_name='Direction agreement',
        description='Share of the real significant DE genes whose predicted log2 fold change has the real sign; no value when the reference has fewer than 10 significant genes with a log2 fold change.',
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'model_direction_match': MetricText(
        display_name='Predicted-gene direction agreement',
        description='Share of the predicted significant DE genes whose real log2 fold change has the predicted sign; no value when the reference has fewer than 10 genes with a log2 fold change.',
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_precision': MetricText(
        display_name='Direction precision',
        description='Share of the predicted significant genes with a real direction for which the prediction commits to the real sign; no value when the reference has fewer than 10 genes with a nonzero log2 fold change.',
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_sensitivity': MetricText(
        display_name='Direction sensitivity',
        description="Share of the real significant genes reached by the longest prefix of the prediction's ranking of them whose sign purity is at least 1 − α/2, with α the significance threshold de.p_adj_threshold (0.975 at the default 0.05); no value when the reference has fewer than 10 significant genes.",
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_sensitivity_universe': MetricText(
        display_name='Direction sensitivity, all genes',
        description='As direction_sensitivity, with the prefix taken over all shared genes, so it can exceed 1; no value when the reference has fewer than 10 significant genes.',
        family='direction',
        min=0.0,
        max=None,
    ),
    'direction_fidelity': MetricText(
        display_name='Direction fidelity',
        description="Chance-corrected share of the predicted significant genes with a real direction whose predicted sign matches, (k/n − q)/d, with n the number of those genes, k how many of them match, q the majority-sign fraction of the real significant genes that have a direction and d = max(1 − q, 0.05), every count excluding the perturbation's own gene; when the reference has no significant gene it is 1 if n = 0 and 0 otherwise; when it has some, no value if none of them has a direction or n = 0.",
        family='direction',
        min=-20.0,
        max=1.0,
    ),
    'direction_fidelity_raw': MetricText(
        display_name='Direction fidelity, raw',
        description="Share of the predicted significant genes with a real direction whose predicted sign matches, excluding the perturbation's own gene; no value when the prediction has no such gene.",
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_coverage': MetricText(
        display_name='Direction coverage',
        description="Number of predicted significant genes with a real direction divided by the number of real significant genes, both excluding the perturbation's own gene; no value when the reference has no significant gene.",
        family='direction',
        min=0.0,
        max=None,
    ),
    'direction_yield': MetricText(
        display_name='Direction yield',
        description="Chance-corrected number of correctly signed predicted significant genes per real significant gene, (k − q·n)/(N·d), with N the number of real significant genes, n the number of predicted significant genes with a real direction, k how many of those have the real sign, q the majority-sign fraction of the real significant genes that have a direction and d = max(1 − q, 0.05), every count excluding the perturbation's own gene; no value when the reference has no significant gene or none of them has a direction.",
        family='direction',
        min=None,
        max=None,
    ),
    'direction_yield_raw': MetricText(
        display_name='Direction yield, raw',
        description="Number of correctly signed predicted significant genes divided by the number of real significant genes, both excluding the perturbation's own gene; no value when the reference has no significant gene.",
        family='direction',
        min=0.0,
        max=None,
    ),
    'direction_fidelity_yield': MetricText(
        display_name='Direction fidelity-yield',
        description="direction_fidelity scaled by the smaller of 1 and direction_coverage, every count excluding the perturbation's own gene; when the reference has no significant gene it is 1 if the prediction has no significant gene with a real direction and 0 if it has any; otherwise no value when no real significant gene has a direction, and 0 when some real significant gene has one but the prediction has no significant gene with a real direction.",
        family='direction',
        min=-20.0,
        max=1.0,
    ),
    'direction_fidelity_yield_raw': MetricText(
        display_name='Direction fidelity-yield, raw',
        description="Number of correctly signed predicted significant genes divided by the larger of the number of predicted significant genes with a real direction and the number of real significant genes, every count excluding the perturbation's own gene; no value when both counts are 0.",
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_reach': MetricText(
        display_name='Direction reach',
        description="Share of the real significant genes reached by the longest prefix of the prediction's ranking of them whose sign purity reaches the chance-corrected threshold q + (1 − α)(1 − q), with q the majority-sign fraction of the real significant genes that have a direction and α the significance threshold de.p_adj_threshold (q + 0.95(1 − q) at the default 0.05), every count and ranking excluding the perturbation's own gene; no value when the reference has no significant gene or none of them has a direction.",
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_reach_raw': MetricText(
        display_name='Direction reach, raw',
        description="Share of the real significant genes reached by the longest prefix of the prediction's ranking of them whose sign purity is at least 0.9, every count and ranking excluding the perturbation's own gene; no value when the reference has no significant gene.",
        family='direction',
        min=0.0,
        max=1.0,
    ),
    'direction_reach_unbounded': MetricText(
        display_name='Direction reach, all genes',
        description="As direction_reach, with the prefix taken over all shared genes, excluding the perturbation's own gene, so it can exceed 1; no value when the reference has no significant gene or none of them has a direction.",
        family='direction',
        min=0.0,
        max=None,
    ),
    'direction_reach_unbounded_raw': MetricText(
        display_name='Direction reach, all genes, raw',
        description="As direction_reach_raw, with the prefix taken over all shared genes, excluding the perturbation's own gene, so it can exceed 1; no value when the reference has no significant gene.",
        family='direction',
        min=0.0,
        max=None,
    ),
    'overlap_adjusted': MetricText(
        display_name='DE overlap, chance-corrected',
        description='Matthews correlation between the real significant genes and the same number of top predicted significant genes over all tested genes; no value when the reference has fewer than 10 significant genes or no other gene.',
        family='agreement',
        min=-1.0,
        max=1.0,
    ),
    'precision_adjusted': MetricText(
        display_name='DE precision, chance-corrected',
        description='Markedness of the predicted significant set against the real one over all tested genes; −1 when the prediction calls no gene or every gene.',
        family='agreement',
        min=-1.0,
        max=1.0,
    ),
    'sig_recall_adjusted': MetricText(
        display_name='DE recall, chance-corrected',
        description='Informedness of the predicted significant set against the real one over all tested genes; no value when the reference has fewer than 10 significant genes or no non-significant gene.',
        family='agreement',
        min=-1.0,
        max=1.0,
    ),
    'sig_mcc': MetricText(
        display_name='DE significance MCC',
        description='Matthews correlation between the real and predicted significant sets over all tested genes; no value when the reference has fewer than 10 significant genes or no non-significant gene.',
        family='agreement',
        min=-1.0,
        max=1.0,
    ),
    'cell_cov_corr': MetricText(
        display_name='Co-expression correlation',
        description="Pearson correlation between the upper triangles of the real and predicted gene–gene correlation matrices over the perturbation's top 100 responding genes; no value when either side has fewer than 50 cells, fewer than 20 responding genes vary on both sides, or either side's gene–gene correlations do not vary.",
        family='covariance',
        min=-1.0,
        max=1.0,
    ),
    'cell_cov_eig_ratio': MetricText(
        display_name='Co-expression eigenvalue ratio',
        description="Ratio of the leading eigenvalue of the predicted gene–gene correlation matrix to the real one's, over the same genes as cell_cov_corr; no value when either side has fewer than 50 cells or fewer than 20 responding genes vary on both sides.",
        family='covariance',
        min=0.0,
        max=None,
    ),
    'cell_cov_ccc': MetricText(
        display_name='Co-expression concordance',
        description="Lin's concordance between the real and predicted all-gene correlation matrices over the gene pairs significantly correlated on both sides; no value when either side has fewer than 50 cells or fewer than 3 such pairs remain.",
        family='covariance',
        min=-1.0,
        max=1.0,
    ),
}

STATISTICS: dict[str, dict] = {
    'count': {
        'display_name': 'Count',
        'description': "Number of the metric's values along its axis, null values included.",
        'combine': 'sum',
    },
    'null_count': {
        'display_name': 'Null count',
        'description': "Number of null values along the metric's axis.",
        'combine': 'sum',
    },
    'mean': {
        'display_name': 'Mean',
        'description': "Arithmetic mean of the metric's values along its axis, excluding null values.",
        'combine': 'mean',
    },
    'std': {
        'display_name': 'Standard deviation',
        'description': "Sample standard deviation of the metric's values along its axis, excluding null values; null when fewer than 2 values remain.",
        'combine': 'none',
    },
    'min': {
        'display_name': 'Minimum',
        'description': "Smallest of the metric's values along its axis, excluding null values.",
        'combine': 'none',
    },
    'max': {
        'display_name': 'Maximum',
        'description': "Largest of the metric's values along its axis, excluding null values.",
        'combine': 'none',
    },
    'median': {
        'display_name': 'Median',
        'description': "Median of the metric's values along its axis, excluding null values.",
        'combine': 'none',
    },
    'value': {
        'display_name': 'Value',
        'description': "The metric's single value for the scoring call, computed over the whole context.",
        'combine': 'mean',
    },
    'ratio_of_sums': {
        'display_name': 'Ratio of sums',
        'description': "Sum of the metric's numerator over the scored perturbations divided by the sum of its denominator, over the perturbations where both are finite.",
        'combine': 'mean',
    },
}

AXES: dict[str, dict] = {
    'perturbation': {
        'display_name': 'Perturbation',
        'description': 'One perturbation of the scored context, labeled as in the input data.',
    },
}

FACETS: dict[str, dict] = {
    'family': {
        'display_name': 'Family',
        'description': 'The topic group of the metric.',
    },
    'kind': {
        'display_name': 'Input',
        'description': 'What the metric reads: a differential-expression table (de), pseudobulk expression profiles (anndata), or single cells (cell).',
    },
    'de_method': {
        'display_name': 'DE method',
        'description': 'The differential-expression test that produced the DE tables.',
    },
}
