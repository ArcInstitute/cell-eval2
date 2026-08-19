# The vcc2026 Metric Suite

2026/08/18

> Source document. `build.sh` converts this file to LaTeX and builds `vcc2026-metrics.pdf`.
> Edit the Markdown; the TeX under `build/` is generated.

- [0. Overview](#0-overview)
- [1. Perturbation discrimination](#1-perturbation-discrimination)
- [2. Expression error](#2-expression-error)
- [3. DGE direction fidelity](#3-dge-direction-fidelity)
- [4. DGE direction reach](#4-dge-direction-reach)
- [5. DGE significant set agreement](#5-dge-significant-set-agreement)
- [6. DGE fold-change accuracy](#6-dge-fold-change-accuracy)

---

## 0. Overview

This document describes the six `vcc2026` metrics and how they are combined into the competition
score. Parameter values are those of the packaged `configs/vcc2026.yaml`, which is the configuration
the competition is scored under.

**Submissions.** A submission is a matrix of predicted single-cell expression whose gene axis and
perturbation labels match the reference exactly, including the non-targeting control label; the
number of predicted cells per perturbation is unconstrained. Values must be raw, untransformed
counts: non-negative, integral, and with no cell total above $`10^{6}`$. A matrix failing any of these
is rejected rather than scored. Counts are required because the expression metrics compare
group-summed profiles, which cannot be recovered from log-normalized data.

**Evaluation data.** The reference (real) measurements are pooled CRISPR-interference screens read out by
single-cell RNA sequencing, in several cellular contexts, each scored independently against its own
reference. Per context the panel holds 300 target constructs, one per gene, and a pooled
non-targeting control drawn from 46 constructs; every perturbation is downsampled to exactly 400 cells and
to a median depth of 20,000 UMI per cell, on a common axis of 18,533 genes. Cell count and
sequencing depth move every metric in the suite, so fixing them removes two sources of variation
that belong to the experiment rather than to the prediction.

**Controls.** The reference's control cells are the origin on both sides: the predicted effect is
measured against them, and they form the reference group of the differential-expression test for the
prediction as well as for the reference.

**Gene restrictions.** The perturbed gene's own row is excluded from all six metrics. Knocking a
gene down and then reporting that gene as changed is the premise of the experiment rather than a
prediction, and the row is worth 6–11 % of the expression metric's range on the reference contexts.
The four differential-expression metrics apply two further restrictions, a low-expression filter and
a significance threshold, both defined with the test in section 3.

**The metrics.** The competition scores the six metrics below: two computed from the expression
profiles directly, and four from a differential-expression table. The `de_wilcoxon_` prefix records
that the table comes from a Wilcoxon rank-sum test, and no other test is used. Five of the six are
means over the perturbations of a context; `expr_mse_unbiased_capped_norm` is a single ratio of two
panel-wide sums and has no per-perturbation value.

| metric | what it measures | section |
|---|---|---|
| `pds_cosine` | separability of predicted profiles | 1 |
| `expr_mse_unbiased_capped_norm` | size of the expression error | 2 |
| `de_wilcoxon_direction_fidelity_yield_raw` | correctness of predicted directions | 3 |
| `de_wilcoxon_direction_reach_raw` | depth over which directions stay correct | 4 |
| `de_wilcoxon_sig_jaccard` | agreement of responding-gene sets | 5 |
| `de_wilcoxon_lfc_nmae` | accuracy of predicted fold changes | 6 |

**Baseline and replicate.** Each metric is reported against two values measured on the same
reference data and distributed with it. The baseline $`b`$ is the context's mean perturbation
response: an equal-weight average, over the constructs of that context passing a cell-count and
knockdown-efficiency filter, of the per-construct mean count vector, assigned identically to every
perturbation and then scored as an ordinary submission. The replicate $`r`$ is the reference compared
with itself: cells are split into two disjoint halves per perturbation and per control, one half is
scored against the other, and five such splits (seeds derived from a pinned base seed of 0) are
averaged; each half uses its own control cells, so the two quantities whose agreement is measured
are not correlated through a shared reference.

**The competition score.** Writing $`u`$ for a metric's aggregate over the panel, each of the six is
placed on the scale

```math
s \;=\; \frac{u-b}{r-b}
```

so that 0 is the baseline and 1 is a repeat of the experiment, and a context's score is the
equal-weight mean of the six. Values above 1 occur and are reported rather than clipped, because the
replicate is estimated from half-depth data. Negative values can also occur,
for a submission worse than the baseline, and are likewise reported. Four of the six members are
unclamped; the expression error is clamped to $`[0, 1]`$ and the fold-change error is floored at
$`-6`$. Each member's scaled range is stated in its own section.

## 1. Perturbation discrimination

`pds_cosine` measures whether a predicted profile is identifiable: is the prediction for a given
perturbation closer to that perturbation's measured profile than to any other perturbation's?

**Definition.** A group's expression profile is formed by summing counts over its cells,
normalizing the result to a fixed total $`\mathrm{TS} = 5\times10^{4}`$, and log-transforming:

```math
b_{p,g} \;=\; \log\!\left(1 + \mathrm{TS}\,\frac{P_{p,g}}{\sum_{g'} P_{p,g'}}\right),
\qquad P_{p,g} \;=\; \sum_{c \in \mathcal{C}_p} y_{c,g},
```

where $`\mathcal{C}_p`$ is the set of cells labelled $`p`$ and $`y_{c,g}`$ the raw count of gene $`g`$ in
cell $`c`$. The effect of a perturbation is its profile minus the reference control profile, on both
sides: $`\delta_q = b_q - b_{\mathrm{ctrl}}`$ for the reference and
$`\hat{\delta}_p = \hat{b}_p - b_{\mathrm{ctrl}}`$ for the prediction. For each perturbation $`p`$ the
predicted effect is compared with every one of the $`n = 300`$ measured effects by cosine distance,

```math
d_{p,q} \;=\; 1 - \frac{\langle \hat{\delta}_p,\; \delta_q \rangle}
{\lVert \hat{\delta}_p \rVert \; \lVert \delta_q \rVert}.
```

Both vectors are evaluated with the coordinate of $`p`$'s own target gene removed, and $`d_{p,q} = 1`$
whenever either has zero norm. The score is one minus the normalized rank of the correct match,

```math
\mathrm{PDS}_p \;=\; 1 - \frac{k_p}{n-1}, \qquad
k_p \;=\; \#\{q : d_{p,q} < d_{p,p}\} \;+\; \tfrac{1}{2}\left(\#\{q : d_{p,q} = d_{p,p}\} - 1\right),
```

so that a tied block of candidates shares the average of the positions it spans. The metric's value
for a context is the mean of $`\mathrm{PDS}_p`$ over its 300 perturbations.

**Theoretical limits.** $`k_p`$ ranges over $`[0, n-1]`$, so $`\mathrm{PDS}_p \in [0, 1]`$: it is 1 when
the prediction is strictly closest to its own measured effect and 0 when it is farther from its own
than from all 299 others. A prediction carrying no information about which perturbation is which
ranks uniformly and scores 0.5 in expectation. Two degenerate predictions land there exactly rather
than in expectation. A prediction that emits the same profile for every perturbation produces one
shared distance row, so the ranks are a permutation of $`\{0,\dots,n-1\}`$ and the mean is exactly
0.5. A prediction whose effect is zero, the reference control pasted onto every perturbation, has
zero norm, so its entire row ties at distance 1 and every perturbation takes the mid-rank
$`(n-1)/2`$, i.e. 0.5 individually and not merely on average. The rank is taken over the 300
constructs actually scored, so the same submission evaluated against a different panel is a
different quantity.

**Practical values.** Over the three reference contexts:

| quantity | range |
|---|---|
| baseline $`b`$ | 0.510 – 0.530 |
| replicate $`r`$ | 0.929 – 0.985 |

The baseline sits 0.010–0.030 above the 0.5 no-information point, so a predictor handed the
context's mean response and nothing target-specific is very nearly unidentifiable, which is the
property this metric exists to detect. The replicate shows that half of a context's cells recover
the identity of nearly every perturbation. Its standard deviation over the five splits is 0.5–1.2 %
of the span $`r-b`$ that the score divides by, and because the replicate is already near the ceiling
there is little headroom above it: a submission reproducing the reference exactly scores 1.03–1.17.
Together with the direction reach of section 4 that is the least headroom of any unclamped member;
the other three exceed 1.5 on every context.

**Scaled range.** The scaled value is not clamped at either end. A submission below the baseline
therefore scores negative and is reported so: the control-pasting submission above, at 0.5, sits
0.010–0.030 below the baseline and scores $`-0.07`$ to $`-0.02`$, while the metric's own floor of 0 maps
to $`-1.17`$ to $`-1.26`$.

## 2. Expression error

`expr_mse_unbiased_capped_norm` measures how far the predicted expression profile is from the
measured one, as a fraction of how far the measured profile is from the control. A squared distance
between two profiles estimated from finite samples is inflated by sampling noise on both sides, so
the metric subtracts an estimate of that inflation before taking the ratio.

**The sampling correction.** For a group with $`n`$ cells, let $`S_p = \sum_g P_{p,g}`$ be its total
count and $`\ell_i`$ the total of cell $`i`$. Deleting one cell moves the normalizing denominator of
the profile $`b_p`$ and therefore every gene of it, including genes in which that cell had no reads,
so the correction is a delete-one jackknife rather than a closed-form variance:

```math
C_p \;=\; \frac{n-1}{n}\sum_g \sum_i \left(v_{ig} - \bar{v}_g\right)^2,
\qquad v_{ig} \;=\; \log\!\left(1 + \mathrm{TS}\,\frac{P_{p,g} - y_{ig}}{S_p - \ell_i}\right),
```

with $`\bar{v}_g`$ the mean of $`v_{ig}`$ over the group's cells, and $`C_p = 0`$ for a group of fewer
than two cells. $`C_p`$ estimates the part of a squared distance that would be present even if the
two sides agreed exactly in expectation. It is computed the same way for the prediction, written
$`\hat{C}_p`$, and for the reference control, written $`C_{\mathrm{ctrl}}`$.

**Definition.** Two per-perturbation quantities are formed, both reported. The numerator is the
corrected squared distance between the predicted and measured profiles,

```math
N_p \;=\; \frac{1}{G_p}\left( \lVert \hat{b}_p - b_p \rVert^2 \;-\; \min\!\left(\hat{C}_p,\; C_p\right) \;-\; C_p \right)
\qquad (\texttt{expr\_mse\_unbiased\_capped}),
```

and the denominator is the corrected squared distance from the measured perturbation to the
measured control,

```math
D_p \;=\; \frac{1}{G_p}\left( \lVert b_p - b_{\mathrm{ctrl}} \rVert^2 \;-\; C_p \;-\; C_{\mathrm{ctrl}} \right)
\qquad (\texttt{expr\_distance\_unbiased}).
```

Both squared distances run over every gene except the perturbation's own target gene, and $`G_p`$ is
the number of genes actually summed. Both legs drop the same gene: with only the numerator excluding
it, a submission that predicted the control exactly would no longer read 1. The sampling
corrections are left whole, which is the one approximation here: the cached moments store $`C_p`$ as a
scalar, so the excluded gene's share of it is not recoverable. That share is an ordinary $`1/G`$ for
the variance even though the gene dominates the distance, and leaving it in understates the
denominator sum by 0.007 to 0.013 %, against the 6 to 11 % of the metric's range the exclusion
itself removes. The $`\min`$ in the numerator
caps the correction credited to the prediction at the correction the reference itself earns, so a
submission cannot lower its error by claiming more sampling noise than the measurement has. The
scored metric is the ratio of the two sums over the panel,

```math
\mathrm{MSE} \;=\; \frac{\sum_p N_p}{\sum_p D_p},
```

which is why it has no per-perturbation value: a ratio of sums is not the mean of anything the
per-perturbation frame could carry. $`D_p`$ reads the measured data only, so it is identical for
every submission scored against a given context.

**Theoretical limits.** Lower is better. Both $`N_p`$ and $`D_p`$ are signed, because each subtracts an
estimate that can exceed the plug-in distance it corrects; a negative row is a statement that the
correction cannot resolve a shift at that depth, not that the perturbation is null. Two reference
points follow from the construction rather than from a fitted constant. A perfect prediction has
expected numerator 0, so perfection is 0. A prediction that emits the control unchanged has
expected numerator equal to the expected denominator, so no skill is 1. Because the denominator is
itself corrected, that holds whatever the depth of the reference. There is no upper bound. Both
statements are exact for the true sampling variances and approximate for the jackknife that
estimates them, which is measured 0.32 % high at $`\mathrm{TS} = 5\times10^{4}`$; a control-emitting
submission therefore reads slightly above 1 rather than exactly 1. The cap can also push a
submission above 1 on its own: a prediction whose own correction would exceed the reference's
forfeits the excess.

**Practical values.** Over the three reference contexts:

| quantity | range |
|---|---|
| baseline $`b`$ | 0.986 – 0.992 |
| replicate $`r`$ | 0.028 – 0.045 |

The baseline sits within 0.014 of the no-skill point of 1, so on this metric a predictor handed the
context's mean response is barely distinguishable from one that pastes the control. The replicate
shows that half of a context's cells leave 3–5 % of the control-pasting error, so the span $`r-b`$ is
about 0.95, the widest of the six on every context. That span is what the score divides by, which is
worth separating from the replicate itself: the standard deviation of $`r`$ over the five splits is
28–45 % of $`r`$, but only 0.8–2.2 % of the span. Excluding the target gene matters more here than
elsewhere: an adversary predicting the control everywhere and the truth at its own target gene was
taking 6–11 % of the scaled range before the exclusion was applied, because that gene is the
largest-moving one in 57–66 % of perturbations and the correction consumes roughly half of the raw
distance it is read against.

**Scaled range.** The scaled value is clamped to $`[0, 1]`$ at both ends, the only member clamped at
either end other than the fold-change error's floor, and the only one bounded above. The upper bound follows from the metric already carrying its own no-skill point: a raw value
of 1 means the prediction is no better than the control, so a scaled value past the baseline
carries nothing a ranking uses. The consequence at the other end is that a submission reproducing
the reference exactly scores exactly 1.000 here while the other five exceed 1.

## 3. DGE direction fidelity

Four of the six metrics are computed from a differential-expression table rather than from the
expression profiles directly. This section defines that table, then the first metric built on it:
`de_wilcoxon_direction_fidelity_yield_raw`, which measures whether the genes a submission calls as
responding move in the right direction, and penalizes a submission that calls far fewer genes than
the measurement found.

**The test.** For each perturbation, each gene is tested on its own for a difference between that
perturbation's cells and the control cells, by a two-sided Wilcoxon rank-sum test (equivalently the
Mann–Whitney U test). The test pools the two groups' values for that gene, replaces them by their
ranks, and asks whether the perturbation's cells sit systematically higher or lower in that order
than the control's; the p-value comes from the normal approximation to the rank sum, which is
accurate at the group sizes here. A rank test is the appropriate instrument for this data on two
counts. It assumes nothing about how a gene's counts are distributed, and single-cell counts are
zero-inflated and over-dispersed rather than any convenient family; and its statistic is unmoved by
how large an outlying cell is, only by where that cell falls in the order, where a mean-based test
would be dragged by it. The values ranked are counts normalized per cell to a total of $`10^{6}`$,
and that normalization matters even for a rank test: dividing each cell by its own library size
reorders cells that differ in depth.

Being two-sided, the test reports only *whether* a gene moved, never in which direction or by how
much. Both of those come from the log₂ fold change,

```math
\mathrm{lfc}_{p,g} \;=\; \log_2 \frac{m^{\mathrm{pert}}_{p,g} + \epsilon}{m^{\mathrm{ctrl}}_{g} + \epsilon},
\qquad \epsilon = 10^{-9},
```

with $`m`$ the arithmetic mean of the same normalized values over the cells of the group, and
$`\epsilon`$ keeping the ratio finite where a group's mean is zero. The same procedure with the same
parameters is applied to the submission and to the reference, and on both sides the comparison
group is the measured control.

**Low-expression filter.** A gene is tested for a given perturbation only if its mean expression
exceeds 5 counts per million in that perturbation's cells or in the control cells. A gene with
essentially no reads in either group carries no evidence about the perturbation, while still
consuming multiple-testing budget and so making the correction more conservative for every gene
that does carry evidence.

The filter reads each side's own matrix, so the two sides need not retain the same genes: a gene a
submission expresses below the cutoff leaves the submission's table while remaining in the
reference's, and that pair is then absent from the comparison. Measured on one reference context,
this accounts for 1.7 % of the reference-significant pairs for the baseline arm and 0.2 % for a
half-data arm.

**Multiple testing and significance.** Each perturbation is a family of thousands of simultaneous
tests, one per gene that survived the filter, so a raw p-value cannot be read as evidence on its
own. We use Benjamini–Hochberg correction, which controls the *false discovery rate*: the expected
fraction of false positives among the genes called, rather than the probability of making any false
call at all. A gene is *significant* for perturbation $`p`$ when
$`p^{\mathrm{adj}}_{p,g} < \alpha`$ with $`\alpha = 0.05`$, so that no more than 5 % of the genes
called for that perturbation are expected to be false.

Two scoping decisions are load-bearing. The correction is computed *after* the low-expression
filter, over the surviving pairs only, so discarding an uninformative gene does not merely exclude
it: it lowers the number of hypotheses $`m`$ for every other gene in that perturbation, and so makes
the survivors easier to call. And it is computed *within* each perturbation rather than once
across the panel, so $`m`$ is that perturbation's own gene count and the threshold means the same
thing for a strongly responding
perturbation as for a weak one.

**Definition.** Fix a perturbation $`p`$. A gene is *adjudicable* when the reference assigns it a
defined, non-zero log₂ fold change, so that the measurement can say which way it moved. Let
$`n_{\mathrm{real}}`$ be the number of genes the reference calls significant: the budget of genes
that responded. Let the submission's call set be the genes it calls significant that are also
adjudicable, with $`n_{\mathrm{pred}}`$ its size and $`k`$ the number of them whose predicted and
reference fold changes carry the same sign. A gene the submission calls but leaves without a
direction, through a null, NaN or exactly zero predicted fold change, stays in $`n_{\mathrm{pred}}`$
and scores as a miss rather than being dropped. Both counts exclude the perturbation's own target
gene. The metric is

```math
F_p \;=\; \frac{k}{\max\left(n_{\mathrm{pred}},\; n_{\mathrm{real}}\right)},
```

which factors exactly into a directional precision and a coverage term capped at 1:

```math
F_p \;=\; \frac{k}{n_{\mathrm{pred}}} \cdot \min\!\left(1,\; \frac{n_{\mathrm{pred}}}{n_{\mathrm{real}}}\right).
```

Calling more genes than the reference found buys nothing, since the coverage term saturates;
calling fewer costs proportionally. The metric's value for a context is the mean of $`F_p`$ over the
perturbations where it is defined.

**Theoretical limits.** $`k \le n_{\mathrm{pred}} \le \max(n_{\mathrm{pred}}, n_{\mathrm{real}})`$,
so $`F_p \in [0, 1]`$. It reaches 1 only when every call is correct *and* the submission calls at
least as many genes as the reference found; either half alone is not enough. A submission assigning
directions at random scores 0.5 in expectation once its coverage reaches 1, and less in proportion
to its coverage below that. Three cases have explicit conventions. A submission that calls nothing
has $`n_{\mathrm{pred}} = 0`$ and therefore $`k = 0`$, so it scores 0 whenever the reference found
anything: silence is not rewarded. If the reference found nothing either, the value is $`0/0`$ and
the perturbation is dropped from the mean rather than scored, which is why the cohort is 295–300 of
the 300 rather than a fixed 300. If the reference found nothing but the submission called genes
anyway, the coverage term is inactive and the value is the bare fraction correct, a coin flip
against noise.

**Practical values.** Over the three reference contexts:

| quantity | range |
|---|---|
| baseline $`b`$ | 0.506 – 0.523 |
| replicate $`r`$ | 0.718 – 0.825 |

The baseline sits at the chance level of 0.5: a predictor carrying no target-specific information
does no better than a coin flip here. The replicate is well short of 1, so at half depth the
measurement does not reproduce its own directional calls, which is the reason the scale is measured
rather than assumed. The span $`r-b`$ is 0.20–0.32, the narrowest of the six on every context, so a
given change in the raw value moves this member's contribution to the score more than it would
elsewhere; the standard deviation of $`r`$ over the five splits is correspondingly 2.4–4.4 % of the
span, among the largest of the six.

**Scaled range.** The scaled value is not clamped at either end. A submission below chance
therefore scores negative, and the metric's own floor of 0 maps to $`-1.58`$ to $`-2.68`$: the deepest
negative any of the four unclamped members can reach, which follows from this member having the
narrowest span.

## 4. DGE direction reach

`de_wilcoxon_direction_reach_raw` measures how far down a submission's own ranking its directional
calls stay reliable. Section 3 asks what fraction of the calls are correct; this asks how deep the
correct ones extend before the ordering degrades.

**Definition.** The pool is the budget of section 3: the genes the reference calls significant for
perturbation $`p`$, with $`p`$'s own target gene removed, of which $`n_{\mathrm{real}}`$ is the count.
Those genes are ordered by the submission's own confidence, its significant calls first, then by
ascending predicted adjusted p-value, then by ascending predicted p-value, then by descending
$`|\mathrm{lfc}|`$, with the gene name breaking what remains. Walking down that order and counting
only adjudicable genes, let $`P(k)`$ be the fraction of the first $`k`$ whose predicted sign matches the
reference's. The reach depth is the deepest prefix whose purity still clears a fixed threshold,

```math
k^{*} \;=\; \max\left\{\,k \;:\; P(k) \ge P_0 \right\}, \qquad P_0 \;=\; 0.9,
```

taking $`k^{*} = 0`$ when no prefix clears it, and the metric is that depth as a fraction of the
budget:

```math
R_p \;=\; \frac{k^{*}}{n_{\mathrm{real}}}.
```

$`P_0 = 0.9`$ is a chosen pass mark. It reads directly: a prefix clears it while at most one call in
ten is wrong, so the shallowest depth at which a single error can be tolerated is 10. A wrong call
at position $`j`$ leaves every prefix before it clean and therefore qualifying, so $`k^{*} \ge j-1`$;
only a wrong call at the very first position can drive $`k^{*}`$ to zero, and only then if the
ranking never recovers. "Deepest" is literal rather than "first": purity is not monotone in $`k`$, so
a prefix that dips below $`P_0`$ and recovers still counts at its deeper crossing -- a leading
miss followed by nine matches gives $`P(10) = 9/10 = 0.9`$, and $`k^{*}`$ reaches 10. The metric's
value for a context is the mean of $`R_p`$ over the perturbations where the budget is non-empty.

**Theoretical limits.** The ranking pool is the reference's own budget, so
$`k^{*} \le n_{\mathrm{real}}`$ and $`R_p \in [0, 1]`$. It is 1 when the submission orders the whole budget with
purity at or above 0.9 all the way down, and 0 when no prefix reaches 0.9. The metric is not a
restatement of section 3: fidelity is insensitive to the order of the calls,
so a submission that identifies the right genes but ranks them poorly scores well there and
badly here. On the other hand, a submission that does not calibrate FDR properly, but provides correct significance ranking,
will score poorly on fidelity and high on reach.

**Practical values.** Over the three reference contexts:

| quantity | range |
|---|---|
| baseline $`b`$ | 0.046 – 0.096 |
| replicate $`r`$ | 0.798 – 0.960 |

The baseline is near zero: a predictor with no target-specific information sustains the pass mark
over the first gene or two and no further. The replicate is 0.80–0.96, so a half-depth repeat keeps
its ordering pure over nearly the whole budget, and the span $`r-b`$ lies between 0.71 and 0.91,
second only to the expression error.

**Scaled range.** The scaled value is not clamped at either end, but there is little room on either
side of it. Below, the baseline is already near zero, so the metric's own floor of 0 maps to no worse than
$`-0.13`$. Above, the raw metric is bounded by 1 and the replicate is already 0.80–0.96, so a
submission reproducing the reference exactly scores between 1.04 and 1.29.

## 5. DGE significant set agreement

`de_wilcoxon_sig_jaccard` measures whether a submission identifies the same genes as responding
that the measurement did, charging a missed gene and an invented one alike.

**Definition.** Let $`R_p`$ be the set of genes the reference calls significant for perturbation $`p`$
and $`\hat{R}_p`$ the set the submission calls significant, both formed under the filter and
threshold of section 3, and both with $`p`$'s own target gene removed. The metric is the Jaccard
index of the two,

```math
J_p \;=\; \frac{\left|R_p \cap \hat{R}_p\right|}{\left|R_p \cup \hat{R}_p\right|},
```

counting each (perturbation, gene) pair once. Unlike a recall, whose denominator is $`|R_p|`$, and
unlike a precision, whose denominator is $`|\hat{R}_p|`$, the Jaccard index is symmetric: a gene the
submission fails to call and a gene it calls without cause cost the same. The metric's value for a
context is the mean of $`J_p`$ over its 300 perturbations.

**Theoretical limits.** $`J_p \in [0, 1]`$, reaching 1 when the two sets are identical and 0 when
they are disjoint. An empty union, meaning neither side called anything, is defined as 1: both
sides agree there was no response. That convention is why the metric returns a finite value for
every perturbation, so its cohort is the full 300 where section 3's is not, and it interacts with
target-gene exclusion: a perturbation whose only reference-significant gene was its own target now
has an empty reference set, so a submission calling nothing for it scores 1 where it would
otherwise have scored 0. There is no chance correction. Two independent sets of sizes $`a`$ and
$`\hat{a}`$ drawn from $`G`$ genes overlap by about $`(a\hat{a}/G)\,/\,(a + \hat{a} - a\hat{a}/G)`$,
which at the set sizes seen here is 0.006 to 0.012, so the metric credits that much free agreement.

**Practical values.** Over the three reference contexts:

| quantity | range |
|---|---|
| baseline $`b`$ | 0.021 – 0.037 |
| replicate $`r`$ | 0.296 – 0.405 |

The baseline is close to the chance level above: the mean-response predictor's significant set
overlaps the reference's by only a few per cent. The replicate is 0.30–0.41, far short of 1, so at
half depth the measurement's two halves agree on well under half of the genes in their union. Most
of the set disagreement a submission is charged with is therefore the reference's own instability
rather than the submission's error, which is exactly what the scale removes. The span $`r-b`$ is
0.27–0.37 and the standard deviation of $`r`$ over the five splits is 1.0–2.3 % of it.

**Scaled range.** The scaled value is not clamped at either end. Below, the baseline is near zero,
so the metric's own floor of 0 maps to only $`-0.08`$ to $`-0.11`$. Above, because the replicate is
low, the headroom is the largest of the six: a submission reproducing the reference exactly
scores 2.60–3.56 here, against 1.00 for the expression error and 1.03–1.17 for perturbation
discrimination. A submission whose set agreement exceeds a half-depth replicate's therefore gains
more on the average than an equally large excess would gain elsewhere.

## 6. DGE fold-change accuracy

`de_wilcoxon_lfc_nmae` measures whether a submission gets the size of the response right, where
sections 3 to 5 read only its direction and its membership.

**Definition.** The gate is the reference's own significant set for perturbation $`p`$: genes the
reference calls significant that also carry a finite reference log₂ fold change, with $`p`$'s target
gene removed. Write $`S_p`$ for that set. The metric is the mean absolute error of the predicted fold
changes over the gate, normalized by the mean absolute reference fold change over the same gate,

```math
\mathrm{NMAE}_p \;=\;
\frac{\sum_{g \in S_p} \left| \widehat{\mathrm{lfc}}_{p,g} - \mathrm{lfc}_{p,g} \right|}
{\sum_{g \in S_p} \left| \mathrm{lfc}_{p,g} \right|}.
```

A gene in the gate that the submission's table does not carry, or carries as null or non-finite, is
read as a predicted fold change of zero, that is, as a prediction of no change. The metric's value
for a context is the mean of $`\mathrm{NMAE}_p`$ over the perturbations it returns.

**Gate size.** A perturbation is scored only if its gate holds at least 10 genes; a ratio over a
handful of just-over-threshold genes is noise rather than measurement. A perturbation whose gate is
empty, or whose denominator is zero, is omitted for the same reason. All three conditions read the
reference alone, so the surviving set is identical for every submission: it ranges 209–263 of
the 300 on the three reference contexts. That property is load-bearing rather than incidental, and
it is why a non-finite prediction is filled with zero rather than masked. Masking would let a
submission emit non-finite values until a perturbation fell below the gate size and disappeared
from its own aggregate.

**Theoretical limits.** Lower is better and there is no upper bound. Perfection is 0. The no-skill
point is 1 and follows from the definition rather than from the data: a submission predicting zero
fold change everywhere makes the numerator equal the denominator term by term. That identity is
exact in fold-change space. Because the competition computes the submission's fold changes against
the measured control rather than against the submission's own control cells, a submission
broadcasting the true control is not compared against itself and lands near 1 rather than at it,
measured within 1 % above it on the three reference contexts.

**Practical values.** Over the three reference contexts:

| quantity | range |
|---|---|
| baseline $`b`$ | 1.0004 – 1.0017 |
| replicate $`r`$ | 0.378 – 0.465 |

The baseline sits at 1.000 to within 0.002, so the mean-response predictor's fold changes are, on
average over the gate, no closer to the truth than predicting no change at all would be. The
replicate is 0.38–0.47: a half-depth repeat reproduces its own fold changes with between a third
and a half of the error a null prediction makes. The span $`r-b`$ is 0.54–0.62, and the standard
deviation of $`r`$ over the five splits is 0.7–3.3 % of it, the largest of the six on one context,
which follows from this member's smaller and more variable cohort.

**The replicate estimator.** This is the one member whose replicate is not the ordinary split-half
comparison. A half of the data calls far fewer genes significant, so a split-half gate would be
21–35 % smaller than the gate the metric itself uses, and the two numbers would be means over
different cohorts. Its anchor instead takes the gate and the denominator from the full reference
table and only the two fold-change vectors from the halves, so its cohort matches the metric's
exactly. The reference bundle records which estimator produced each anchor, and a mismatched
pairing is refused rather than scored.

**Scaled range.** The scaled value is linear, as for every other member, and floored at $`-6`$. This
is the one floor in the suite that is not simply a metric's own range: the fold-change error is
unbounded above, so without it a single badly scaled submission could move the average without
limit. Linear means a submission further from the truth than the baseline is charged in proportion:
a raw value of 2.0 scores $`-1.61`$ to $`-1.86`$ on the reference contexts, and the floor binds only at
a raw value of 4.2 to 4.7. There is no explicit high clamp, but the member is bounded above all the
same: a normalized absolute error cannot fall below zero, so no submission can score past the
value earned at a raw error of 0 -- 1.61–1.87 on the reference contexts, which is what a
submission reproducing the reference exactly earns.
