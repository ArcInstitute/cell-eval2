# DE tables: what "unthresholded" means

`cell_eval2` computes one differential-expression (DE) table per side, `de_real` and
`de_pred`, and then scores metrics *on top of* those tables. Several downstream knobs
(`de.p_adj_threshold`, `de.min_abs_log2fc`, top-*k* selection, ...) decide which rows a
metric looks at, so "the DE table" is ambiguous unless the stage is named. This page fixes
the vocabulary used by the `score_cellstream(..., write_de=True)` output contract
([#387](https://github.com/ArcInstitute/cell_eval2/issues/387)).

## Definition

An **unthresholded DE table** is the complete output of `de_compute.compute_de` for the
configured backend, for **every** `(target, feature)` pair the backend returns, exactly as
handed to the scoring layer. Concretely, on the gpudge path it is the frame after
`_finalize_gpudge_de` has applied, in this order:

1. the v1 zero-mean LFC clip (`de.clip_value`, when not `None`), recomputed from gpudge's
   own `target_mean` / `ref_mean`;
2. the **pre-DE reference-only CPM gate** (`filter.filter_gene_min_cpm_cell`) and its
   per-target Benjamini-Hochberg adjustment;
3. `de.normalize_de_schema` (the `fdr -> p_adj` alias, the derived
   `abs_log2_fold_change` column, the required-column check);
4. global BH when `de.fdr_scope == "global"`.

and **before** any of the following has run:

| Applied later, NOT part of the table | Where it happens |
|---|---|
| `de.p_adj_threshold` significance gate | `de._rank_matrix` / `PreparedDE` |
| `de.min_abs_log2fc` post-FDR effect-size floor | `de.apply_lfc_floor` |
| `de.nan_lfc_policy` (`mask` forces `p_adj = 1` on NaN-LFC rows) | `de.apply_nan_policy`, from `score_piece` |
| `de.sort_by` ranking and top-*k* (`_top50/_top100/...`) selection | `de._rank_matrix` |
| `de.auc_pval_floor` / `auc_pval_floor_value` | the `pr_auc` / `roc_auc` metrics |
| target-gene exclusion and `target_gene_map` resolution | `_RefBundle.target_resolution`, PDS |
| any metric-specific gene selection | the metric functions |

So "unthresholded" does **not** mean "unfiltered": a configured CPM gate is part of DEG
computation (it decides which rows exist and re-adjusts `p_adj` per target) and therefore
remains in effect. It means no *evaluation* threshold has been applied.

## Columns

The canonical required columns are `target`, `feature`, `log2_fold_change` and `p_adj`
(`de.REQUIRED_COLS`). The gpudge backend additionally supplies `target_mean`, `ref_mean`,
`target_ncells`, `ref_ncells`, `p_value` and its Mann–Whitney statistic `Ueffect` (named `U` by
older gpudge releases); `normalize_de_schema` adds `abs_log2_fold_change`. The table written by
the reference bundle (`real_de.parquet`) and emitted by `write_de=True` carries all of them in
this order:

```
target, feature, target_mean, ref_mean, target_ncells, ref_ncells,
log2_fold_change, p_value, Ueffect, p_adj, abs_log2_fold_change
```

On a gpudge release that names the statistic `U`, that column is `U` instead of `Ueffect`:
column names, like dtypes and null semantics, are preserved as the backend produced them — nothing
is renamed, cast or filled. Another backend supplies its own columns, so two tables share a schema
only when the same backend (and release) produced them.

## Row order

Rows are emitted in the order the backend produced them, concatenated over the
perturbation batches the streaming reference builder iterates (`plan_pert_batches`
walks perturbations in sorted-label order, so targets are contiguous and sorted). The
table is never re-sorted for serialization; consumers that need a specific order must
sort themselves.

`de_pred.parquet`'s row order is **generation order**, not a lexicographic sort of the
per-piece file names: `score_cellstream` tracks each pred batch's piece (`ctx_0`, `ctx_1`,
…, `ctx_10`, …) by the batch number assigned as the loop runs, and `de_export.merge_de_pieces`
concatenates the pieces in that order — so the file's row order is batch order composed with
gpudge's own per-batch row order, never a glob-then-sort over piece file names (which would
place `ctx_10` before `ctx_2`).

## Outputs of `score_cellstream(write_de=True)`

`write_de=True` (requires `outdir` and at least one DE metric in the selection; off by default)
makes `score_cellstream` write two DE tables into `outdir`, beside its usual `per_pert.parquet` /
`per_context.parquet` / `overall.parquet`:

- **`de_real.parquet`** — the reference builder's full unthresholded real-DE table (the frame
  described above): the reference bundle's own `real_de.parquet` rewritten row group by row
  group through the pinned writer below — the same rows, values, column types and order,
  re-encoded under the contract instead of copied with whatever writer produced the bundle file.
- **`de_pred.parquet`** — the complete unthresholded predicted DE table, built from the per-piece
  pred DE tables via `de_export.merge_de_pieces`: a bounded-memory streaming concatenation (one
  piece's row groups resident at a time, never a whole-table `pl.concat`) — a production `de_pred`
  is on the order of 16.5M rows.

Both Parquet files go through the one pinned writer contract in `de_export.py`
(`de_export.write_de_parquet` / `de_export.merge_de_pieces`): `DE_PARQUET_COMPRESSION = "zstd"` at
`DE_PARQUET_COMPRESSION_LEVEL = 3`, Parquet statistics enabled, a `DE_PARQUET_ROW_GROUP_ROWS =
250_000`-row bound on row groups (a rewritten source's groups are split at that bound, never
merged, so they can be smaller), the exact generated row order (never re-sorted), and no added
index column — replacing reliance on whatever a library's default compression/row-grouping happens
to be.

**Publication is rollback-on-error, not atomic-as-a-group.** All five files are staged under
temporary names in `outdir` first — the two DE tables before the run's temporary reference
directory is removed, the three score frames once they are assembled — and only then published
together: each pre-existing destination is moved aside as `<dest>.bak.<tag>`, the five are
`os.replace`d into place in sequence, and the backups are deleted. A failure before publication
leaves `outdir` exactly as it was, with no temp debris; a failure during publication — a
`KeyboardInterrupt` included — unwinds by removing every remaining temp file, unlinking any
destination already published, and restoring the moved-aside backups. An `outdir` that is not a
directory, or a destination in it that is not a regular file (any symlink included), is
refused before any archive is opened.

The publish **commits** when its last rename completes, and nothing after that is rolled back —
not even an interrupt that lands before the call returns. Deleting the backups afterwards is best
effort: a backup that cannot be deleted (logged as a warning), or one an interrupt leaves behind,
stays beside the new outputs as a `<dest>.bak.<tag>` copy of the previous file — stale, and safe
to delete.

The guarantee is rollback on error, not atomic visibility of the five files as one generation: a
reader in another process can momentarily see some new files beside old ones, and a crash between
two `os.replace` calls leaves exactly that mix on disk. Recovery also relies on the filesystem
behaving while it runs: recognising the commit needs readable file metadata, and a rollback that
itself fails or is interrupted can leave files to inspect by hand — when it knows that, it raises
an error naming them.

**With the ground-truth DE cache on** (`score_cellstream(..., ground_truth_cache=<root>)`), the
publish carries a sixth file, `ground_truth_de_cache.json` — the cache lookup's provenance, staged
right after the two DE tables and published, and rolled back, with the other five — and
`de_real.parquet` is a byte copy of the validated cache entry, which the same pinned writer
produced ([`ground_truth_cache.md`](ground_truth_cache.md) §10). With the cache off, `outdir`
receives exactly the five files above.

## Related

- `docs/metrics.md` §4 for the metrics computed on these tables.
