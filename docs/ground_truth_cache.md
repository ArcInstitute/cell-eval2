# The ground-truth DEG cache

## 0. Quick start (operators)

**Prerequisites.** The ground-truth archive must carry a strong `payload_sha256` in its
Cellstream manifest — written only when the archive was produced (or concatenated) with
`payload_checksum=True`. Without it, the cache fails closed before any DE runs (§4.1); there is
no structural fallback, only regenerating or re-concatenating the archive.

**First run (a miss).** With `score_cellstream(..., ground_truth_cache=<root>)` (the policy
defaults to `"readwrite"`, §3), the first call against a new (archive, semantics) pair computes
the real-DE table once, under the entry's lock, and publishes it. A `write_de=True` run's
`ground_truth_de_cache.json` reports `status="miss_computed"`, and the
`cellstream: ground-truth DE ...` log line shows the same status plus timing. Expect the full
real-DE compute cost on this call.

**Warm run (a hit).** A later call against the same archive with the same resolved settings reads
and validates the committed entry — every check in §6.4, including the artifact's full-file
SHA-256 and the null/duplicate scans — and reports `status="hit"` (or `"hit_after_wait"` if it
raced another writer for the same cache ID). The real-DE backend is never invoked on a hit.

**Outputs.** With the cache on, a `write_de=True` run publishes `ground_truth_de_cache.json`
beside the five files `docs/de_tables.md` describes, in the same rollback-on-error publish (§10);
with the cache off it publishes exactly those five. The cache entry itself lives under the
`ground_truth_cache` root, independently of `outdir`, at
`<root>/ground-truth-deg/v1/<xx>/<cache-id>/{de_real.parquet, manifest.json}` (§6.1).

**On failure:**

| Exception | Meaning | What to do |
|---|---|---|
| `GroundTruthFingerprintError` | The archive has no strong `payload_sha256`. | Regenerate or re-concatenate the archive with `payload_checksum=True` — there is no fallback. |
| `GroundTruthCacheCorruptError` | A committed entry failed validation. | Never auto-repaired or overwritten — remove the exact entry directory (below); the next `readwrite` call rebuilds it. |
| `GroundTruthCacheMissError` | A `readonly` caller hit a miss. | Either pre-warm the entry from a `readwrite` caller elsewhere, or switch this caller to `readwrite`. |
| `GroundTruthCacheFilesystemError` | Locking or atomic rename is unsupported/broken on this filesystem, or a writer's own self-validation failed. | Run `cell-eval2 cache doctor <root>` against the same mount. |

**Removing an entry.** Never edit an entry's files in place — the cache re-validates content on
every hit and treats any hand-edit as corruption. Remove the entry directory instead, and only
while no `score_cellstream` run is using this root: a running scorer links the entry into its
temporary bundle and reads it again when it publishes, so removing it mid-run fails that run.
Remove the sibling lock file too only when no writer can be running: a writer that holds or
awaits that lock would otherwise be joined by one that locks a fresh file, and the two would
compute concurrently — still correct, but no longer single-flight.

```
<root>/ground-truth-deg/v1/<cache_id[:2]>/<cache_id>/         # the entry directory
<root>/ground-truth-deg/v1/<cache_id[:2]>/.<cache_id>.lock    # its sibling lock file
```

**Checking things.** `cell-eval2 cache doctor <root>` self-checks a root's filesystem primitives
(locking, atomic rename, the Parquet round-trip) without computing any DE (§9). `cell-eval2 cache
inspect <entry-or-id> --root <root>` validates one entry's identity self-consistency and content
integrity and prints its manifest summary (§9).

The rest of this document is the full contract: identity, format, concurrency and operations.

## 1. What is cached and why

The ground-truth DEG cache is a content-addressed, immutable cache of the **unthresholded real
(ground-truth) differential-expression table** computed on the Cellstream `.shad` scoring path
(`score_cellstream`). See [`docs/de_tables.md`](de_tables.md) for the exact meaning of
"unthresholded" and the canonical 11 columns of that table.

The real-DE stage is expensive and, for a fixed ground-truth archive and a fixed real-DE
configuration, produces the *same* table regardless of which prediction, model or evaluation run
is being scored against it. Every model evaluated against the same ground truth pays that cost
again unless the table is cached and reused.

Only `de_real.parquet` is cached in this version. Predicted DE, real pseudobulk, control matrices,
metric results and model outputs are **not** cached — they are per-run temporary artifacts, and in
particular a prediction-side control block (`control_source="pred"`) must never reach the shared
cache (see [§7](#7-concurrency-and-atomicity-11)).

**Where it is wired.** With the cache on, `cellstream.score_cellstream` obtains the real-DE
table through `GroundTruthDEGCache.get_or_compute` inside the streaming reference builder
(`partition_inmem._build_reference_streaming_core`, via its `real_de_provider` hook). With the
cache off the builder gets no provider and runs exactly as it did before this cache existed. The
identity
primitives and the storage class are independently usable from `cell_eval2.ground_truth_cache`.
That module is a package internally (`identity` / `integrity` / `store` submodules) — the public
import paths (everything importable as `cell_eval2.ground_truth_cache.<name>`, and the names
re-exported from `cell_eval2` itself) are unchanged by that split.

## 2. Terminology

| Term | Meaning |
|---|---|
| Ground truth / real data | The observed expression data used as the reference side of evaluation. |
| Real DEG table | The output of the configured real-data DE backend for every `(target, feature)` pair it returns. |
| Unthresholded | No `p_adj` cutoff, effect-size cutoff, metric-specific gene selection, or top-*k* selection applied. A configured pre-DE CPM filter (`filter.filter_gene_min_cpm_cell`) is part of DEG computation and remains in effect. |
| Strong fingerprint | A digest covering the archive's expression payload bytes plus its ordered feature and perturbation-label axes. |
| DEG semantics | The canonical configuration + implementation identity for every factor capable of changing `de_real.parquet`. |
| Cache ID | SHA-256 of the strong ground-truth fingerprint and the DEG semantics together. |
| Cache hit | A committed entry exists for the cache ID and passes manifest, schema and content-integrity validation. |
| Cache miss | No committed entry exists for the requested cache ID. |

## 3. Turning it on (`ground_truth_cache`, `ground_truth_cache_policy`)

Two keyword arguments of `score_cellstream` control this cache. They are deliberately **not**
`EvalConfig` fields:

- **`ground_truth_cache: str | os.PathLike | None = None`** — the cache root. `None` means the
  cache is **off**: the real-DE table is computed inline exactly as before this cache existed,
  nothing is fingerprinted, no cache key or identity document is built, no cache I/O happens at
  all, and no sidecar is written. Otherwise the root must be an absolute path after
  `os.path.expanduser`; `cell_eval2` manages only `<root>/ground-truth-deg/v1/` beneath it (§12).
  The root may be `outdir` itself or any other directory in it, but not one of the files the run
  publishes into `outdir` (`de_real.parquet`, `de_pred.parquet`, the three score frames,
  `ground_truth_de_cache.json`), nor a path inside one (`ValueError`).
- **`ground_truth_cache_policy: Literal["readwrite", "readonly"] = "readwrite"`** — the operational
  policy once a root is set:
  - **`readwrite`** (default): read and validate an existing entry; on a miss, acquire the
    per-entry lock, compute the table exactly once, and publish it.
  - **`readonly`**: read and validate an existing entry only; a miss raises
    `GroundTruthCacheMissError` naming the cache ID and entry path. Never creates the cache root,
    entries, or lock files. `"readonly"` with no root is refused (`ValueError`) rather than
    silently ignored.

Both arguments are checked before any archive is opened (§11).

There is no `refresh` policy. Entries are immutable — a change to the resolved real-DE semantics
produces a different cache ID (a new entry), never an in-place update. An operator who needs to
rebuild a specific entry removes that exact entry directory out of band; nothing in the cache
overwrites a committed entry automatically.

**Why not `EvalConfig`.** Because they are not configuration fields, neither can move
`config_digest`, a result-cache key, a partial sidecar's `config_hash`, or the competition preset.
And `cache_real` keeps its single meaning — the root of the general artifact cache (`CacheStore`)
that `compute_metrics` and the shard-streaming path use; `score_cellstream` ignores it.

## 4. Ground-truth identity

Identity is a strong **fingerprint** of the source archive combined with the resolved **real-DE
semantics** that produced the table. There is deliberately no structural fallback: the ground-truth
cache never reproduces the old behaviour of `cell_source.cell_fingerprint`, which can fall back to
a structural (shape-only) fingerprint that collides when matrix values change without changing
shape, genes or perturbations. A false miss is acceptable; a false hit is not.

### 4.1 Strong fingerprint prerequisite (spec §8.2)

The archive's Cellstream manifest must carry `payload_sha256` — a strong content checksum written
only when the archive was produced (or concatenated) with `payload_checksum=True`. If a cache root
is configured and the checksum is absent,
`ground_truth_cache.fingerprint_ground_truth_cell_archive` raises `GroundTruthFingerprintError`
**before any DE runs**, naming the archive and instructing the caller to regenerate or concatenate
it with `payload_checksum=True` (cellstream / shardad writers). This is never a structural
fallback — an archive without a strong checksum is simply not cache-eligible.

### 4.2 Canonical sequence digest (§8.3)

Ordered string sequences (`var_names`, `obs[pert_col]`) are hashed with
`ground_truth_cache.sequence_sha256`:

```
sha256(domain || 0x00 || u64be(count) || (u64be(len(utf8)) || utf8)*)
```

`domain` defaults to `DOMAIN_SEQUENCE = b"cell-eval2-string-sequence-v1"`. Every element must
already be `str` — no `None`/bytes/int coercion; a non-string element raises
`GroundTruthContractError`. The length prefixes make the digest immune to delimiter collisions in
feature or perturbation names.

### 4.3 Fingerprint document (§8.4)

`GroundTruthFingerprint.from_axes` (called by `fingerprint_ground_truth_cell_archive`) builds the
version-1 document with exactly the keys in `FINGERPRINT_DOCUMENT_KEYS_V1`, in this order:

```
fingerprint_algorithm, archive_format, payload_sha256, n_obs, n_vars,
feature_axis_sha256, perturbation_axis_sha256, pert_col
```

`fingerprint_algorithm` is the constant `FINGERPRINT_ALGORITHM =
"cell-eval2-ground-truth-cellstream-v1"`, `archive_format` is `ARCHIVE_FORMAT =
"cellstream-cell-archive"`. `n_obs`/`n_vars` are derived from the axes themselves (never passed
independently), so they cannot disagree with the digests. `feature_axis_sha256` /
`perturbation_axis_sha256` are `sequence_sha256(var_names)` / `sequence_sha256(pert_labels)`, where
`pert_labels` come from `store.obs[pert_col]` in the archive's **storage row order** (not sorted).
The document's own digest is `document_sha256(document)` (`GroundTruthFingerprint.sha256`) — SHA-256
of `canonical_json_bytes(document)` (§8.4/§9.3: sorted keys, compact separators, Unicode preserved,
non-finite numbers forbidden, floats via Python's shortest round-trip repr).
`validate_fingerprint_document` enforces the exact key set and per-field types (64-lowercase-hex
digests, non-negative `int` counts, non-empty `pert_col`).

`fingerprint_ground_truth_cell_archive(store_or_path, *, pert_col)` accepts either an already-open
cell store (duck-typed on `.manifest`/`.var`/`.obs`) or a path, opening and closing it itself in the
latter case (including on error). Deliberately **excluded** from the fingerprint: archive path,
filename, device, inode, mtime; the producing model's checkpoint identity; the evaluation run ID
and display context label; the prediction archive's identity.

## 5. Real-DE semantics and cache ID

### 5.1 Semantic document (§9.1)

`ground_truth_cache.build_real_de_semantics` is the **only** place semantic keys are produced. It
takes the run's `EvalConfig` plus the already-resolved values the caller (`score_cellstream`) is
responsible for supplying (`control`, `input_type`, `target_sum`, `device`, `de_backend`,
`precision`, `native_gpu_normalize`) — not the raw, possibly-unresolved `EvalConfig` fields, since
v1 autodetection, a data-derived median target sum, and the archive's own `control_value` column
can all change what actually ran. It asserts `cfg.de.replicate_col is None` (deseq2-only; the
deseq2 backend never runs on the cellstream real-DE path) before building anything, raising
`GroundTruthContractError` otherwise. The result is a `RealDESemantics` (validated document +
`document_sha256`).

The document has exactly the 22 keys in `SEMANTIC_DOCUMENT_KEYS_V1`, in this order:

```
cache_format_version, real_de_semantics_version, cell_eval2_version, gpudge_version,
cellstream_decoder_version, de_backend, config_version, input_type, target_sum, pert_col,
control, mean_calc, epsilon, clip_value, fdr_scope, filter_gene_min_cpm_cell, validate_input,
max_counts_per_cell, allow_fractional_real, native_gpu_normalize, device, precision
```

`validate_semantic_document` enforces this exact key set (missing/extra keys both fail), that
`cache_format_version == CACHE_FORMAT_VERSION` and `real_de_semantics_version ==
REAL_DE_SEMANTICS_VERSION`, non-empty string types for the version/backend/label fields, bool types
for the three flag fields, and finite-number-or-null for the nullable numeric fields
(`target_sum`, `clip_value`, `filter_gene_min_cpm_cell`).

`producer_versions()` fills `cell_eval2_version`, `gpudge_version` and `cellstream_decoder_version`
from `importlib.metadata.version(...)` (falling back to the `shardad` distribution for
`cellstream_decoder_version` on pre-rename images); a missing required distribution is a hard
`GroundTruthContractError` — the cache cannot key an entry without knowing what produced it.

**Storage churn is a deliberate consequence, not an oversight.** `cell_eval2_version`,
`gpudge_version` and `cellstream_decoder_version` are all semantic keys, so every `cell_eval2`,
`gpudge` or `cellstream` release — even one that changes nothing about the DE math — starts a
fresh generation of cache IDs: every entry keyed under the old versions becomes unreachable
(never invalidated or deleted in place). The separate, manually-bumped
`real_de_semantics_version` exists for the narrower case of a change to the real-DE
row/value/dtype/ordering contract *within* one release, where bumping a whole package version
would be overkill. See [§12](#12-operations) for the expected storage impact and how to reclaim
an old generation.

**Versions identify releases, not code.** `importlib.metadata` reports the version an install was
made with, and an editable (development) install keeps that version while its code changes. Its
cache entries can therefore be served to code that would compute a different table. A
development checkout must use a fresh cache root whenever its DE-relevant code changes (or run
with the cache off); a shared root is for released installs.

`resolve_de_precision(x_dtype, *, input_type, native_gpu_normalize, target_sum)` derives
`precision`: it mirrors `de_compute._to_linear` and gpudge's native-normalize branch of
`de_compute.compute_de`. On the `score_cellstream` path (float32 cell-layout archives,
`native_gpu_normalize=True`) this resolves to `"float32"` — the dtype gpudge actually ranks on the
GPU; its float64 accumulators are an implementation detail already pinned by `gpudge_version`.

### 5.2 Included and excluded settings (§9.2)

The semantic-key builder is an **explicit allowlist**
(`tests/test_ground_truth_cache_classification.py` checks
`tests/fixtures/ground_truth_cache/eval_config_classification.yaml` covers every `EvalConfig`
field), not `cfg.to_dict()`. 14 resolved `EvalConfig`-derived settings plus 8 inputs that are not
`EvalConfig` fields at all make up the 22 semantic keys:

**Included (14, from `EvalConfig`, resolved values):**

| `EvalConfig` field | Semantic key | Why |
|---|---|---|
| `pert_col` | `pert_col` | Defines the target populations. |
| `control` (resolved) | `control` | Defines the reference population. |
| `version` | `config_version` | Drives real-side input-type autodetection (v1); conservative compatibility guard. |
| `input_type` (resolved) | `input_type` | The effective type handed to `compute_de`. |
| `target_sum` (resolved) | `target_sum` | Passed to `compute_de` after `norm.resolve_target_sum`. |
| `max_counts_per_cell` | `max_counts_per_cell` | Scale-limit validation gate; changes accepted input. |
| `validate_input` | `validate_input` | Gates `_validate_stream_slice` entirely. |
| `device` (resolved) | `device` | The resolved numerical execution class (always `"cuda"` on this path). |
| `filter.filter_gene_min_cpm_cell` | `filter_gene_min_cpm_cell` | The pre-DE reference-only CPM gate; changes which rows exist and their per-target BH `p_adj`. |
| `de.backend` (resolved) | `de_backend` | `_require_partition_config` requires this to resolve to `gpudge` on this path. |
| `de.mean_calc` | `mean_calc` | Arithmetic vs. geometric group means change `log2_fold_change`. |
| `de.epsilon` | `epsilon` | LFC pseudocount. |
| `de.clip_value` | `clip_value` | The v1 zero-mean LFC clip, recomputed in `_finalize_gpudge_de`. |
| `de.fdr_scope` | `fdr_scope` | Changes `p_adj`; forced to `per_pert` on this path but kept for cross-path safety. |

**Not `EvalConfig` fields (8 `non_config_inputs`):**

| Semantic key | Source |
|---|---|
| `cache_format_version` | Constant `1`. |
| `real_de_semantics_version` | Manual guard — bumped on any change to real-DE rows/values/dtypes/ordering. |
| `cell_eval2_version` | `importlib.metadata.version("cell-eval2")`. |
| `gpudge_version` | `importlib.metadata.version("gpudge")`. |
| `cellstream_decoder_version` | `importlib.metadata.version("cellstream")` (or `"shardad"` on pre-rename images). |
| `allow_fractional_real` | Constant `false` — `run._val_allow_fractional` always returns `False` for `side="real"`. |
| `native_gpu_normalize` | Constant `true` on the `score_cellstream` path. |
| `precision` | The dtype the DE backend consumes (`"float32"` for gpudge); approved 2026-09-16. |

**Notably excluded**, with a reason recorded in the classification YAML: `metrics` (never reaches
`compute_de`), `control_source` (only decides the *pred*-side control; the shared cache holds only
`de_real.parquet`), `num_threads`/`pert_chunk`/`gather_threads` (GPU/PDS/decode parallelism —
results are identical regardless of thread count), `outdir`/`cache_real`/`cache_pred`/
`cache_strict` (operational, already excluded from `config_hash`), every
`discrimination.*` field (PDS only), `de.p_adj_threshold`/`de.min_abs_log2fc`/`de.sort_by`/
`de.nan_lfc_policy`/`de.auc_pval_floor*` (applied downstream of the unthresholded table),
`target_gene_map` (resolved after the table is loaded), `bulk_target_sum` (pseudobulk comparator
only). Two fields are **folded** into an included key rather than excluded outright:
`allow_discrete` and `autodetect_input_type` only affect which `input_type` gets resolved, so the
resolved `input_type` value already captures their effect.

**Prohibited:** `de.replicate_col` must be `None` — it is deseq2-only (pseudobulk replicates), and
the deseq2 backend cannot run on the cellstream real-DE path. `build_real_de_semantics` asserts
this and raises `GroundTruthContractError` rather than silently ignoring a non-null value.

### 5.3 Cache ID (§9.3)

`ground_truth_cache.cache_key_document(fingerprint_doc, semantics_doc)` validates both documents,
then builds:

```json
{
  "cache_key_algorithm": "cell-eval2-real-deg-cache-key-v1",
  "ground_truth_fingerprint": { "...": "the complete §8.4 document" },
  "de_semantics": { "...": "the complete §9.1 document" }
}
```

`cache_key_algorithm` is the constant `CACHE_KEY_ALGORITHM =
"cell-eval2-real-deg-cache-key-v1"`. `ground_truth_cache.cache_id(fingerprint_doc, semantics_doc)`
is `document_sha256` of that key document — a lowercase-hex SHA-256 string.
`GroundTruthDEGCacheRequest` is **identity-only**: a frozen dataclass of just `fingerprint` and
`semantics`. It exposes the cache ID as its `.cache_id` property, and `.entry_relpath` as
`entry_relpath(self.cache_id)`. `root` and `policy` are not request fields — they belong to the
`GroundTruthDEGCache` instance (§8), which is the single source of truth for where an entry lives
and whether this call may create one; a `readonly` cache can never create anything, whatever a
caller's own request says.

## 6. On-disk format

### 6.1 Layout (§10.1)

```text
<root>/
  ground-truth-deg/
    v1/
      ab/
        abcdef...<64-char cache id>/
          de_real.parquet
          manifest.json
        .abcdef...<64-char cache id>.lock
```

`entry_relpath(cache_id)` returns `ground-truth-deg/v{CACHE_FORMAT_VERSION}/<cache_id[:2]>/<cache_id>`
and rejects anything that is not exactly 64 lowercase hex characters (path-traversal safety). The
lock file (`GroundTruthDEGCache.lock_path`) is a **sibling** of the entry directory, named
`.<cache_id>.lock`, so it can exist before the entry is committed; a writer stages under
`.tmp-<cache_id>-<pid>-<uuid>` in the same parent directory and publishes with one `os.rename`.
`ENTRY_TABLE_NAME = "de_real.parquet"`, `ENTRY_MANIFEST_NAME = "manifest.json"`.

Root handling: with `policy="readwrite"`, `GroundTruthDEGCache.__init__` creates
`<root>/ground-truth-deg/v{CACHE_FORMAT_VERSION}/` (`os.makedirs(..., exist_ok=True)`) but not the
per-prefix subdirectories, which are created lazily on first write. With `policy="readonly"`, the
constructor **never creates anything** — it raises `GroundTruthCacheFilesystemError` immediately if
`root` does not already exist as a directory.

### 6.2 Parquet contract (§10.2)

`de_real.parquet` is written through `de_export.write_de_parquet`, the same pinned writer through
which `score_cellstream(write_de=True)` publishes its DE tables (see `docs/de_tables.md`'s
"Outputs of `score_cellstream(write_de=True)`" section):

- `DE_PARQUET_COMPRESSION = "zstd"`, `DE_PARQUET_COMPRESSION_LEVEL = 3`.
- Parquet statistics enabled (`write_statistics=True`).
- `DE_PARQUET_ROW_GROUP_ROWS = 250_000` — a target row-group size, with the final group naturally
  smaller.
- The exact generated row order (never re-sorted for serialization).
- No index column added solely for serialization.

The canonical columns, in order, are the same 11 as `docs/de_tables.md`:

```
target, feature, target_mean, ref_mean, target_ncells, ref_ncells,
log2_fold_change, p_value, U, p_adj, abs_log2_fold_change
```

Two corrections from the original spec text, confirmed against the actual gpudge output: the
backend's statistic column is `U` (named
`Ueffect` on some gpudge versions, which is exactly why `gpudge_version` is a semantic key), not
`test_statistic`; and `abs_log2_fold_change` — added by `de.normalize_de_schema` inside
`compute_de` — **is** canonical unthresholded output, not a downstream-only derived column, and
belongs in the cached table.

### 6.3 Manifest (§10.3)

`manifest.json` is operational metadata, not a second scientific payload, written via
`_atomic_write_json_line` (canonical JSON + trailing newline, same-directory tmp file +
`os.replace`). Fields:

```json
{
  "cache_format": "cell-eval2-ground-truth-deg",
  "cache_format_version": 1,
  "cache_id": "<64-char digest>",
  "created_at_utc": "2026-09-16T00:00:00Z",
  "ground_truth_fingerprint": { "...": "the §8.4 document" },
  "ground_truth_fingerprint_sha256": "<digest>",
  "de_semantics": { "...": "the §9.1 document" },
  "de_semantics_sha256": "<digest>",
  "artifact": {
    "path": "de_real.parquet",
    "sha256": "<file digest>",
    "size_bytes": 0,
    "row_count": 0,
    "columns": [{"name": "target", "arrow_type": "large_string", "nullable": false}],
    "compression": "zstd",
    "compression_level": 3,
    "target_row_group_rows": 250000
  },
  "perturbations": {
    "count": 0,
    "ordered_values_sha256": "<digest of the sorted, distinct `target` values>"
  },
  "producer": {"cell_eval2": "...", "gpudge": "...", "cellstream": "...", "polars": "...", "pyarrow": "..."}
}
```

`created_at_utc` is informational only and is not part of the cache ID. `producer` here is
**best-effort** (`_producer_versions_best_effort`; a missing distribution records `"unknown"`
rather than raising) — it is diagnostic provenance, unlike `producer_versions()`'s hard error when
building the semantic document, since the versions that matter for identity are already pinned in
`de_semantics`. `CACHE_FORMAT = "cell-eval2-ground-truth-deg"`.

### 6.4 Entry validation (§10.4)

`_validate_entry_files` (shared by `GroundTruthDEGCache.validate_entry`, bound to a request, and
`inspect_entry`, which passes `request=None` and skips the fingerprint/semantics-equality check)
runs, in order:

1. **Entry type** — the entry path is a real directory: not a symlink (even one to a valid copy
   of the entry inside the root), a regular file or a dangling link.
2. **Containment** — the entry directory's realpath resolves under the cache root.
3. **Regular files** — `manifest.json` and `de_real.parquet` exist and are regular files (a symlink
   or other special file is itself corruption).
4. **Format/version/ID** — `manifest["cache_format"] == CACHE_FORMAT`,
   `cache_format_version == CACHE_FORMAT_VERSION`, `manifest["cache_id"]` equals the requested ID.
5. **Fingerprint/semantics equality** (only when a `request` is supplied) — the manifest's
   `ground_truth_fingerprint`/`de_semantics` documents re-hash to their own recorded digests, *and*
   match the request's `fingerprint.document`/`semantics.document` exactly.
6. **Artifact bytes** — `artifact.path` is `de_real.parquet`, and on-disk `size_bytes` and `sha256`
   match the manifest's `artifact` block.
7. **Parquet metadata** — codec is `["ZSTD"]` for every column chunk, row count matches, column
   names and Arrow types match the manifest exactly, and every column in `de.REQUIRED_COLS`
   (`target`, `feature`, `log2_fold_change`, `p_adj`) is present with a DE type, whatever the
   manifest says: `target`/`feature` strings, `log2_fold_change`/`p_adj` floating point.
8. **Content** — `target`/`feature` have zero nulls; no duplicate `(target, feature)` row.
9. **Perturbation universe** — the sorted, distinct `target` values' count and
   `sequence_sha256` match `manifest["perturbations"]`.

Every failure raises `GroundTruthCacheCorruptError`, naming the cache ID, the failed check, and the
exact path. A malformed or mismatched committed entry is **corrupt, never a miss** — neither
`readwrite` nor `readonly` overwrites it automatically; check 1 makes that hold for anything at
the entry path, which `lookup` therefore never reports as a miss.

**All nine checks run on every hit, not only at write time — by design, not by oversight.** A
lookup that skipped the artifact-bytes, content or perturbation-universe checks on a warm hit
would be faster, but a false hit — silently serving a table that no longer matches its own
manifest — is the one failure this cache must never produce. Measured at ≈2.2 s for a 1.2 GB /
28M-row entry (full-file SHA-256 plus the null/duplicate scans included), which is the validation
cost every `readwrite`/`readonly` hit pays before `get_or_compute` returns `status="hit"`.

## 7. Concurrency and atomicity (§11)

`GroundTruthDEGCache` is a dedicated content-addressed store — the existing general `CacheStore` is
not reused as the top-level implementation because it does not single-flight the compute operation
and could repoint or evict content under one logical key.

`get_or_compute(request, compute)`:

1. `lookup(request)` — an **unlocked** existence + validation check. Returns a `hit` immediately if
   a valid entry exists; raises `GroundTruthCacheCorruptError` if an entry directory exists but
   fails validation (never silently treated as a miss).
2. Under `readonly`, a miss raises `GroundTruthCacheMissError` naming the cache ID and entry path —
   `compute` is never called.
3. Under `readwrite`: open the per-entry `.lock` file and acquire an exclusive `fcntl.flock`
   (blocking; the wait is timed as `lock_wait_seconds`). A failure to acquire the lock raises
   `GroundTruthCacheFilesystemError` pointing at `cell-eval2 cache doctor <root>`.
4. **Recheck under the lock** — another writer may have published while this one waited; a hit here
   is returned as `status="hit_after_wait"`. This second check is what prevents duplicate DEG
   computation between concurrent cold callers.
5. Clean up any stale staging directories for this cache ID
   (`STALE_STAGING_AGE_S = 86_400` seconds; best-effort, logged, never blocks the new write).
6. Create a unique staging directory, call `compute()` **exactly once** (must return a
   `polars.DataFrame`, else `GroundTruthContractError`), and capture the sorted distinct `target`
   values before the table is written.
7. Write `de_real.parquet` via `de_export.write_de_parquet`, `fsync` it, build `manifest.json` and
   `fsync` it.
8. **Self-validate** the staged entry (`validate_entry`) before publishing — a failure here is a
   `GroundTruthCacheFilesystemError` (a writer bug, not external corruption), never a silent
   publish of a bad entry.
9. `fsync` the staging directory, then `os.rename(staging, entry)`. If the rename loses a publish
   race (the entry now exists), the staged copy is discarded and the winner's entry is returned as
   `hit_after_wait`.
10. `fsync` the parent directory, release the lock, return `status="miss_computed"` with
    `lock_wait_seconds`/`compute_seconds`/`write_seconds`/`validate_seconds` populated.

Readers of an already-committed entry **never take a lock** — publication is a same-filesystem
atomic rename, so a reader either sees nothing or a fully valid entry. A test seam,
`_fault_hook` (gated on the `CELL_EVAL2_GTC_FAULT` environment variable, or a direct monkeypatch of
the module attribute), hard-kills the process at `after_parquet`, `after_manifest`, `before_rename`
or `after_rename` to exercise crash recovery.

Shared-cache writes require a working POSIX advisory lock (`fcntl.flock`) and a same-filesystem
atomic rename; if either is unsupported, `readwrite` fails with `GroundTruthCacheFilesystemError`
rather than silently writing an unlocked shared entry. `cell-eval2 cache doctor <root>` diagnoses
exactly this (§9).

`fcntl` itself is imported **lazily**: `import cell_eval2`, fingerprinting an archive, and a
`readonly`/hit-only lookup all work on any platform, including one without `fcntl`. Only a
`readwrite` miss (which must take the per-entry lock before computing) or `cell-eval2 cache
doctor` (which exercises locking directly) import it, and only those paths can raise
`GroundTruthCacheFilesystemError` for its absence.

## 8. Using it from Python

**This is the only way to turn the cache on.** No `cell-eval2` subcommand runs
`score_cellstream` (§9), so a caller passes the two keyword arguments to it directly:

```python
from cell_eval2 import score_cellstream, MemBudget
from cell_eval2.config import EvalConfig

# metrics="de": score_cellstream cannot serve the moments metrics in the default "full" set
cfg = EvalConfig(metrics="de", device="cuda", pert_col="target_gene")
result = score_cellstream(
    "pred.shad", "real.shad",
    config=cfg,
    mem_budget=MemBudget(host_bytes=64 * 2**30, gpu_bytes=40 * 2**30),
    outdir="results/",
    write_de=True,   # -> results/ground_truth_de_cache.json beside the five write_de outputs
    ground_truth_cache="/shared/gt-cache",    # the cache root (§12)
    ground_truth_cache_policy="readwrite",    # or "readonly" once the cache is populated
)

import json
with open("results/ground_truth_de_cache.json") as fh:
    sidecar = json.load(fh)
print(sidecar["status"], sidecar["cache_id"])   # e.g. "hit" / "64-hex..."
```

A producer that only needs the strong archive identity — for example, an archive writer
persisting its ground-truth fingerprint alongside a published real archive — uses the
fingerprinting helper directly, without touching the cache storage at all:

```python
from cell_eval2 import fingerprint_ground_truth_cell_archive

fp = fingerprint_ground_truth_cell_archive("real.shad", pert_col="perturbation")
print(fp.sha256, fp.document)
```

For lower-level cache access (inspection tooling, tests, or a producer that wants to pre-warm an
entry outside `score_cellstream`), use `ground_truth_cache.GroundTruthDEGCache` directly.
`GroundTruthDEGCacheRequest` carries **identity only** — `fingerprint` and `semantics` — while the
`GroundTruthDEGCache` instance owns `root` and `policy` and is the single source of truth for
whether this call may create an entry: a `readonly` cache can never create anything, regardless of
what the request or caller ask for.

```python
from cell_eval2.ground_truth_cache import (
    GroundTruthDEGCache, GroundTruthDEGCacheRequest, fingerprint_ground_truth_cell_archive,
    build_real_de_semantics, RealDESemantics,
)

fingerprint = fingerprint_ground_truth_cell_archive("real.shad", pert_col="perturbation")
semantics = build_real_de_semantics(
    cfg, control="non-targeting", input_type="lognorm", target_sum=None,
    device="cuda", de_backend="gpudge", precision="float32", native_gpu_normalize=True,
)
cache = GroundTruthDEGCache("/shared/gt-cache", policy="readwrite")
request = GroundTruthDEGCacheRequest(fingerprint=fingerprint, semantics=semantics)
cache_result = cache.get_or_compute(request, compute=lambda: my_real_de_dataframe)
print(cache_result.status, cache_result.table_path)
```

## 9. CLI (`cell-eval2 cache ...`)

```text
cell-eval2 cache inspect <entry-or-id> --root <root>
cell-eval2 cache doctor <root>
```

- **`cache inspect`** calls `ground_truth_cache.inspect_entry(entry_or_id, root=root)`.
  `entry_or_id` is either a 64-hex cache ID (resolved under `--root` via `entry_relpath`) or an
  entry directory path directly. It always checks **identity self-consistency** — both the
  `ground_truth_fingerprint` and `de_semantics` documents are present and schema-valid, each
  re-hashes to its own recorded digest, and `manifest["cache_id"]` equals `sha256` of the §5.3 key
  document built from the two — plus every file-level check (regular files, format/version,
  artifact size/sha256, parquet codec/rows/columns, content nulls/duplicates,
  perturbation-universe digest). The only thing it skips is equality against a specific
  `GroundTruthDEGCacheRequest`, since `cache inspect` has none to compare against. It never
  raises, even for a malformed entry — invalid JSON, invalid UTF-8, a missing required field are
  reported exactly like a corrupt one: `problems` carries the message(s) and `valid` is `false`.
  It prints `{"entry_dir", "cache_id", "valid", "problems", "manifest"}` as JSON (the `manifest`
  summary omits the `ground_truth_fingerprint`/`de_semantics` blocks) and exits `0` if valid, `2`
  otherwise.
- **`cache doctor`** calls `ground_truth_cache.doctor(root)`. It self-checks a cache root's
  filesystem primitives — directory creation, `flock` exclusion between a parent and a child
  process, a Parquet Zstandard round-trip, and a same-filesystem atomic directory rename — entirely
  inside a scratch directory it creates and removes under
  `<root>/ground-truth-deg/v{CACHE_FORMAT_VERSION}/.doctor-<pid>-<uuid>`. It computes **no DE**. It
  never raises for a failed probe; it prints `{"root", "checks": [...], "ok": ...}` as JSON and
  exits `0` if every check passed, `1` otherwise.

Both subcommands are registered under `cell-eval2 cache <inspect|doctor>` in `cli.py`
(`cache_sub = cache.add_subparsers(dest="cache_command", ...)`). There is no delete/GC subcommand
in this version — see [§12](#12-operations).

**`cache inspect`/`cache doctor` are the only CLI subcommands that touch this cache.** None of
the scoring subcommands (`run`, `baseline`, `prep-cache`, `prep-real-bundle`, `score`) invokes
`score_cellstream`, and none has a flag for this cache: their `--cache-real` is the general
artifact cache's root (`CacheStore`), a different system. The ground-truth cache is turned on only
by calling `score_cellstream` from Python (§8).

## 10. Provenance sidecar (§12.5)

A `write_de=True` `score_cellstream` run **with the cache on** publishes
`ground_truth_de_cache.json`: staged right after `de_real.parquet`/`de_pred.parquet` and published
together with them and the three score frames — six files under the one rollback-on-error
guarantee `docs/de_tables.md` describes. With the cache off no sidecar is written, and `outdir`
receives exactly the five files of `docs/de_tables.md`.
`GroundTruthDEGCacheResult.to_sidecar_dict(producer=...)` is the schema:

```json
{
  "schema_version": 1,
  "status": "hit|hit_after_wait|miss_computed",
  "policy": "readwrite|readonly",
  "cache_id": "<digest or null>",
  "cache_entry": "<path or null>",
  "ground_truth_fingerprint_sha256": "<digest or null>",
  "de_semantics_sha256": "<digest or null>",
  "de_real_sha256": "<digest or null>",
  "de_real_rows": 0,
  "lock_wait_seconds": 0.0,
  "compute_seconds": 0.0,
  "producer": {"cell_eval2": "...", "gpudge": "..."}
}
```

The sidecar reports the outcome of that run's lookup: `status="hit"` / `"hit_after_wait"` /
`"miss_computed"`, with `cache_id`, `cache_entry`, both identity digests, and (on a miss)
`lock_wait_seconds`/`compute_seconds`, straight from the live `GroundTruthDEGCacheResult`.
`de_real_sha256`/`de_real_rows` describe the cache entry's table, and with the cache on the
published `de_real.parquet` IS that table: a byte copy, checked against the entry's SHA-256 as it
is copied (an entry that changed after this run validated it raises
`GroundTruthCacheCorruptError`, and nothing is published). So `de_real_sha256` is also the
published file's SHA-256, whichever pyarrow version wrote the entry.

The sidecar describes the run that wrote it, and only that run. A later run into the same `outdir`
that writes no sidecar (the cache off, or `write_de=False`) leaves an earlier sidecar in place,
just as a `write_de=False` run leaves an earlier run's DE tables. If its `de_real_sha256` differs
from the SHA-256 of `outdir/de_real.parquet`, the sidecar is stale. If they match, the published
table is the entry's bytes, but the other fields may still describe an earlier run. Use a fresh
`outdir` per run when the sidecar must describe the files beside it.

## 11. Failure behaviour

| Condition | Behaviour |
|---|---|
| `ground_truth_cache is None` | Compute normally, exactly as before this cache existed: no fingerprint, no semantic document, no cache I/O of any kind, no sidecar. |
| `ground_truth_cache_policy` not `"readwrite"`/`"readonly"`, or `"readonly"` with no root | `ValueError`, before any archive is opened. |
| Root not absolute after `os.path.expanduser` | `GroundTruthContractError`, before any archive is opened. |
| Root is, or lies inside, a file the run publishes into `outdir` (compared after `realpath`) | `ValueError`, before any archive is opened; nothing is created. |
| `readonly` root does not exist | `GroundTruthCacheFilesystemError`, before any archive is opened; nothing is created. |
| Strong payload checksum absent, cache enabled | Fail before any DE: `GroundTruthFingerprintError`. |
| Real archive replaced between `score_cellstream`'s two opens of it (the fingerprinted handle and the one the reference build streams from), cache enabled | Fail before any DE: `GroundTruthFingerprintError`. Nothing is computed or cached. |
| Anything but a real directory at an entry path (a symlink, a regular file, a dangling link) | `GroundTruthCacheCorruptError` (check `entry_type`), at once: never a miss, never overwritten. |
| An exception or an interrupt during compute or write | The unpublished staging directory is removed and the lock released; the error propagates. |
| A `GroundTruthFingerprint` / `RealDESemantics` built with a digest that is not its document's | `GroundTruthContractError`. |
| `readonly`, miss | Fail with cache ID + entry path, no compute: `GroundTruthCacheMissError`. Never creates the root, entries or lock files. |
| `readwrite`, miss | Single-flight compute under an exclusive `flock`, then atomic publish: `status="miss_computed"`. |
| Lock unsupported / permission denied / rename fails for a reason other than a lost race | `GroundTruthCacheFilesystemError`, naming the path; diagnose with `cell-eval2 cache doctor <root>`. |
| Two writers race to publish the same cache ID | The loser discards its staged copy and returns the winner's entry: `status="hit_after_wait"`. |
| Committed entry fails validation | `GroundTruthCacheCorruptError`, naming the cache ID/check/path. Never treated as a miss, never overwritten automatically. |
| Stale staging directory (older than `STALE_STAGING_AGE_S`) | Removed best-effort under the entry's lock before a new write; cleanup failure is logged and does not block the new staging directory. |
| `compute()` returns something other than a `polars.DataFrame` | `GroundTruthContractError`. |
| `cfg.de.replicate_col is not None` (cache enabled) | `GroundTruthContractError`, raised by `build_real_de_semantics` before any DE runs. With the cache off no semantic document is built, so nothing is enforced there. |
| A required producer distribution (`cell-eval2`/`gpudge`/`cellstream`) is not installed, cache enabled | `GroundTruthContractError` from `producer_versions()` — the cache cannot key an entry without knowing what produced it. With the cache off `producer_versions()` is never called. |

## 12. Operations

**Production root.** `cell_eval2` itself only ever creates and manages the
`ground-truth-deg/v{CACHE_FORMAT_VERSION}/` subtree **beneath** whatever path is passed as
`ground_truth_cache` (§6.1) — point it at the parent cache directory (e.g. `/shared/gt-cache`),
not at a path already ending in `ground-truth-deg/`, or entries will nest one level too deep. A
shared root needs to live on a filesystem every writer and reader can reach.

**NFS and locking.** `flock` on NFS requires a working lock manager. Run
`cell-eval2 cache doctor <root>` against the production mount before relying on it for concurrent
writers — it exercises exactly the parent/child `flock` exclusion, the Parquet round-trip, and the
atomic rename that a real writer needs. A node-local root (e.g. under `/tmp`) is fine for testing
and does not need `doctor` for correctness, only for confidence.

**Quota.** The cache accumulates immutably — nothing evicts or refreshes an entry automatically.
Budget roughly ≈1.2 GB per production-sized entry (a ~17M-row `de_real.parquet` plus its
manifest) times the number of distinct (ground-truth archive, real-DE semantics) combinations
that will ever be evaluated.

**Storage churn across releases (§5.1, deliberate).** `cell_eval2_version`, `gpudge_version` and
`cellstream_decoder_version` are semantic keys, so every `cell_eval2`, `gpudge` or `cellstream`
release — including one that changes nothing about the DE math — starts a fresh generation of
cache IDs: every entry keyed under the old versions becomes unreachable, not deleted. Budget
roughly one full generation's worth of entries per release. The separate, manually-bumped
`real_de_semantics_version` is for a narrower change to the real-DE row/value/dtype/ordering
contract *within* one release.

To reclaim an old generation, remove entries whose recorded `de_semantics.cell_eval2_version`
names a version no longer in service — while nothing is using the root (§0's rule for removing an
entry), and after confirming with `cache inspect` that nothing still depends on them:

```bash
find <root>/ground-truth-deg/v1 -mindepth 2 -maxdepth 2 -type d | while read -r d; do
  jq -e '.de_semantics.cell_eval2_version == "0.17.0"' "$d/manifest.json" >/dev/null 2>&1 && echo "$d"
done | xargs -r rm -rf
```

(`manifest.json`'s top-level `producer.cell_eval2`, §6.3, is diagnostic-only best-effort
provenance — match on `de_semantics.cell_eval2_version` for an authoritative selection.) Removing
an entry's directory also orphans its sibling `.<cache_id>.lock` file; remove that too, once no
writer can be running.

**Immutability and eviction.** There is no refresh policy and no delete/GC CLI command in this
version. An operator who needs to remove a specific entry deletes its exact directory
(`<root>/ground-truth-deg/v1/<xx>/<cache_id>/`) and its lock file out of band (§0); nothing in the
cache does this automatically, and a changed real-DE semantics simply produces a different cache
ID rather than invalidating the old one.

**Backups.** Not needed — this is derived data, reproducible from the strong source archive plus
the recorded semantics.

**Security.** The manifest stores only digests, counts and package versions — no source archive
path, filename, username, credential, model path, or full observation identifier is written to a
shared manifest (contrast the fingerprint document's deliberate exclusion of the archive path,
§4.3). Cache IDs are validated as exactly 64 lowercase hex characters everywhere they cross a
filesystem-path boundary (`entry_relpath`, `GroundTruthDEGCache.entry_dir`), and entry containment
is re-checked against the resolved cache root on every read (`_check_containment`) to refuse a path
that escapes the root through a symlink. The `readwrite` constructor makes the same check before it
creates `ground-truth-deg/v1/`, so a symlinked component that leaves the root creates nothing.

## 13. Related

- [`docs/de_tables.md`](de_tables.md) — the "unthresholded" definition, the canonical 11 columns,
  row order, and the `score_cellstream(write_de=True)` output contract this cache builds on.
- [`tests/fixtures/ground_truth_cache/README.md`](../tests/fixtures/ground_truth_cache/README.md)
  — the frozen cross-repository fixture kit (archives, golden vectors, the stdlib-only reference
  implementation `gtc_reference.py`) that the archive producer and CellEval2 must both reproduce
  byte for byte.
- [#236](https://github.com/ArcInstitute/cell_eval2/issues/236),
  [#387](https://github.com/ArcInstitute/cell_eval2/issues/387),
  [#388](https://github.com/ArcInstitute/cell_eval2/issues/388) — the streaming real-DE cache, the
  `write_de` output contract, and its Zstandard/bounded-memory hardening.
- `tests/fixtures/ground_truth_cache/eval_config_classification.yaml` and
  `tests/test_ground_truth_cache_classification.py`,
  `tests/test_ground_truth_cache_golden.py`,
  `tests/test_ground_truth_cache_identity.py`,
  `tests/test_ground_truth_cache_store.py`,
  `tests/test_ground_truth_cache_cli.py` — the classification allowlist and the tests that pin it,
  the golden identity vectors, and the storage/CLI test suites.
