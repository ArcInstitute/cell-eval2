# Ground-truth DEG cache — Phase 0 contract kit

Frozen cross-repository fixture + golden identity vectors for the ground-truth
cache contract ([`docs/ground_truth_cache.md`](../../../docs/ground_truth_cache.md);
the § numbers in this kit are the original design spec's: §8 ground-truth identity,
§9 DEG semantic identity and cache ID, §17.1 unit tests). It is the reference that
CellEval2 (the cache owner) and any producer recording the fingerprint must
reproduce byte for byte.

| File | What it is |
|---|---|
| `gtc_reference.py` | Stdlib-only reference implementation: canonical JSON, length-prefixed ordered-string digest, fingerprint document, semantic-document validation, cache ID, entry path. |
| `golden_vectors.json` | Golden vectors generated from `gtc_reference.py` (+ the fixture manifest). Every `expected_*` value is a contract. |
| `fixture_manifest.json` | Identity record of each archive: file SHA-256, cellstream manifest subset, ordered `var_names`, `obs[perturbation]` in storage order, axis digests, fingerprint document + digest. |
| `archives/real.shad` | Ground truth, written with `payload_checksum=True` (cache-eligible). 320 cells x 150 genes, 6 perturbations + `non-targeting`, raw integer counts. |
| `archives/real_nochecksum.shad` | Same cells, no payload checksum. Cache-enabled evaluation must refuse it (§8.2). |
| `archives/real_onevalue.shad` | One count changed, with checksum. Must produce a different fingerprint (§17.1). |
| `archives/pred_a.shad` | Prediction A (resampled from the real means), no checksum (predictions need none). |
| `archives/pred_b.shad` | Prediction B (null model), with checksum. |
| `make_fixture.py` | Deterministic generator (seed `20260916`, cellstream 0.10.0). Regenerates identical `payload_sha256` values; file bytes may differ only in the manifest's `created_at`. |
| `make_golden_vectors.py` | Regenerates `golden_vectors.json`. |

Perturbation labels equal gene names (`TG01..TG06`) so CellEval2's target-gene
resolution needs no `target_gene_map`. The archives were written by cellstream
0.10.0 and are read by this repository's tests with cellstream 0.9.1.

Consumers: copy `gtc_reference.py` + `golden_vectors.json` (and the archives if
needed) verbatim, or point a test at this directory, as
`tests/test_ground_truth_cache_golden.py` does; its cellstream-dependent checks
skip when cellstream is not importable.

Regenerate (needs `cellstream>=0.10.0 anndata numpy scipy pandas`):

```bash
python tests/fixtures/ground_truth_cache/make_fixture.py
python tests/fixtures/ground_truth_cache/make_golden_vectors.py
```
