# `pre_247` — the metric as it stood before the #247 cap

Two **verbatim** source files, vendored so that
`tests/test_expr_mse_unbiased_ratio.py::test_expr_mse_unbiased_reproduces_the_pre_247_metric_exactly`
can reconstruct the old `delta.mse_unbiased` without reading this repository's history.

| file here | was | at |
|---|---|---|
| `moments.py` | `src/cell_eval2/moments.py` | `810215c~1` |
| `delta.py` | `src/cell_eval2/metrics/delta.py` | `810215c~1` |

`810215c` is `feat(#247)!: cap the prediction's sampling correction at the real side's`, so
`810215c~1` = **`6a40e433841fb05261cc786a4d7dcbf0c8884d2b`** — the last revision before the cap.

## Why vendored and not `git show`

The test used to shell out to `git show 810215c~1:<path>`, which ties it to one specific object
in one specific `.git`. That object does not exist in a tree copy, a source distribution, or any
repository other than the `ArcInstitute/cell_eval2` archive — and the old guard **raised** rather
than skipped in a full clone whose revision did not resolve, by design. Reading the two files
from disk makes the characterization portable and removes CI's `fetch-depth: 0`.

## Do not edit these files

They are a **characterization baseline**: the test asserts that today's `mse_unbiased` returns
bit-identical values to this code. Edit a fixture and the test still passes — while
characterizing something else. That is the one failure mode that would make it worthless, so the
loader verifies the SHA-256 of each file on every run (`PRE_247_SHA256` in the test module):

```
3c55e97a98024ffa59facc0c5e000a0c7464297974acc6672fc19af82709cf91  moments.py
cf2c10b89eef43c52a1f166dc0c98c6b0d2fe0d6d05b72ab223acd07b1412e23  delta.py
```

Ruff lints them like any other file in the tree; they pass the pinned rule set (`E4,E7,E9,F`)
unmodified. If a future rule widening flags them, exclude the directory in `pyproject.toml`
rather than reformatting the fixture.

To restore or re-derive them, from a clone of the archive repository:

```bash
git show 6a40e433841fb05261cc786a4d7dcbf0c8884d2b:src/cell_eval2/moments.py \
    > tests/fixtures/pre_247/moments.py
git show 6a40e433841fb05261cc786a4d7dcbf0c8884d2b:src/cell_eval2/metrics/delta.py \
    > tests/fixtures/pre_247/delta.py
```

Only `mse_unbiased` is exercised. The loader strips `delta.py`'s relative imports and injects
the two `moments` primitives it needs, so `prep.pseudobulk` and the `safety.*` helpers the file
also imported are never resolved.
