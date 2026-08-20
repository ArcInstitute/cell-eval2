"""Deterministic, partition-stable on-the-fly noise vehicle for streamed blocks.

Wraps a ``(csr, labels)`` block iterator (e.g. ``stream.iter_blocks``). ``level=0.0``
is an exact identity passthrough. The per-shard RNG is seeded from ``seed`` XOR the
shard index, so the noise applied to a given shard is independent of how the stream
is partitioned across runs (partition-stable).
"""

from __future__ import annotations

import numpy as np

_KINDS = ("gaussian", "downsample")


def noise_block(X, *, kind, level, seed, shard_idx):
    """Noise one block's CSR ``X`` -> CSR. ``level == 0.0`` returns ``X`` unchanged.

    Single source of truth for the per-shard RNG (``seed`` XOR ``shard_idx``) and the noise
    math, so a caller streaming its own blocks (e.g. the row-store simulator, which needs both
    the clean and the noised block from one decode) reuses the exact behavior ``noise_blocks``
    yields. See the module docstring for the partition-stability rationale."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level!r}")
    if level == 0.0:
        return X
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(shard_idx))
    Xs = X.tocsr()
    # New matrix shares indices/indptr; only the data array is fresh (no index-array copy).
    if kind == "gaussian":
        # Multiplicative lognormal on nonzeros: preserves nonneg, scales with `level`. NOTE (F7.2):
        # exp(N(0, level)) has mean exp(level^2/2) > 1, so this inflates the expected library size
        # (~13% at level=0.5) rather than being mean-preserving -- intended as a scale perturbation.
        scaled = Xs.data * np.exp(rng.normal(0.0, level, size=Xs.data.shape))
        if np.issubdtype(Xs.dtype, np.integer):
            # Round (not truncate toward zero) and clip to the dtype range so a large scaled count
            # cannot silently wrap/overflow an integer count dtype (F7.1).
            info = np.iinfo(Xs.dtype)
            scaled = np.clip(np.rint(scaled), info.min, info.max)
        new_data = scaled.astype(Xs.dtype)
        return Xs.__class__((new_data, Xs.indices, Xs.indptr), shape=Xs.shape)
    # downsample
    # ⚠️ KNOWN, and RULED (2026-08-19, Alex) to stay: this `np.allclose` is at numpy's DEFAULT
    # RELATIVE tolerance, so the guard below does not deliver what its message promises. MEASURED:
    # `50000.5` PASSES it and is then truncated to 50000 by the `astype(np.int64)` two lines down --
    # the exact silent truncation the message exists to prevent. `norm._INT_ATOL` fixed the same
    # defect in the submission gate; this one is left because `noise=` reaches no scored path (no CLI
    # flag, `None` default, callers are `tools/scale` and `tools/rowstore` only) and is expected to be
    # retired with the rowstore format. A fix would also have to ROUND rather than truncate the dust
    # it accepts, which moves numbers here. Do not re-file this as a new bug.
    if not np.allclose(Xs.data, np.rint(Xs.data)):
        raise ValueError(
            "downsample noise requires integer count data "
            "(fractional/normalized input would be silently truncated)"
        )
    keep = np.clip(1.0 - level, 0.0, 1.0)
    new_data = rng.binomial(Xs.data.astype(np.int64), keep).astype(Xs.dtype)
    # eliminate_zeros() mutates indices/indptr in place, so this path must NOT share
    # them with the input (Xs may be the caller's matrix) -- copy before pruning.
    Xc = Xs.__class__((new_data, Xs.indices.copy(), Xs.indptr.copy()), shape=Xs.shape)
    Xc.eliminate_zeros()
    return Xc


def noise_blocks(blocks, *, kind, level, seed):
    """Wrap a ``(csr, labels)`` block iterator, noising each block. Delegates to ``noise_block``
    per shard (single source of truth); ``level == 0.0`` is an exact identity passthrough."""
    for shard_idx, (X, labels) in enumerate(blocks):
        yield noise_block(X, kind=kind, level=level, seed=seed, shard_idx=shard_idx), labels
