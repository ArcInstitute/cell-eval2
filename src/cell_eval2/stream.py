"""Streaming source for packed cellstream (``SHPK``) ``.shad`` archives.

Reads a packed archive shard-by-shard so the full matrix never has to be
resident. The optional ``cellstream`` dependency (extra ``scale``) is imported
lazily inside each function, so importing this module never requires cellstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ShadMeta:
    n_obs: int
    n_vars: int
    var_names: np.ndarray
    group_by: str
    perts: np.ndarray


def is_shad(path) -> bool:
    """True if ``path`` is a packed cellstream (SHPK) file."""
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
        return False
    from cellstream.packed import is_packed_file

    return bool(is_packed_file(path))


def shad_metadata(path) -> ShadMeta:
    from cellstream.read import ShardedArchive

    a = ShardedArchive(path)
    # Fetch each accessor ONCE. `a.obs` / `a.var` are decoding properties on the packed reader, so
    # `getattr(a, "obs")` already evaluates one, and re-accessing `a.obs` decodes it a SECOND time
    # -- var in particular routes through header_adata() (read_h5ad on the header) every access.
    obs_attr = getattr(a, "obs", None)
    obs = obs_attr() if callable(obs_attr) else obs_attr
    var_attr = getattr(a, "var", None)
    var = var_attr() if callable(var_attr) else var_attr
    group_by = a.manifest.get("group_by")
    perts = np.unique(obs[group_by].to_numpy().astype(str))
    return ShadMeta(
        int(a.n_obs),
        int(a.n_vars),
        np.asarray(var.index.values, dtype=str),
        str(group_by),
        perts,
    )


def shad_var_names(path) -> np.ndarray:
    """Just the gene axis of a packed ``.shad``, without ``shad_metadata``'s extra work.

    ``shad_metadata`` decodes ``obs`` as well and runs ``np.unique`` over EVERY cell label to
    build ``perts`` -- millions of strings on a production archive. The gene-axis validation added
    by the 2026-07-25 ultrareview needs only ``var``, so it skips all of that.

    This does NOT avoid reading ``obs``: cellstream's packed ``var()`` goes through
    ``header_adata()``, which reads a header h5ad that carries the full obs, and unlike ``obs()``
    it has no columnar fast path. Measured 119 ms vs shad_metadata's 281 ms on a 200k-cell
    archive; the residual still scales with n_obs. A ``/var``-only reader belongs in cellstream.
    """
    from cellstream.read import ShardedArchive

    a = ShardedArchive(path)
    var_attr = getattr(a, "var", None)          # fetched ONCE: `a.var` may be a decoding property
    var = var_attr() if callable(var_attr) else var_attr
    return np.asarray(var.index.values, dtype=str)


def shad_fingerprint(path) -> str:
    """Stable identity of a packed ``.shad`` archive for cache keys + partition-aggregate safety.

    Prefers the writer's content fingerprint from the manifest **only when it is a
    plain string** (a content digest); otherwise falls back to a deterministic
    hash of structural metadata. (cellstream's manifest ``writer_fingerprint`` is a
    dict of writer provenance -- ``cellstream_version`` (``shardad_version`` before the
    rename), worker counts -- which is *not* a stable content identity and would embed a
    version string into the cache key, so it is deliberately ignored here in favour of the
    structural hash. The ``isinstance(wf, str)`` test below is what makes both spellings take
    the same branch.)
    """
    import hashlib

    from cellstream.read import ShardedArchive

    a = ShardedArchive(path)
    wf = a.manifest.get("writer_fingerprint")
    if isinstance(wf, str) and wf:
        return f"shad:{wf}:{a.manifest.get('group_by')}"
    m = shad_metadata(path)
    h = hashlib.sha256()
    for part in (m.n_obs, m.n_vars, m.group_by, *sorted(m.perts.tolist())):
        h.update(str(part).encode())
        h.update(b"\x00")
    for gene in m.var_names:  # gene set/order: distinct panels must not collide on a structural hash
        h.update(str(gene).encode())
        h.update(b"\x00")
    return "shad:" + h.hexdigest()


def iter_blocks(path, *, pert_col):
    """Yield ``(X_csr float32, pert_labels str)`` per group-shard; ~one shard resident."""
    from cellstream.read import ShardedArchive

    a = ShardedArchive(path)
    for gs in a.iter_group_shards():
        block = gs.to_anndata()
        X = block.X
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        if X.dtype != np.float32:  # frugal cast: new data array, shared indices/indptr
            X = X.__class__((X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape)
        labels = block.obs[pert_col].to_numpy().astype(str)
        yield X, labels


def read_reference_block(path, *, pert_col):
    """Return ``(X_csr float32, labels str)`` for the archive's reference shard, or ``None`` if the
    archive has none.

    The reference shard (the control pool, e.g. ``non-targeting``) is deliberately EXCLUDED by
    ``iter_blocks`` / ``iter_group_shards`` -- a consumer that needs a COMPLETE cell set (e.g. the
    row-store simulator, which must include control cells for the scorer's reference) reads it via
    this helper and concatenates it with the ``iter_blocks`` stream. Output mirrors ``iter_blocks``."""
    from cellstream.read import ShardedArchive

    a = ShardedArchive(path)
    ref = a.read_reference()
    if ref is None:
        return None
    X = ref.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if X.dtype != np.float32:  # frugal cast: new data array, shared indices/indptr
        X = X.__class__((X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape)
    labels = ref.obs[pert_col].to_numpy().astype(str)
    return X, labels
