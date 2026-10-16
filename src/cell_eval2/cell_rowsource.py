"""Row-access lazy view over cells, for the raw-cell (``kind='cell'``) metrics.

Ported from arcbench-nf ``bin/_cell_source.py``. The scoring pipeline's own
``cell_source.py`` is GROUP-oriented (whole-perturbation blocks + streaming-bulk
accumulators, built for pseudobulk/DE). The ``cell_cov_*`` metrics instead need
RANDOM ROW ACCESS -- per-perturbation cell gathers and column-sliced streaming --
which the group API cannot express. This module supplies that narrow interface.

Two concrete sources implement it: ``AnnDataSource`` wraps an in-memory/backed
AnnData (the ``compute_metrics`` path hands it ``pred_ad``/``real_ad``), and
``CellStoreSource`` streams a cell-layout ``.shad`` archive through the cellstream
store's ``gather_rows`` -- so a cell metric reads only the cells it samples, without
materializing the whole archive.

Memory contract: no implementation materializes the full X. Metrics pull only what
they need -- ``gather_rows`` of the sampled cells, or ``iter_row_chunks`` over a
column subset -- plus cached global stats (``col_mean``/``row_sum``/``global_max``)
computed once and reused across every metric sharing the same source instance.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


# -- Generic X reductions (backed lazy / scipy sparse / dense) -----------------
def _row_sum(X) -> np.ndarray:
    """Per-row sum (n_obs,) over any of: backed lazy, scipy sparse, dense."""
    r = X.sum(axis=1)
    return np.asarray(r).reshape(-1).astype(np.float64, copy=False)


def _col_sum(X) -> np.ndarray:
    """Per-column sum (n_vars,)."""
    m = X.sum(axis=0)
    return np.asarray(m).reshape(-1).astype(np.float64, copy=False)


def _global_max(X) -> float:
    """Single scalar X.max() across backed / sparse / dense."""
    m = X.max()
    return float(m.item()) if hasattr(m, "item") else float(m)


def _densify_cols(chunk, cols: np.ndarray | None) -> np.ndarray:
    """Column-slice (optional) + densify + contiguous float32 for one row block."""
    if cols is not None:
        chunk = chunk[:, cols]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    else:
        chunk = np.asarray(chunk)   # np.matrix (legacy AnnData) -> ndarray, not a matrix view
    return np.ascontiguousarray(chunk, dtype=np.float32)


#: Dense float32 bytes per chunk of the global-stats pass (`CellSource._ensure_global_stats`).
_GLOBAL_STATS_CHUNK_BYTES = 256 * 2**20


class CellSource:
    """Read-only lazy view. Concrete sources implement ``gather_rows`` and
    ``iter_row_chunks``; the base derives global stats from a single streaming
    pass and caches them (so a metric that needs them reads X once). ``obs`` / ``var`` /
    ``var_names`` / ``n_obs`` / ``n_vars`` are set by subclasses and are small
    (metadata only)."""

    obs: pd.DataFrame
    var: pd.DataFrame
    var_names: pd.Index
    n_obs: int
    n_vars: int

    def __init__(self) -> None:
        self._col_sum: np.ndarray | None = None
        self._row_sum: np.ndarray | None = None
        self._max: float | None = None

    # -- required per-impl ----------------------------------------------------
    def gather_rows(self, rows: np.ndarray):
        """Return the given rows (all genes), IN THE REQUESTED ORDER, as scipy
        sparse or dense."""
        raise NotImplementedError

    def iter_row_chunks(
        self, chunk_size: int, cols: np.ndarray | None = None,
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        """Yield (start, end, dense float32 chunk) over ALL rows, optionally
        restricted to ``cols``. RAW values -- no normalization (callers scale
        using the full-gene ``row_sum``)."""
        raise NotImplementedError

    # -- derived global stats (single cached streaming pass) ------------------
    def _ensure_global_stats(self) -> None:
        if self._col_sum is not None:
            return
        csum = np.zeros(self.n_vars, dtype=np.float64)
        rsum = np.empty(self.n_obs, dtype=np.float64)
        gmax = 0.0
        # Rows per chunk from a byte budget, not a fixed count: 50,000 dense float32 rows are
        # 3.7 GB at 18.5k genes (Gemini, #395).
        chunk_rows = max(1, _GLOBAL_STATS_CHUNK_BYTES // (4 * max(self.n_vars, 1)))
        for start, end, chunk in self.iter_row_chunks(chunk_rows, cols=None):
            csum += chunk.sum(axis=0, dtype=np.float64)
            rsum[start:end] = chunk.sum(axis=1, dtype=np.float64)
            if chunk.size:
                gmax = max(gmax, float(chunk.max()))
        self._col_sum, self._row_sum, self._max = csum, rsum, gmax

    def col_mean(self) -> np.ndarray:
        self._ensure_global_stats()
        return self._col_sum / max(self.n_obs, 1)

    def row_sum(self) -> np.ndarray:
        self._ensure_global_stats()
        return self._row_sum

    def global_max(self) -> float:
        self._ensure_global_stats()
        return self._max

    def should_normalize(self, input_type: str) -> bool:
        """Whether metrics should CP10k+log1p this source. ``counts`` -> always,
        ``lognorm`` -> never, ``auto`` -> the historical max>25 heuristic."""
        if input_type == "counts":
            return True
        if input_type == "lognorm":
            return False
        return self.global_max() > 25

    def close(self) -> None:
        pass


class AnnDataSource(CellSource):
    """Wrap an in-memory or backed AnnData. Delegates every read straight to
    ``adata.X`` so the numerics match a direct AnnData computation exactly. A backed X is read
    row block by row block, never materialized."""

    def __init__(self, adata) -> None:
        super().__init__()
        self._adata = adata
        self.obs = adata.obs
        self.var = adata.var
        self.var_names = adata.var_names
        self.n_obs = adata.n_obs
        self.n_vars = adata.n_vars

    def gather_rows(self, rows: np.ndarray):
        rows = np.asarray(rows)
        if (not getattr(self._adata, "isbacked", False) or rows.size < 2
                or np.all(rows[:-1] <= rows[1:])):
            return self._adata.X[rows]
        # A BACKED X is read through h5py, which takes increasing indices only: gather sorted,
        # then return the rows in the requested order, as the base class promises.
        order = np.argsort(rows, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return self._adata.X[rows[order]][inverse]

    def iter_row_chunks(self, chunk_size, cols=None):
        # Rows first, columns per chunk: `X[:, cols]` up front would read every row of a backed X
        # before the first chunk, so `chunk_size` would bound nothing (Copilot, #395).
        X = self._adata.X
        for start in range(0, self.n_obs, chunk_size):
            end = min(start + chunk_size, self.n_obs)
            yield start, end, _densify_cols(X[start:end], cols)

    # Cheap direct reductions on a RESIDENT X -- matches a direct AnnData call. A backed X
    # takes the base class's chunked pass instead, so the memory contract holds (Gemini).
    def col_mean(self) -> np.ndarray:
        if getattr(self._adata, "isbacked", False):
            return super().col_mean()
        return _col_sum(self._adata.X) / max(self.n_obs, 1)

    def row_sum(self) -> np.ndarray:
        if getattr(self._adata, "isbacked", False):
            return super().row_sum()
        return _row_sum(self._adata.X)

    def global_max(self) -> float:
        if getattr(self._adata, "isbacked", False):
            return super().global_max()
        return _global_max(self._adata.X)


class CellStoreSource(CellSource):
    """Row-access view over a cell-layout ``.shad`` archive that STREAMS instead of
    materializing. The slot ``cell_rowsource``'s header anticipated: cell metrics need random
    per-perturbation row gathers, which a ``cellstream`` ``CellStore`` already serves through
    ``gather_rows`` (memory-mapped, ~12 ms warm for a 500-row block on the NGN2 archive), so a
    whole-archive AnnData never has to be built. The ~G×G correlation metric reads only the
    ≤``max_cells`` cells it samples per perturbation, not the full 1.3M×18k matrix.

    Opens the archive through ``_cell_archive`` (the sole-importer shim) so this module keeps no
    direct ``cellstream`` dependency. ``obs`` / ``var`` are the archive's metadata frames (small);
    every X read is a ``gather_rows`` against the store. The caller owns the lifetime -- call
    :meth:`close` (or let ``dispatch_cell_metrics``' streaming path close it) to release the mmap.
    """

    def __init__(self, path, *, gather_threads: int = -1) -> None:
        super().__init__()
        from ._cell_archive import materialized_x_dtype, open_cell_store

        # `EvalConfig.gather_threads` (-1 = auto): the store's own default decodes serially.
        self._gather_threads = gather_threads
        self._store = open_cell_store(path)
        try:
            # Gather at the dtype a MATERIALIZED read has, so a block is value-identical to the
            # rows of `load_anndata(path)`: the store's own default is float32 for every integer
            # archive, whose row sums round above 2**24 where a uint32 matrix's do not. None (an
            # unknown cellstream) keeps the store default.
            self._out_dtype = materialized_x_dtype(self._store)
            self.obs = self._store.obs
            var = self._store.var
            # A cell archive carries gene ids on var.index; some are written with an empty var
            # frame, in which case synthesize a positional index so var_names is still length
            # n_vars.
            self.var = (var if var is not None
                        else pd.DataFrame(index=np.arange(self._store.n_vars)))
            self.var_names = self.var.index
            self.n_obs = int(self._store.n_obs)
            self.n_vars = int(self._store.n_vars)
        except BaseException:
            # The caller never receives an object to close, so release the mmap here.
            self._store.close()
            raise

    def gather_rows(self, rows: np.ndarray):
        rows = np.asarray(rows)
        if rows.size == 0:
            import scipy.sparse as _sp
            return _sp.csr_matrix((0, self.n_vars), dtype=np.float32)
        # The store sees monotonic ids (its gather is cheapest, and order-safe, on sorted input),
        # but the rows come back IN THE REQUESTED ORDER, as the base class promises. CellsPrep
        # already sorts what it gathers, so the reorder below is for any other caller.
        if np.all(rows[:-1] <= rows[1:]):
            return self._gather(rows)
        order = np.argsort(rows, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return self._gather(rows[order])[inverse]

    def _gather(self, rows):
        from ._threads import resolve_gather_threads

        n_threads = resolve_gather_threads(len(rows), self._gather_threads)
        if self._out_dtype is None:
            return self._store.gather_rows(rows, n_threads=n_threads)
        return self._store.gather_rows(rows, n_threads=n_threads, out_dtype=self._out_dtype)

    def iter_row_chunks(self, chunk_size, cols=None):
        for start in range(0, self.n_obs, chunk_size):
            end = min(start + chunk_size, self.n_obs)
            block = self._gather(np.arange(start, end))
            yield start, end, _densify_cols(block, cols)

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
