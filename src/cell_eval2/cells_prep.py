"""Shared preprocessing for the sampled-cell (Part B) metrics -- issue #386 §4.

The Part B families read BOTH sides as single cells in counts and compare them
per perturbation. Three things are shared across those families and are defined
once here so the families cannot drift apart on them:

  1. **The lognorm space.** ``log1p(1e4 · x / l)`` per cell -- CP10K. ⚠️ This is NOT the
     run's ``bulk_lognorm`` comparator and does NOT read ``cfg.bulk_target_sum`` or
     ``cfg.target_sum``: the comparator is a per-GROUP pseudobulk space
     (``moments.DEFAULT_BULK_TARGET_SUM``, 5e4) and these metrics are about CELLS. Spec §4
     fixes the constant, deliberately.
  2. **The scored gene set ``G_p``** (spec §2.3): every gene on both sides, minus
     the perturbation's OWN target gene -- resolved through
     ``distances.resolve_exclusion_columns`` with panel scope, so a run has ONE
     notion of "the target gene" shared with ``pds_*`` and the chance-corrected DE
     metrics -- minus genes with zero counts in BOTH the real control and the real
     perturbation (nothing to compare on a column that is zero everywhere).
  3. **Equal-depth seeded subsampling.** Both sides are cut to the SAME cell count
     per perturbation, ``min(n_real, n_pred, max_cells)``. Spec §4 asks for "at most
     ``max_cells`` cells, seeded, so cost is bounded and both sides sit at the same
     depth"; equal depth is the load-bearing half. Every Part B metric is a
     comparison between two quantities ESTIMATED from cells, and the sampling noise
     in such an estimate is a function of n -- a pred side with 4x the real side's
     cells would produce a systematically tighter correlation matrix (B4) or a
     narrower projected distribution (B2) for reasons that have nothing to do with
     the model. Capping each side independently at ``max_cells`` would leave that
     confound in for every perturbation where the two sides' depths differ.

Memory contract: no function here materializes a side's full X. Per-group summaries
(:class:`GroupStats`) stream in row chunks; cell blocks are gathered for ONE group at a
time and column-sliced to the metric's gene subset BEFORE densifying, so a
[500 cells x 100 genes] block is what a family actually holds -- not [500 x 18,533].
⚠️ Library sizes come from the FULL panel row sum, not from the sliced columns, so
slicing early does not change a single lognorm value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .distances import resolve_exclusion_columns

#: The per-cell normalization target, fixed by spec §4. See the module docstring for why
#: this is a constant here and not `cfg.target_sum` / `cfg.bulk_target_sum`.
CP10K = 1e4

#: Dense float64 bytes per chunk when streaming a group's summary. A ROW count would not bound
#: memory: 20,000 rows x 18,533 genes densify to 3 GB in float64, and the CP10K step holds two
#: more of that size, about 9 GB for one 20,000-cell control group. 256 MiB is ~1,800 rows at
#: that width; groups are usually far smaller than one chunk.
_STATS_CHUNK_BYTES = 256 * 2**20


def lognorm(counts, lib) -> np.ndarray:
    """``log1p(CP10K · counts / lib)``, dense float32.

    ``counts`` is a [n, k] count block (sparse or dense), ``lib`` the [n] FULL-panel
    library size of those same rows -- which is why the caller passes it in rather than
    summing ``counts``: ``counts`` is usually a column subset and its own row sums are
    not library sizes. A zero-library cell (possible on a filtered panel) yields zeros
    rather than a divide-by-zero.
    """
    dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    out = np.asarray(dense, dtype=np.float32)
    safe = np.where(np.asarray(lib, dtype=np.float64) > 0, lib, 1.0).astype(np.float32)
    return np.log1p(CP10K * (out / safe[:, None]))


def pert_rng(seed: int, pert: str) -> np.random.Generator:
    """A generator seeded by ``(seed, pert)``, independent of iteration order.

    A single ``default_rng(seed)`` advanced once per perturbation would make every
    perturbation's cell sample depend on how many perturbations were scored before it,
    so adding one perturbation to a panel would move every later one's value. The digest
    is taken over the pair so the same perturbation draws the same cells in the ceiling
    run, the baseline run, and the scoring run.
    """
    digest = hashlib.sha256(f"{int(seed)}:{pert}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class GroupStats:
    """One perturbation's (or the control's) per-gene summary on one side.

    ``cp10k_mean`` is the mean over cells of ``CP10K · x_cg / l_c`` -- the CP10K
    pseudobulk the ``log2FC`` in :meth:`CellsPrep.real_log2fc` is taken on.
    ``nonzero_cells`` is how many cells express each gene, which serves both the
    "at least ``min_nonzero_cells`` real cells" gene filters and the §2.3 zero-column
    drop (a count column is all-zero exactly when no cell expresses it).
    """

    n_cells: int
    cp10k_mean: np.ndarray      # [G] float64
    nonzero_cells: np.ndarray   # [G] int64


def group_stats(source, rows: np.ndarray, *, validate=None) -> GroupStats:
    """:class:`GroupStats` for ``rows`` of ``source``, streamed in row chunks. ``validate``, when
    given, is called on every raw block gathered (see :class:`CellsPrep`)."""
    n_vars = int(source.n_vars)
    cp10k_sum = np.zeros(n_vars, dtype=np.float64)
    nonzero = np.zeros(n_vars, dtype=np.int64)
    rows = np.asarray(rows)
    chunk = max(1, _STATS_CHUNK_BYTES // (8 * max(n_vars, 1)))
    for start in range(0, rows.size, chunk):
        block = source.gather_rows(rows[start:start + chunk])
        if validate is not None:
            validate(block)
        dense = block.toarray() if sp.issparse(block) else np.asarray(block)
        dense = np.asarray(dense, dtype=np.float64)
        lib = dense.sum(axis=1)
        safe = np.where(lib > 0, lib, 1.0)
        cp10k_sum += (CP10K * dense / safe[:, None]).sum(axis=0)
        nonzero += (dense != 0).sum(axis=0)
    n = max(rows.size, 1)
    return GroupStats(n_cells=int(rows.size), cp10k_mean=cp10k_sum / n, nonzero_cells=nonzero)


@dataclass(frozen=True)
class CellPair:
    """Both sides' lognorm cells for one perturbation, at equal depth, over ``cols``."""

    real: np.ndarray    # [n, k] lognorm float32
    pred: np.ndarray    # [n, k] lognorm float32
    cols: np.ndarray    # [k] column indices into the shared gene axis
    n_cells: int        # the common depth n


class CellsPrep:
    """The shared Part B view over one (pred, real) cell-source pair.

    Built once per dispatch and memoized in ``run.dispatch_cell_metrics``'s
    ``_cell_metric_cache`` (see :func:`prep_for`), so every Part B family in a run sees
    the same gene sets, the same cells, and the same per-group summaries. Only the CHEAP
    per-group summaries and index arrays are cached; cell blocks are gathered on demand
    (see the module docstring's memory contract).
    """

    def __init__(
        self, pred_source, real_source, *, pert_col: str, control: str,
        genes: np.ndarray | None = None, target_gene_map: dict[str, str] | None = None,
        input_type_pred: str = "counts", input_type_real: str = "counts",
        max_cells: int = 500, seed: int = 0, validate_cells=None,
    ) -> None:
        for side, it in (("pred", input_type_pred), ("real", input_type_real)):
            if it != "counts":
                raise ValueError(
                    f"the sampled-cell (Part B) metrics need raw counts, but the {side} side "
                    f"resolved to input_type={it!r}. They are defined on counts (spec §2.1): "
                    "a per-cell library size and a CP10K lognorm cannot be recovered from an "
                    "already-normalized matrix, and expm1-ing one back would invent counts. "
                    "Score these metrics on a counts pair, or deselect them."
                )
        self._pred = pred_source
        self._real = real_source
        self._pert_col = pert_col
        self._control = str(control)
        self._max_cells = int(max_cells)
        self._seed = int(seed)
        # `validate_cells(side, block)`: called on every RAW block gathered from either side,
        # before any arithmetic. The dispatch passes the run's input validation here, because the
        # cell metrics read only the cells they sample: on the cell-layout fast path, and for a
        # backed input, no whole-matrix pass has validated those cells. None = no checks.
        self._validate_cells = validate_cells

        self._genes = (np.asarray(real_source.var_names, dtype=str) if genes is None
                       else np.asarray(genes, dtype=str))
        if self._genes.size != int(real_source.n_vars):
            raise ValueError(
                f"gene labels ({self._genes.size}) do not match the real side's feature "
                f"dimension ({int(real_source.n_vars)})"
            )
        self._real_idx = self._index_by_pert(real_source)
        self._pred_idx = self._index_by_pert(pred_source)
        # Perturbations scorable at all: present, non-control, on BOTH sides. `io.validate_pair`
        # already requires identical label sets on the production path; a direct caller may not
        # have gone through it, so the intersection is taken rather than assumed.
        self._perts = [p for p in sorted(set(self._real_idx) & set(self._pred_idx))
                       if p != self._control]
        # Target-gene exclusion is resolved LAZILY (see `_excl_by_pert`): a metric that scores over
        # ALL genes without excluding the perturbed gene (cell_cov_ccc) must not require a
        # target_gene_map at all, and resolving here would raise on a symbol-vs-ENSG panel before
        # such a metric ever ran. The resolution itself is unchanged; only its timing is.
        self._target_gene_map = target_gene_map
        self._excl_by_pert_cache: dict[str, int] | None = None
        self._stats: dict[tuple[str, str], GroupStats] = {}

    @property
    def _excl_by_pert(self) -> dict[str, int]:
        """``{pert -> its own target-gene column}``, resolved on first use and cached.

        The ONE definition of "this perturbation's own gene", shared with pds_* and the
        chance-corrected DE family. Gated on the whole real panel (panel scope, spec §2.3), so a
        zero-resolve construct-ID panel raises here -- but only for a metric that actually asks for
        the exclusion (``scored_genes``), not for one that scores over all genes."""
        if self._excl_by_pert_cache is None:
            labels = list(self._perts)
            excl = resolve_exclusion_columns(
                labels, self._genes, target_gene_map=self._target_gene_map, gate_labels=labels,
                who="the sampled-cell (Part B) metrics' target-gene exclusion",
                escape="or deselect the cell_* metrics.",
            ) if labels else {}
            self._excl_by_pert_cache = {labels[i]: col for i, col in excl.items()}
        return self._excl_by_pert_cache

    # -- metadata -------------------------------------------------------------
    @property
    def genes(self) -> np.ndarray:
        return self._genes

    @property
    def perts(self) -> list[str]:
        """Non-control perturbations present on both sides, sorted."""
        return list(self._perts)

    def _index_by_pert(self, source) -> dict[str, np.ndarray]:
        col = source.obs[self._pert_col].astype(str).to_numpy()
        if col.size == 0:
            return {}       # `bounds` below would be [0] and index an empty array (Gemini)
        order = np.argsort(col, kind="stable")
        sorted_col = col[order]
        bounds = np.flatnonzero(np.r_[True, sorted_col[1:] != sorted_col[:-1]])
        return {str(sorted_col[b]): np.sort(order[b:e]) for b, e in
                zip(bounds, np.r_[bounds[1:], sorted_col.size])}

    def rows(self, side: str, pert: str) -> np.ndarray:
        # Indexed, not `.get(..., empty)`: every caller asks for a perturbation on both sides
        # or for the real control, which `io.validate_pair` guarantees, so a miss is a caller
        # bug -- and an empty control would silently become the log2FC's reference (Gemini).
        idx = (self._real_idx if side == "real" else self._pred_idx)
        return idx[str(pert)]

    def _source(self, side: str):
        return self._real if side == "real" else self._pred

    # -- cached per-group summaries -------------------------------------------
    def stats(self, side: str, pert: str) -> GroupStats:
        key = (side, str(pert))
        if key not in self._stats:
            validate = (None if self._validate_cells is None
                        else (lambda block, _side=side: self._validate_cells(_side, block)))
            self._stats[key] = group_stats(self._source(side), self.rows(side, pert),
                                           validate=validate)
        return self._stats[key]

    def real_log2fc(self, pert: str) -> np.ndarray:
        """``log2((cp10k_mean_p + 1) / (cp10k_mean_ctrl + 1))`` from the REAL side only.

        CP10K with pseudocount 1 (spec §4 B4 / §3 A2). Real-only by design: the gene
        classes these metrics are evaluated on must not depend on the submission, or a
        model could choose its own scored gene set.
        """
        p = self.stats("real", pert).cp10k_mean
        c = self.stats("real", self._control).cp10k_mean
        return np.log2((p + 1.0) / (c + 1.0))

    # -- the scored gene set --------------------------------------------------
    def scored_genes(self, pert: str) -> np.ndarray:
        """``G_p`` as column indices (spec §2.3): all genes, minus this perturbation's own
        target gene, minus genes with zero counts in BOTH the real control and real ``p``."""
        keep = np.ones(self._genes.size, dtype=bool)
        col = self._excl_by_pert.get(str(pert))
        if col is not None:
            keep[col] = False
        zero_both = ((self.stats("real", pert).nonzero_cells == 0)
                     & (self.stats("real", self._control).nonzero_cells == 0))
        keep &= ~zero_both
        return np.flatnonzero(keep)

    # -- equal-depth cell blocks ---------------------------------------------
    def depth(self, pert: str) -> int:
        """The common per-side cell count for ``pert``: ``min(n_real, n_pred, max_cells)``."""
        n = min(self.rows("real", pert).size, self.rows("pred", pert).size)
        return int(min(n, self._max_cells)) if self._max_cells else int(n)

    def pair(self, pert: str, cols: np.ndarray | None = None) -> CellPair:
        """Both sides' lognorm cells for ``pert`` over ``cols``, at :meth:`depth` rows each.
        ``cols=None`` means ALL genes (no column slice) -- the case ``cell_cov_ccc`` uses, which
        scores over the whole panel and excludes no gene.

        One seeded draw per (perturbation, side): ``pert_rng`` is re-derived per side from
        the same (seed, pert) pair and then advanced, so the two sides draw DIFFERENT cells
        (they are different datasets) while both are reproducible from the seed alone.
        """
        col_idx = (np.arange(self._genes.size) if cols is None
                   else np.asarray(cols, dtype=np.int64))
        n = self.depth(pert)
        blocks = {}
        for side in ("real", "pred"):
            rows = self.rows(side, pert)
            if rows.size > n:
                rng = pert_rng(self._seed, f"{pert}|{side}")
                rows = np.sort(rng.choice(rows, size=n, replace=False))
            source = self._source(side)
            full = source.gather_rows(rows)
            if self._validate_cells is not None:
                self._validate_cells(side, full)
            lib = np.asarray(full.sum(axis=1)).reshape(-1).astype(np.float64)
            blocks[side] = lognorm(full if cols is None else full[:, col_idx], lib)
        return CellPair(real=blocks["real"], pred=blocks["pred"], cols=col_idx, n_cells=n)


_CACHE_KEY = "cells_prep"


def prep_for(cache: dict, **kwargs) -> CellsPrep:
    """The run's single :class:`CellsPrep`, built on first use and memoized in ``cache``.

    ``cache`` is ``dispatch_cell_metrics``'s per-dispatch ``_cell_metric_cache``. Keyed on
    the construction kwargs that change what it computes, so a family asking for a
    different ``max_cells`` or ``seed`` gets its own instance instead of silently reusing
    another family's.
    """
    key = (_CACHE_KEY, kwargs.get("max_cells"), kwargs.get("seed"))
    prep = cache.get(key)
    if prep is None:
        prep = cache[key] = CellsPrep(**kwargs)
    return prep
