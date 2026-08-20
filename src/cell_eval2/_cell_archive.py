"""Sole importer of the cellstream cell-layout API.

The rename this module was built to absorb has happened: the private ``shardad`` package is
now the public ``cellstream``, and confining the cell-layout imports here is what kept that
swap to the import lines. Note ``cellstream`` is a FRESH public repository, not a rename of
the private one, so shardad issue numbers do not carry over and are not cited as public links.
Layout is detected by manifest, never by file extension (the cell-archive extension is
``.csa``).
"""
from __future__ import annotations

import os
from pathlib import Path

import anndata as ad

# The public floor `[scale]` installs. Every cell-layout capability this module gates on exists in
# cellstream's first public release, so one constant covers both gates. It names 0.9.1 rather than
# 0.9.0 to match what `[scale]` actually resolves -- the error text tells the reader to reinstall
# with `cell_eval2[scale]`, so the version it names must be the one that gives them. The .1 is a
# Python 3.13 guardrail; the reasoning is in pyproject.toml.
_MIN_CELLSTREAM = "0.9.1"


def _is_shpk(p: Path) -> bool:
    """True if the file starts with the cellstream ``SHPK`` container magic, checked
    WITHOUT importing cellstream so ``cell_layout`` can raise a clear error when the
    package is absent. The magic did NOT change in the shardad->cellstream rename, so
    archives written by either package are recognised here."""
    try:
        with p.open("rb") as fh:
            return fh.read(4) == b"SHPK"
    except OSError:
        return False


def cell_layout(path: str | os.PathLike) -> bool:
    """True iff ``path`` is a cellstream cell-layout (``layout="cell"``) packed archive.

    Checks the 4-byte ``SHPK`` container magic first (never the file suffix), so a
    plain ``.h5ad`` returns ``False`` WITHOUT importing cellstream, and ``.shad``,
    ``.csa``, and any future extension resolve by content.

    Fails clearly for a packed archive that cannot be materialized here rather
    than letting the caller fall through to ``anndata.read_h5ad`` and surface a
    cryptic HDF5 error:
    - cellstream not installed + an SHPK packed archive -> ``ImportError``;
    - a packed archive whose manifest layout is not ``"cell"`` (e.g. shard-layout)
      -> ``ValueError``.
    A non-packed file (a real ``.h5ad``) still returns ``False`` so it loads normally.
    """
    p = Path(path)
    if not p.is_file() or not _is_shpk(p):
        return False  # not an SHPK packed archive (plain h5ad / other): no cellstream import
    # An SHPK packed archive: cellstream is needed to read its manifest / materialize it.
    try:
        from cellstream.packed.reader import PackedArchive
    except ImportError:
        raise ImportError(
            f"{p} is a cellstream packed archive, but the cellstream cell-layout reader could "
            "not be imported — 'cellstream' is not installed or is too old (needs the "
            f"cell-layout API, cellstream>={_MIN_CELLSTREAM}). Install or upgrade cellstream to score "
            "cell-layout archives."
        ) from None
    layout = PackedArchive(os.fspath(p)).manifest.get("layout")
    if layout == "cell":
        return True
    raise ValueError(
        f"{p} is a cellstream packed archive with layout={layout!r}, not 'cell'; "
        "load_anndata accepts h5ad or cell-layout archives — shard-layout archives "
        "are scored via the streaming path, not this loader."
    )


def materialize_cell(path: str | os.PathLike) -> ad.AnnData:
    """Materialize a cell-layout archive to a full in-memory AnnData.

    ``cellstream.read_h5ad`` detects the packed cell layout and decodes it via the
    cell bulk reader (``CellStore.to_anndata()`` is a lazy view, not a
    materialization, so it is not used here). Read serially (``n_workers=1``):
    it sidesteps the reader's spawn worker-pool (robust, no multiprocessing
    surprises), and Stage-1 materialize is the small/medium correctness path —
    out-of-core scale is the streaming path, not this loader.
    """
    from cellstream import read_h5ad
    return read_h5ad(os.fspath(path), n_workers=1)


def open_cell_store(path: str | os.PathLike):
    """Open a cell-layout archive as a lazy ``cellstream.cell.CellStore`` (reads only).

    The sole ``cellstream.cell`` import site (mirrors ``materialize_cell``'s isolation
    of the top-level ``cellstream.read_h5ad``). The caller owns the store and must
    ``.close()`` it.

    Fails loudly if cellstream is present but too old for cell-layout streaming — a
    ``CellStore`` without ``gather_rows_adata``, or whose method lacks ``n_threads``
    support, which the streaming scorer calls with no fallback. Without these checks
    the missing capability surfaces as a cryptic ``AttributeError`` or ``TypeError``
    on the first batch; the ``[scale]`` extra pins a new-enough cellstream, so this
    guards only a bring-your-own install.
    """
    import inspect

    from cellstream.cell import open_cell

    store = open_cell(os.fspath(path))
    # Two capability gates. Both name ``_MIN_CELLSTREAM``, not two different versions: under
    # shardad these arrived separately (0.7.0 and 0.7.1) but that provenance is not expressible
    # against cellstream's own version line, whose FIRST public release already has both. The
    # gates stay separate anyway because the ``what`` string is what tells the user which
    # capability is missing; the version is only where to upgrade to. Checked here, once, because
    # the gather sites (cell_eval2.cellstream.CellBatchSource, cell_source.cell_reference/...) call
    # with no fallback -- without this the miss surfaces as an AttributeError/TypeError on the
    # first batch, deep inside the scoring loop.
    missing = None
    if not hasattr(store, "gather_rows_adata"):
        missing = ("gather_rows_adata", _MIN_CELLSTREAM)
    else:
        # Probe by BINDING n_threads as a keyword -- exactly how the gather sites call it --
        # instead of checking for a literal parameter named "n_threads". This accepts a
        # **kwargs-forwarding wrapper, and an UNINSPECTABLE callable (inspect.signature can raise
        # TypeError/ValueError for some C-backed methods) is treated as "unknown -> reject", so
        # the store fails loud+closed here rather than leaking and erroring cryptically on the
        # first batch (Checkpoint-2 codex review).
        try:
            inspect.signature(store.gather_rows_adata).bind_partial(n_threads=1)
        except (TypeError, ValueError):
            missing = ("gather_rows_adata(..., n_threads=...)", _MIN_CELLSTREAM)
    if missing is not None:
        try:
            store.close()
        except Exception:
            pass
        what, version = missing
        raise ImportError(
            "cellstream is installed but too old to stream-score cell-layout archives: "
            f"its CellStore has no '{what}' (present since cellstream {version}). "
            f"Upgrade cellstream to >={version} — e.g. reinstall with `cell_eval2[scale]`."
        )
    return store


def cell_group_spans(store) -> dict[str, tuple[int, int]]:
    """``{label: (start, stop)}`` storage-order row spans for every group in a cell archive.

    THE ONE PLACE cell_eval2 touches a cellstream INTERNAL (``CellStore._load_groups``). It
    lives here because this module is already the sole cellstream importer, so the coupling is
    confined to one file. A public replacement is tracked in Arc's internal cellstream tracker
    (filed as shardad#248 before the public repo existed, so there is no public issue to link);
    replace this when it lands.

    Why not ``store.obs[group_by].value_counts()``: that route is both more expensive (it
    materializes and RETAINS the whole obs parquet on the store) and NOT equivalent. The
    cellstream writer stringifies the group-by labels when it builds group identities, so a
    value-count over the raw column drops NaN/pd.NA (cellstream keeps those as a "nan"/"<NA>"
    group), leaves two keys for distinct raw values cellstream merged into one group by
    stringifying, and on a
    categorical column emits zero-count categories that are not groups at all. The groups record
    is exact by construction.

    Returns ``{}`` when the internal is absent or the archive is ungrouped -- callers then pass
    ``n_rows=None`` to the thread resolver and get the conservative small-read default rather
    than crashing. That fallback is for a MISSING accessor ONLY: errors raised BY
    ``_load_groups`` (corrupt groups JSON, I/O failure) propagate, because swallowing them would
    silently downgrade a broken archive to "unknown group sizes" and score it anyway.
    """
    loader = getattr(store, "_load_groups", None)
    if loader is None:
        return {}
    rec = loader()          # NOT `or {}`: a None/garbage record must surface, not be laundered
    return {str(g["label"]): (int(g["start"]), int(g["stop"])) for g in rec["groups"]}


def cell_reference_row_count(store) -> int | None:
    """Rows in the archive's reference (control) pool, or ``None`` if it has none.

    The reference groups lead contiguously from row 0, so the pool is ``[0, max stop)`` over the
    recorded reference label(s) -- the same arithmetic ``CellStore.read_reference`` itself does.
    Needed because ``read_reference`` takes no row ids, so a caller cannot otherwise size the
    read it is about to issue. ``"reference"`` is a single label, a list of labels, or ``None``.
    Same internal-access caveat as :func:`cell_group_spans` -- the ``None`` fallback is for a
    MISSING accessor only, and loader errors propagate.

    The label lookup is EXACT, mirroring ``read_reference``, which does
    ``max(recs[label]["stop"] for label in ref_labels)`` and raises ``KeyError`` for a reference
    label with no group record. Skipping unknown labels here would return a SMALLER count than
    the read actually gathers -- under-threading the largest read in the pipeline -- and would
    mask an archive inconsistency that the very next call (``read_reference``) raises on anyway.
    """
    loader = getattr(store, "_load_groups", None)
    if loader is None:
        return None
    rec = loader()          # NOT `or {}`: a None/garbage record must surface, not be laundered
    ref = rec.get("reference")
    if ref is None:
        return None
    ref_labels = ref if isinstance(ref, list) else [ref]
    stops = {str(g["label"]): int(g["stop"]) for g in rec["groups"]}
    # No empty-list special case: read_reference does max(...) over ref_labels, so `reference: []`
    # raises ValueError there -- mirror that, don't silently return None for a read that crashes.
    return max(stops[str(label)] for label in ref_labels)   # KeyError/ValueError == read_reference's
