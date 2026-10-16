"""Bounded-memory, contract-pinned parquet I/O for gene-level DE tables.

Ground-truth DEG cache, Phase 1 output contract: hardens the ``write_de`` path that
``cellstream.score_cellstream`` writes ``de_real.parquet`` / ``de_pred.parquet`` through.

Those bytes are the cache-key surface a later commit builds on, so the on-disk contract has to
be EXPLICIT and TESTABLE -- codec, compression level, row-group size, statistics -- rather than
whatever ``polars.DataFrame.write_parquet``'s defaults happen to be today. A silent change to
any of those (e.g. polars bumping its default zstd level, or letting pyarrow pick its own
row-group size) would change size/latency for every cache consumer without changing a single
value in the table, and nothing downstream would notice. ``pyarrow.parquet.ParquetWriter`` is
the writer in this dependency set that takes codec/level/row-group-size and per-column-chunk
statistics as explicit arguments, so this module wraps it rather than ``write_parquet``.

A production ``de_pred`` table is ~16.5M rows (spec), so both writers below are STREAMING: one
frame, or one piece's row group, resident at a time -- never a ``pl.concat`` of everything.

pyarrow is imported lazily inside each function: it is a heavier, opt-in dependency (see the
``scale``/``dev`` comments in pyproject.toml), and importing it at module scope would make every
importer of ``cell_eval2`` pay for it even when this module is never used.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import polars as pl

#: Pinned, not "whatever the writer defaults to" -- see the module docstring.
DE_PARQUET_COMPRESSION = "zstd"
DE_PARQUET_COMPRESSION_LEVEL = 3
DE_PARQUET_ROW_GROUP_ROWS = 250_000


@dataclass(frozen=True)
class ParquetWriteReport:
    """What got written, so a caller (a sidecar, a test) can assert on it without re-reading
    the table's data columns."""

    rows: int
    size_bytes: int
    sha256: str
    columns: tuple[dict, ...]


def sha256_file(path: str) -> str:
    """Streaming SHA-256 (1 MiB chunks) -- a DE table can be tens of millions of rows, so this
    must never hold the file in memory to hash it."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _columns_report(schema) -> tuple[dict, ...]:
    return tuple(
        {"name": schema.field(i).name, "arrow_type": str(schema.field(i).type),
         "nullable": schema.field(i).nullable}
        for i in range(len(schema))
    )


def _report_for(dest: str, schema) -> ParquetWriteReport:
    """Build the report from the CLOSED file on disk (not from in-memory state), so the report
    always describes what a later reader will actually see."""
    import pyarrow.parquet as pq

    with pq.ParquetFile(dest) as pf:
        rows = pf.metadata.num_rows
    return ParquetWriteReport(
        rows=rows, size_bytes=os.path.getsize(dest), sha256=sha256_file(dest),
        columns=_columns_report(schema),
    )


def write_de_parquet(frames: Iterable[pl.DataFrame], dest: str) -> ParquetWriteReport:
    """Write every frame in ``frames``, in order, to ONE parquet file at ``dest`` under the
    pinned contract (module docstring).

    ``dest`` is a plain destination path -- staging and publication are the caller's job
    (``cellstream._StagedOutputs`` stages to a temp path in the same directory and
    ``os.replace``s it into place). No index column is added and nothing is sorted: the
    generated row order IS the artifact.

    The first frame's arrow schema is authoritative; every later frame's schema must
    ``.equals()`` it, or this raises ``ValueError`` naming the offending frame's index and both
    schemas -- a silent schema drift between pieces would otherwise merge into one table with
    columns that mean different things in different row ranges.
    """
    import pyarrow.parquet as pq

    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("write_de_parquet: no frames") from None

    first_tbl = first.to_arrow()
    schema = first_tbl.schema
    writer = pq.ParquetWriter(
        dest, schema, compression=DE_PARQUET_COMPRESSION,
        compression_level=DE_PARQUET_COMPRESSION_LEVEL, write_statistics=True,
    )
    try:
        writer.write_table(first_tbl, row_group_size=DE_PARQUET_ROW_GROUP_ROWS)
        for i, frame in enumerate(it, start=1):
            tbl = frame.to_arrow()
            if not tbl.schema.equals(schema):
                raise ValueError(
                    f"write_de_parquet: frame {i} schema does not match frame 0's schema\n"
                    f"frame 0 schema: {schema}\nframe {i} schema: {tbl.schema}"
                )
            writer.write_table(tbl, row_group_size=DE_PARQUET_ROW_GROUP_ROWS)
    finally:
        writer.close()  # always closed, even on a mid-write ValueError -- no dangling fd/handle
    return _report_for(dest, schema)


def merge_de_pieces(piece_paths: Sequence[str], dest: str, *,
                    report: bool = True) -> ParquetWriteReport | None:
    """Bounded-memory concatenation of the parquet files at ``piece_paths``, in the given
    order, into ONE parquet file at ``dest`` under the same pinned contract.

    Streams one row group at a time (``ParquetFile.read_row_group``) rather than
    ``pl.read_parquet`` + ``pl.concat`` -- the whole point, at a ~16.5M-row de_pred, is that no
    piece's full table and no concatenated whole is ever resident together.

    Order is exactly ``piece_paths``' order (the caller is responsible for it --
    ``score_cellstream`` passes batch generation order, never a glob-then-sort over piece file
    names) -- this function does not re-sort. Every piece's arrow schema must match the first
    piece's, or this raises ``ValueError`` naming the mismatched piece's path.

    ``report=False`` skips the ``ParquetWriteReport`` and returns ``None``: the report hashes the
    whole output, a second full read that a caller with no use for it should not pay for.
    """
    import pyarrow.parquet as pq

    paths = list(piece_paths)
    if not paths:
        raise ValueError("merge_de_pieces: no pieces")

    first_schema = None
    writer = None
    try:
        for p in paths:
            # Each piece's ParquetFile is closed as soon as this piece is merged (the `with`
            # exits at the bottom of the loop body), rather than staying open until the whole
            # merge finishes -- a production merge can walk thousands of pieces.
            with pq.ParquetFile(p) as pf:
                if first_schema is None:
                    first_schema = pf.schema_arrow
                    writer = pq.ParquetWriter(
                        dest, first_schema, compression=DE_PARQUET_COMPRESSION,
                        compression_level=DE_PARQUET_COMPRESSION_LEVEL, write_statistics=True,
                    )
                elif not pf.schema_arrow.equals(first_schema):
                    raise ValueError(
                        f"merge_de_pieces: piece {p!r} schema does not match "
                        f"{paths[0]!r}'s schema\n{paths[0]}: {first_schema}\n{p}: "
                        f"{pf.schema_arrow}"
                    )
                for i in range(pf.num_row_groups):
                    writer.write_table(pf.read_row_group(i),
                                       row_group_size=DE_PARQUET_ROW_GROUP_ROWS)
    finally:
        if writer is not None:
            writer.close()
    return _report_for(dest, first_schema) if report else None


def read_de_parquet_metadata(path: str) -> dict:
    """Metadata-only inspection of the parquet file at ``path`` -- no data columns are loaded --
    for validating the on-disk contract (codec, row-group size) without paying for a read.

    Returns ``{"rows", "row_groups", "columns", "codecs", "max_row_group_rows"}``; ``codecs`` is
    the sorted set of distinct per-column-chunk compression names seen anywhere in the file
    (e.g. ``["ZSTD"]``), not just the first chunk's.
    """
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as pf:
        meta = pf.metadata
        schema = meta.schema.to_arrow_schema()
        columns = [
            {"name": schema.field(i).name, "arrow_type": str(schema.field(i).type),
             "nullable": schema.field(i).nullable}
            for i in range(len(schema))
        ]
        codecs = set()
        max_row_group_rows = 0
        for rg in range(meta.num_row_groups):
            row_group = meta.row_group(rg)
            max_row_group_rows = max(max_row_group_rows, row_group.num_rows)
            for col in range(row_group.num_columns):
                codecs.add(row_group.column(col).compression)
    return {
        "rows": meta.num_rows,
        "row_groups": meta.num_row_groups,
        "columns": columns,
        "codecs": sorted(codecs),
        "max_row_group_rows": max_row_group_rows,
    }
