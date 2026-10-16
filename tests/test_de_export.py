"""Tests for de_export's pinned parquet-write contract (ground-truth-cache Phase 0, commit 3):
explicit zstd level 3, statistics on, 250k-row row groups, bounded-memory piece merge, exact
row-order preservation. CPU-only -- no cellstream/gpudge involved.
"""
import hashlib

import polars as pl
import pytest

pytest.importorskip("pyarrow")

from cell_eval2.de_export import (  # noqa: E402
    DE_PARQUET_ROW_GROUP_ROWS,
    ParquetWriteReport,
    merge_de_pieces,
    read_de_parquet_metadata,
    sha256_file,
    write_de_parquet,
)


def _frame(targets, feature="g0"):
    n = len(targets)
    return pl.DataFrame({
        "target": targets,
        "feature": [feature] * n,
        "log2_fold_change": [float(i) for i in range(n)],
        "p_adj": [0.01 * (i + 1) for i in range(n)],
    })


def _hashlib_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _codecs_of(path) -> set[str]:
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(path).metadata
    codecs = set()
    for rg in range(meta.num_row_groups):
        row_group = meta.row_group(rg)
        for col in range(row_group.num_columns):
            codecs.add(row_group.column(col).compression)
    return codecs


def test_write_de_parquet_writes_one_file_with_pinned_contract(tmp_path):
    frames = [_frame(["A"]), _frame(["B", "C"]), _frame(["D", "E", "F"])]
    dest = str(tmp_path / "de.parquet")

    report = write_de_parquet(frames, dest)

    assert isinstance(report, ParquetWriteReport)
    assert report.rows == 6
    assert report.size_bytes > 0
    assert report.sha256 == _hashlib_sha256(dest) == sha256_file(dest)
    assert [c["name"] for c in report.columns] == ["target", "feature",
                                                    "log2_fold_change", "p_adj"]
    assert _codecs_of(dest) == {"ZSTD"}

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(dest)
    assert pf.metadata.row_group(0).column(0).statistics is not None

    got = pl.read_parquet(dest)
    assert got["target"].to_list() == ["A", "B", "C", "D", "E", "F"]


def test_write_de_parquet_large_frame_uses_bounded_row_groups(tmp_path):
    n = 600_000
    frame = pl.DataFrame({
        "target": [f"t{i}" for i in range(n)],
        "feature": ["g0"] * n,
        "log2_fold_change": [0.0] * n,
        "p_adj": [1.0] * n,
    })
    dest = str(tmp_path / "big.parquet")

    write_de_parquet([frame], dest)
    meta = read_de_parquet_metadata(dest)

    assert meta["rows"] == n
    assert meta["max_row_group_rows"] <= DE_PARQUET_ROW_GROUP_ROWS
    assert meta["row_groups"] >= 3
    assert meta["codecs"] == ["ZSTD"]


def test_write_de_parquet_schema_mismatch_raises(tmp_path):
    frames = [_frame(["A"]), pl.DataFrame({"target": ["B"], "feature": ["g0"]})]
    dest = str(tmp_path / "de.parquet")

    with pytest.raises(ValueError, match=r"frame 1"):
        write_de_parquet(frames, dest)


def test_write_de_parquet_empty_iterable_raises(tmp_path):
    dest = str(tmp_path / "de.parquet")
    with pytest.raises(ValueError, match=r"no frames"):
        write_de_parquet([], dest)


def test_merge_de_pieces_preserves_order_and_contract(tmp_path):
    piece_frames = [_frame(["A", "B"]), _frame(["C"]), _frame(["D", "E"])]
    piece_paths = []
    for i, fr in enumerate(piece_frames):
        p = tmp_path / f"piece_{i}.parquet"
        fr.write_parquet(p)
        piece_paths.append(str(p))

    dest = str(tmp_path / "merged.parquet")
    report = merge_de_pieces(piece_paths, dest)

    expected_targets = [t for fr in piece_frames for t in fr["target"].to_list()]
    got = pl.read_parquet(dest)
    assert got["target"].to_list() == expected_targets
    assert _codecs_of(dest) == {"ZSTD"}
    assert report.rows == len(expected_targets)
    assert report.size_bytes > 0
    assert report.sha256 == sha256_file(dest)


def test_merge_de_pieces_row_order_preserved_for_unsorted_input(tmp_path):
    # Deliberately NOT in target-sorted order -- merge_de_pieces must not sort.
    piece_frames = [_frame(["Z", "M"]), _frame(["A"]), _frame(["Q", "B", "Y"])]
    piece_paths = []
    for i, fr in enumerate(piece_frames):
        p = tmp_path / f"unsorted_{i}.parquet"
        fr.write_parquet(p)
        piece_paths.append(str(p))

    dest = str(tmp_path / "merged_unsorted.parquet")
    merge_de_pieces(piece_paths, dest)

    expected = [t for fr in piece_frames for t in fr["target"].to_list()]
    got = pl.read_parquet(dest)["target"].to_list()
    assert got == expected
    assert got != sorted(got)  # sanity: the input really was unsorted


def test_merge_de_pieces_schema_mismatch_names_the_path(tmp_path):
    ok = tmp_path / "ok.parquet"
    _frame(["A"]).write_parquet(ok)
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"target": ["B"], "feature": ["g0"]}).write_parquet(bad)

    with pytest.raises(ValueError, match=r"bad\.parquet"):
        merge_de_pieces([str(ok), str(bad)], str(tmp_path / "merged.parquet"))


def test_merge_de_pieces_no_pieces_raises(tmp_path):
    with pytest.raises(ValueError, match=r"no pieces"):
        merge_de_pieces([], str(tmp_path / "merged.parquet"))


def test_merge_de_pieces_report_false_skips_the_output_hash(tmp_path, monkeypatch):
    """report=False returns None without hashing the output -- the report's sha256 is a second
    full read of a file that can be a production-sized de_pred (Copilot on #398). The merged
    file itself is unchanged."""
    import cell_eval2.de_export as de_export

    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _frame(["A", "A"]).write_parquet(a)
    _frame(["B"]).write_parquet(b)
    with_report = tmp_path / "with_report.parquet"
    report = merge_de_pieces([str(a), str(b)], str(with_report))
    assert isinstance(report, ParquetWriteReport)

    hashed = []
    monkeypatch.setattr(de_export, "sha256_file", lambda p: hashed.append(p) or "x")
    without = tmp_path / "without_report.parquet"
    assert merge_de_pieces([str(a), str(b)], str(without), report=False) is None
    assert hashed == []
    assert pl.read_parquet(without).equals(pl.read_parquet(with_report))


def test_merge_de_pieces_rewrites_a_large_source_row_group_under_the_contract(tmp_path, monkeypatch):
    """merge_de_pieces is also how score_cellstream rewrites the POLARS-written reference table
    into de_real.parquet, so its own output must meet the contract whatever the source looked
    like: a source row group larger than the bound is split to <= DE_PARQUET_ROW_GROUP_ROWS,
    every column chunk carries statistics, and the writer is opened at zstd level 3. The other
    merge tests use tiny pieces, so dropping row_group_size, statistics or the level from the
    merge would pass them all (Codex review of the #389 split)."""
    import pyarrow.parquet as pq
    from polars.testing import assert_frame_equal
    import cell_eval2.de_export as de_export

    n = DE_PARQUET_ROW_GROUP_ROWS + 50_001
    src = tmp_path / "source.parquet"
    frame = _frame([f"T{i % 7}" for i in range(n)])
    frame.write_parquet(src, row_group_size=n)          # ONE row group, over the bound
    with pq.ParquetFile(src) as pf:
        assert pf.metadata.num_row_groups == 1 and pf.metadata.row_group(0).num_rows == n

    opened = []
    real_writer = pq.ParquetWriter

    def _spy(*a, **k):
        opened.append(k)
        return real_writer(*a, **k)

    monkeypatch.setattr(pq, "ParquetWriter", _spy)
    dest = tmp_path / "merged.parquet"
    merge_de_pieces([str(src)], str(dest))
    monkeypatch.undo()

    assert [(k.get("compression"), k.get("compression_level"), k.get("write_statistics"))
            for k in opened] == [(de_export.DE_PARQUET_COMPRESSION,
                                  de_export.DE_PARQUET_COMPRESSION_LEVEL, True)]
    meta = read_de_parquet_metadata(str(dest))
    assert meta["rows"] == n and meta["codecs"] == ["ZSTD"]
    assert meta["max_row_group_rows"] <= DE_PARQUET_ROW_GROUP_ROWS
    with pq.ParquetFile(dest) as pf:
        assert pf.metadata.num_row_groups >= 2
        for rg in range(pf.metadata.num_row_groups):
            for col in range(pf.metadata.num_columns):
                assert pf.metadata.row_group(rg).column(col).statistics is not None, (rg, col)
    assert_frame_equal(pl.read_parquet(dest), frame, check_exact=True)   # content and order
