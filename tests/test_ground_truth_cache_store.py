"""Storage-half tests for the ground-truth DEG cache (``GroundTruthDEGCache``,
``inspect_entry``, ``doctor``).

CPU-only: ``pyarrow`` is required (the pinned parquet writer), never GPU/cellstream. All
subprocess-based tests use ``sys.executable`` with ``PYTHONPATH`` pointed at this worktree's
``src/`` so the child sees the same in-development ``cell_eval2``.
"""
from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("pyarrow")

from cell_eval2 import ground_truth_cache as gtcache  # noqa: E402
from cell_eval2.config import EvalConfig  # noqa: E402

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


#: Every subprocess bound in this module only has to catch a HANG: a child imports cell_eval2
#: cold, which alone has taken 10-14 s on a loaded node, so a tight bound fails a correct run.
_CHILD_TIMEOUT_S = 120


def _expect_line(proc, expected, *, timeout=_CHILD_TIMEOUT_S):
    """Read one stdout line from ``proc`` and require it to be ``expected``, within ``timeout``
    seconds. A watchdog kills the child at the deadline, which ends any blocked read -- a partial
    line with no newline included -- at EOF. On any other line the child is killed FIRST and its
    stderr collected after: reading the pipes of a child still waiting on this process would
    deadlock."""
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    try:
        line = proc.stdout.readline()
    finally:
        watchdog.cancel()
    if line.strip() == expected:
        return
    proc.kill()
    _out, err = proc.communicate(timeout=10)
    pytest.fail(f"expected {expected!r} from the child within {timeout} s, got {line!r}; "
                f"stderr:\n{err}")


def _env_with_src() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = SRC_DIR if not existing else f"{SRC_DIR}{os.pathsep}{existing}"
    return env


_GOOD_PAYLOAD = "ab" * 32
_VERSIONS = {"cell-eval2": "0.17.0", "gpudge": "0.9.1", "cellstream": "0.10.0"}


def make_de_frame(targets, n_features: int) -> pl.DataFrame:
    """Deterministic 11-column DE frame; sorted targets are laid out contiguously (each
    target's ``n_features`` rows are adjacent, in target-sorted order)."""
    sorted_targets = sorted(targets)
    rows_target: list[str] = []
    rows_feature: list[str] = []
    for t in sorted_targets:
        for i in range(n_features):
            rows_target.append(t)
            rows_feature.append(f"g{i}")
    n = len(rows_target)
    return pl.DataFrame({
        "target": rows_target,
        "feature": rows_feature,
        "target_mean": [float(i) for i in range(n)],
        "ref_mean": [0.5] * n,
        "target_ncells": [10] * n,
        "ref_ncells": [20] * n,
        "log2_fold_change": [float(i % 5) - 2.0 for i in range(n)],
        "p_value": [0.01 * ((i % 10) + 1) for i in range(n)],
        "U": [float(i) for i in range(n)],
        "p_adj": [0.02 * ((i % 10) + 1) for i in range(n)],
        "abs_log2_fold_change": [abs(float(i % 5) - 2.0) for i in range(n)],
    })


def _make_request(
    root: str, policy: str, *, targets=("A", "B", "C"), pert_col: str = "perturbation",
    n_features: int = 4, versions=None, payload_sha256: str = _GOOD_PAYLOAD,
) -> gtcache.GroundTruthDEGCacheRequest:
    fp = gtcache.GroundTruthFingerprint.from_axes(
        payload_sha256=payload_sha256,
        var_names=[f"g{i}" for i in range(n_features)],
        pert_labels=[*targets, "non-targeting"],
        pert_col=pert_col,
    )
    cfg = EvalConfig.v1()
    cfg.pert_col = pert_col
    sem = gtcache.build_real_de_semantics(
        cfg, control="non-targeting", input_type="counts", target_sum=10_000.0,
        device="cuda", de_backend="gpudge", precision="float32", native_gpu_normalize=True,
        versions=versions or _VERSIONS,
    )
    # `root`/`policy` stay on THIS helper's signature (every call site below passes them,
    # mirroring `GroundTruthDEGCache(root, policy=...)`), but `GroundTruthDEGCacheRequest`
    # itself is identity-only now (P1 fix) -- it no longer accepts either.
    return gtcache.GroundTruthDEGCacheRequest(fingerprint=fp, semantics=sem)


def _publish(root: str, *, targets=("A", "B", "C"), n_features=4, **kwargs):
    """Publish one fresh, valid entry; returns (cache, request, result)."""
    req = _make_request(root, "readwrite", targets=targets, n_features=n_features, **kwargs)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    calls = []

    def compute():
        calls.append(1)
        return make_de_frame(targets, n_features)

    result = cache.get_or_compute(req, compute)
    assert result.status == "miss_computed"
    assert len(calls) == 1
    return cache, req, result


# ── 1. readwrite miss -> publish -> hit ─────────────────────────────────────────────


def test_readwrite_miss_publishes_and_second_call_hits(tmp_path):
    root = str(tmp_path / "cache")
    cache, req, r1 = _publish(root, targets=("B", "A", "C"))

    assert r1.status == "miss_computed"
    assert r1.de_real_rows == 12  # 3 targets * 4 features
    expected_entry = os.path.join(root, gtcache.entry_relpath(req.cache_id))
    assert r1.entry_dir == expected_entry
    assert os.path.isdir(expected_entry)
    assert os.path.isfile(os.path.join(expected_entry, "de_real.parquet"))
    manifest_path = os.path.join(expected_entry, "manifest.json")
    assert os.path.isfile(manifest_path)

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["cache_format"] == gtcache.CACHE_FORMAT
    assert manifest["cache_format_version"] == gtcache.CACHE_FORMAT_VERSION
    assert manifest["cache_id"] == req.cache_id
    assert manifest["artifact"]["row_count"] == 12
    assert manifest["artifact"]["sha256"] == r1.de_real_sha256
    assert manifest["perturbations"]["count"] == 3
    assert manifest["perturbations"]["ordered_values_sha256"] == gtcache.sequence_sha256(["A", "B", "C"])
    assert set(manifest["producer"]) >= {"cell_eval2", "gpudge", "cellstream", "polars", "pyarrow"}

    meta = gtcache.read_de_parquet_metadata(os.path.join(expected_entry, "de_real.parquet"))
    assert meta["codecs"] == ["ZSTD"]

    # staging fully cleaned up, lock file present
    parent = os.path.dirname(expected_entry)
    leftovers = [n for n in os.listdir(parent) if n.startswith(".tmp-")]
    assert leftovers == []
    assert os.path.isfile(cache.lock_path(req.cache_id))

    calls = []

    def compute_again():
        calls.append(1)
        return make_de_frame(("B", "A", "C"), 4)

    r2 = cache.get_or_compute(req, compute_again)
    assert r2.status == "hit"
    assert calls == []
    assert r2.table_path == r1.table_path
    t1 = pl.read_parquet(r1.table_path)
    t2 = pl.read_parquet(r2.table_path)
    assert t1.equals(t2)


# ── 2. readonly policy ───────────────────────────────────────────────────────────────


def test_readonly_hit_after_readwrite_publish(tmp_path):
    root = str(tmp_path / "cache")
    _cache, req, r1 = _publish(root)

    ro = gtcache.GroundTruthDEGCache(root, policy="readonly")
    hit = ro.lookup(req)
    assert hit is not None
    assert hit.status == "hit"
    assert hit.de_real_sha256 == r1.de_real_sha256


def test_readonly_miss_raises_and_touches_nothing(tmp_path):
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readonly")
    ro = gtcache.GroundTruthDEGCache(root, policy="readonly")

    before = sorted(os.walk(root))

    def _must_not_be_called():
        raise AssertionError("compute must never be called under readonly")

    with pytest.raises(gtcache.GroundTruthCacheMissError) as exc_info:
        ro.get_or_compute(req, _must_not_be_called)
    msg = str(exc_info.value)
    assert req.cache_id in msg
    assert ro.entry_dir(req.cache_id) in msg

    after = sorted(os.walk(root))
    assert before == after


def test_readonly_ctor_rejects_missing_root(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    with pytest.raises(gtcache.GroundTruthCacheFilesystemError):
        gtcache.GroundTruthDEGCache(missing, policy="readonly")
    assert not os.path.exists(missing)


def test_readonly_ctor_never_creates_versioned_subdir(tmp_path):
    root = str(tmp_path / "cache")
    os.makedirs(root)
    gtcache.GroundTruthDEGCache(root, policy="readonly")
    assert os.listdir(root) == []


# ── 3. corruption matrix ─────────────────────────────────────────────────────────────


def _fresh_entry(tmp_path, name):
    root = str(tmp_path / name)
    cache, req, result = _publish(root)
    return cache, req, result.entry_dir


def _assert_corrupt_and_untouched(cache, req, entry_dir):
    parquet = os.path.join(entry_dir, "de_real.parquet")
    manifest = os.path.join(entry_dir, "manifest.json")
    before = {}
    for p in (parquet, manifest):
        if os.path.exists(p):
            st = os.lstat(p)
            before[p] = (st.st_mtime_ns, st.st_size)
    with pytest.raises(gtcache.GroundTruthCacheCorruptError):
        cache.lookup(req)
    with pytest.raises(gtcache.GroundTruthCacheCorruptError):
        cache.get_or_compute(req, lambda: (_ for _ in ()).throw(AssertionError("must not recompute")))
    for p, (mtime_ns, size) in before.items():
        st = os.lstat(p)
        assert (st.st_mtime_ns, st.st_size) == (mtime_ns, size), f"{p} was modified"


def test_corrupt_flipped_byte_in_parquet(tmp_path):
    cache, req, entry_dir = _fresh_entry(tmp_path, "c1")
    parquet = os.path.join(entry_dir, "de_real.parquet")
    with open(parquet, "r+b") as fh:
        fh.seek(20)
        b = fh.read(1)
        fh.seek(20)
        fh.write(bytes([b[0] ^ 0xFF]))
    _assert_corrupt_and_untouched(cache, req, entry_dir)


def test_corrupt_manifest_row_count(tmp_path):
    cache, req, entry_dir = _fresh_entry(tmp_path, "c2")
    manifest_path = os.path.join(entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["artifact"]["row_count"] += 1
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    _assert_corrupt_and_untouched(cache, req, entry_dir)


def test_corrupt_manifest_cache_id(tmp_path):
    cache, req, entry_dir = _fresh_entry(tmp_path, "c3")
    manifest_path = os.path.join(entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["cache_id"] = "0" * 64
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    _assert_corrupt_and_untouched(cache, req, entry_dir)


@pytest.mark.parametrize("columns", [[None], "target", [{"name": 1, "arrow_type": "large_string"}]],
                         ids=["null-member", "not-a-list", "non-string-name"])
def test_corrupt_manifest_columns_not_objects(tmp_path, columns):
    """A manifest whose artifact.columns is not a list of {name, arrow_type} objects is corrupt
    -- never an AttributeError/TypeError escaping the lookup."""
    cache, req, entry_dir = _fresh_entry(tmp_path, "cols")
    manifest_path = os.path.join(entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["artifact"]["columns"] = columns
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    _assert_corrupt_and_untouched(cache, req, entry_dir)


@pytest.mark.parametrize("path", ["../other.parquet", "/abs/de_real.parquet", "other.parquet", None],
                         ids=["traversal", "absolute", "other-name", "missing"])
def test_corrupt_manifest_artifact_path(tmp_path, path):
    """artifact.path is part of the on-disk format, and validation always reads de_real.parquet:
    a manifest naming any other path contradicts its own entry and is corrupt (Copilot, PR #399)."""
    cache, req, entry_dir = _fresh_entry(tmp_path, "apath")
    manifest_path = os.path.join(entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if path is None:
        manifest["artifact"].pop("path")
    else:
        manifest["artifact"]["path"] = path
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    _assert_corrupt_and_untouched(cache, req, entry_dir)


@pytest.mark.parametrize("kind", ["file", "dangling-symlink", "symlink-out-of-root",
                                  "symlink-to-a-valid-entry-in-root"])
@pytest.mark.parametrize("policy", ["readonly", "readwrite"])
def test_a_non_directory_at_the_entry_path_is_corrupt_not_a_miss(tmp_path, policy, kind):
    """Only real directories published by rename belong at an entry's path. Anything else there --
    a regular file, a dangling symlink, a symlink out of the root, even a symlink to a valid copy
    of the entry inside the root -- is corruption, reported at once and left in place: never a
    readonly miss, never a readwrite compute whose publish then fails, never a hit through an alias
    of mutable storage (Copilot, PR #399)."""
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, policy, targets=("A",), n_features=1)
    entry = gtcache.GroundTruthDEGCache(root, policy="readwrite").entry_dir(req.cache_id)
    os.makedirs(os.path.dirname(entry), exist_ok=True)
    if kind == "file":
        with open(entry, "w", encoding="utf-8"):
            pass
    elif kind == "dangling-symlink":
        os.symlink(str(tmp_path / "nowhere"), entry)
    elif kind == "symlink-out-of-root":
        (tmp_path / "elsewhere").mkdir()
        os.symlink(str(tmp_path / "elsewhere"), entry)
    else:
        gtcache.GroundTruthDEGCache(root, policy="readwrite").get_or_compute(
            req, lambda: make_de_frame(("A",), 1))
        aside = os.path.join(root, "aside", os.path.basename(entry))
        os.makedirs(os.path.dirname(aside))
        os.rename(entry, aside)
        os.symlink(aside, entry)
    cache = gtcache.GroundTruthDEGCache(root, policy=policy)
    with pytest.raises(gtcache.GroundTruthCacheCorruptError):
        cache.lookup(req)
    with pytest.raises(gtcache.GroundTruthCacheCorruptError):
        cache.get_or_compute(req, lambda: (_ for _ in ()).throw(AssertionError("must not compute")))
    assert os.path.lexists(entry)
    assert gtcache.inspect_entry(req.cache_id, root=root)["valid"] is False


@pytest.mark.parametrize("column, values", [
    ("feature", lambda n: list(range(n))),
    ("p_adj", lambda n: ["0.5"] * n),
    ("log2_fold_change", lambda n: ["1.0"] * n),
], ids=["int-feature", "string-p_adj", "string-lfc"])
def test_required_de_columns_must_have_de_types(tmp_path, column, values):
    """The manifest's column types come from the very frame being cached, so matching them proves
    nothing about the frame itself. The required DE columns are checked on their own: target and
    feature are strings, log2_fold_change and p_adj floating point. A compute() returning anything
    else fails the write's self-validation and publishes nothing (Copilot, PR #399). (A non-string
    target never gets that far: the perturbation-universe digest refuses it first.)"""
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readwrite", targets=("A",), n_features=2)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    frame = make_de_frame(("A",), 2)
    frame = frame.with_columns(pl.Series(column, values(frame.height)))
    with pytest.raises(gtcache.GroundTruthCacheFilesystemError, match="parquet_required_column_types"):
        cache.get_or_compute(req, lambda: frame)
    assert not os.path.lexists(cache.entry_dir(req.cache_id))


@pytest.mark.parametrize("path, root, inside", [
    ("/ground-truth-deg/v1/ab/x", "/", True),
    ("/a/b/c", "/a/b", True),
    ("/a/b", "/a/b", True),
    ("/a/bc/x", "/a/b", False),
    ("/elsewhere", "/a/b", False),
], ids=["fs-root", "below", "equal", "sibling-prefix", "outside"])
def test_containment_is_by_path_component(path, root, inside):
    """Containment compared ``root + os.sep`` as a string prefix, which refused every entry under
    a filesystem-root cache root ("/" + "/" is "//"). It is by path components now, in validation
    and in entry_dir alike, and a sibling sharing a prefix is still outside (Copilot, PR #399)."""
    from cell_eval2.ground_truth_cache import integrity

    cid = "ab" * 32
    if inside:
        assert integrity._check_containment(path, root, cid) == path
    else:
        with pytest.raises(gtcache.GroundTruthCacheCorruptError, match="containment"):
            integrity._check_containment(path, root, cid)
    # entry_dir reads nothing but self.root, so an instance rooted at "/" needs no directory there.
    cache = object.__new__(gtcache.GroundTruthDEGCache)
    cache.root = "/"
    assert cache.entry_dir(cid) == f"/ground-truth-deg/v{gtcache.CACHE_FORMAT_VERSION}/ab/{cid}"


@pytest.mark.parametrize("field", ["cache_format_version", "artifact.row_count",
                                   "perturbations.count"])
def test_corrupt_manifest_boolean_in_an_integer_field(tmp_path, field):
    """True == 1, so a manifest with `true` in an integer field matched a version of 1, or a
    one-row / one-target entry's counts. A malformed manifest is corrupt (Copilot, PR #399)."""
    root = str(tmp_path / "cache")
    cache, req, result = _publish(root, targets=("A",), n_features=1)   # one row, one target
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    node = manifest
    *parents, leaf = field.split(".")
    for k in parents:
        node = node[k]
    assert node[leaf] == 1
    node[leaf] = True
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    _assert_corrupt_and_untouched(cache, req, result.entry_dir)


def test_strip_trailing_strips_the_alternate_separator_too(monkeypatch):
    """On Windows "/" (os.altsep) separates paths as well as "\\"; the entry-type check must strip
    both, or `link/` would still be followed (Copilot, PR #399). Simulated with altsep="\\"."""
    from cell_eval2.ground_truth_cache import integrity

    monkeypatch.setattr(os, "altsep", "\\")
    assert integrity._strip_trailing("a/b\\") == "a/b"
    assert integrity._strip_trailing("a/b\\/.") == "a/b"


@pytest.mark.parametrize("path, expected", [
    ("C:/", "C:/"), ("C:\\", "C:\\"), ("C:\\.", "C:\\"), ("C:/./", "C:/"),
    ("C:\\cache\\entry\\", "C:\\cache\\entry"), ("C:/cache/entry/.", "C:/cache/entry"),
    ("\\\\server\\share\\", "\\\\server\\share\\"), ("\\\\server\\share\\x\\.", "\\\\server\\share\\x"),
])
def test_strip_trailing_keeps_windows_drive_roots(monkeypatch, path, expected):
    """Under Windows path rules, stripping must keep a drive or UNC root's separator: "C:" alone
    means the current directory on drive C, a different directory (codex, PR #399)."""
    import ntpath

    from cell_eval2.ground_truth_cache import integrity

    with monkeypatch.context() as m:
        m.setattr(os, "path", ntpath)
        m.setattr(os, "sep", "\\")
        m.setattr(os, "altsep", "/")
        got = integrity._strip_trailing(path)
    assert got == expected


def test_corrupt_missing_manifest(tmp_path):
    cache, req, entry_dir = _fresh_entry(tmp_path, "c4")
    os.remove(os.path.join(entry_dir, "manifest.json"))
    with pytest.raises(gtcache.GroundTruthCacheCorruptError):
        cache.lookup(req)


def test_corrupt_parquet_replaced_with_symlink(tmp_path):
    cache, req, entry_dir = _fresh_entry(tmp_path, "c5")
    parquet = os.path.join(entry_dir, "de_real.parquet")
    copy_target = os.path.join(str(tmp_path), "copy_of_de_real.parquet")
    with open(parquet, "rb") as src, open(copy_target, "wb") as dst:
        dst.write(src.read())
    os.remove(parquet)
    os.symlink(copy_target, parquet)
    with pytest.raises(gtcache.GroundTruthCacheCorruptError):
        cache.lookup(req)
    assert stat.S_ISLNK(os.lstat(parquet).st_mode)


def test_corrupt_wrong_codec(tmp_path):
    cache, req, entry_dir = _fresh_entry(tmp_path, "c6")
    parquet = os.path.join(entry_dir, "de_real.parquet")
    frame = pl.read_parquet(parquet)
    # polars' own default compression is also zstd -- use snappy so the codec check has
    # something to actually catch (same logical content, different codec).
    frame.write_parquet(parquet, compression="snappy")
    # Update the manifest's artifact sha256/size so the EARLIER artifact_bytes check (which
    # runs before the codec check) passes -- otherwise the byte-identity check would catch
    # the rewrite first and the codec check would never run. This isolates check 6.
    manifest_path = os.path.join(entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["artifact"]["sha256"] = gtcache.sha256_file(parquet)
    manifest["artifact"]["size_bytes"] = os.path.getsize(parquet)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    with pytest.raises(gtcache.GroundTruthCacheCorruptError, match="parquet_codec"):
        cache.lookup(req)


def test_corrupt_duplicate_target_feature_row(tmp_path):
    root = str(tmp_path / "c7")
    req = _make_request(root, "readwrite", targets=("A", "B"), n_features=2)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    entry_dir = cache.entry_dir(req.cache_id)
    os.makedirs(entry_dir)
    frame = make_de_frame(("A", "B"), 2)
    dup_row = frame.head(1)
    frame_with_dup = pl.concat([frame, dup_row])
    parquet_path = os.path.join(entry_dir, "de_real.parquet")
    report = gtcache.write_de_parquet([frame_with_dup], parquet_path)
    targets = sorted(frame_with_dup["target"].unique().to_list())
    manifest = {
        "cache_format": gtcache.CACHE_FORMAT,
        "cache_format_version": gtcache.CACHE_FORMAT_VERSION,
        "cache_id": req.cache_id,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "ground_truth_fingerprint": dict(req.fingerprint.document),
        "ground_truth_fingerprint_sha256": req.fingerprint.sha256,
        "de_semantics": dict(req.semantics.document),
        "de_semantics_sha256": req.semantics.sha256,
        "artifact": {
            "path": "de_real.parquet", "sha256": report.sha256, "size_bytes": report.size_bytes,
            "row_count": report.rows, "columns": [dict(c) for c in report.columns],
            "compression": "zstd", "compression_level": gtcache.DE_PARQUET_COMPRESSION_LEVEL,
            "target_row_group_rows": gtcache.DE_PARQUET_ROW_GROUP_ROWS,
        },
        "perturbations": {"count": len(targets), "ordered_values_sha256": gtcache.sequence_sha256(targets)},
        "producer": {"cell_eval2": "0.17.0"},
    }
    with open(os.path.join(entry_dir, "manifest.json"), "wb") as fh:
        fh.write(gtcache.canonical_json_bytes(manifest))
        fh.write(b"\n")
    with pytest.raises(gtcache.GroundTruthCacheCorruptError, match="duplicate_target_feature"):
        cache.lookup(req)


# ── 4. entry_dir rejections ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_id", ["../x", "A" * 64, "0" * 63])
def test_entry_dir_rejects_bad_cache_id(tmp_path, bad_id):
    cache = gtcache.GroundTruthDEGCache(str(tmp_path), policy="readwrite")
    with pytest.raises(gtcache.GroundTruthContractError):
        cache.entry_dir(bad_id)


def test_entry_dir_rejects_symlinked_prefix_dir_escaping_root(tmp_path):
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = gtcache.GroundTruthDEGCache(str(root), policy="readwrite")
    cache_id = "a" * 64
    prefix_dir = os.path.dirname(os.path.join(str(root), gtcache.entry_relpath(cache_id)))
    os.makedirs(os.path.dirname(prefix_dir), exist_ok=True)
    os.symlink(str(outside), prefix_dir)
    with pytest.raises(gtcache.GroundTruthCacheFilesystemError):
        cache.entry_dir(cache_id)


# ── 5. two ids coexist ───────────────────────────────────────────────────────────────


def test_two_entries_coexist(tmp_path):
    root = str(tmp_path / "cache")
    cache1, req1, r1 = _publish(root, targets=("A", "B"), n_features=2)
    cache2, req2, r2 = _publish(root, targets=("X", "Y", "Z"), n_features=2)
    assert req1.cache_id != req2.cache_id
    assert os.path.isdir(r1.entry_dir)
    assert os.path.isdir(r2.entry_dir)
    hit1 = cache1.lookup(req1)
    hit2 = cache2.lookup(req2)
    assert hit1.de_real_rows == 4
    assert hit2.de_real_rows == 6


# ── 6. hit_after_wait ────────────────────────────────────────────────────────────────


_CHILD_SLOW_PUBLISH_SRC = """
import sys, time
sys.path.insert(0, sys.argv[1])
from cell_eval2 import ground_truth_cache as gtcache
from cell_eval2.config import EvalConfig

root = sys.argv[2]

def _make_request(root, policy):
    fp = gtcache.GroundTruthFingerprint.from_axes(
        payload_sha256="ab" * 32, var_names=["g0", "g1", "g2", "g3"],
        pert_labels=["A", "B", "C", "non-targeting"], pert_col="perturbation",
    )
    cfg = EvalConfig.v1()
    cfg.pert_col = "perturbation"
    sem = gtcache.build_real_de_semantics(
        cfg, control="non-targeting", input_type="counts", target_sum=10000.0, device="cuda",
        de_backend="gpudge", precision="float32", native_gpu_normalize=True,
        versions={"cell-eval2": "0.17.0", "gpudge": "0.9.1", "cellstream": "0.10.0"},
    )
    return gtcache.GroundTruthDEGCacheRequest(fingerprint=fp, semantics=sem)


def make_de_frame(targets, n_features):
    import polars as pl
    sorted_targets = sorted(targets)
    rt, rf = [], []
    for t in sorted_targets:
        for i in range(n_features):
            rt.append(t)
            rf.append(f"g{i}")
    n = len(rt)
    return pl.DataFrame({
        "target": rt, "feature": rf, "target_mean": [float(i) for i in range(n)],
        "ref_mean": [0.5] * n, "target_ncells": [10] * n, "ref_ncells": [20] * n,
        "log2_fold_change": [float(i % 5) - 2.0 for i in range(n)],
        "p_value": [0.01 * ((i % 10) + 1) for i in range(n)], "U": [float(i) for i in range(n)],
        "p_adj": [0.02 * ((i % 10) + 1) for i in range(n)],
        "abs_log2_fold_change": [abs(float(i % 5) - 2.0) for i in range(n)],
    })


def gated_compute():
    print("locked", flush=True)                      # holding the entry's lock, nothing published
    assert sys.stdin.readline().strip() == "go"      # until the parent is about to block on it
    return make_de_frame(["A", "B", "C"], 4)


req = _make_request(root, "readwrite")
cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
result = cache.get_or_compute(req, gated_compute)
print(result.status, flush=True)
"""


def test_hit_after_wait(tmp_path, monkeypatch):
    """Deterministic: the child computes while holding the entry's lock and publishes only when
    this process -- having missed -- is about to block on that lock, so the recheck under the
    lock is what finds the entry, however the two processes are scheduled. Every wait is
    bounded: the readiness line by ``_expect_line``, the contended lookup by a watchdog."""
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readwrite")

    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SLOW_PUBLISH_SRC, SRC_DIR, root],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_env_with_src(),
    )
    try:
        _expect_line(child, "locked")

        real_fcntl = gtcache.store._flock_module()

        class _ReleaseChildThenLock:
            def __getattr__(self, name):
                return getattr(real_fcntl, name)

            def flock(self, fd, op):
                if op == real_fcntl.LOCK_EX:
                    child.stdin.write("go\n")
                    child.stdin.flush()
                return real_fcntl.flock(fd, op)

        monkeypatch.setattr(gtcache.store, "_flock_module", lambda: _ReleaseChildThenLock())
        cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
        calls = []

        def must_not_be_called():
            calls.append(1)
            return make_de_frame(("A", "B", "C"), 4)

        # A child stalled while holding the lock would block this call for ever: the watchdog
        # kills it at the deadline, which releases the lock, and the assertions below then fail.
        watchdog = threading.Timer(_CHILD_TIMEOUT_S, child.kill)
        watchdog.start()
        try:
            result = cache.get_or_compute(req, must_not_be_called)
        finally:
            watchdog.cancel()
        assert result.status == "hit_after_wait"
        assert result.lock_wait_seconds > 0
        assert calls == []

        out, err = child.communicate(timeout=_CHILD_TIMEOUT_S)
        assert child.returncode == 0, err
        assert out.strip() == "miss_computed"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


# ── 7. concurrent cold writers ───────────────────────────────────────────────────────


_CHILD_CONCURRENT_SRC = """
import os, sys, time
sys.path.insert(0, sys.argv[1])
from cell_eval2 import ground_truth_cache as gtcache
from cell_eval2.config import EvalConfig

root, counter_path, role = sys.argv[2], sys.argv[3], sys.argv[4]

fp = gtcache.GroundTruthFingerprint.from_axes(
    payload_sha256="ab" * 32, var_names=["g0", "g1"], pert_labels=["A", "non-targeting"],
    pert_col="perturbation",
)
cfg = EvalConfig.v1()
cfg.pert_col = "perturbation"
sem = gtcache.build_real_de_semantics(
    cfg, control="non-targeting", input_type="counts", target_sum=10000.0, device="cuda",
    de_backend="gpudge", precision="float32", native_gpu_normalize=True,
    versions={"cell-eval2": "0.17.0", "gpudge": "0.9.1", "cellstream": "0.10.0"},
)
req = gtcache.GroundTruthDEGCacheRequest(fingerprint=fp, semantics=sem)


def compute():
    import polars as pl
    line = f"{os.getpid()}\\n"
    fd = os.open(counter_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, line.encode())
    os.close(fd)
    if role == "first":
        print("computing", flush=True)                  # holding the entry's lock
        assert sys.stdin.readline().strip() == "go"     # until "second" is blocked on it
    return pl.DataFrame({
        "target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0], "p_adj": [0.01],
    })


if role == "second":
    real_fcntl = gtcache.store._flock_module()

    class _AnnounceThenLock:
        def __getattr__(self, name):
            return getattr(real_fcntl, name)

        def flock(self, fd, op):
            if op == real_fcntl.LOCK_EX:
                print("waiting", flush=True)            # missed, about to block on the lock
            return real_fcntl.flock(fd, op)

    gtcache.store._flock_module = lambda: _AnnounceThenLock()

cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
result = cache.get_or_compute(req, compute)
print(result.status, flush=True)
"""


def test_concurrent_cold_writers_compute_exactly_once(tmp_path):
    """Deterministic: "first" computes while holding the entry's lock; "second" starts, misses,
    and blocks on that lock; only then is "first" released to publish. Exactly one compute, and
    the second writer's recheck under the lock finds the entry."""
    root = str(tmp_path / "cache")
    os.makedirs(root)
    counter_path = str(tmp_path / "counter.txt")
    env = _env_with_src()

    def spawn(role):
        return subprocess.Popen(
            [sys.executable, "-c", _CHILD_CONCURRENT_SRC, SRC_DIR, root, counter_path, role],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env,
        )

    first, second = spawn("first"), None
    try:
        _expect_line(first, "computing")
        second = spawn("second")
        _expect_line(second, "waiting")
        first.stdin.write("go\n")
        first.stdin.flush()
        out1, err1 = first.communicate(timeout=_CHILD_TIMEOUT_S)
        out2, err2 = second.communicate(timeout=_CHILD_TIMEOUT_S)
        assert first.returncode == 0, err1
        assert second.returncode == 0, err2
        assert (out1.strip(), out2.strip()) == ("miss_computed", "hit_after_wait")
    finally:
        for child in (first, second):
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    with open(counter_path, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln]
    assert len(lines) == 1


# ── 8. kill / fault injection ────────────────────────────────────────────────────────


_CHILD_FAULT_SRC = """
import os, sys
sys.path.insert(0, sys.argv[1])
from cell_eval2 import ground_truth_cache as gtcache
from cell_eval2.config import EvalConfig
import polars as pl

root = sys.argv[2]

fp = gtcache.GroundTruthFingerprint.from_axes(
    payload_sha256="ab" * 32, var_names=["g0"], pert_labels=["A", "non-targeting"],
    pert_col="perturbation",
)
cfg = EvalConfig.v1()
cfg.pert_col = "perturbation"
sem = gtcache.build_real_de_semantics(
    cfg, control="non-targeting", input_type="counts", target_sum=10000.0, device="cuda",
    de_backend="gpudge", precision="float32", native_gpu_normalize=True,
    versions={"cell-eval2": "0.17.0", "gpudge": "0.9.1", "cellstream": "0.10.0"},
)
req = gtcache.GroundTruthDEGCacheRequest(fingerprint=fp, semantics=sem)
print(req.cache_id, flush=True)

cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
cache.get_or_compute(req, lambda: pl.DataFrame(
    {"target": ["A"], "feature": ["g0"], "log2_fold_change": [1.0], "p_adj": [0.01]}
))
print("finished-without-fault", flush=True)
"""


def _run_fault_child(root, fault_stage):
    env = _env_with_src()
    env["CELL_EVAL2_GTC_FAULT"] = fault_stage
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_FAULT_SRC, SRC_DIR, root],
        capture_output=True, text=True, env=env, timeout=_CHILD_TIMEOUT_S,
    )
    lines = proc.stdout.strip().splitlines()
    cache_id = lines[0] if lines else None
    return proc.returncode, cache_id, proc.stderr


@pytest.mark.parametrize("stage", ["after_parquet", "after_manifest", "before_rename"])
def test_fault_injection_leaves_staging_and_no_entry(tmp_path, stage):
    root = str(tmp_path / "cache")
    os.makedirs(root)
    returncode, cache_id, stderr = _run_fault_child(root, stage)
    assert returncode == 3, stderr
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    entry_dir = cache.entry_dir(cache_id)
    assert not os.path.isdir(entry_dir)
    parent = os.path.dirname(entry_dir)
    staging_dirs = [n for n in os.listdir(parent) if n.startswith(f".tmp-{cache_id}-")]
    assert len(staging_dirs) == 1

    # A normal run with stale_staging_age_s=0 cleans up the orphaned staging dir and
    # publishes normally.
    cache2 = gtcache.GroundTruthDEGCache(root, policy="readwrite", stale_staging_age_s=0)
    req = _make_request(root, "readwrite", targets=("A",), n_features=1)
    assert req.cache_id == cache_id
    result = cache2.get_or_compute(req, lambda: make_de_frame(("A",), 1))
    assert result.status == "miss_computed"
    staging_dirs_after = [n for n in os.listdir(parent) if n.startswith(f".tmp-{cache_id}-")]
    assert staging_dirs_after == []
    assert os.path.isdir(entry_dir)


def test_fault_injection_after_rename_leaves_valid_entry(tmp_path):
    root = str(tmp_path / "cache")
    os.makedirs(root)
    returncode, cache_id, stderr = _run_fault_child(root, "after_rename")
    assert returncode == 3, stderr
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    entry_dir = cache.entry_dir(cache_id)
    assert os.path.isdir(entry_dir)
    req = _make_request(root, "readwrite", targets=("A",), n_features=1)
    assert req.cache_id == cache_id
    hit = cache.lookup(req)
    assert hit is not None
    assert hit.status == "hit"


# ── 9. flock failure ─────────────────────────────────────────────────────────────────

# `fcntl` is imported LAZILY now (`store._flock_module`, only inside the write/doctor
# paths -- see module split, P1 fix "do not make every cell_eval2 import depend on POSIX
# fcntl"), so this test imports the real module directly (this venv is POSIX) rather than
# reaching for a `gtcache.fcntl` module attribute that no longer exists.
import fcntl as _real_fcntl  # noqa: E402


class _RaisingFcntl:
    LOCK_EX = _real_fcntl.LOCK_EX
    LOCK_UN = _real_fcntl.LOCK_UN
    LOCK_NB = _real_fcntl.LOCK_NB

    @staticmethod
    def flock(_fd, _op):
        raise OSError(errno.ENOLCK, "no locks available (injected)")


def test_flock_failure_wraps_and_leaves_no_staging(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readwrite", targets=("A",), n_features=1)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    # The supported patch point post-split: `store._flock_module` (bare-name lookup inside
    # `store.get_or_compute`), not a `fcntl` attribute on the package itself.
    monkeypatch.setattr(gtcache.store, "_flock_module", lambda: _RaisingFcntl)

    with pytest.raises(gtcache.GroundTruthCacheFilesystemError, match="doctor"):
        cache.get_or_compute(req, lambda: make_de_frame(("A",), 1))

    entry_dir = cache.entry_dir(req.cache_id)
    parent = os.path.dirname(entry_dir)
    assert not os.path.isdir(entry_dir)
    if os.path.isdir(parent):
        assert [n for n in os.listdir(parent) if n.startswith(".tmp-")] == []


def test_flock_module_unavailable_raises_filesystem_error_mentioning_fcntl(tmp_path, monkeypatch):
    """P1 fix: on a platform without `fcntl` (e.g. Windows), `_flock_module()` raises a
    targeted `GroundTruthCacheFilesystemError` naming `fcntl` -- not an `ImportError` deep
    inside `get_or_compute`, and not at `import cell_eval2` time (see
    test_ground_truth_cache_identity.py's subprocess-based import test)."""
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readwrite", targets=("A",), n_features=1)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")

    def _raise_import_error():
        raise gtcache.GroundTruthCacheFilesystemError(
            "POSIX advisory locks (fcntl) are unavailable on this platform"
        )

    monkeypatch.setattr(gtcache.store, "_flock_module", _raise_import_error)

    with pytest.raises(gtcache.GroundTruthCacheFilesystemError, match="fcntl"):
        cache.get_or_compute(req, lambda: make_de_frame(("A",), 1))

    entry_dir = cache.entry_dir(req.cache_id)
    parent = os.path.dirname(entry_dir)
    assert not os.path.isdir(entry_dir)
    # `_flock_module()` is resolved before the lock file or staging dir is created, so the
    # 2-hex-prefix directory (created lazily on first write) never even appears here.
    assert not os.path.isdir(parent)


# ── 10. compute raising ──────────────────────────────────────────────────────────────


def test_compute_raising_cleans_up_staging(tmp_path):
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readwrite", targets=("A",), n_features=1)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")

    def boom():
        raise ValueError("computation exploded")

    with pytest.raises(ValueError, match="computation exploded"):
        cache.get_or_compute(req, boom)

    entry_dir = cache.entry_dir(req.cache_id)
    assert not os.path.isdir(entry_dir)
    parent = os.path.dirname(entry_dir)
    if os.path.isdir(parent):
        assert [n for n in os.listdir(parent) if n.startswith(".tmp-")] == []


@pytest.mark.parametrize("where", ["compute", "after_parquet", "after_manifest"])
def test_an_interrupt_cleans_up_staging(tmp_path, monkeypatch, where):
    """A KeyboardInterrupt -- during compute, or after the parquet or the manifest is written --
    removes the unpublished staging directory too, not only an Exception does. Hits never sweep
    stale staging, so debris left here could outlive every later run (codex, checkpoint 2)."""
    root = str(tmp_path / "cache")
    os.makedirs(root)
    req = _make_request(root, "readwrite", targets=("A",), n_features=1)
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")

    def compute():
        if where == "compute":
            raise KeyboardInterrupt
        return make_de_frame(("A",), 1)

    def _interrupt_at(stage):
        if stage == where:
            raise KeyboardInterrupt

    monkeypatch.setattr(gtcache.store, "_fault_hook", _interrupt_at)
    with pytest.raises(KeyboardInterrupt):
        cache.get_or_compute(req, compute)

    entry_dir = cache.entry_dir(req.cache_id)
    assert not os.path.isdir(entry_dir)
    parent = os.path.dirname(entry_dir)
    assert [n for n in os.listdir(parent) if n.startswith(".tmp-")] == []


# ── 11. doctor ────────────────────────────────────────────────────────────────────────


def test_doctor_reports_ok(tmp_path):
    report = gtcache.doctor(str(tmp_path))
    assert report["ok"] is True
    assert report["root"] == str(tmp_path)
    names = {c["name"] for c in report["checks"]}
    assert {"mkdir", "flock_exclusion", "parquet_zstd_roundtrip", "atomic_rename", "cleanup"} <= names
    assert all(c["ok"] for c in report["checks"])
    # doctor cleans up fully after itself
    versioned = os.path.join(str(tmp_path), "ground-truth-deg", f"v{gtcache.CACHE_FORMAT_VERSION}")
    if os.path.isdir(versioned):
        assert [n for n in os.listdir(versioned) if n.startswith(".doctor-")] == []


def test_doctor_counts_only_contention_as_lock_exclusion(tmp_path, monkeypatch):
    """Only EACCES/EAGAIN from the parent's non-blocking flock show the child's lock excluding it.
    Any other error (ENOLCK on a mount without a lock daemon, ENOTSUP, EIO) is a locking failure,
    and doctor must not report it as healthy exclusion (Copilot, PR #399)."""
    import errno as _errno
    import fcntl as _fcntl

    class _NoLocks:
        LOCK_EX, LOCK_NB, LOCK_UN = _fcntl.LOCK_EX, _fcntl.LOCK_NB, _fcntl.LOCK_UN

        @staticmethod
        def flock(fd, op):
            raise OSError(_errno.ENOLCK, "No locks available")

    monkeypatch.setattr(gtcache.store, "_flock_module", lambda: _NoLocks)
    report = gtcache.doctor(str(tmp_path))
    assert report["ok"] is False
    assert not any(c["name"] == "flock_exclusion" and c["ok"] for c in report["checks"])
    probe = [c for c in report["checks"] if c["name"] == "flock_probe"]
    assert probe and not probe[0]["ok"] and "No locks available" in probe[0]["detail"]


def test_fsync_dir_tolerates_only_unsupported_fsync(tmp_path, monkeypatch):
    """Directory fsync is best-effort only where a filesystem refuses to fsync a directory at all
    (EINVAL/ENOTSUP), as its docstring says. Failing to OPEN the directory is not such a case and
    must surface, not silently skip the durability step (Copilot, PR #399)."""
    import errno as _errno

    with pytest.raises(FileNotFoundError):
        gtcache.store._fsync_dir(str(tmp_path / "missing"))

    def _unsupported(fd):
        raise OSError(_errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(os, "fsync", _unsupported)
    gtcache.store._fsync_dir(str(tmp_path))   # tolerated


def test_readwrite_constructor_creates_nothing_through_a_symlinked_namespace(tmp_path):
    """A <root>/ground-truth-deg symlink pointing outside the root must not make the readwrite
    constructor create v1/ out there: containment is checked before anything is created
    (Copilot, PR #399)."""
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), str(root / "ground-truth-deg"))
    with pytest.raises(gtcache.GroundTruthCacheFilesystemError, match="outside the cache root"):
        gtcache.GroundTruthDEGCache(str(root), policy="readwrite")
    assert os.listdir(outside) == []


def test_doctor_refuses_a_namespace_that_escapes_the_root(tmp_path):
    """doctor makes its scratch directory under <root>/ground-truth-deg/v1. A symlinked component
    pointing outside the root -- a namespace GroundTruthDEGCache refuses -- must fail doctor's
    mkdir probe and create nothing out there, not report a healthy filesystem (Copilot, PR #399)."""
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), str(root / "ground-truth-deg"))
    report = gtcache.doctor(str(root))
    assert report["ok"] is False
    mkdir = [c for c in report["checks"] if c["name"] == "mkdir"]
    assert mkdir and not mkdir[0]["ok"] and "outside the cache root" in mkdir[0]["detail"]
    assert os.listdir(outside) == []


def test_doctor_canonicalizes_the_root_before_creating(tmp_path):
    """doctor works under the same canonical root as the store: a root spelled
    <tmp>/results/de_real.parquet/../cache must not leave a stray de_real.parquet directory --
    doctor's cleanup removes only its own scratch directory (codex, PR #399)."""
    (tmp_path / "results").mkdir()
    root = os.path.join(str(tmp_path), "results", "de_real.parquet", "..", "cache")
    report = gtcache.doctor(root)
    assert report["ok"] is True
    assert not os.path.lexists(tmp_path / "results" / "de_real.parquet")


def test_readwrite_constructor_canonicalizes_the_root_before_creating(tmp_path):
    """A root spelled through a missing component and `..` -- <tmp>/results/de_real.parquet/../cache
    -- names <tmp>/results/cache, which is what score_cellstream's output-alias check resolves it
    to. Creating it must not leave a stray de_real.parquet directory on the way, which would
    surface only when the run publishes (Copilot, PR #399)."""
    (tmp_path / "results").mkdir()
    root = os.path.join(str(tmp_path), "results", "de_real.parquet", "..", "cache")
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    assert not os.path.lexists(tmp_path / "results" / "de_real.parquet")
    expected = os.path.realpath(str(tmp_path / "results" / "cache"))
    assert cache.root == expected
    assert os.path.isdir(os.path.join(expected, "ground-truth-deg", f"v{gtcache.CACHE_FORMAT_VERSION}"))


def test_doctor_bounds_the_lock_holders_readiness_wait(tmp_path, monkeypatch):
    """A lock-holder child that never reports ready -- its blocking flock hanging on a broken
    mount, the very condition doctor diagnoses -- must not hang doctor: the readiness wait is
    bounded, the child is killed, and the probe is recorded as failed (Copilot, PR #399). The
    stand-in child sleeps 30 s, so an unbounded wait would take that long."""
    import time

    monkeypatch.setattr(gtcache.store, "_DOCTOR_LOCK_CHILD_SRC", "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(gtcache.store, "_DOCTOR_READY_TIMEOUT_S", 2.0, raising=False)
    t0 = time.monotonic()
    report = gtcache.doctor(str(tmp_path))
    elapsed = time.monotonic() - t0
    assert elapsed < 20, f"doctor waited {elapsed:.1f} s for a child that never got the lock"
    assert report["ok"] is False
    probe = [c for c in report["checks"] if c["name"] == "flock_probe"]
    assert probe and not probe[0]["ok"] and "did not report ready" in probe[0]["detail"]


# ── 12. inspect_entry ────────────────────────────────────────────────────────────────


def test_inspect_entry_valid(tmp_path):
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    info = gtcache.inspect_entry(req.cache_id, root=root)
    assert info["valid"] is True
    assert info["problems"] == []
    assert info["cache_id"] == req.cache_id
    assert info["entry_dir"] == result.entry_dir
    assert "ground_truth_fingerprint" not in info["manifest"]
    assert "de_semantics" not in info["manifest"]
    assert info["manifest"]["ground_truth_fingerprint_sha256"] == req.fingerprint.sha256
    assert info["manifest"]["de_semantics_sha256"] == req.semantics.sha256


def test_inspect_entry_invalid_after_corruption(tmp_path):
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    parquet = os.path.join(result.entry_dir, "de_real.parquet")
    with open(parquet, "r+b") as fh:
        fh.seek(10)
        b = fh.read(1)
        fh.seek(10)
        fh.write(bytes([b[0] ^ 0xFF]))

    info = gtcache.inspect_entry(req.cache_id, root=root)
    assert info["valid"] is False
    assert len(info["problems"]) == 1
    assert "artifact_sha256" in info["problems"][0] or "artifact_size" in info["problems"][0]


def test_inspect_entry_by_directory_path(tmp_path):
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    info = gtcache.inspect_entry(result.entry_dir, root=root)
    assert info["valid"] is True
    assert info["cache_id"] == req.cache_id


@pytest.mark.parametrize("suffix", ["", "/", "/.", "/./"])
def test_inspect_entry_by_path_refuses_a_symlinked_entry_however_spelled(tmp_path, suffix):
    """lstat("link/") follows the link, so a trailing separator (or a terminal "/.") once let an
    entry path that is a symlink -- here to a valid copy inside the root -- pass the entry-type
    check when inspected by directory path. A real entry directory stays valid however it is
    spelled (codex, PR #399)."""
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    entry = result.entry_dir
    assert gtcache.inspect_entry(entry + suffix, root=root)["valid"] is True
    aside = os.path.join(root, "aside", os.path.basename(entry))
    os.makedirs(os.path.dirname(aside))
    os.rename(entry, aside)
    os.symlink(aside, entry)
    assert gtcache.inspect_entry(entry + suffix, root=root)["valid"] is False


def test_inspect_entry_unreadable_artifact_is_invalid_without_raising(tmp_path, monkeypatch):
    """inspect_entry never raises for a failed check: an artifact it cannot read (hashing it
    raises PermissionError -- injected, so the test needs no root-dependent chmod gate) is
    reported as an invalid entry, not a traceback out of `cell-eval2 cache inspect` (Gemini,
    PR #399)."""
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)

    def _unreadable(path):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(gtcache.integrity, "sha256_file", _unreadable)
    info = gtcache.inspect_entry(req.cache_id, root=root)
    assert info["valid"] is False
    assert any("Permission denied" in p for p in info["problems"])


# ── 13. inspect_entry identity self-consistency (P1 fix) ──────────────────────────────
#
# Before this fix, `inspect_entry` passed `request=None` into `_validate_entry_files`,
# which skipped the fingerprint/semantics check ENTIRELY -- so an entry whose identity
# documents were missing, schema-invalid, or simply did not hash to its own recorded
# `cache_id` was reported `valid: True` as long as the Parquet bytes matched. These tests
# pin `_check_identity_self_consistency`, which now runs unconditionally in both
# `inspect_entry` and `GroundTruthDEGCache.validate_entry`.


def test_inspect_entry_missing_identity_field_is_invalid(tmp_path):
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    del manifest["ground_truth_fingerprint"]
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    info = gtcache.inspect_entry(req.cache_id, root=root)
    assert info["valid"] is False
    assert len(info["problems"]) == 1
    assert "identity_self_consistency" in info["problems"][0]


def test_inspect_entry_cache_id_not_digest_of_its_documents_is_invalid(tmp_path):
    """A manifest whose top-level `cache_id` still equals the id under check (so the
    simpler `_check_format_and_id` check alone would pass) but whose recomputed
    `cache_id(fingerprint_doc, semantics_doc)` does not equal it -- i.e. the recorded id is
    not actually the digest of the documents sitting next to it. Only
    `_check_identity_self_consistency` catches this."""
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    # Mutate the fingerprint document AND its own recorded per-document digest together (so
    # the "document_sha256(doc) == recorded sha" sub-check still passes on its own) -- but
    # leave the top-level `cache_id` field untouched, so it silently stops being the digest
    # of {fingerprint, semantics} even though it still equals `req.cache_id`/the directory's
    # own id (what `_check_format_and_id` compares against).
    mutated_fp = dict(manifest["ground_truth_fingerprint"])
    mutated_fp["pert_col"] = "some_other_pert_col"
    manifest["ground_truth_fingerprint"] = mutated_fp
    manifest["ground_truth_fingerprint_sha256"] = gtcache.document_sha256(mutated_fp)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    info = gtcache.inspect_entry(req.cache_id, root=root)
    assert info["valid"] is False
    assert len(info["problems"]) == 1
    assert "identity_self_consistency" in info["problems"][0]


def test_inspect_entry_malformed_columns_is_invalid_without_raising(tmp_path):
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["artifact"]["columns"] = [None]
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    report = gtcache.inspect_entry(req.cache_id, root=root)
    assert report["valid"] is False
    assert len(report["problems"]) == 1 and "artifact_columns" in report["problems"][0]


def test_inspect_entry_invalid_utf8_manifest_returns_invalid_without_raising(tmp_path):
    """`_load_manifest_json` now catches `UnicodeDecodeError` alongside `OSError`/
    `ValueError` (json.JSONDecodeError is itself a ValueError), and `inspect_entry`'s
    documented contract -- never raise for a failed check -- holds even for a manifest
    that is not valid UTF-8 at all."""
    root = str(tmp_path / "cache")
    _cache, req, result = _publish(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, "wb") as fh:
        fh.write(b"\xff\xfe this is neither valid utf-8 nor json")

    info = gtcache.inspect_entry(req.cache_id, root=root)  # must not raise
    assert info["valid"] is False
    assert len(info["problems"]) == 1
    assert "manifest_json" in info["problems"][0]


def test_check_request_equality_rejects_fingerprint_and_semantics_mismatch():
    """Direct unit test of the (b) optional check (`_check_request_equality`):
    `GroundTruthDEGCache.validate_entry` always supplies a request; `inspect_entry` never
    does (covered by the (a) self-consistency tests above). A real end-to-end mismatch is
    unreachable through the full entry-file pipeline without a SHA-256 collision -- once
    (a) has established `cache_id(fp_doc, sem_doc) == manifest["cache_id"] ==
    request.cache_id`, `fp_doc`/`sem_doc` and the request's own documents are guaranteed
    (short of a collision) to be the same content -- so this exercises the function
    directly against a hand-built manifest."""
    from cell_eval2.ground_truth_cache.integrity import _check_request_equality

    fp1 = gtcache.GroundTruthFingerprint.from_axes(
        payload_sha256=_GOOD_PAYLOAD, var_names=["g0"], pert_labels=["c"],
        pert_col="perturbation",
    )
    fp2 = gtcache.GroundTruthFingerprint.from_axes(
        payload_sha256=_GOOD_PAYLOAD, var_names=["g0", "g1"], pert_labels=["c", "A"],
        pert_col="perturbation",
    )
    cfg = EvalConfig.v1()
    cfg.pert_col = "perturbation"
    sem = gtcache.build_real_de_semantics(
        cfg, control="non-targeting", input_type="counts", target_sum=10_000.0,
        device="cuda", de_backend="gpudge", precision="float32", native_gpu_normalize=True,
        versions=_VERSIONS,
    )
    req = gtcache.GroundTruthDEGCacheRequest(fingerprint=fp1, semantics=sem)

    fingerprint_mismatch_manifest = {
        "ground_truth_fingerprint": dict(fp2.document), "de_semantics": dict(sem.document),
    }
    with pytest.raises(gtcache.GroundTruthCacheCorruptError, match="fingerprint_mismatch"):
        _check_request_equality(fingerprint_mismatch_manifest, req, "manifest.json", "x" * 64)

    other_sem_doc = dict(sem.document)
    other_sem_doc["control"] = "a-different-control"
    semantics_mismatch_manifest = {
        "ground_truth_fingerprint": dict(fp1.document), "de_semantics": other_sem_doc,
    }
    with pytest.raises(gtcache.GroundTruthCacheCorruptError, match="semantics_mismatch"):
        _check_request_equality(semantics_mismatch_manifest, req, "manifest.json", "x" * 64)
