"""CLI tests for the `cache inspect` / `cache doctor` subcommands."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")

from cell_eval2.cli import _build_parser, main  # noqa: E402
from cell_eval2 import ground_truth_cache as gtcache  # noqa: E402

from test_ground_truth_cache_store import _make_request, make_de_frame  # noqa: E402


def test_parser_accepts_cache_doctor(tmp_path):
    args = _build_parser().parse_args(["cache", "doctor", str(tmp_path)])
    assert args.command == "cache"
    assert args.cache_command == "doctor"
    assert args.root == str(tmp_path)


def test_parser_accepts_cache_inspect(tmp_path):
    cache_id = "a" * 64
    args = _build_parser().parse_args(["cache", "inspect", cache_id, "--root", str(tmp_path)])
    assert args.command == "cache"
    assert args.cache_command == "inspect"
    assert args.entry_or_id == cache_id
    assert args.root == str(tmp_path)


def test_parser_requires_cache_subcommand(tmp_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["cache"])


def test_cli_cache_doctor_prints_ok_json_and_exits_zero(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["cache", "doctor", str(tmp_path)])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["ok"] is True


def _publish_one(root):
    req = _make_request(root, "readwrite")
    cache = gtcache.GroundTruthDEGCache(root, policy="readwrite")
    result = cache.get_or_compute(req, lambda: make_de_frame(("A", "B", "C"), 4))
    return req, result


def test_cli_cache_inspect_valid_entry_exits_zero(tmp_path, capsys):
    root = str(tmp_path / "cache")
    req, result = _publish_one(root)

    with pytest.raises(SystemExit) as exc_info:
        main(["cache", "inspect", req.cache_id, "--root", root])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["valid"] is True
    assert payload["cache_id"] == req.cache_id


def test_cli_cache_inspect_corrupted_entry_exits_two(tmp_path, capsys):
    import os

    root = str(tmp_path / "cache")
    req, result = _publish_one(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["cache_id"] = "0" * 64
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    with pytest.raises(SystemExit) as exc_info:
        main(["cache", "inspect", req.cache_id, "--root", root])
    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["valid"] is False


def test_cli_cache_inspect_invalid_utf8_manifest_exits_two_without_traceback(tmp_path, capsys):
    """P1 fix: an invalid-UTF-8 manifest must not turn `cache inspect` into an unhandled
    traceback (`_load_manifest_json` now catches `UnicodeDecodeError` too) -- it reports
    `valid: false` as ordinary JSON and exits 2, same as any other corrupt entry."""
    import os

    root = str(tmp_path / "cache")
    req, result = _publish_one(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, "wb") as fh:
        fh.write(b"\xff\xfe not valid utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["cache", "inspect", req.cache_id, "--root", root])
    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["valid"] is False
    assert len(payload["problems"]) == 1


def test_cli_cache_inspect_malformed_columns_exits_two_without_traceback(tmp_path, capsys):
    """A manifest whose artifact.columns holds a null member is reported as an invalid entry
    (JSON, exit 2), not an AttributeError traceback."""
    import os

    root = str(tmp_path / "cache")
    req, result = _publish_one(root)
    manifest_path = os.path.join(result.entry_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["artifact"]["columns"] = [None]
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    with pytest.raises(SystemExit) as exc_info:
        main(["cache", "inspect", req.cache_id, "--root", root])
    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert "artifact_columns" in payload["problems"][0]
