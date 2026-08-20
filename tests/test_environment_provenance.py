"""``run_meta``'s ``environment`` block: what actually computed the numbers.

The artifacts named the DE engine (``resolved_de_backend: 'gpudge'``) and no version anywhere
-- not in ``manifest.json``, ``baseline_meta.json`` or ``anchor_meta.json`` -- while gpudge
drove four of the six scored ``vcc2026`` members. These tests pin the record that closes that,
and above all they pin its ONE way of going wrong: the classification must never carry the URL
or the path it was derived from. A leak there is invisible to the publish sweeps, because the
offending value exists only at runtime.
"""
from __future__ import annotations

import json
import os

import pytest

from cell_eval2.baseline import (
    _ENVIRONMENT_PEERS,
    _distribution_provenance,
    _environment_record,
    _package_provenance,
    build_run_meta,
    config_digest,
)
from cell_eval2.config import DEParams, EvalConfig

#: The leaky strings the classifier is handed: an absolute filesystem path and a repository URL
#: -- SHAPED exactly like what the reference venv's `direct_url.json` actually holds (measured
#: 2026-08-19), which is what must not survive into an artifact.
#:
#: ⚠️ The home-directory path is NEUTRALISED, not verbatim. `tests/**` SHIPs and the real one is
#: a `check_publish_set.SWEEP_TOKENS` entry, so writing it out would make this file a publish
#: blocker -- the same trap `PUBLISH_SET.txt` already records against `test_gate_manifest.py`'s
#: forbidden-token tuple. The username carries no test power; being absolute is the whole point.
#: The VCS URL stays literal for a different reason: it names a PUBLIC repository, so it is not a
#: sweep token. It is no longer observable either -- `scale` stopped being a git URL in #358, so no
#: cell_eval2 dependency has a VCS `direct_url.json` for the test to read off a real venv.
_EDITABLE_URL = "file:///home/somebody/projects/code/gpudge"
_VCS_URL = "https://github.com/ArcInstitute/cellstream.git"


class _FakeDist:
    """Just enough ``importlib.metadata.Distribution`` for the classifier: a ``direct_url.json``
    payload (or ``None``) and a version. Faked rather than read off the host, so the
    classification is tested on all five cases on every machine -- CI has no editable gpudge
    and no VCS install to observe."""

    def __init__(self, direct_url, version="1.2.3", raw=...):
        # `direct_url=None` means NO `direct_url.json` at all, which is a different case from a
        # file whose contents are the literal `null` -- pass `raw="null"` for that one. (Conflating
        # the two is how the first version of the non-object test asserted the wrong thing.)
        self._raw = raw if raw is not ... else (
            None if direct_url is None else json.dumps(direct_url))
        self._version = version

    @property
    def version(self):
        return self._version

    def read_text(self, name):
        return self._raw if name == "direct_url.json" else None


_CASES = {
    "release": None,
    "git": {"url": _VCS_URL,
            "vcs_info": {"vcs": "git", "commit_id": "f" * 40, "requested_revision": "v0.7.1"}},
    "local-editable": {"url": _EDITABLE_URL, "dir_info": {"editable": True}},
    "local": {"url": "file:///opt/wheels/gpudge-0.8.0.whl",     # neutralised, see _EDITABLE_URL
              "archive_info": {"hash": "sha256=" + "a" * 64}},
    "archive": {"url": "https://example.invalid/wheels/gpudge-0.8.0-py3-none-any.whl",
                "archive_info": {"hash": "sha256=" + "b" * 64}},
}

#: Every token the classifier is allowed to produce. Asserted against, so a sixth one cannot
#: appear without a test saying so.
_TOKENS = frozenset({"release", "git", "hg", "svn", "bzr", "vcs", "local-editable", "local",
                     "archive"})


@pytest.mark.parametrize("expected,direct_url", sorted(_CASES.items()))
def test_provenance_classification_covers_every_install_shape(expected, direct_url):
    assert _distribution_provenance(_FakeDist(direct_url)) == expected


@pytest.mark.parametrize("vcs,expected", [("git", "git"), ("hg", "hg"), ("svn", "svn"),
                                          ("bzr", "bzr"), ("fossil", "vcs"), (None, "vcs")])
def test_a_non_git_checkout_is_not_called_git(vcs, expected):
    """⚠️ Answering `git` for a mercurial checkout would be a small lie of exactly the kind this
    field exists to prevent. An unrecognised `vcs` becomes the generic token rather than an
    arbitrary string read out of a file on disk."""
    info = {"url": _VCS_URL, "vcs_info": {"commit_id": "f" * 40}}
    if vcs is not None:
        info["vcs_info"]["vcs"] = vcs
    assert _distribution_provenance(_FakeDist(info)) == expected


@pytest.mark.parametrize("scheme", ["file", "File", "FILE"])
def test_the_file_scheme_is_matched_case_insensitively(scheme):
    """A URI scheme is case-insensitive (RFC 3986 3.1), so `FILE:///...` is a LOCAL install and
    must not be reported as a remote `archive`."""
    assert _distribution_provenance(_FakeDist(
        {"url": f"{scheme}:///opt/wheels/gpudge-0.8.0.whl",
         "archive_info": {"hash": "sha256=" + "c" * 64}})) == "local"


def test_a_git_checkout_of_a_local_path_is_still_git():
    """`pip install git+file:///path/to/repo` -- `vcs_info` wins over the `file:` URL, because it
    IS a VCS install and the token must say so."""
    assert _distribution_provenance(_FakeDist(
        {"url": "git+file:///srv/mirrors/gpudge.git",
         "vcs_info": {"vcs": "git", "commit_id": "d" * 40}})) == "git"


@pytest.mark.parametrize("raw", ["[]", "null", "3", '"a string"', '[{"url": "file:///x"}]'])
def test_a_direct_url_that_is_not_an_object_is_UNCLASSIFIED_not_release(raw):
    """⚠️ Gemini (PR #347) proposed `return "release"` here. That would be wrong: the mere PRESENCE
    of `direct_url.json` is what rules `release` OUT, so answering `release` for a payload we could
    not read puts the most-trusted token on the least-known case. `None` says "there is direct-URL
    provenance and it could not be read" -- true, and useful."""
    assert _distribution_provenance(_FakeDist(None, raw=raw)) is None


def test_a_version_that_raises_still_records_the_provenance(monkeypatch):
    """The two halves are independent. `dist.version` reads METADATA and can raise on a corrupt one
    (Gemini, PR #347); nulling the classification too would discard information we have."""
    from cell_eval2 import baseline

    class NoVersion(_FakeDist):
        @property
        def version(self):
            raise ValueError("corrupt METADATA")

    monkeypatch.setattr(baseline.Distribution, "from_name",
                        staticmethod(lambda name: NoVersion(None)))
    assert _package_provenance("polars") == {"version": None, "provenance": "release"}


def test_a_non_editable_local_directory_is_local_not_editable():
    """`dir_info` WITHOUT `editable` is a plain `pip install ./path` -- a snapshot, so its
    version metadata is trustworthy, unlike the editable case. The two must not collapse."""
    assert _distribution_provenance(
        _FakeDist({"url": _EDITABLE_URL, "dir_info": {}})) == "local"


@pytest.mark.parametrize("direct_url", [_CASES["git"], _CASES["local-editable"],
                                        _CASES["local"], _CASES["archive"]])
def test_the_url_never_survives_the_classification(direct_url):
    """⚠️ THE ONE WAY THIS CHANGE COULD LEAK. The classifier is handed an absolute filesystem
    path and a repository URL -- the SHAPES `direct_url.json` really holds, with the username
    neutralised (see `_EDITABLE_URL`); only a fixed token may come back."""
    got = _distribution_provenance(_FakeDist(direct_url))
    assert got in _TOKENS, got
    for leak in (direct_url["url"], "file:", "http", "github.com", "/home/", "somebody",
                 "ArcInstitute"):
        assert leak not in got


def _leaks(node) -> list[str]:
    """Every substring in a recorded value that would betray a path, a URL or a host."""
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            found += _leaks(k) + _leaks(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            found += _leaks(v)
    elif isinstance(node, str):
        for leak in ("://", "file:", "/", "\\", "@", "github", "somebody",
                     "ArcInstitute"):
            if leak in node:
                found.append(f"{leak!r} in {node!r}")
    return found


def test_the_environment_record_leaks_no_path_or_url(monkeypatch):
    """⚠️ THE TEST THAT MATTERS. Asserted on the WHOLE block, keys and values, against
    path-shaped and URL-shaped payloads -- a runtime-only value is invisible to the publish
    sweeps, so this is the only thing standing between an internal path and a shipped bundle.

    It fakes `Distribution.from_name`, NOT `_package_provenance`, so the real production path
    runs end to end: patching the helper would skip the very function that would acquire a new
    leaking field (Codex, checkpoint 2). The SCHEMA is pinned too, for the same reason -- an
    extra key carrying a path would otherwise pass a value-only check."""
    from cell_eval2 import baseline

    monkeypatch.setattr(baseline.Distribution, "from_name", staticmethod(
        lambda name: _FakeDist(_CASES["local-editable" if name == "gpudge" else "git"],
                               version="0.7.0")))
    record = _environment_record("gpudge")
    assert record["gpudge"] == {"version": "0.7.0", "provenance": "local-editable"}
    for key, entry in record.items():
        assert set(entry) == {"version", "provenance"}, key      # exact schema, no extra field
        assert entry["provenance"] in _TOKENS
    assert _leaks(record) == []


def test_an_unknown_backend_token_never_becomes_a_key():
    """⚠️ The "never records a path" contract has to hold on the FUNCTION, not on validation up
    the stack. A `.get(token, token)` fallback would have put a caller-supplied string straight
    into a shipped artifact as a dict KEY -- where `_leaks` on the values alone would miss it."""
    leaky = "file:///etc/shadow"
    record = _environment_record(leaky)
    assert leaky not in record
    assert set(record) == set(_ENVIRONMENT_PEERS)
    assert _leaks(record) == []


def test_the_real_record_on_this_host_leaks_nothing_either():
    """The faked case proves the classifier; this proves whatever this machine actually has --
    the versions themselves are recorded verbatim, and a PEP 440 version carries no separator,
    but that is an invariant worth failing on rather than trusting."""
    assert _leaks(_environment_record("scanpy")) == []


def test_a_missing_package_is_null_rather_than_fatal(monkeypatch):
    """CI installs no gpudge, so its entry is legitimately null there. Best-effort is the
    contract: `anchor._version()`'s "never lose an anchor to provenance", applied to a run."""
    from importlib.metadata import PackageNotFoundError

    from cell_eval2 import baseline

    def boom(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(baseline.Distribution, "from_name", staticmethod(boom))
    assert _package_provenance("gpudge") is None
    record = _environment_record("gpudge")
    assert set(record) == {"gpudge", *_ENVIRONMENT_PEERS}
    assert all(v is None for v in record.values())


def test_an_unreadable_direct_url_keeps_the_version_and_nulls_the_class(monkeypatch):
    """A broken sidecar must not cost the version. `provenance=None` says "unclassified",
    which is honest; dropping the entry would be strictly less information."""
    from cell_eval2 import baseline

    class Broken(_FakeDist):
        def read_text(self, name):
            raise OSError("permission denied")

    monkeypatch.setattr(baseline.Distribution, "from_name",
                        staticmethod(lambda name: Broken(None, version="9.9.9")))
    assert _package_provenance("polars") == {"version": "9.9.9", "provenance": None}


def test_a_package_that_raises_nulls_ONLY_ITS_OWN_ENTRY(monkeypatch):
    """Per-target isolation: one distribution with unreadable metadata must not truncate the key
    set. A short record leaves a reader unable to tell "absent" from "merely unrecorded"."""
    from cell_eval2 import baseline

    def boom(name):
        if name == "scipy":
            raise RuntimeError("boom")
        return {"version": "1.0.0", "provenance": "release"}

    monkeypatch.setattr(baseline, "_package_provenance", boom)
    record = _environment_record("gpudge")
    assert set(record) == {"gpudge", *_ENVIRONMENT_PEERS}       # complete, despite the failure
    assert record["scipy"] is None
    assert record["polars"] == {"version": "1.0.0", "provenance": "release"}


def test_nothing_in_the_block_can_raise(monkeypatch):
    """The outer guard, exercised where it matters: a failure resolving the backend->distribution
    map must still yield a run."""
    from cell_eval2 import de_compute

    monkeypatch.delattr(de_compute, "_BACKEND_MODULE")
    assert _environment_record("gpudge") == {}


def _de_cfg(**kw):
    return EvalConfig(metrics=["de_wilcoxon_overlap"], device="cpu",
                      de=DEParams(backend="scanpy"), **kw)


def test_build_run_meta_stamps_the_block_and_names_the_resolved_backend(synthetic_pair):
    """The record is keyed by the token `resolved_de_backend` reports, so the two fields
    cross-reference by name rather than by a reader's guess."""
    pred, real = synthetic_pair
    meta = build_run_meta(_de_cfg(), real, pred)
    assert meta["resolved_de_backend"] == "scanpy"
    env = meta["environment"]
    assert meta["resolved_de_backend"] in env
    for peer in _ENVIRONMENT_PEERS:
        assert peer in env
        assert set(env[peer]) == {"version", "provenance"}
        assert env[peer]["version"]                      # installed here, so non-null
        assert env[peer]["provenance"] in _TOKENS      # _CASES would fail on an hg peer
    assert _leaks(env) == []


def test_a_run_that_supplies_its_de_tables_records_the_stack_and_no_engine(synthetic_pair):
    """`resolved_de_backend` is None when no engine ran -- and then there is no engine version
    to record either. The numeric stack is still what computed everything else."""
    pred, real = synthetic_pair
    meta = build_run_meta(EvalConfig(metrics=["delta_mse"], device="cpu"), real, pred)
    assert meta["resolved_de_backend"] is None
    assert set(meta["environment"]) == set(_ENVIRONMENT_PEERS)


def test_the_deseq2_token_is_keyed_by_the_token_and_versioned_by_its_distribution():
    """`deseq2` is the only backend whose token differs from its distribution (`deseq2_gpu`).
    The key follows `resolved_de_backend`; the version must come from the distribution."""
    record = _environment_record("deseq2")
    assert "deseq2" in record and "deseq2_gpu" not in record
    # Unconditional on purpose: where `deseq2_gpu` is absent (CI, every minimal install) BOTH
    # sides are None, so the equality still says what it means -- the entry is whatever that
    # DISTRIBUTION resolves to. An `if record["deseq2"] is not None` guard would make the
    # assertion silently vanish in the one environment nobody watches.
    assert record["deseq2"] == _package_provenance("deseq2_gpu")


def test_the_block_cannot_move_config_digest(synthetic_pair, monkeypatch):
    """`config_digest` takes the CONFIG, never the meta dict -- so the environment can differ
    between two runs of the same configuration without making them refuse to pair. Faked to a
    different environment rather than argued from the signature."""
    from cell_eval2 import baseline

    pred, real = synthetic_pair
    cfg = _de_cfg()
    before = build_run_meta(cfg, real, pred)
    monkeypatch.setattr(baseline, "_environment_record",
                        lambda backend: {"polars": {"version": "0.0.1",
                                                    "provenance": "local-editable"}})
    after = build_run_meta(cfg, real, pred)
    assert after["environment"] != before["environment"]
    assert after["config_digest"] == before["config_digest"]
    assert before["config_digest"] == config_digest(
        cfg, comparator=before["comparator"], de_real=None)
    # Every compared field, not just the digest: `check_submission` and
    # `cli._check_baseline_config` must see two identical runs.
    from cell_eval2.real_bundle import MANIFEST_RECORDED_ONLY, SUBMISSION_PEERS
    for f in SUBMISSION_PEERS + MANIFEST_RECORDED_ONLY:
        assert after.get(f) == before.get(f), f


def test_environment_is_absent_from_the_compared_and_recorded_tuples():
    """⚠️ #291, measured: `read_real_bundle` requires every name in BOTH tuples to be present
    in the manifest it reads, so adding this field to either makes all three frozen official
    val bundles unreadable outright. It stays out by construction, and this is the guard."""
    from cell_eval2.real_bundle import MANIFEST_RECORDED_ONLY, SUBMISSION_PEERS

    assert "environment" not in SUBMISSION_PEERS + MANIFEST_RECORDED_ONLY

def test_a_bundle_whose_baseline_meta_predates_this_field_still_reads(counts_bundle_inputs,
                                                                     tmp_path):
    """A build carrying `environment` must not make a build PREDATING it unreadable -- the three
    frozen official bundles are exactly that shape. Hermetic: a real bundle is built, then its
    `baseline_meta.json` is rewritten without the field, standing in for a pre-#338 artifact with
    no host path and no environment gate involved."""
    from dataclasses import replace

    from cell_eval2.config import EvalConfig
    from cell_eval2.real_bundle import build_real_bundle, read_real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    root = str(tmp_path / "b")
    build_real_bundle(real, baseline_pred,
                      config=replace(EvalConfig.from_preset("vcc2026")),
                      outdir=root, bundle_id="legacy-shape-test")

    meta_path = os.path.join(root, "baseline_meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert "environment" in meta, "the field must be there before we take it away"
    # `manifest.json` is left ALONE: it never carried the field in the first place (#291 --
    # adding it to either required tuple makes the frozen bundles unreadable), so a bundle with
    # no `environment` in its baseline_meta is precisely the pre-#338 shape.
    with open(meta_path, "w") as fh:
        json.dump({k: v for k, v in meta.items() if k != "environment"}, fh, indent=2)

    bundle = read_real_bundle(root)
    assert "environment" not in bundle.baseline_meta
    assert "environment" not in bundle.manifest
    assert bundle.manifest["real_bundle_id"] == "legacy-shape-test"


# ⚠️ THE THREE FROZEN `-r2` OFFICIAL BUNDLES ARE VERIFIED BY MEASUREMENT, NOT BY A TEST HERE.
# Measured 2026-08-19 on this branch: `read_real_bundle` opens `vcc2026-valA-r2`, `-valB-r2` and
# `-valC-r2` read-only and all three still read, so a build carrying `environment` does not make
# a build predating it unreadable. It is deliberately NOT tracked as a test, for two reasons that
# both point outside this change: the bundles live under a `check_publish_set.SWEEP_TOKENS` path
# and `tests/**` SHIPs, so the constant alone would be a publish blocker needing a DROP rule in
# `PUBLISH_SET.txt` -- another branch's file, and #335's session owns it -- and the skip it would
# need is a new environment gate in `internal:tests/gated_modules.toml` (#344 has since resolved
# that FIX by DROPping the table, so a public reader has no such file at all).
# `test_environment_is_absent_from_the_compared_and_recorded_tuples` above pins the mechanism that
# actually keeps an old bundle readable, and it needs no host and no gate. (Those bundles will
# still refuse to SCORE -- #342 moved `competition_digest` and #343 moved it again with
# `rule_version` 2 -> 3, so their `rule_digest` is stale twice over and the `-r3` rebuild is the
# release pass's job, not this one's.)
