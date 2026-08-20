"""`score --real-bundle` — the fatal gate and the enrolment it guards.

The load-bearing test in this file is `test_a_competition_bundle_moves_the_average`: it is the
one that would have caught BOTH of part C-1's dead features on its first run, because a check
that is wrongly False here does not warn -- it exits non-zero.
"""
import json
from dataclasses import dataclass

import polars as pl
import pytest

from cell_eval2 import competition
from cell_eval2.baseline import build_run_meta
from cell_eval2.config import EvalConfig
from cell_eval2.real_bundle import (MANIFEST, build_real_bundle, check_submission,
                                    read_real_bundle)
from cell_eval2.run import aggregate_metrics_wide, compute_metrics, metric_output_names
from cell_eval2.score import score_metrics


@dataclass(frozen=True)
class _Submission:
    agg: str
    meta: dict
    meta_path: str


@pytest.fixture
def submission(counts_bundle_inputs, tmp_path):
    """A `run`-shaped submission: the aggregate CSV, its run_meta dict, and that meta on disk.

    Built under the SAME config as the bundle, because `check_submission` compares nine peer
    fields (plus a one-sided check on `de_pred_fingerprint`) -- a submission produced under a
    different config is exactly what the gate exists to refuse, so a fixture that produced one
    could only ever test the failure path.
    """
    _baseline_pred, real, submission_pred = counts_bundle_inputs
    cfg = EvalConfig.from_preset("vcc2026")
    agg = aggregate_metrics_wide(
        compute_metrics(submission_pred, real, config=cfg),
        metrics=metric_output_names(cfg),
    )
    agg_path = tmp_path / "submission_agg.csv"
    agg.write_csv(agg_path)
    meta = build_run_meta(cfg, real, submission_pred)
    meta_path = tmp_path / "run_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return _Submission(agg=str(agg_path), meta=meta, meta_path=str(meta_path))


@pytest.fixture
def competition_bundle(counts_bundle_inputs, tmp_path):
    baseline_pred, real, _submission_pred = counts_bundle_inputs
    root = tmp_path / "competition-bundle"
    build_real_bundle(real, baseline_pred, config=EvalConfig.from_preset("vcc2026"),
                      outdir=root, bundle_id="vcc2026-CCL_x-r1")
    return read_real_bundle(root)


@pytest.fixture
def diagnostic_bundle(counts_bundle_inputs, tmp_path):
    """Diagnostic through a foreign ANCHOR PARAMETER, never a different metric profile.

    `config_digest` covers the resolved metric list, so changing metrics would make the
    submission gate refuse the bundle instead of exercising diagnostic enrolment. With
    `base_seed=123`, `rule_digest` is None and `rule_mismatches` names `base_seed` and
    `derived_seeds`, while all nine submission peers (and the one-sided `de_pred_fingerprint`
    check) still agree.
    """
    baseline_pred, real, _submission_pred = counts_bundle_inputs
    root = tmp_path / "diagnostic-bundle"
    build_real_bundle(real, baseline_pred, config=EvalConfig.from_preset("vcc2026"),
                      outdir=root, bundle_id="vcc2026-CCL_x-r1-diagnostic", base_seed=123)
    return read_real_bundle(root)


@pytest.fixture
def loose_anchor_dir(competition_bundle):
    """Reuse the competition bundle's root as a plain loose anchor directory.

    A bundle directory is byte-format identical to what `run --anchor` writes:
    `anchor_agg.parquet`, `anchor_splits.parquet`, and `anchor_meta.json`. Therefore
    `read_anchor`/`resolve_anchor` open it unchanged, and reuse avoids a second five-split build.
    """
    return competition_bundle.root


@pytest.fixture
def anchor_expect(submission):
    from cell_eval2.score import expect_from_run_meta
    return expect_from_run_meta(submission.meta)


def test_a_competition_bundle_moves_the_average(competition_bundle, submission):
    """Alex 2026-08-13: from_replicate REPLACES from_baseline in avg_score."""
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta=submission.meta)
    row = out.filter(pl.col("metric") == "avg_score")
    body = out.filter(pl.col("metric") != "avg_score")
    # ⚠️ The membership is asserted against the CANONICAL SIX, never against "whatever rows
    # came back non-null". Deriving the expectation from the implementation's own output makes
    # five members, seven members and enrolled diagnostics all pass.
    scored = body.filter(pl.col("from_replicate").is_not_null())["metric"].to_list()
    assert sorted(scored) == sorted(competition.competition_members())
    # ⚠️ The four `vcc2026` diagnostics get NO ROW AT ALL, and an earlier draft asserted the
    # opposite -- that they appear as null rows -- which is deterministically red.
    # `score_metrics` skips an unscored metric outright (`score.py:528`,
    # `if not policy.scored: continue`), `_replicate_entries` skips it for the same reason, and
    # row restoration only restores entries it holds. So the frame's non-avg rows are EXACTLY
    # the six. This is the honest invariant, and it is stronger than the null-row one: it also
    # fails if a diagnostic ever leaks into the average.
    DIAGNOSTICS = {"expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased",
                   "expr_real_mass_ratio"}
    assert sorted(body["metric"].to_list()) == sorted(competition.competition_members())
    assert not (set(body["metric"].to_list()) & DIAGNOSTICS)
    assert row["from_replicate"].item() == pytest.approx(
        body.filter(pl.col("metric").is_in(list(competition.competition_members())))
            ["from_replicate"].mean())
    # The frozen frame order, which the rule's `member_order_in_frame` claims.
    assert [m for m in body["metric"].to_list() if m in set(competition.competition_members())] \
        == competition.competition_payload()["member_order_in_frame"]
    # The old headline is not merely demoted -- it is REMOVED, so an unmigrated consumer reads
    # a null rather than a well-formed, no-longer-official number under the label it reads.
    assert row["from_baseline"].item() is None
    # Per-metric from_baseline survives as a diagnostic.
    assert out.filter(pl.col("metric") == "pds_cosine")["from_baseline"].item() is not None
    assert out["real_bundle_id"].to_list()[0] == competition_bundle.manifest["real_bundle_id"]
    assert out["anchor_source"].to_list()[0] == "real_bundle"


def test_a_diagnostic_bundle_reports_but_does_not_enrol(diagnostic_bundle, submission, caplog):
    out = score_metrics(submission.agg, real_bundle=diagnostic_bundle.root,
                        user_meta=submission.meta)
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() is not None
    assert out.filter(pl.col("metric") == "avg_score")["from_replicate"].item() is not None
    assert "diagnostic" in caplog.text


def test_a_STALE_rule_digest_is_fatal(competition_bundle, submission, tmp_path):
    """Distinguishable from a diagnostic bundle only because the field is PRESENT: a boolean
    could not tell "never claimed the label" from "claimed it under a rule that has moved"."""
    path = f"{competition_bundle.root}/{MANIFEST}"
    man = json.loads(open(path).read())
    man["rule_digest"] = "0" * 64
    open(path, "w").write(json.dumps(man))
    with pytest.raises(ValueError, match="rule"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta=submission.meta)


@pytest.mark.parametrize("field,bad", [
    ("config_digest", "deadbeef"), ("cell_eval2_version", "0.0.1"),
    ("comparator", "lognorm"), ("source_fingerprint", "deadbeef"),
    ("resolved_de_backend", "nonesuch"),
])
def test_a_submission_that_disagrees_with_the_bundle_is_REFUSED(
        competition_bundle, submission, field, bad):
    """Fatal, with no escape hatch (Alex 2026-08-13). This is what makes every check in the
    gate exercised for real: a wrongly-False check exits non-zero on the first bundle."""
    with pytest.raises(ValueError, match=field):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, field: bad})


def test_the_error_names_EVERY_disagreeing_field(competition_bundle, submission):
    with pytest.raises(ValueError) as e:
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, "config_digest": "x",
                                 "comparator": "lognorm"})
    assert "config_digest" in str(e.value) and "comparator" in str(e.value)


# --- what the refusal MESSAGE says -- fixture-free, `check_submission` is a pure function ---

#: A pairing that agrees on everything. Values are arbitrary: the function compares the two
#: dicts to each other, never against the world.
_AGREED = {
    "cell_eval2_version": "0.13.0", "config_digest": "d" * 8, "comparator": "bulk_lognorm",
    "source_fingerprint": "s" * 8, "source_fingerprint_strict": True,
    "resolved_device": "cpu", "resolved_de_backend": None,
    "input_type_real_effective": "counts", "de_real_fingerprint": None,
    "de_pred_fingerprint": None,
}


def _refusal(**user_overrides):
    with pytest.raises(ValueError) as e:
        check_submission({**_AGREED, "real_bundle_id": "b"}, {**_AGREED, **user_overrides})
    return str(e.value)


def test_a_peer_only_refusal_does_NOT_mention_de_pred():
    """Copilot, PR #298. The `--de-pred` remedy used to be appended to the shared sentence, so
    a submission failing only on `config_digest` was told to drop a flag that would fix
    nothing -- and the opt-in was advertised as though it were an escape hatch from the peer
    comparison, which it is not."""
    said = _refusal(config_digest="nonesuch")
    assert "config_digest" in said
    assert "--de-pred" not in said                       # covers the opt-in's name too


def test_a_supplied_table_refusal_names_BOTH_remedies():
    said = _refusal(de_pred_fingerprint="deadbeef")
    assert "without --de-pred" in said
    assert "--diagnostic-supplied-de-pred" in said and "does NOT enrol" in said


def test_a_mixed_refusal_names_every_problem_and_still_offers_the_opt_in():
    """The supplied table IS one of the causes here, so the opt-in is relevant even though it
    alone would not make this pairing pass -- and every problem is enumerated first."""
    said = _refusal(de_pred_fingerprint="deadbeef", config_digest="nonesuch")
    assert "config_digest" in said and "de_pred_fingerprint" in said
    assert "--diagnostic-supplied-de-pred" in said


# --- #192: the PRED side's effective input type is not a pairing field --------------------

def test_a_DIFFERENT_pred_side_input_type_is_ACCEPTED(competition_bundle, submission):
    """#192. The prediction is the one side a pairing is expected to differ on: the baseline
    arm is a fractional mean of counts pulled back to `counts` by the matrix-space lock, and a
    VCC submission is commonly log-normalized. Both sides are converted into the same metric
    space before any metric is computed.

    ⚠️ There is no waiver on this path at all -- `--allow-config-mismatch` alongside
    `--real-bundle` is a usage error -- so before this change such a submission was simply
    un-scoreable, not merely warned about."""
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta={**submission.meta,
                                   "input_type_pred_effective": "lognorm"})
    same = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                         user_meta=submission.meta)
    # and it changes NOTHING: the field never entered a number, only the gate
    assert out.equals(same)


def test_the_REAL_side_input_type_is_STILL_a_pairing_field(competition_bundle, submission):
    """The other edge of #192, and the one that keeps the change honest. The real side is what
    the two runs genuinely share, so a difference there IS fatal. Without this, dropping
    `input_type_real_effective` as well would leave the suite green."""
    with pytest.raises(ValueError, match="input_type_real_effective"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta,
                                 "input_type_real_effective": "lognorm"})


def test_the_manifest_still_RECORDS_the_pred_side_input_type(competition_bundle):
    """Recorded, not compared. `read_real_bundle` still requires it, so a bundle written
    without it -- which is what dropping the field outright would produce -- is refused."""
    assert competition_bundle.manifest["input_type_pred_effective"] == "counts"
    path = f"{competition_bundle.root}/{MANIFEST}"
    with open(path) as fh:
        man = json.load(fh)
    del man["input_type_pred_effective"]
    with open(path, "w") as fh:
        json.dump(man, fh)
    with pytest.raises(ValueError, match="input_type_pred_effective"):
        read_real_bundle(competition_bundle.root)


# --- #291: a SUPPLIED pred-side DE table is not a submission ------------------------------

def test_a_SUPPLIED_pred_side_DE_table_is_REFUSED(competition_bundle, submission):
    """#291's gate half. A supplied `--de-pred` table is not derived from the submitted cells,
    and it can omit `(target, feature)` pairs that `_direction_frame`'s inner join would
    otherwise score as misses -- worth 0.0 -> 0.9625 on `direction_reach_raw` in the repro
    (measured at the pre-`REACH_PURITY_FLOOR` floor 0.975; 0.0 -> 0.8875 at REACH_PURITY_FLOOR).

    The submission is otherwise IDENTICAL to the passing one: only `de_pred_fingerprint`
    moves, so a green result here cannot come from some other peer disagreeing."""
    with pytest.raises(ValueError, match="de_pred_fingerprint"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef"})


def test_the_supplied_DE_refusal_names_the_flag_that_produced_it(competition_bundle,
                                                                 submission):
    """A hex digest names nothing actionable. The message has to name the flag, because the
    remedy is to drop it and re-run."""
    with pytest.raises(ValueError) as e:
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef"})
    assert "--de-pred" in str(e.value)


def test_a_run_meta_WITHOUT_de_pred_fingerprint_is_REFUSED(competition_bundle, submission):
    """FAIL-CLOSED, the rule the peer loop already follows: an absent key is a mismatch, not a
    match. Without this, deleting the field from a hand-edited run_meta.json restores the exact
    hole the check closes."""
    meta = {k: v for k, v in submission.meta.items() if k != "de_pred_fingerprint"}
    with pytest.raises(ValueError, match="de_pred_fingerprint"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root, user_meta=meta)


def test_the_opt_in_SCORES_but_does_NOT_enrol(competition_bundle, submission):
    """#291's opt-in buys the RUN, not the label. `--de-pred` is the isolator the metric
    campaigns are built on and those arms want the bundle's scale, so refusing outright would
    move the rig off the bundle path -- but enrolling would stamp a competition average on a
    number the gate exists to disqualify.

    ⚠️ The assertion is the ENROLMENT, not the absence of an exception. Under enrolment
    `avg_score`'s `from_baseline` is NULLED and `from_replicate` carries the average
    (#276 part C); a diagnostic result keeps `from_baseline`. A test that only checked "it did
    not raise" would pass on the plain waiver this deliberately is not."""
    meta = {**submission.meta, "de_pred_fingerprint": "deadbeef"}
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta=meta, diagnostic_supplied_de_pred=True)
    row = out.filter(pl.col("metric") == "avg_score")
    assert row["from_baseline"].item() is not None          # NOT enrolled
    assert row["from_replicate"].item() is not None         # still reported

    enrolled = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                             user_meta=submission.meta)
    assert enrolled.filter(pl.col("metric") == "avg_score")["from_baseline"].item() is None
    # ...and the numbers themselves are untouched: only the enrolment moved
    assert (row["from_replicate"].item()
            == enrolled.filter(pl.col("metric") == "avg_score")["from_replicate"].item())


def test_the_opt_in_still_stamps_the_bundle_provenance(competition_bundle, submission):
    """It WAS scored against that bundle, and the frame has to keep saying so -- the downgrade
    is the enrolment, not the provenance."""
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef"},
                        diagnostic_supplied_de_pred=True)
    assert out["real_bundle_id"][0] == "vcc2026-CCL_x-r1"
    assert out["anchor_source"][0] == "real_bundle"


def test_the_opt_in_does_NOT_waive_anything_else(competition_bundle, submission):
    """It is not `--allow-config-mismatch` by another name. A peer mismatch alongside a
    supplied table is still fatal, and so is a MISSING `de_pred_fingerprint` -- the opt-in is
    scoped to a table that was actually supplied, so a hand-edited run_meta cannot reach it
    by deleting the key."""
    with pytest.raises(ValueError, match="config_digest"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef",
                                 "config_digest": "nonesuch"},
                      diagnostic_supplied_de_pred=True)
    absent = {k: v for k, v in submission.meta.items() if k != "de_pred_fingerprint"}
    with pytest.raises(ValueError, match="missing from run_meta.json"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta=absent, diagnostic_supplied_de_pred=True)


def test_the_opt_in_is_INERT_on_a_clean_submission(competition_bundle, submission):
    """No waiver taken means no downgrade. Passing the flag on a submission that never
    supplied a table must not quietly cost it its enrolment -- which is what deriving the
    downgrade from the FLAG rather than from the waivers TAKEN would do."""
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta=submission.meta, diagnostic_supplied_de_pred=True)
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() is None


def test_the_opt_in_is_REFUSED_without_a_bundle(competition_bundle, submission):
    """The loose path has no enrolment to downgrade, so accepting the flag there would let a
    harness believe it had marked a run diagnostic when nothing did. Refused, like
    `anchor_expect`.

    ⚠️ Matched on the PARAMETER spelling (`real_bundle`, not `--real-bundle`), which is what
    separates this from the CLI test below: the library message names library arguments and
    the CLI message names flags, so neither assertion can be satisfied by the other's path."""
    with pytest.raises(ValueError, match="applies only to real_bundle"):
        score_metrics(submission.agg, competition_bundle.baseline_agg,
                      diagnostic_supplied_de_pred=True)


def test_the_CLI_ALSO_refuses_the_flag_without_a_bundle(tmp_path):
    """⚠️ NOT covered by the `score_metrics` test above, and that is the whole point: the CLI
    only forwards the flag inside the `--real-bundle` branch, so DELETING its refusal would
    make the flag silently ignored on the loose path -- `score_metrics` would be called
    without the kwarg and never see it. The library test stays green through that deletion.

    Fixture-free on purpose: the refusal fires in the dispatch BEFORE `_check_baseline_config`
    opens anything, so neither path has to exist and this costs no bundle build."""
    from cell_eval2.cli import main

    with pytest.raises(SystemExit, match="applies only to --real-bundle"):
        main(["score", "--user-agg", str(tmp_path / "u.csv"),
              "--baseline-agg", str(tmp_path / "b.csv"),
              "--diagnostic-supplied-de-pred"])


def test_the_opt_in_WARNS_and_says_the_number_is_not_a_competition_score(
        competition_bundle, submission, caplog):
    """The frame does not record the downgrade -- the bundle is a competition bundle and its
    manifest carries the real `rule_digest` either way -- so the warning is the only thing that
    says so at the moment it happens. It has to name the waiver AND the consequence."""
    with caplog.at_level("WARNING"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef"},
                      diagnostic_supplied_de_pred=True)
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "de_pred_fingerprint" in said and "--de-pred" in said
    assert "NOT a competition score" in said


def test_BOTH_reasons_are_reported_when_a_diagnostic_bundle_takes_a_waiver(
        diagnostic_bundle, submission, caplog):
    """The two reasons not to enrol are unrelated -- the BUNDLE can be diagnostic and the
    SUBMISSION can be -- and an `if/elif` chain reports only the first.

    ⚠️ This is the only test that fails if the enrolment block is reverted to `elif`: every
    other case has exactly one reason in play, so the chain and the independent conditions
    agree on all of them."""
    with caplog.at_level("WARNING"):
        score_metrics(submission.agg, real_bundle=diagnostic_bundle.root,
                      user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef"},
                      diagnostic_supplied_de_pred=True)
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "de_pred_fingerprint" in said                 # the SUBMISSION's reason
    assert "DIAGNOSTIC bundle" in said and "base_seed" in said   # the BUNDLE's reason


def test_the_CLI_flag_reaches_the_scorer_and_withholds_ENROLMENT(
        competition_bundle, submission, tmp_path):
    """End to end through `main`, because the flag's whole point is that the CAMPAIGNS invoke
    `cell-eval2 score` from bash -- a library-only opt-in would leave that workflow just as
    broken, and nothing below the CLI can catch a subparser/dispatch mistake.

    Both arms: refused without the flag, scored-but-not-enrolled with it."""
    from cell_eval2.cli import main

    meta_path = tmp_path / "supplied_run_meta.json"
    with open(meta_path, "w") as fh:
        json.dump({**submission.meta, "de_pred_fingerprint": "deadbeef"}, fh)
    argv = ["score", "--user-agg", submission.agg, "--user-meta", str(meta_path),
            "--real-bundle", competition_bundle.root, "-o", str(tmp_path / "scored.csv")]

    with pytest.raises(ValueError, match="de_pred_fingerprint"):
        main(argv)
    main([*argv, "--diagnostic-supplied-de-pred"])

    got = pl.read_csv(tmp_path / "scored.csv").filter(pl.col("metric") == "avg_score")
    assert got["from_baseline"].item() is not None      # NOT enrolled
    assert got["from_replicate"].item() is not None     # still reported


def test_the_check_is_NOT_a_peer_so_a_LEGACY_manifest_still_reads(competition_bundle,
                                                                  submission):
    """⚠️ THE REGRESSION THIS GUARDS. The issue proposes adding `de_pred_fingerprint` to
    `SUBMISSION_PEERS`; `read_real_bundle` requires every peer to be PRESENT in the manifest,
    so that one line stops all three official val bundles from being readable AT ALL -- not
    scored differently, not warned about: `read_real_bundle` raises.

    Deleting the key here is not a hypothetical edit. It reproduces the on-disk state of
    `vcc2026-val{A,B,C}-r1` exactly, and the assertion below is the one that fails the moment
    the field becomes a peer."""
    path = f"{competition_bundle.root}/{MANIFEST}"
    with open(path) as fh:
        man = json.load(fh)
    man.pop("de_pred_fingerprint", None)              # absent already; make the state explicit
    assert "de_pred_fingerprint" not in man
    with open(path, "w") as fh:
        json.dump(man, fh)

    read_real_bundle(competition_bundle.root)                        # still READABLE
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta=submission.meta)                   # still SCOREABLE
    assert out.filter(pl.col("metric") == "avg_score")["from_replicate"].item() is not None
    with pytest.raises(ValueError, match="de_pred_fingerprint"):     # and still GATED
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta={**submission.meta, "de_pred_fingerprint": "deadbeef"})


def test_a_bundle_needs_the_submission_meta(competition_bundle, submission):
    with pytest.raises(ValueError, match="user_meta"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root)


def test_a_bundle_and_the_loose_flags_are_mutually_exclusive(competition_bundle, submission):
    with pytest.raises(ValueError, match="both ends"):
        score_metrics(submission.agg, competition_bundle.baseline_agg,
                      real_bundle=competition_bundle.root, user_meta=submission.meta)


def test_score_time_overrides_never_reach_the_competition_column(
        competition_bundle, submission, tmp_path, caplog):
    """A score-time override moves from_baseline and must NOT move from_replicate -- the same
    rule a frozen scale has, for the same reason: a CLI flag that can move the headline number
    makes the frozen rule meaningless."""
    # ⚠️ The submission must be made BAD on the overridden member first: every knob here only
    # shapes the region BELOW the baseline, and the fixture's submission is a good prediction --
    # so on it the knob moves nothing at all, and a bare `tuned != plain` assertion is red for
    # a correct implementation while a bare "from_replicate unchanged" assertion is green for
    # an implementation that ignores the knob everywhere. Force the floor to engage, then pin
    # BOTH sides.
    #
    # ⚠️ The knob is `clamp_low`, NOT `penalty_cap`. It was `penalty_cap=1.0` while
    # `de_wilcoxon_lfc_nmae` carried the Box-Cox tail; under `scoring.ERROR_LINEAR` the cap is
    # inert on this member (its floor is a DECLARED `clamp_low`, not one derived from the cap),
    # so a cap-only override would leave both columns at -6.0 and the test would pass without
    # ever exercising a moved value. Cap-only inertness is pinned separately below.
    agg = pl.read_csv(submission.agg)
    base_nmae = pl.read_csv(competition_bundle.baseline_agg).filter(
        pl.col("statistic") == "mean")["de_wilcoxon_lfc_nmae"].item()
    bad = agg.with_columns(
        pl.when(pl.col("statistic") == "mean").then(pl.lit(base_nmae * 20.0))
          .otherwise(pl.col("de_wilcoxon_lfc_nmae")).alias("de_wilcoxon_lfc_nmae"))
    bad_path = str(tmp_path / "bad_agg.csv")
    bad.write_csv(bad_path)

    kw = dict(real_bundle=competition_bundle.root, user_meta=submission.meta)
    plain = score_metrics(bad_path, **kw)
    tuned = score_metrics(bad_path, **kw, clamp_low=-1.0)
    capped = score_metrics(bad_path, **kw, penalty_cap=1.0)

    def row(df):
        return df.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae")

    # the knob BITES on from_baseline: floored at the policy's -6 vs the supplied -1...
    assert row(plain)["from_baseline"].item() == pytest.approx(-6.0)
    assert row(tuned)["from_baseline"].item() == pytest.approx(-1.0)
    # ...and reaches from_replicate on NEITHER call.
    assert tuned["from_replicate"].to_list() == plain["from_replicate"].to_list()
    assert row(plain)["from_replicate"].item() == pytest.approx(-6.0)
    assert "policy-frozen" in caplog.text
    # `penalty_cap` alone is inert on this member now -- both columns, both ends. Pinned so the
    # reason the knob above changed is visible rather than looking like a stylistic edit.
    assert row(capped)["from_baseline"].item() == pytest.approx(-6.0)
    assert capped["from_replicate"].to_list() == plain["from_replicate"].to_list()


def test_a_caller_supplied_anchor_expect_is_REFUSED(competition_bundle, submission):
    """⚠️ Without this test the whole gate could be deleted and the suite would stay green.
    An expectation built from anywhere but the submission lets the anchor validate against
    itself, which passes for any artifact whatsoever -- so `real_bundle` refuses a supplied
    one rather than quietly overriding it."""
    from cell_eval2.score import expect_from_run_meta

    with pytest.raises(ValueError, match="anchor_expect"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta=submission.meta,
                      anchor_expect=expect_from_run_meta(submission.meta))


def test_adding_a_SCALE_changes_only_the_scale_column(competition_bundle, submission):
    """⚠️ `--scale` is orthogonal only if the two restorations agree. The scale path restores a
    dropped metric on the metric/score LISTS before the frame exists (`score.py:594-618`); the
    anchor path restores on the FRAME. With a row-removing override and a scale both in play,
    the anchor would find the row already present, skip its own restoration, and leave the
    frame in an order the frozen rule does not describe -- which also changes the order the
    average is summed in."""
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG

    kw = dict(real_bundle=competition_bundle.root, user_meta=submission.meta,
              overrides={"de_wilcoxon_sig_jaccard":
                         replace(CATALOG["de_wilcoxon_sig_jaccard"].scoring, scored=False)})
    plain = score_metrics(submission.agg, **kw)
    scaled = score_metrics(submission.agg, **kw, scale="low-random_high-1_v10")
    assert scaled["metric"].to_list() == plain["metric"].to_list()
    assert scaled["from_replicate"].to_list() == plain["from_replicate"].to_list()
    assert "low-random_high-1_v10" in scaled.columns


def test_a_per_metric_override_cannot_narrow_the_competition_average(
        competition_bundle, submission):
    """The other half of policy-freezing: `overrides={m: Scoring(scored=False)}` removes a
    metric's OUTPUT ROW, which would leave the competition average covering five members under
    a six-member anchor. Row restoration puts it back; this pins that it does."""
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG

    spec = CATALOG["de_wilcoxon_sig_jaccard"].scoring
    out = score_metrics(submission.agg, real_bundle=competition_bundle.root,
                        user_meta=submission.meta,
                        overrides={"de_wilcoxon_sig_jaccard": replace(spec, scored=False)})
    body = out.filter(pl.col("metric") != "avg_score")
    scored = body.filter(pl.col("from_replicate").is_not_null())["metric"].to_list()
    assert sorted(scored) == sorted(competition.competition_members())
    assert body.filter(pl.col("metric") == "de_wilcoxon_sig_jaccard")["from_baseline"] \
               .item() is None                       # restored row: no baseline value
    # ...and the frame order still matches the frozen rule after restoration.
    assert [m for m in body["metric"].to_list() if m in set(competition.competition_members())] \
        == competition.competition_payload()["member_order_in_frame"]


def test_ANY_anchor_requires_the_mean_statistic(competition_bundle, submission,
                                                loose_anchor_dir, anchor_expect):
    """The anchor's `replicate` is a mean of five aggregates; reading the user side at the
    median would divide a median gap by a mean span and call it a score.

    ⚠️ BOTH boundaries. The guard is scoped to the ANCHOR block -- wider than the bundle,
    narrower than the general path -- so a bundle AND a plain validated `--anchor` must both
    refuse median. Testing only the bundle leaves the scope unpinned in the direction that
    matters, since a bundle-only guard would pass that test while letting a loose anchor
    through."""
    with pytest.raises(ValueError, match="comparison_statistic"):
        score_metrics(submission.agg, real_bundle=competition_bundle.root,
                      user_meta=submission.meta, comparison_statistic="median")
    with pytest.raises(ValueError, match="comparison_statistic"):
        score_metrics(submission.agg, competition_bundle.baseline_agg,
                      anchor=loose_anchor_dir, anchor_expect=anchor_expect,
                      comparison_statistic="median")


def test_a_non_mean_statistic_without_ANY_anchor_still_works():
    """⚠️ The other edge of the same scope. Beside the `lfc_nmae_ref` guard the refusal would
    reach every baseline run -- a behaviour change nobody asked for.

    ⚠️ Deliberately NOT the competition aggregate. `aggregate_metrics_wide` gives a DERIVED
    metric a real value at `mean` only and NaN at every other statistic, so a vcc2026 pair
    read at the median has a NaN baseline for `expr_mse_unbiased_capped_norm` and raises
    `degenerate baseline` (measured) -- which would make this test green for a reason that has
    nothing to do with the anchor guard."""
    user = pl.DataFrame({"statistic": ["mean", "median"], "expr_mae": [0.30, 0.30]})
    base = pl.DataFrame({"statistic": ["mean", "median"], "expr_mae": [0.50, 0.50]})
    out = score_metrics(user, base, comparison_statistic="median")
    assert "from_baseline" in out.columns


def test_the_owned_columns_are_reserved():
    from cell_eval2.scales import RESERVED_COLUMNS

    assert {"from_replicate", "anchor_source", "anchor_digest", "real_bundle_id",
            "real_bundle_digest"} <= RESERVED_COLUMNS


def test_the_CLI_path_actually_reaches_the_scorer(competition_bundle, submission, tmp_path,
                                                  capsys):
    """⚠️ End-to-end through `main`, because the two defects this guards against are both in
    the CLI and invisible from `score_metrics`: `cli.py:601` rejects a score command carrying
    neither --baseline-agg nor --scale (which is exactly a bundle-only call), and `kw` is
    initialized after the point the bundle branch has to write into it."""
    from cell_eval2.cli import main

    out = tmp_path / "scored.csv"
    main(["score", "--user-agg", submission.agg, "--user-meta", submission.meta_path,
          "--real-bundle", competition_bundle.root, "-o", str(out)])
    got = pl.read_csv(out)
    assert got.filter(pl.col("metric") == "avg_score")["from_replicate"].item() is not None
    assert got.filter(pl.col("metric") == "avg_score")["from_baseline"].item() is None


@pytest.mark.parametrize("flag,val", [("--baseline-agg", "b.csv"), ("--anchor", "a/"),
                                      ("--anchor-cache", "c/"), ("--baseline-meta", "m.json"),
                                      ("--allow-config-mismatch", None)])
def test_the_CLI_refuses_every_conflicting_flag(competition_bundle, submission, flag, val):
    from cell_eval2.cli import main

    argv = ["score", "--user-agg", submission.agg, "--user-meta", submission.meta_path,
            "--real-bundle", competition_bundle.root, flag]
    if val is not None:
        argv.append(val)
    with pytest.raises(SystemExit, match="real-bundle"):
        main(argv)
