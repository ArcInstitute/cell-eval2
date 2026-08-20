"""#266 / #182 / #193 -- the counts-vs-lognorm contract on the three drivers, plus the MEASURED
reachability facts that decide how far each fix may go.

Chunk 6's first task was a reachability check on these three, because their competition
reachability was inferred from #288 rather than measured. What the measurement found:

* The frozen `vcc2026` rule REJECTS a log1p submission on the in-memory path -- `validate_input:
  true` + `allow_fractional_counts: false` raise "declared input_type='counts' but values are
  fractional". #193's own mechanism (`to_normalization` returning lognorm unchanged) additionally
  needs the effective type to BE lognorm, which needs `autodetect_input_type`, which the rule pins
  off. So #193 as written is not competition-reachable.
* `allow_fractional_counts` is in `baseline.DIGEST_EXEMPT_FIELDS`, so flipping it does NOT move
  `config_digest` -- but it DOES move `anchor.semantic_identity`, which `real_bundle`'s bundle path
  compares via `score.expect_from_run_meta`. The competition bundle path therefore still refuses
  it; the loose `--baseline-agg` path does not. Pinned below, because the two gates disagreeing
  about which fields are semantic is the thing that could silently change.
* The shard/cell/cellstream drivers hard-code `autodetect=True`, so the frozen rule's
  `autodetect_input_type: false` does NOT bind them. That is why #288's "unreachable because
  autodetect is inside the frozen rule" does not transfer to this chunk -- and it is moot only
  because no streaming driver can reach a bundle at all (no CLI dispatch; `build_run_meta` is
  called only from cli.py's `run`).
"""
from dataclasses import replace

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cell_eval2.config import EvalConfig

PERTS = ["non-targeting", "GENE_A", "GENE_B", "GENE_C"]


def _counts(seed, n_per=40, n_genes=16):
    rng = np.random.default_rng(seed)
    X = rng.poisson(4.0, size=(n_per * len(PERTS), n_genes)).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.obs["target"] = np.repeat(PERTS, n_per)
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"GENE_{chr(65 + i)}" if i < 3 else f"g{i}" for i in range(n_genes)]
    return a


def _lognorm(a, target=1e6):
    b = a.copy()
    X = np.asarray(b.X.todense(), dtype=np.float64)
    libs = X.sum(axis=1, keepdims=True)
    libs[libs == 0] = 1.0
    b.X = sp.csr_matrix(np.log1p(X * (target / libs)).astype(np.float32))
    return b


# --- the frozen rule's own guards (the reachability measurement, pinned) ------------------------

def test_the_frozen_rule_rejects_a_log1p_submission_on_the_in_memory_path():
    """The gate that makes #193 unreachable under `vcc2026`, and it is not the one #288 named:
    `validate_input` + `allow_fractional_counts=False`, not `autodetect_input_type`."""
    from cell_eval2.run import compute_metrics
    cfg = replace(EvalConfig.from_preset("vcc2026"), device="cpu", outdir=None)
    with pytest.raises(ValueError, match="declared input_type='counts' but values are fractional"):
        compute_metrics(_lognorm(_counts(1)), _counts(2), config=cfg)


def test_the_frozen_rule_types_a_log1p_submission_as_counts_so_193s_mechanism_cannot_fire():
    """#193 needs the effective type to BE lognorm; the rule pins autodetect off, so the declared
    'counts' wins and the failure mode is double normalization, not a scale mismatch. Recorded
    because it is why the fix here is a warning on the MIXED case and not a change to
    to_normalization."""
    from cell_eval2.run import _effective_autodetect, _effective_input_type
    cfg = EvalConfig.from_preset("vcc2026")
    ln = _lognorm(_counts(1))
    assert cfg.autodetect_input_type is False
    assert not _effective_autodetect(cfg, side="pred")
    assert _effective_input_type(ln, cfg, side="pred") == "counts"


def test_allow_fractional_counts_is_invisible_to_config_digest_but_NOT_to_the_anchor():
    """The two gates disagree about which fields are semantic, and only one of them closes the
    hole. `allow_fractional_counts` is the single flag that disables the guard above, and
    `baseline.DIGEST_EXEMPT_FIELDS` deliberately exempts it (the baseline arm sets it pred-side and
    must still pair with an ordinary run) -- so `config_digest` cannot see it. The competition
    bundle path is saved by `anchor.semantic_identity`, which lists it in `_SEMANTIC_FIELDS` and IS
    compared, via `score.expect_from_run_meta` -> `anchor.validate_anchor`.

    Pinned so that removing it from either place shows up here as a failure rather than as a
    submission that scores 0.003 where it should score 0.811."""
    from cell_eval2.anchor import _SEMANTIC_FIELDS, semantic_identity
    from cell_eval2.baseline import DIGEST_EXEMPT_FIELDS, config_digest
    from cell_eval2.catalog import resolve_metrics

    base = replace(EvalConfig.from_preset("vcc2026"), device="cpu")
    loose = replace(base, allow_fractional_counts=True)
    assert "allow_fractional_counts" in DIGEST_EXEMPT_FIELDS
    assert (config_digest(base, comparator="bulk_lognorm")
            == config_digest(loose, comparator="bulk_lognorm")), \
        "config_digest is expected NOT to see this flag -- the baseline arm depends on that"

    assert "allow_fractional_counts" in _SEMANTIC_FIELDS
    real = _counts(2)
    names, _ = resolve_metrics(base.metrics, version=base.version)
    assert semantic_identity(base, real, names) != semantic_identity(loose, real, names), \
        "the ANCHOR gate must still see it, or the bundle path has no guard at all"


def test_no_streaming_driver_writes_the_run_meta_the_bundle_path_requires():
    """Why #266/#182 cannot reach a competition bundle regardless of the gate: `score --real-bundle`
    requires the submission's own run_meta.json (`score.expect_from_run_meta`), and
    `baseline.build_run_meta` is called from exactly one place -- cli.py's `run`, the in-memory
    path. The streaming drivers write partials, which carry no such record."""
    import inspect

    from cell_eval2 import cli
    assert "build_run_meta" in inspect.getsource(cli)
    for mod_name in ("scale", "cellstream", "h5ad_manifest", "rowstore", "partition"):
        mod = __import__(f"cell_eval2.{mod_name}", fromlist=["x"])
        assert "build_run_meta" not in inspect.getsource(mod), mod_name


# --- #182: the contract is enforced where it is stated -----------------------------------------

def test_compute_de_streaming_refuses_non_counts_itself():
    """#182's second half: the raw-counts contract was STATED in this function's docstring and
    checked nowhere. It took no input_type argument at all, so a lognorm archive was handed a
    library-size normalization target and the DE numbers came back plausible and meaningless.
    Raises BEFORE the backend/GPU check, so it is testable on CPU."""
    from cell_eval2.de_compute import compute_de_streaming
    with pytest.raises(NotImplementedError, match="requires a RAW-COUNTS archive"):
        compute_de_streaming(
            "unused.shad", backend="gpudge", reference=None, groupby="target",
            mean_calc="arithmetic", epsilon=1e-9, target_sum=1e6, clip_value=None,
            fdr_scope="per_pert", filter_gene_min_cpm_cell=5.0, input_type="lognorm",
        )


def test_compute_de_streaming_defaults_to_counts_so_existing_callers_are_unchanged():
    """The default must not turn every current call site into a raise; it fails later, at the
    backend gate, exactly as before."""
    from cell_eval2.de_compute import compute_de_streaming
    with pytest.raises(ValueError, match="requires the gpudge backend"):
        compute_de_streaming(
            "unused.shad", backend="pdex", reference=None, groupby="target",
            mean_calc="arithmetic", epsilon=1e-9, target_sum=1e6, clip_value=None,
            fdr_scope="per_pert", filter_gene_min_cpm_cell=5.0,
        )


# --- #266: the shard gate now covers the DE family ---------------------------------------------

def _shad(tmp_path, name, adata):
    pytest.importorskip("cellstream")
    from cellstream import write_sharded
    p = tmp_path / f"{name}.shad"
    write_sharded(adata, str(p), group_by="target", reference="non-targeting", overwrite=True)
    return str(p)


@pytest.mark.parametrize("pred_ln,real_ln", [(True, False), (False, True), (True, True)])
def test_score_streaming_refuses_a_lognorm_shad_for_DE_ONLY_metrics(tmp_path, pred_ln, real_ln):
    """#266 exactly: `metrics=["de_wilcoxon_overlap"]` on a lognorm .shad with a NUMERIC
    target_sum must raise. It previously returned plausible DE numbers computed from re-normalized
    lognorm values -- `main` had no non-counts gate for the DE family at all, and #264 narrowed
    only the anndata one.

    Parameterized over WHICH side is lognorm, as the issue asks: a gate inspecting one side passes
    a both-sides test.
    """
    from cell_eval2.scale import score_streaming
    pred = _shad(tmp_path, "pred", _lognorm(_counts(1)) if pred_ln else _counts(1))
    real = _shad(tmp_path, "real", _lognorm(_counts(2)) if real_ln else _counts(2))
    cfg = replace(EvalConfig.v2(), metrics=["de_wilcoxon_overlap"], device="cpu",
                  validate_input=False)
    assert cfg.target_sum == 1e6, "the defect needs a NUMERIC target_sum"
    with pytest.raises(ValueError, match=r"requires raw counts on BOTH sides.*DE metrics"):
        score_streaming(pred, real, config=cfg)


def test_score_streaming_still_refuses_a_lognorm_shad_for_the_anndata_family(tmp_path):
    """The pre-existing half of the gate must keep working, and keep naming the anndata metrics."""
    from cell_eval2.scale import score_streaming
    pred = _shad(tmp_path, "pred", _lognorm(_counts(1)))
    real = _shad(tmp_path, "real", _counts(2))
    cfg = replace(EvalConfig.v2(), metrics=["pds_cosine"], device="cpu", validate_input=False)
    with pytest.raises(ValueError, match="requires raw counts on BOTH sides.*anndata metrics"):
        score_streaming(pred, real, config=cfg)


def test_the_gate_message_names_the_fractional_counts_false_positive(tmp_path):
    """MEASURED while checking reachability: `guess_is_lognorm` tests only for a fractional
    per-cell total, so a SCALED or AVERAGED counts matrix -- a dispersed baseline arm, say --
    resolves to 'lognorm' and trips this gate too. The gate cannot distinguish the two, so it says
    so instead of asserting the data is log-normalized."""
    from cell_eval2 import norm as _norm
    from cell_eval2.scale import score_streaming
    frac = _counts(1)
    frac.X = frac.X.multiply(1.37).tocsr()          # fractional COUNTS, not log1p
    assert _norm.guess_is_lognorm(frac), "fixture must actually look fractional"
    pred = _shad(tmp_path, "predfrac", frac)
    real = _shad(tmp_path, "realc", _counts(2))
    cfg = replace(EvalConfig.v2(), metrics=["de_wilcoxon_overlap"], device="cpu",
                  validate_input=False, allow_fractional_counts=True)
    with pytest.raises(ValueError, match="fractional COUNTS rather than log-normalized"):
        score_streaming(pred, real, config=cfg)


def test_a_counts_shad_pair_is_unaffected_by_the_widened_gate(tmp_path, monkeypatch):
    """The gate must not fire on the ordinary case. Stops at the DE call (gpudge needs a GPU), so
    reaching it is the assertion."""
    import cell_eval2.scale as scale
    pred = _shad(tmp_path, "pred", _counts(1))
    real = _shad(tmp_path, "real", _counts(2))
    cfg = replace(EvalConfig.v2(), metrics=["de_wilcoxon_overlap"], device="cpu",
                  validate_input=False)

    class _Reached(Exception):
        pass

    monkeypatch.setattr(scale, "compute_de_streaming",
                        lambda *a, **k: (_ for _ in ()).throw(_Reached()))
    with pytest.raises(_Reached):
        scale.score_streaming(pred, real, config=cfg)


def test_the_shard_peek_ignores_the_frozen_rules_autodetect_pin(tmp_path):
    """The finding that breaks the #288-based inference: `_shard_effective_input_type` hard-codes
    autodetect=True, so this driver resolves from the DATA even under a config that pins
    `autodetect_input_type: false`. That is deliberate -- the shard accumulator takes no
    input_type -- and it is exactly why the gate above is effective. Pinned so the two facts stay
    connected."""
    from cell_eval2.scale import _shard_effective_input_type
    frozen = EvalConfig.from_preset("vcc2026")
    assert frozen.autodetect_input_type is False and frozen.input_type == "counts"
    ln = _shad(tmp_path, "ln", _lognorm(_counts(1)))
    assert _shard_effective_input_type(ln, frozen, side="pred") == "lognorm"


# --- #193: the in-memory mixed-scale warning ---------------------------------------------------

def test_mixed_effective_types_under_a_numeric_target_sum_warn(caplog):
    """#193's option 1, the smallest honest change. A raise is not available: a counts-real /
    lognorm-pred pair is an explicitly SUPPORTED path (#155 spec 8) and a test pins that it still
    runs."""
    import logging

    from cell_eval2.run import _warn_mixed_library_scale
    cfg = replace(EvalConfig.v2(), target_sum=1e6)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        _warn_mixed_library_scale(cfg, {"pred": "lognorm", "real": "counts"})
    assert "effective input types DIFFER" in caplog.text
    assert "14.9x-25.2x" in caplog.text          # the measured cost, not a vague caution
    assert "target_sum=None" in caplog.text      # and the remedy


@pytest.mark.parametrize("types,target_sum", [
    ({"pred": "counts", "real": "counts"}, 1e6),      # agree -> nothing to warn about
    ({"pred": "lognorm", "real": "lognorm"}, 1e6),    # agree
    ({"pred": "lognorm", "real": "counts"}, None),    # None resolves to the control median
])
def test_the_mixed_scale_warning_does_not_fire_when_it_should_not(caplog, types, target_sum):
    import logging

    from cell_eval2.run import _warn_mixed_library_scale
    cfg = replace(EvalConfig.v2(), target_sum=target_sum)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        _warn_mixed_library_scale(cfg, types)
    assert "effective input types DIFFER" not in caplog.text


def test_the_mixed_scale_warning_cannot_fire_under_the_frozen_rule():
    """Not asserted from the config text but derived through the resolver both sides use: the rule
    pins autodetect off, so both effective types are the declared 'counts' whatever the data is,
    and the warning's precondition (they differ) is unreachable."""
    from cell_eval2.run import _effective_input_type
    cfg = EvalConfig.from_preset("vcc2026")
    for pred, real in ((_lognorm(_counts(1)), _counts(2)), (_counts(1), _lognorm(_counts(2)))):
        eff = {"pred": _effective_input_type(pred, cfg, side="pred"),
               "real": _effective_input_type(real, cfg, side="real")}
        assert eff["pred"] == eff["real"] == "counts"


# --- the cell-layout sibling: a DECLARED-only gate is not a gate --------------------------------

def _csad(tmp_path, name, adata):
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive
    p = tmp_path / f"{name}.csad"
    # codec="zstd": the default pfordelta codec needs pyfastpfor, which the scale extra does not
    # pull in, and this test is about the gate rather than the encoding.
    write_cell_archive(adata, str(p), group_by="target", reference="non-targeting",
                       overwrite=True, codec="zstd")
    return str(p)


@pytest.mark.parametrize("pred_ln,real_ln", [(True, False), (False, True), (True, True)])
def test_score_streaming_cell_refuses_lognorm_archives_under_a_DECLARED_counts_config(
        tmp_path, pred_ln, real_ln):
    """The same hole as #266 on the cell layout, and worse: `score_streaming_cell`'s gate keyed on
    `cfg.input_type` -- the DECLARED value -- so a config declaring counts over lognorm archives
    walked straight through the very check whose message says the values "would be silently
    mis-normalized". Neither streaming driver validates input at all (`scale.py` has no
    `validate_input`/`check_scale_limit` call), so nothing downstream caught it either.
    """
    from cell_eval2.scale import score_streaming_cell
    pred = _csad(tmp_path, "pred", _lognorm(_counts(1)) if pred_ln else _counts(1))
    real = _csad(tmp_path, "real", _lognorm(_counts(2)) if real_ln else _counts(2))
    cfg = replace(EvalConfig.v2(), metrics=["expr_mae"], device="cpu", input_type="counts",
                  validate_input=False)
    with pytest.raises(NotImplementedError, match=r"stored values are not counts.*DECLARED"):
        score_streaming_cell(pred, real, config=cfg)


def test_score_streaming_cell_keeps_its_declared_lognorm_message(tmp_path):
    """A caller who DECLARES lognorm should still get the "counts-only, use the materialize path"
    message rather than a data-shaped one -- which is why the effective check sits after the
    declared one."""
    from cell_eval2.scale import score_streaming_cell
    cfg = replace(EvalConfig.v2(), metrics=["expr_mae"], device="cpu", input_type="lognorm")
    with pytest.raises(NotImplementedError, match="is counts-only: both its gpudge DE"):
        score_streaming_cell("unused", "unused", config=cfg)


def test_score_streaming_cell_accepts_a_counts_pair(tmp_path):
    """The gate must not fire on the ordinary case. metrics=["expr_mae"] dispatches no DE, so this
    runs to completion on CPU."""
    from cell_eval2.scale import score_streaming_cell
    pred = _csad(tmp_path, "predc", _counts(1))
    real = _csad(tmp_path, "realc", _counts(2))
    cfg = replace(EvalConfig.v2(), metrics=["expr_mae"], device="cpu", input_type="counts",
                  validate_input=False)
    out = score_streaming_cell(pred, real, config=cfg)
    assert set(out["metric"].unique()) == {"expr_mae"}
    assert out.height == len(PERTS) - 1          # one row per non-control perturbation


# --- codex-review round 2: the allow_discrete bypass, and the empty-selection path ---------------

@pytest.mark.parametrize("driver", ["shard", "cell"])
def test_allow_discrete_TRUE_cannot_talk_a_lognorm_archive_past_either_gate(tmp_path, driver):
    """`resolve_input_type` returns "counts" the moment `allow_discrete` is set, WITHOUT inspecting
    a value (its `if allow_discrete: return "counts"` short-circuit) -- so honouring the config's
    value let a genuinely lognorm archive declare its way past the very gate that was just added.
    Round 2 of codex-review; the round-1 tests all used the default False and could not see it."""
    from cell_eval2.scale import score_streaming, score_streaming_cell
    cfg = replace(EvalConfig.v2(), metrics=["expr_mae"], device="cpu", input_type="counts",
                  validate_input=False, allow_discrete=True)
    if driver == "shard":
        p, r = _shad(tmp_path, "p", _lognorm(_counts(1))), _shad(tmp_path, "r", _counts(2))
        with pytest.raises(ValueError, match="requires raw counts on BOTH sides"):
            score_streaming(p, r, config=cfg)
    else:
        p, r = _csad(tmp_path, "p", _lognorm(_counts(1))), _csad(tmp_path, "r", _counts(2))
        with pytest.raises(NotImplementedError, match="stored values are not counts"):
            score_streaming_cell(p, r, config=cfg)


def test_the_public_strict_check_also_ignores_allow_discrete(tmp_path):
    """`strict=True` is advertised as "does this archive agree with what I declared". With
    allow_discrete honoured it answered "counts" for a lognorm archive and never raised, i.e. the
    advertised check was vacuous for that config (codex-review round 2). An explicit
    allow_discrete= still wins, so a caller who wants the ordinary resolver can ask for it."""
    from cell_eval2 import cell_archive_input_type
    ln = _csad(tmp_path, "ln", _lognorm(_counts(1)))
    cfg = replace(EvalConfig.v2(), input_type="counts", allow_discrete=True)
    assert cell_archive_input_type(ln, config=cfg) == "counts"          # ordinary resolution
    with pytest.raises(ValueError, match="resolve to input_type='lognorm'"):
        cell_archive_input_type(ln, config=cfg, strict=True)            # safety resolution
    # explicit override is still honoured
    assert cell_archive_input_type(ln, config=cfg, strict=True, allow_discrete=True) == "counts"


def test_h5ad_manifest_pair_guard_cannot_be_bypassed_by_allow_discrete(tmp_path):
    """`h5ad_manifest._resolve_input_type_h5ad` feeds the pred/real equality guard in
    `score_h5ad_manifest`, so honouring allow_discrete let a counts/lognorm pair resolve as
    counts/counts and walk through the guard that exists to refuse it."""
    import anndata as ad

    from cell_eval2 import h5ad_manifest
    ln = _lognorm(_counts(1))
    path = tmp_path / "ln.h5ad"
    ad.AnnData(X=ln.X, obs=ln.obs, var=ln.var).write_h5ad(path)
    cfg = replace(EvalConfig.v2(), input_type="counts", allow_discrete=True)
    counts_path = tmp_path / "counts.h5ad"
    c = _counts(2)
    ad.AnnData(X=c.X, obs=c.obs, var=c.var).write_h5ad(counts_path)
    # A PAIR, not one call (codex-review round 3): the single-call version would also pass if
    # allow_discrete=True made BOTH files resolve to "lognorm", which would still bypass the
    # equality guard. What matters is that the two resolve DIFFERENTLY, so the guard can see it.
    assert h5ad_manifest._resolve_input_type_h5ad(str(path), cfg=cfg) == "lognorm"
    assert h5ad_manifest._resolve_input_type_h5ad(str(counts_path), cfg=cfg) == "counts"


def test_an_EMPTY_metric_selection_still_refuses_a_lognorm_shad(tmp_path):
    """codex-review round 2 corrected an "unreachable" claim of mine: with an empty resolved
    selection both unsafe lists are empty, the #266 gate does not fire, and #155's target_sum guard
    is what stops the run. Pinned so the retained guard is not dead code."""
    from cell_eval2.scale import score_streaming
    ln = _shad(tmp_path, "ln", _lognorm(_counts(1)))
    cfg = replace(EvalConfig.v2(), metrics=[], target_sum=None, device="cpu",
                  input_type="lognorm", validate_input=False)
    with pytest.raises(NotImplementedError, match="target_sum=None requires a RAW-COUNTS archive"):
        score_streaming(ln, ln, config=cfg)


# --- RULED 2026-08-17: keep the digest exemption, WARN when the allowance is load-bearing -------

def test_the_fractional_allowance_WARNS_when_it_actually_let_something_through(caplog):
    """The ruling's whole point is that the gap was SILENT. Measured harm: the same log1p
    submission scored as counts moves every scored member (sig_jaccard 0.811 -> 0.003) with a
    byte-identical `config_digest`. Removing the exemption would break every baseline pairing, so
    the ruling is warn-and-keep -- which is only worth anything if the warning actually fires.

    Asserted on the CONTENT, not merely that something was logged: the message has to name the
    exemption, or a reader has no way to know why the pairing check will stay quiet."""
    import logging

    from cell_eval2 import run

    frac = _lognorm(_counts(7))          # fractional values, declared below as counts
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        run._validate_input_once(frac, "counts", allow_fractional=True)
    assert "LOAD-BEARING" in caplog.text
    assert "DIGEST_EXEMPT_FIELDS" in caplog.text
    assert "config_digest" in caplog.text


def test_the_fractional_allowance_is_SILENT_when_inert(caplog):
    """The other half, and the one that keeps this from becoming a line people learn to ignore.
    An all-integer matrix would have passed validation with the flag OFF, so the allowance changed
    nothing and there is nothing to report.

    This is the assertion that fails if the warning is ever moved out from behind
    `_is_all_integer` -- e.g. keyed on the flag alone, which would fire on every counts run that
    merely PASSES the flag, including every strict-integer baseline rebuild.

    ⚠️ Asserts NO warning record at all, not merely the absence of one phrase (codex-review round 8,
    P3): the first version only excluded "LOAD-BEARING", so a reworded -- or entirely unrelated --
    warning on inert integer input would have passed it. Silence is the claim, so silence is what is
    checked."""
    import logging

    from cell_eval2 import run

    ints = _counts(8)                    # genuinely integral: the flag is inert here
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        run._validate_input_once(ints, "counts", allow_fractional=True)
    offenders = [r.getMessage() for r in caplog.records
                 if r.name == "cell_eval2.run" and r.levelno >= logging.WARNING]
    assert offenders == [], f"expected total silence on inert input, got {offenders}"


def test_the_warnings_claim_about_the_digest_is_still_TRUE():
    """The message asserts a fact about another module's constant. Couple them, so removing the
    exemption turns the warning into a lie HERE rather than in a user's log: if
    `allow_fractional_counts` ever leaves `DIGEST_EXEMPT_FIELDS`, this fails and the text must be
    rewritten (the run would then correctly fail the pairing check instead)."""
    import inspect

    from cell_eval2 import run
    from cell_eval2.baseline import DIGEST_EXEMPT_FIELDS

    src = inspect.getsource(run._warn_fractional_allowance_used)
    assert "DIGEST_EXEMPT_FIELDS" in src, "the message must name where the exemption lives"
    assert "allow_fractional_counts" in DIGEST_EXEMPT_FIELDS, \
        "the warning tells users config_digest cannot see this flag -- it now can, so fix the text"


def test_the_allowance_is_PRED_side_only_so_the_warning_cannot_describe_the_real_side():
    """A scope claim the warning's wording depends on: it says "the pred side" without being told
    which side it is on. That is sound only because `_val_allow_fractional` returns the flag for
    'pred' and False for 'real' -- the real side is always validated strictly."""
    from cell_eval2 import run

    cfg = replace(EvalConfig.v2(), allow_fractional_counts=True, device="cpu")
    assert run._val_allow_fractional(cfg, side="pred") is True
    assert run._val_allow_fractional(cfg, side="real") is False


def test_a_results_cache_hit_on_a_BACKED_pred_is_not_left_silent(caplog):
    """codex-review round 8, P1 -- the coverage hole in the warning above, which I had missed.

    A backed (path) pred skips the up-front validation loop (run.py:1377-1381), and a results-cache
    hit returns (run.py:1447-1468) before the deferred validation sites run. So a warm run on path
    inputs classifies nothing and the loud warning cannot fire. Forcing it to fire would make a
    cache hit read X -- the one cost the results cache exists to avoid -- so the cache-hit path gets
    a scan-free notice that reports the FLAG rather than its effect.

    Three arms, because the value of this notice is entirely in not misfiring: it must fire for a
    backed pred, stay silent for an in-memory one (already classified up front, so warning again
    would double-report), and stay silent with the flag off."""
    import logging

    from cell_eval2 import run

    class _Backed:
        isbacked = True

    cfg = replace(EvalConfig.v2(), allow_fractional_counts=True, device="cpu")

    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        run._warn_fractional_allowance_unclassified(cfg, _Backed())
    assert "RESULTS CACHE" in caplog.text
    assert "DIGEST_EXEMPT_FIELDS" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        run._warn_fractional_allowance_unclassified(cfg, _counts(9))   # in-memory: not backed
        run._warn_fractional_allowance_unclassified(
            replace(cfg, allow_fractional_counts=False), _Backed())    # flag off
    offenders = [r.getMessage() for r in caplog.records
                 if r.name == "cell_eval2.run" and r.levelno >= logging.WARNING]
    assert offenders == [], f"the cache-hit notice must not misfire, got {offenders}"


def test_the_cache_hit_notice_is_actually_WIRED_to_the_cache_hit_branch():
    """The test above exercises the helper; this one pins that `_run_metrics` CALLS it on the
    results-cache-hit path. Without this, deleting the call site leaves the helper fully tested and
    entirely dead -- which is exactly how a guard ends up passing its own tests while protecting
    nothing (that shape has appeared repeatedly in this review)."""
    import inspect

    from cell_eval2 import run

    src = inspect.getsource(run._run_metrics)
    hit = src.index('cached = pred_store.get("results"')
    ret = src.index("return cached", hit)
    assert "_warn_fractional_allowance_unclassified" in src[hit:ret], \
        "the notice must be emitted inside the results-cache-hit branch, before it returns"
