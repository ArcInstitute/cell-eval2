"""The golden end-to-end scoring pin: a committed fixture scored through the real path.

⚠️ WHY THIS FILE EXISTS. `competition_digest()` pins each member's scoring POLICY and
`tests/test_scales.py` pins the scale registry, but until #381 nothing in the suite pinned an
actual SCORE -- so a dependency bump or an accidental change to a metric's arithmetic could
reissue every published number silently. Nothing enforces `uv.lock` at install time, which is the
exposure #338 reported. ⚠️ This PR pins `pdex==0.3.0`, the engine behind the FOUR `de_*` members
specifically -- the other two are computed elsewhere -- so a residual exposure remains in the rest
of the numeric stack, including `numpy`, `scipy`, `scanpy` and `polars` (which computes the
`aggregate_metrics_wide` means on this very path). All are declared as lower bounds and reach a
fresh CI runner at whatever version resolves that day; that list is illustrative, not exhaustive.

It is also what backstops #381: `cell_eval2_version` is no longer compared on the bundle path,
so the tripwire is this file rather than a version string. That is a better trade in both
directions -- it fires on every push instead of once per release, and it fires on the number
rather than on a proxy.

⚠️ THE BACKEND AND DEVICE ARE PINNED, AND THAT IS LOAD-BEARING. `EvalConfig.de.backend` and
`device` both default to `"auto"`, which resolves BY HOST -- `pdex`/`cpu` on a CI runner,
`gpudge`/`cuda` on an Arc GPU node. The GPU gate runs this whole suite on such a node, so an
`auto` config here would compute different DE numbers there and this file would be red on every
gate run: #224's failure mode exactly, where a test left at `auto` only ever exercised one of two
branches. The resolved values are ASSERTED below so that removing the pin fails loudly instead of
quietly changing the numbers.

Pinning them does not make the bundle diagnostic: `competition._RULE_EXCLUDED` contains `device`
and `_RULE_EXCLUDED_NESTED` contains `("de", "backend")`, so `rule_config_hash` normalizes both
away and the bundle carries the real `rule_digest`.

⚠️ ONE MEMBER IS NOT BIT-REPRODUCIBLE. `de_wilcoxon_lfc_nmae` moves by 1 ULP run-to-run on a
SINGLE host, through a threaded floating-point reduction in the BLAS/OpenMP layer -- measured over
12 runs across four threading configurations (default, `num_threads=1`, `OMP_NUM_THREADS=1`,
`POLARS_MAX_THREADS=1`) and both CI dependency stacks (py3.11 with numpy 2.4.6/scipy 1.17.1/scanpy
1.11.5, py3.12 with numpy 2.5.2/scipy 1.18.1/scanpy 1.12.3). Setting all four knobs to 1 makes the
whole record bit-identical, including across those two stacks -- but this file does not do that,
because the knobs must be set before numpy and polars are imported and a conftest hook is too
late. So the mode is per-member DATA with its reason recorded, never a blanket `rtol`: a future
loosening then shows up in review as a mode change rather than as a widened constant.
"""
import math
from dataclasses import replace

import polars as pl
import pytest

from cell_eval2 import competition
from cell_eval2.baseline import build_run_meta
from cell_eval2.config import EvalConfig
from cell_eval2.real_bundle import build_real_bundle, read_real_bundle
from cell_eval2.run import aggregate_metrics_wide, compute_metrics, metric_output_names
from cell_eval2.score import score_metrics


@pytest.fixture(scope="module")
def counts_bundle_inputs_module():
    """`conftest.counts_bundle_inputs` is function-scoped and a module-scoped fixture cannot
    depend on it -- so both delegate to ONE plain function in `tests/_helpers.py` (Step 0). Never
    `from conftest import`: pytest loads conftest as a PLUGIN, and importing it directly breaks
    under `--import-mode=importlib`. `tests/_helpers.py`'s own docstring records that Copilot
    flagged exactly this at seven sites on PR #209, and `pyproject.toml`'s
    `pythonpath = ["tools", "tests"]` exists to make `_helpers` importable under ANY import mode.
    """
    from _helpers import counts_bundle_inputs

    return counts_bundle_inputs()


EXACT = "exact"
ULP = "ulp"

#: Per-member ULP budget. A member ABSENT from this dict is compared bit-exactly. Keeping the
#: budget PER MEMBER is what makes the "a future loosening shows up as a per-row change" claim
#: true: a single shared constant would widen both tolerant rows at once while both kept the same
#: mode and the same recorded rationale -- exactly the invisible loosening the mode table exists to
#: prevent (Copilot, PR #382).
#:
#: Four ULPs is 4x the 1 ULP measured across 12 runs. A window this narrow cannot hide a real
#: regression: real changes in this repo move members by 1e-3 to 0.9, twelve-plus orders of
#: magnitude above it.
ULP_BUDGET = {
    "de_wilcoxon_lfc_nmae": 4,   # the one member that jitters; see the module docstring
    "avg_score": 4,              # DERIVED from it -- see the note on GOLDEN below
}

#: metric -> (from_baseline, from_replicate, mode). Measured single-threaded and verified
#: bit-identical on py3.11 and py3.12.
#:
#: `de_wilcoxon_lfc_nmae` is `ULP` because it is the one member that jitters (see the module
#: docstring). `avg_score` is `ULP` for a DERIVED reason and not because it was observed to move
#: -- it never moved in 12 runs. It is the mean of six members, one of which jitters, so a 1-ULP
#: move in one of six shifts the mean by ~1/6 ULP; that rounds away UNLESS the true mean sits near
#: a rounding boundary. These particular values keep it clear of one, which makes its stability a
#: property of these numbers rather than of the arithmetic. Asserting it exactly would buy nothing
#: -- the six members are each asserted individually, five of them exactly -- and would risk a
#: rare flake that is expensive to diagnose.
GOLDEN = {
    "de_wilcoxon_direction_fidelity_yield_raw": (0.9704761904761906, 1.2937951543538633, EXACT),
    "de_wilcoxon_direction_reach_raw":          (1.0,                1.0245506356861025, EXACT),
    "de_wilcoxon_sig_jaccard":                  (0.6878876341050253, 3.6016986293466267, EXACT),
    "expr_mse_unbiased_capped_norm":            (1.0,                0.9982689365791101, EXACT),
    "pds_cosine":                               (1.0,                1.0,                EXACT),
    "de_wilcoxon_lfc_nmae":                     (0.8081499602056582, 1.4773040275015314, ULP),
}

#: `avg_score` carries no `from_baseline`: #276 part C NULLS it deliberately, so that an
#: unmigrated consumer reads a null rather than a well-formed, no-longer-official number under
#: the label it already reads.
GOLDEN_AVG_SCORE = 1.5659362305778723

#: Four identities worth freezing. THREE are genuine SUBMISSION-TIME gate inputs
#: (`config_digest` is a compared peer; `real_fingerprint` and `anchor_semantic_identity` are what
#: `validate_anchor` compares). `anchor_config_hash` is NOT: it is compared only in
#: `build_real_bundle`'s leg-consistency check and merely required to be PRESENT at score time, so
#: it is a build-time consistency and provenance identity (codex checkpoint-1 round 2 P2).
#: `cache.config_hash` hashes deterministic JSON over primitive config values and carries no
#: package version -- its `version` field is the scoring convention `"v2"` -- so the constant is
#: release-neutral. All four measured stable across both CI dependency stacks and across the one
#: KNOWN host-dependent resolution branch, which the pin above closes:
#: `baseline.config_digest` RESOLVES `device="auto"` and the DE backend before digesting, so an
#: unpinned config would digest `gpudge`/`cuda` on a GPU host and `pdex`/`cpu` on a CI runner.
#:
#: ⚠️ That is NOT the same as host-independence, and the difference matters for exactly one of the
#: four: `cache.fingerprint_adata` hashes generated NumPy arrays and native byte buffers, so
#: `real_fingerprint` could still move on a different architecture. CONFIRMED on GitHub's
#: `ubuntu-latest` runners under both matrix legs (3.11 and 3.12) before this PR merged, which is
#: a second machine and not merely a second venv -- so the pin is not resting on one host. Other
#: architectures remain uncovered, and a runner-specific change in this value is NOT to be waved
#: through as harmless (Copilot, PR #382).
#:
#: ⚠️ `anchor_digest` is DELIBERATELY ABSENT, and pinning it would defeat this file's own purpose:
#: `anchor.anchor_digest` hashes `cell_eval2_version` into the digest, so a frozen hex here would
#: go red on every package-version bump -- reintroducing exactly the coupling #381 removes, inside
#: the test meant to backstop it. PROVED: the official `-r4` anchor frame hashed with `"0.15.0"`
#: reproduces `-r3`'s `anchor_digest` byte for byte, with byte-identical parquets either side. It is
#: also not a gate -- `score_metrics` recomputes it as output provenance and nothing compares it.
#: `anchor_config_hash` is the version-neutral identity that belongs here instead
#: (codex checkpoint-1 P1).
GOLDEN_DIGESTS = {
    "config_digest": "23ea1f2e71afef1a3bd691e173bc9d3644f18f5367ec141777d4d548956ac33d",
    "anchor_config_hash": "dc264f8e7de4c22c67a1231f085d9a9522f942e358e87d689ee549c8253ce990",
    "real_fingerprint": "61bc2128999f094e4b72f6301da89e145b99560459311fdf390c22b4e5536583",
    "anchor_semantic_identity": "c7254a5808154f5206263dcfb56cdd034dac8328cb0d7a662ac9af08271c2fcb",
}


def _pinned_cfg():
    """The competition preset with the two host-resolved fields nailed down."""
    cfg = EvalConfig.from_preset("vcc2026")
    return replace(cfg, device="cpu", de=replace(cfg.de, backend="pdex"))


def _assert_cell(metric, column, got, want, mode):
    # ⚠️ VALIDATE THE MODE FIRST. Dispatching on `mode == EXACT` and treating everything else as
    # ULP means a typo in the table -- or an accidental edit of the `ULP` constant -- silently
    # converts a bit-exact pin into a tolerant one, which is the exact failure this table exists to
    # make visible. Checked before the `want is None` short-circuit so a null row cannot skip it
    # either (Copilot, PR #382).
    assert mode in (EXACT, ULP), f"{metric}.{column}: unknown comparison mode {mode!r}"
    if want is None:
        assert got is None, f"{metric}.{column}: expected null, got {got!r}"
        return
    assert got is not None, f"{metric}.{column}: expected {want!r}, got null"
    if mode == EXACT:
        assert metric not in ULP_BUDGET, (
            f"{metric} is EXACT in GOLDEN but carries a ULP_BUDGET entry; one of the two is wrong")
        assert got == want, f"{metric}.{column}: expected {want!r} exactly, got {got!r}"
        return
    budget_ulps = ULP_BUDGET.get(metric)
    assert budget_ulps is not None, (
        f"{metric} is ULP-mode in GOLDEN but has no ULP_BUDGET entry; the budget is per member")
    # ⚠️ `math.ulp(0.0)` is 5e-324, so a 4-ULP budget on a zero expectation admits errors of
    # EITHER sign whose magnitude is at most the fourth positive subnormal (the comparison is on
    # `abs`) -- not bit-exact, but so far below any real signal that the row's
    # DECLARED tolerance would bear no relation to what it enforces. Fail loudly rather than
    # substituting a floor: Gemini suggested `1e-16`, which is an ABSOLUTE tolerance (`atol`)
    # picked here rather than measured, and this file's doctrine is per-member tolerances each
    # with a recorded reason. No current ULP row is zero; if one ever is, its tolerance has to be
    # decided and justified deliberately, not inherited from a fallback.
    assert want != 0.0, (
        f"{metric}.{column} is ULP-mode with a ZERO expectation; math.ulp(0.0) is 5e-324, so the "
        "budget would be meaningless. Give this row an explicit absolute tolerance and record why.")
    budget = budget_ulps * math.ulp(want)
    assert abs(got - want) <= budget, (
        f"{metric}.{column}: {got!r} is {abs(got - want) / math.ulp(want):.1f} ULP from "
        f"{want!r}, over the {budget_ulps} ULP budget"
    )


@pytest.fixture(scope="module")
def golden_scored(counts_bundle_inputs_module, tmp_path_factory):
    """MODULE-SCOPED: the bundle build is ~17 s and six tests read it. A function-scoped fixture
    would pay that six times on each of the two CI matrix legs for no added coverage
    (codex checkpoint-1 P2). `tmp_path_factory` is the module-scoped counterpart of `tmp_path`.

    MEASURED before adopting the wider scope: `build_real_bundle`, `compute_metrics` and
    `score_metrics` leave every SCORE-BEARING part of the three AnnData inputs untouched -- a
    content hash over `X`, `obs_names`, `var_names`, the `obs` columns and the `target` column is
    unchanged after each call, and so are the shapes. They DO attach private validation memos
    (`run._validate_input_once` sets `_validated_inputs`, `_check_scale_limit_once` sets
    `_validated_scale_limits`, and the GPU path can set `_precomputed_row_total_max`), so "mutates
    nothing" would be false (codex checkpoint-1 round 2 P1). It does not matter here: this fixture
    returns only the bundle and the scored frame, never the AnnData, and the six tests read both
    without writing.
    """
    baseline_pred, real, submission_pred = counts_bundle_inputs_module
    cfg = _pinned_cfg()
    tmp_path = tmp_path_factory.mktemp("golden")
    root = tmp_path / "golden-bundle"
    build_real_bundle(real, baseline_pred, config=cfg, outdir=root,
                      bundle_id="vcc2026-golden-r1")
    bundle = read_real_bundle(root)
    agg = aggregate_metrics_wide(compute_metrics(submission_pred, real, config=cfg),
                                metrics=metric_output_names(cfg))
    agg_path = tmp_path / "submission_agg.csv"
    agg.write_csv(agg_path)
    meta = build_run_meta(cfg, real, submission_pred)
    out = score_metrics(str(agg_path), real_bundle=str(root), user_meta=meta)
    return bundle, out


def test_the_pinned_config_actually_resolved(golden_scored):
    """Guard the guard. If `auto` ever leaks back in, every number below is host-dependent and
    this file goes red on the GPU gate for a reason that reads like a product bug."""
    bundle, _out = golden_scored
    assert bundle.manifest["resolved_de_backend"] == "pdex"
    assert bundle.manifest["resolved_device"] == "cpu"


def test_the_bundle_is_a_COMPETITION_bundle(golden_scored):
    """Pinning the device and backend must not make it diagnostic -- both are normalized out of
    `rule_config_hash`. Asserted as a RELATIONSHIP, not against a fourth copy of the frozen hex:
    `tests/test_competition_rule.py` owns that constant."""
    bundle, _out = golden_scored
    assert bundle.manifest["rule_digest"] == competition.competition_digest()
    assert bundle.manifest["rule_mismatches"] == []


def test_the_frozen_identities_hold(golden_scored):
    """The four identities frozen with this fixture. All measured stable across both
    CI dependency stacks."""
    bundle, _out = golden_scored
    for field, want in GOLDEN_DIGESTS.items():
        assert bundle.manifest[field] == want, f"{field} moved: {bundle.manifest[field]!r}"


def test_the_frame_has_exactly_the_expected_ROWS(golden_scored):
    """⚠️ Guards against a vacuous pass. `{r["metric"]: r for r in ...}` collapses duplicates, so
    a frame carrying two `pds_cosine` rows or two `avg_score` rows would score identically to a
    correct one and `.to_dicts()[0]` would silently read the first (codex checkpoint-1 P2)."""
    _bundle, out = golden_scored
    metrics = out["metric"].to_list()
    assert out.height == len(competition.competition_members()) + 1, metrics
    assert len(set(metrics)) == len(metrics), f"duplicate metric rows: {metrics}"
    assert metrics.count("avg_score") == 1


def test_the_scored_members_hold_their_golden_values(golden_scored):
    _bundle, out = golden_scored
    rows = {r["metric"]: r for r in out.iter_rows(named=True)}
    # ⚠️ Membership against the CANONICAL SIX, never against whatever rows came back: deriving
    # the expectation from the output makes five members and seven members both pass.
    body = [m for m in rows if m != "avg_score"]
    assert sorted(body) == sorted(competition.competition_members())
    assert sorted(GOLDEN) == sorted(competition.competition_members())
    for metric, (want_base, want_rep, mode) in GOLDEN.items():
        row = rows[metric]
        _assert_cell(metric, "from_baseline", row["from_baseline"], want_base, mode)
        _assert_cell(metric, "from_replicate", row["from_replicate"], want_rep, mode)


def test_avg_score_holds_its_golden_value(golden_scored):
    _bundle, out = golden_scored
    row = out.filter(pl.col("metric") == "avg_score").to_dicts()[0]
    _assert_cell("avg_score", "from_replicate", row["from_replicate"], GOLDEN_AVG_SCORE, ULP)
    _assert_cell("avg_score", "from_baseline", row["from_baseline"], None, EXACT)
