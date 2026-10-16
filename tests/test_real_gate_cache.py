"""Every cache that could replay a pre-#391 value sees the gate (issue #391, spec §7).

Spec: internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md.
"""
import dataclasses

from cell_eval2.catalog import CATALOG, resolve_metrics
from cell_eval2.config import EvalConfig
from cell_eval2.real_gate import RealGate
from cell_eval2.run import _REAL_GATE_SEMANTICS, _no_drop_policy, _result_config_digest

_COMPETITION_DIGEST = "fb5aa56b7436"        # main 807a60b9 -- #391 must not move it


def test_no_policy_for_either_competition_profile_or_for_v1():
    for profile in ("vcc", "vcc2026"):
        assert _no_drop_policy(resolve_metrics(profile)[0], "v2") is None, profile
    assert _no_drop_policy(resolve_metrics("full")[0], "v1") is None


def _gate(monkeypatch, name, gate, **kw):
    monkeypatch.setitem(CATALOG, name, dataclasses.replace(CATALOG[name], real_gate=gate, **kw))


def test_the_policy_is_by_value_and_sorted(monkeypatch):
    name = "de_wilcoxon_sig_recall"
    _gate(monkeypatch, name, RealGate("sig_set", min_gate_size=7), worst_value=None)
    policy = _no_drop_policy(resolve_metrics("de")[0], "v2", "wilcoxon")
    assert [name, None, ["sig_set", 7, "all"]] in policy
    assert policy == sorted(policy)


def test_a_selection_with_no_gate_has_no_policy(monkeypatch):
    """The term is for runs whose behaviour #391 changed. A worst value without a gate is the old
    fill, unchanged, so it must not move a warm cache key."""
    monkeypatch.setitem(CATALOG, "de_wilcoxon_precision", dataclasses.replace(
        CATALOG["de_wilcoxon_precision"], worst_value=0.0))
    assert _no_drop_policy(["de_wilcoxon_precision"], "v2", "wilcoxon") is None


def test_the_policy_reads_the_spec_the_deseq2_backend_dispatches(monkeypatch):
    """Under de.backend="deseq2" a requested de_wilcoxon_* name dispatches through its
    de_deseq2_* sibling, so a change to the SIBLING's gate must move the policy (codex, round 1)."""
    _gate(monkeypatch, "de_wilcoxon_lfc_spearman", RealGate("rank_corr"), worst_value=-1.0)
    _gate(monkeypatch, "de_deseq2_lfc_spearman", RealGate("rank_corr"), worst_value=-1.0)
    names = ["de_wilcoxon_lfc_spearman"]
    before = _no_drop_policy(names, "v2", "deseq2")
    assert before == [["de_deseq2_lfc_spearman", -1.0, ["rank_corr", 10, "all"]]]
    _gate(monkeypatch, "de_deseq2_lfc_spearman", RealGate("rank_corr", min_gate_size=20),
          worst_value=-1.0)
    assert _no_drop_policy(names, "v2", "deseq2") != before
    assert _no_drop_policy(names, "v2", "wilcoxon") == [
        ["de_wilcoxon_lfc_spearman", -1.0, ["rank_corr", 10, "all"]]]


def test_the_result_cache_key_moves_with_the_policy(monkeypatch):
    cfg = EvalConfig(version="v2", device="cpu")
    names = resolve_metrics("de")[0]
    name = "de_wilcoxon_lfc_spearman"

    def key():
        return _result_config_digest(cfg, de_backend_used=False, comparator="lognorm",
                                     no_drop_policy=_no_drop_policy(names, "v2", "wilcoxon"))

    _gate(monkeypatch, name, RealGate("rank_corr"))
    gated = key()
    assert gated != _result_config_digest(cfg, de_backend_used=False, comparator="lognorm")
    _gate(monkeypatch, name, RealGate("rank_corr", min_gate_size=3))
    assert key() != gated


def test_a_competition_run_keeps_its_cache_key():
    cfg = EvalConfig.from_preset("vcc2026")
    names = resolve_metrics("vcc2026")[0]
    assert _result_config_digest(cfg, de_backend_used=True, comparator="bulk_lognorm",
                                 no_drop_policy=_no_drop_policy(names, cfg.version,
                                                                cfg.de.backend)) == \
        _result_config_digest(cfg, de_backend_used=True, comparator="bulk_lognorm")


def test_the_partition_payload_carries_the_gate(monkeypatch):
    from cell_eval2.partition import _PARTIAL_SEMANTICS_SCHEMA, result_semantics

    name = "de_wilcoxon_lfc_spearman_pos"
    monkeypatch.setitem(CATALOG, name, dataclasses.replace(
        CATALOG[name], real_gate=RealGate("rank_corr", lfc_direction="pos")))
    sem = result_semantics([name, "expr_mae"], comparator="lognorm")
    # At least #391's 6; `test_partition` pins the current number exactly (7 since #397).
    assert sem["schema"] == _PARTIAL_SEMANTICS_SCHEMA >= 6
    assert sem["metric_real_gate"] == [[name, ["rank_corr", 10, "pos"]], ["expr_mae", None]]
    assert sem["real_gate_semantics"] == _REAL_GATE_SEMANTICS == 1


def test_the_competition_digest_does_not_move():
    from cell_eval2.competition import competition_digest

    assert competition_digest().startswith(_COMPETITION_DIGEST)


def test_the_anchor_identity_carries_the_policy_only_where_it_applies(monkeypatch):
    from _helpers import _counts_adata_fp64

    from cell_eval2.anchor import anchor_semantic_params

    _gate(monkeypatch, "delta_pearson", RealGate("delta_var", min_gate_size=None))
    real = _counts_adata_fp64(seed=0, per_group=6, g=5)
    for metrics, present in ((["delta_pearson"], True), (["expr_mae"], False)):
        cfg = EvalConfig(metrics=metrics, version="v2", device="cpu", input_type="counts",
                         pert_col="target")
        names = list(resolve_metrics(cfg.metrics, version=cfg.version)[0])
        params = anchor_semantic_params(cfg, real, names)
        assert ("no_drop_policy" in params) is present, metrics
        assert ("real_gate_semantics" in params) is present, metrics
