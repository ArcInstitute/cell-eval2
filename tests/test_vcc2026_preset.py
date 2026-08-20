"""The competition preset is v2 plus EXACTLY two declared deltas.

`tests/test_preset_yaml_matches_conventions.py` covers the version presets by parametrizing
over `_VERSION_CONVENTIONS`, so it cannot see this one. The trap it was written for applies
here too: `from_dict` backfills anything the YAML omits from the DATACLASS defaults, so a
preset that simply forgot a field would pass a naive "loads without error" test.
"""
import shlex

import pytest

from cell_eval2.config import EvalConfig
from cell_eval2.score import expect_from_run_meta

_DELTAS = {"metrics": ("vcc2026", "full"), "cache_strict": (True, False)}


def _flatten(cfg):
    """Config -> {dotted key: value}. Copied, not imported: cross-module test imports break
    under --import-mode=importlib (the conftest-import trap)."""
    out = {}
    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            else:
                out[f"{prefix}{k}"] = v
    walk(cfg.to_dict())
    return out


def _cfg_for(argv):
    from cell_eval2.cli import _build_cfg, _build_parser

    return _build_cfg(_build_parser().parse_args(shlex.split(argv)))


def test_vcc2026_is_v2_plus_exactly_the_declared_deltas():
    comp = _flatten(EvalConfig.from_preset("vcc2026"))
    v2 = _flatten(EvalConfig.for_version("v2"))
    differing = {k: (comp[k], v2[k]) for k in comp if comp[k] != v2[k]}
    assert differing == _DELTAS, (
        f"the competition preset drifted from v2: {differing}. Every field but `metrics` and "
        "`cache_strict` must match v2 exactly -- the competition scores v2's conventions, and "
        "`cell_eval2_version` is compared exactly, so a v2 change is caught by the version "
        "gate rather than needing a second frozen copy here."
    )


def test_the_yaml_DECLARES_every_field():
    """⚠️ The file's headline claim is "reading this tells you what the competition scores
    at", and only a RAW-YAML test can enforce it: `from_dict` backfills every omitted key from
    the dataclass defaults, so comparing two constructed configs passes no matter how much the
    file leaves out. `configs/v2.yaml` in fact omits allow_fractional_counts,
    autodetect_input_type, validate_input, device, pert_chunk, gather_threads and
    target_gene_map -- copying it verbatim would make the claim false."""
    import yaml
    from importlib.resources import files

    raw = yaml.safe_load(
        files("cell_eval2").joinpath("configs", "vcc2026.yaml").read_text(encoding="utf-8"))
    declared, expected = set(), set(_flatten(EvalConfig.from_preset("vcc2026")))

    def walk(d, prefix=""):
        for k, v in d.items():
            walk(v, f"{prefix}{k}.") if isinstance(v, dict) else declared.add(f"{prefix}{k}")

    walk(raw)
    assert declared == expected, (
        f"vcc2026.yaml omits {sorted(expected - declared)} and invents "
        f"{sorted(declared - expected)}")


def test_autodetect_input_type_stays_off():
    """Spec 3.7: with autodetect OFF, `_effective_input_type` returns the DECLARED type, so
    the fractional baseline arm and a counts submission report the same value and the pairing
    check cannot reject them for a difference neither side chose."""
    assert EvalConfig.from_preset("vcc2026").autodetect_input_type is False


def test_cache_strict_is_load_bearing_not_hygiene():
    """`expect_from_run_meta` REFUSES a metadata-only fingerprint, so a competition run built
    without cache_strict produces artifacts that cannot be anchor-scored at all."""
    assert EvalConfig.from_preset("vcc2026").cache_strict is True
    with pytest.raises(ValueError, match="metadata-only"):
        expect_from_run_meta({"source_fingerprint": "x", "source_fingerprint_strict": False,
                              "cell_eval2_version": "0.13.0", "anchor_semantic_identity": "y",
                              "anchor_metric_names": ["pds_cosine"]})


def test_preset_is_a_config_BASE_that_explicit_flags_override():
    """`--preset` and `--config` are two spellings of the same thing -- the config BASE -- so
    they are mutually exclusive. Only explicit flags and `--set` override a preset."""
    cfg = _cfg_for("run --preset vcc2026 -ar r.h5ad -ap p.h5ad")
    assert cfg.metrics == "vcc2026" and cfg.cache_strict is True
    assert _cfg_for("run --preset vcc2026 --no-cache-strict -ar r.h5ad "
                    "-ap p.h5ad").cache_strict is False
    assert _cfg_for("run --preset vcc2026 --set de.p_adj_threshold=0.01 -ar r.h5ad "
                    "-ap p.h5ad").de.p_adj_threshold == 0.01


def test_preset_and_config_together_are_refused(tmp_path):
    y = tmp_path / "c.yaml"
    EvalConfig.for_version("v2").to_yaml(str(y))
    with pytest.raises(SystemExit, match="both config BASES"):
        _cfg_for(f"run --preset vcc2026 --config {y} -ar r.h5ad -ap p.h5ad")


def test_an_unknown_preset_names_the_flag():
    with pytest.raises(SystemExit, match=r"--preset"):
        _cfg_for("run --preset nope -ar r.h5ad -ap p.h5ad")
