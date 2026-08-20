"""The shipped preset YAMLs and `_VERSION_CONVENTIONS` are TWO sources of truth for the
same thing, and nothing pinned them together until issue #282.

`EvalConfig.from_preset("v1")` reads `configs/v1.yaml`; `EvalConfig.for_version("v1")`
reads the `_VERSION_CONVENTIONS` dict. A field added to one and not the other does not
raise -- `from_dict` backfills anything the YAML omits from the DATACLASS defaults, which
are the v2 values. So an omitted v1 override silently gives v1 the v2 behaviour.

That is not hypothetical: adding `tie_policy` in #282 hit exactly this. The knob was set
in `_VERSION_CONVENTIONS["v1"]` and the v1 preset still resolved to the v2 default until
`configs/v1.yaml` was updated too.
"""
from dataclasses import fields, is_dataclass

import pytest

from cell_eval2.config import _VERSION_CONVENTIONS, EvalConfig


def _flatten(cfg):
    """{dotted field name: value} over the config's dataclass tree."""
    out = {}

    def walk(obj, prefix=""):
        for f in fields(obj):
            v = getattr(obj, f.name)
            if is_dataclass(v):
                walk(v, f"{prefix}{f.name}.")
            else:
                out[f"{prefix}{f.name}"] = v

    walk(cfg)
    return out


@pytest.mark.parametrize("version", sorted(_VERSION_CONVENTIONS))
def test_preset_yaml_resolves_identically_to_the_conventions_dict(version):
    """Every field, not just the ones a caller happens to read."""
    from_yaml = _flatten(EvalConfig.from_preset(version))
    from_dict = _flatten(EvalConfig.for_version(version))
    differing = {k: (from_yaml[k], from_dict[k])
                 for k in from_yaml if from_yaml[k] != from_dict[k]}
    assert not differing, (
        f"configs/{version}.yaml and _VERSION_CONVENTIONS[{version!r}] disagree on "
        f"{sorted(differing)} (yaml, dict): {differing}. Both must be updated together; "
        "an omitted YAML field silently backfills the v2 dataclass default."
    )


def _convention_items():
    """(version, dotted key, expected value) for every override, flattened one level."""
    for version, block in sorted(_VERSION_CONVENTIONS.items()):
        for key, value in block.items():
            items = value.items() if isinstance(value, dict) else [(None, value)]
            for sub, want in items:
                yield version, (f"{key}.{sub}" if sub else key), want


@pytest.mark.parametrize("version,dotted,want", list(_convention_items()))
def test_every_convention_override_survives_the_yaml_round_trip(version, dotted, want):
    """Each override is its OWN test case.

    ⚠️ This was a loop with a `pytest.skip` inside it, which aborts the whole parameter
    instance at the first default-valued field and leaves every later field unchecked --
    a test that silently stops testing. Caught by the Codex review on #282.
    Parameterizing per field means one uninteresting key can never mask the next.
    """
    assert _flatten(EvalConfig.from_preset(version))[dotted] == want, (
        f"{version}: {dotted} did not survive the YAML round trip; "
        f"_VERSION_CONVENTIONS says {want!r}"
    )


@pytest.mark.parametrize("version", sorted(_VERSION_CONVENTIONS))
def test_every_convention_key_is_written_down_in_the_yaml(version):
    """Presence, not just value. A key that resolves correctly only because `from_dict`
    backfilled the dataclass default is invisible to a reader of the preset, and stops
    being correct the moment that default moves. Checking the raw YAML is what makes the
    two sources safe rather than merely consistent today -- and it is what would catch
    `tie_policy` being dropped from v2.yaml, where the backfilled value happens to be
    right and every value-based assertion therefore still passes."""
    import yaml
    from importlib.resources import files

    raw = yaml.safe_load(
        files("cell_eval2").joinpath("configs", f"{version}.yaml").read_text("utf-8")
    )
    missing = []
    for key, value in _VERSION_CONVENTIONS[version].items():
        if isinstance(value, dict):
            missing += [f"{key}.{s}" for s in value if s not in (raw.get(key) or {})]
        elif key not in raw:
            missing.append(key)
    assert not missing, f"configs/{version}.yaml never mentions {missing}"


def test_v1_keeps_the_legacy_tie_policy_and_v2_takes_the_correction():
    """The #282 knob, stated as a fact about the shipped presets rather than a default."""
    assert EvalConfig.from_preset("v1").discrimination.tie_policy == "position"
    assert EvalConfig.from_preset("v2").discrimination.tie_policy == "midrank"
    assert EvalConfig.from_preset("cell-eval-0.7.6").discrimination.tie_policy == "position"
    # the bare default is the corrected one -- EvalConfig() == v2 is a load-bearing invariant
    assert EvalConfig().discrimination.tie_policy == "midrank"
