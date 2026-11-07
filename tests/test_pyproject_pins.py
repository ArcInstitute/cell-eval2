"""The `gpudge` extra pins gpudge exactly.

`settings.de.engine` records the engine name only, so the pin is what fixes the engine version
that a cell_eval2 release computes with.
"""
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pins_exactly(requirement: str) -> bool:
    parsed = Requirement(requirement.strip())
    specifiers = list(parsed.specifier)
    if parsed.name != "gpudge" or len(specifiers) != 1 or specifiers[0].operator != "==":
        return False
    try:
        Version(specifiers[0].version)
    except ValueError:
        return False
    return not specifiers[0].version.endswith(".*")


def _gpudge_requirements() -> list[str]:
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    return [r for r in extras["gpudge"] if Requirement(r).name == "gpudge"]


def test_the_gpudge_extra_pins_gpudge_exactly():
    requirements = _gpudge_requirements()
    assert len(requirements) == 1, requirements
    assert _pins_exactly(requirements[0]), requirements[0]


def test_the_pin_check_refuses_every_other_way_to_name_a_version():
    (live,) = _gpudge_requirements()
    version = str(list(Requirement(live).specifier)[0].version)
    head = version.rsplit(".", 1)[0]
    for loose in (f"gpudge>={version}", f"gpudge=={head}.*", f"gpudge~={head}", "gpudge",
                  f"gpudge>={head},<99", f"gpudge>={version}, ==99", f"gpudge=={version},!=99",
                  "gpudge==nonsense"):
        try:
            refused = not _pins_exactly(loose)
        except Exception:  # an unparsable requirement is refused too
            refused = True
        assert refused, loose
    assert _pins_exactly(f"gpudge=={version}") and _pins_exactly(f"gpudge == {version}")
