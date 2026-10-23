"""Record reproducible settings for #412, design §8.1–§9.

Every field ``baseline.config_digest`` covers, with its final value, minus the
metric selection, plus the run-time facts belongs here. Inert fields stay in:
over-strict, never wrong. The registry covers ``EvalConfig`` so a new field cannot
silently escape recording. A key added after 0.20.0 must declare ``neutral``:
the value reproducing behaviour before the key existed (design §9).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import Distribution

import numpy as np

from .baseline import DIGEST_EXEMPT_FIELDS, _package_provenance
from .catalog import CATALOG
from .config import EvalConfig
from .de_compute import _BACKEND_MODULE
from .run import RunFacts


NO_NEUTRAL = object()


@dataclass(frozen=True)
class SettingKey:
    """Describe recording and replay together to prevent drift (#412, design §9)."""

    path: tuple[str, ...]
    field: tuple[str, ...] | None
    fact: bool = False
    is_float: bool = False
    neutral: object = NO_NEUTRAL


# All 0.20.0 keys are required: inventing defaults for a missing key would claim
# reproducibility for an incomplete document (#412, design §9).
SETTINGS: tuple[SettingKey, ...] = (
    SettingKey(("pert_col",), ("pert_col",), fact=True),
    SettingKey(("control",), ("control",), fact=True),
    SettingKey(("version",), ("version",)),
    SettingKey(("control_source",), ("control_source",)),
    SettingKey(("input_type_real",), ("input_type",), fact=True),
    SettingKey(("input_type_pred",), None, fact=True),
    SettingKey(("max_counts_per_cell",), ("max_counts_per_cell",), is_float=True),
    SettingKey(("target_sum",), ("target_sum",), is_float=True),
    SettingKey(("allow_discrete",), ("allow_discrete",)),
    SettingKey(("autodetect_input_type",), ("autodetect_input_type",)),
    SettingKey(("validate_input",), ("validate_input",)),
    SettingKey(("filter", "filter_gene_min_cpm_cell"),
               ("filter", "filter_gene_min_cpm_cell"), is_float=True),
    SettingKey(("device",), ("device",)),
    SettingKey(("pert_chunk",), ("pert_chunk",)),
    SettingKey(("discrimination", "distance"), ("discrimination", "distance")),
    SettingKey(("discrimination", "rank_denominator"), ("discrimination", "rank_denominator")),
    SettingKey(("discrimination", "exclude_target_gene"), ("discrimination", "exclude_target_gene")),
    SettingKey(("discrimination", "embed_key"), ("discrimination", "embed_key")),
    SettingKey(("discrimination", "tie_policy"), ("discrimination", "tie_policy")),
    SettingKey(("discrimination", "exclusion_scope"), ("discrimination", "exclusion_scope")),
    SettingKey(("de", "p_adj_threshold"), ("de", "p_adj_threshold"), is_float=True),
    SettingKey(("de", "min_abs_log2fc"), ("de", "min_abs_log2fc"), is_float=True),
    SettingKey(("de", "sort_by"), ("de", "sort_by")),
    SettingKey(("de", "method"), ("de", "method")),
    SettingKey(("de", "nan_lfc_policy"), ("de", "nan_lfc_policy")),
    SettingKey(("de", "backend"), ("de", "backend")),
    SettingKey(("de", "replicate_col"), ("de", "replicate_col")),
    SettingKey(("de", "mean_calc"), ("de", "mean_calc")),
    SettingKey(("de", "epsilon"), ("de", "epsilon"), is_float=True),
    SettingKey(("de", "clip_value"), ("de", "clip_value"), is_float=True),
    SettingKey(("de", "fdr_scope"), ("de", "fdr_scope")),
    SettingKey(("de", "auc_pval_floor"), ("de", "auc_pval_floor")),
    SettingKey(("de", "auc_pval_floor_value"), ("de", "auc_pval_floor_value"), is_float=True),
    SettingKey(("de", "engine"), None, fact=True),
    SettingKey(("target_gene_map",), ("target_gene_map",)),
    SettingKey(("bulk_target_sum",), ("bulk_target_sum",), is_float=True),
    SettingKey(("comparator",), None, fact=True),
    SettingKey(("scoring_path",), None, fact=True),
)
NOT_RECORDED: frozenset[str] = frozenset({"metrics"}) | DIGEST_EXEMPT_FIELDS
FACT_PATHS: tuple[str, ...] = (
    "pert_col", "control", "input_type_real", "input_type_pred",
    "comparator", "de.engine", "scoring_path",
)


class SettingsMismatch(ValueError):
    """Refuse replay when resolved facts differ (#412, design §9)."""


def _scalar(value):
    """Keep numeric identity while unwrapping numpy scalars (#412, design §8.2)."""
    return value.item() if isinstance(value, np.generic) else value


def check_recordable(cfg: EvalConfig) -> None:
    """Report every value JSON cannot preserve before scoring (#412, design §8.2).

    ``baseline.config_digest`` stringifies values JSON cannot serialize natively;
    unwrapping them for replay would change that identity. Stringifying map keys
    also changes how ``de.resolve_target_genes`` matches labels.
    """
    errors = []
    for key in SETTINGS:
        if key.field is None:
            continue
        value = cfg
        for part in key.field:
            value = getattr(value, part)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            errors.append(f"{'.'.join(key.path)}={value!r}: pass a native Python value")
    if cfg.target_gene_map is not None:
        for label in cfg.target_gene_map:
            if not isinstance(label, str):
                errors.append(f"target_gene_map key {label!r}: pass a native Python value "
                              "with a string key (quote the label in YAML)")
    if errors:
        raise ValueError("Settings cannot be recorded exactly: " + "; ".join(errors))


def _direct_url(dist: str) -> str | None:
    """Read optional PEP 610 metadata without breaking a run (#412, design §8.2)."""
    try:
        return Distribution.from_name(dist).read_text("direct_url.json")
    except Exception:
        return None


def vcs_commit(raw: str | None) -> str | None:
    """Accept only an immutable commit, never a metadata URL (#412, design §8.2)."""
    if raw is None:
        return None
    try:
        info = json.loads(raw)
        if not isinstance(info, dict):
            return None
        vcs_info = info.get("vcs_info")
        if not isinstance(vcs_info, dict):
            return None
        commit = vcs_info.get("commit_id")
        if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
            return commit
    except Exception:
        return None
    return None


def engine_record(backend: str) -> dict:
    """Identify the engine without publishing paths or URLs (#412, design §8.2).

    Editable and local builds have no PEP 610 VCS commit, so builds sharing a
    version cannot be distinguished. This is documented rather than refused so
    development installs can write documents. On a result-cache hit the record
    describes the installed engine, as ``baseline.build_run_meta`` does.
    """
    dist = _BACKEND_MODULE[backend]
    provenance = _package_provenance(dist)
    return {
        "name": dist,
        "version": provenance["version"] if provenance is not None else None,
        "provenance": provenance["provenance"] if provenance is not None else None,
        "commit": vcs_commit(_direct_url(dist)),
    }


def _leaves(settings, prefix=()):
    # Maps and engine records are single settings; only EvalConfig's parameter
    # groups expand into paths (#412, design §8.2).
    for name, value in settings.items():
        path = prefix + (name,)
        if not prefix and name in ("filter", "discrimination", "de") and isinstance(value, dict):
            yield from _leaves(value, path)
        else:
            yield path


def _python_scalars(value):
    if isinstance(value, dict):
        return {key: _python_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_python_scalars(item) for item in value)
    return _scalar(value)


def build_settings(facts: RunFacts) -> dict:
    """Record final config and resolved facts, detecting drift (#412, design §8)."""
    result = facts.config.to_dict()
    for name in NOT_RECORDED | {"input_type"}:
        result.pop(name, None)
    result["device"] = facts.device
    # baseline.config_digest resolves only real-side DE; engine records either side
    # so a supplied real table replays with the same identity (#412, design §8.2–§9).
    result["de"]["backend"] = (facts.de_backend
                               if facts.de_backend_used and facts.de_real_computed
                               else facts.config.de.backend)
    result["de"]["engine"] = engine_record(facts.de_backend) if facts.de_backend_used else None
    result["input_type_real"] = facts.input_type_real
    result["input_type_pred"] = facts.input_type_pred
    result["comparator"] = facts.comparator
    result["scoring_path"] = facts.scoring_path

    expected = {key.path for key in SETTINGS}
    actual = set(_leaves(result))
    if actual != expected:
        raise RuntimeError(
            f"Settings registry/config drift: missing {sorted(expected - actual)!r}; "
            f"unexpected {sorted(actual - expected)!r}"
        )
    check_recordable(facts.config)
    # check_recordable has excluded scalars baseline.config_digest would stringify;
    # preserve the remaining numeric types on replay (#412, design §8.2).
    return _python_scalars(result)


def _get(value: Mapping, path: tuple[str, ...]):
    for part in path:
        value = value[part]
    return value


def settings_to_config(doc: Mapping) -> EvalConfig:
    """Replay every known setting without inventing identity (#412, design §9)."""
    # Valid JSON such as `[]` or `null` is no document; refuse it as data, not AttributeError
    # (Copilot, #418): the CLI turns a ValueError into a clean exit.
    if not isinstance(doc, Mapping):
        raise ValueError(f"expected a JSON object, got {type(doc).__name__}")
    if doc.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {doc.get('schema_version')!r}")
    suite = doc.get("suite")
    if not isinstance(suite, Mapping):
        raise ValueError("suite must be a mapping")
    if suite.get("name") != "cell_eval2":
        raise ValueError(f"Unsupported suite.name: {suite.get('name')!r}")
    metrics = doc.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    for name, entry in metrics.items():
        if name not in CATALOG:
            raise ValueError(f"Unknown metric: {name}")
        if not isinstance(entry, Mapping) or entry.get("version") != CATALOG[name].version:
            version = entry.get("version") if isinstance(entry, Mapping) else None
            raise ValueError(f"Metric {name} version {version!r} differs from "
                             f"catalog version {CATALOG[name].version!r}")
    settings = suite.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("suite.settings must be a mapping")
    unknown = set(_leaves(settings)) - {key.path for key in SETTINGS}
    if unknown:
        raise ValueError("Unknown settings: " + ", ".join(".".join(p) for p in sorted(unknown)))
    # Document order preserves config_digest for explicit and derived selections;
    # catalog order would change their identity (#412, design §9).
    data = {"metrics": list(metrics)}
    for key in SETTINGS:
        try:
            value = _get(settings, key.path)
        except (KeyError, TypeError):
            if key.neutral is NO_NEUTRAL:
                raise ValueError(f"Missing setting: {'.'.join(key.path)}") from None
            value = key.neutral
        if key.field is not None:
            current = data
            for part in key.field[:-1]:
                current = current.setdefault(part, {})
            current[key.field[-1]] = value
    return EvalConfig.from_dict(data)


def recorded_facts(settings: Mapping) -> dict[str, object]:
    """Keep the resolved facts separate from replayed config (#412, design §9)."""
    return {path: _get(settings, tuple(path.split("."))) for path in FACT_PATHS}


def check_facts(expected: Mapping[str, object], facts: RunFacts) -> None:
    """Refuse all resolved differences before scoring (#412, design §9)."""
    actual = recorded_facts(build_settings(facts))
    differences = [f"{path}: expected {expected[path]!r}, got {actual[path]!r}"
                   for path in FACT_PATHS if expected[path] != actual[path]]
    if differences:
        raise SettingsMismatch("Resolved settings differ: " + "; ".join(differences))


def changed_fields(base: EvalConfig, cfg: EvalConfig) -> list[str]:
    """Detect overrides that change JSON run identity (#412, design §9)."""
    changes = []

    def compare(left, right, prefix=()):
        if isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(left.keys() | right.keys()):
                if not prefix and name in DIGEST_EXEMPT_FIELDS:
                    continue
                path = prefix + (name,)
                if name not in left or name not in right:
                    changes.append(".".join(path))
                else:
                    compare(left[name], right[name], path)
        elif json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
            changes.append(".".join(prefix))

    compare(base.to_dict(), cfg.to_dict())
    return changes


def facts_checker(doc: Mapping) -> Callable[[RunFacts], None]:
    """Give API and CLI replays the same pre-scoring guard (#412, design §9)."""
    return lambda facts: check_facts(recorded_facts(doc["suite"]["settings"]), facts)
