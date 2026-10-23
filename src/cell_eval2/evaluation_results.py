"""Metric shapes and identity-ledger rules for Arc Bench's evaluation_results schema version 1.

The headline statistic and axis describe the shapes that document publishes (#412, design
§5.3/§6.5); the ledger rules keep them stable under contract constraint 11.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .catalog import CATALOG, METRICS_VERSION

if TYPE_CHECKING:
    import polars as pl

    from .config import EvalConfig
    from .run import RunFacts


logger = logging.getLogger(__name__)

FILENAME = "evaluation_results.json"
SCHEMA_VERSION = 1
SUITE_NAME = "cell_eval2"


# Design §6.5: these metrics repeat one context-wide value over perturbation rows.
SCALAR_METRICS = frozenset({"de_wilcoxon_nsig_spearman", "de_deseq2_nsig_spearman"})


def headline_statistic(name: str) -> str:
    """Use the document's shape, not the row aggregation, for identity (design §6.5, #412)."""
    if CATALOG[name].derived is not None:
        return "ratio_of_sums"
    return "value" if name in SCALAR_METRICS else "mean"


def metric_axis(name: str) -> str | None:
    """Share axis names with the writer so constraint 11 catches renames (design §6.5)."""
    return "perturbation" if headline_statistic(name) == "mean" else None


def metric_identity() -> dict:
    """Record every catalog shape for the CI and release checks (design §5.3, #412)."""
    return {
        "metrics_version": METRICS_VERSION,
        "metrics": {
            name: {
                "version": CATALOG[name].version,
                "tolerance": CATALOG[name].tolerance,
                "statistic": headline_statistic(name),
                "direction": CATALOG[name].scoring.direction,
                "axis": metric_axis(name),
            }
            for name in sorted(CATALOG)
        },
    }


def identity_violations(ledger: Mapping, current: Mapping) -> list[str]:
    """Check all published identities, including retired ones (design §5.3, constraint 11).

    Sharing these rules with the release gate prevents a regenerated working ledger from
    hiding a shape change that required a version bump (#412).
    """
    history = ledger["history"]
    active = {k: max(int(n) for n in history[k]) for k in ledger["active"]}
    versions = {k: entry["version"] for k, entry in current["metrics"].items()}
    violations = []
    if current["metrics_version"] < ledger["metrics_version"]:
        old, new = ledger["metrics_version"], current["metrics_version"]
        violations.append(f"metrics_version decreased: {old} -> {new}")
    if versions != active and current["metrics_version"] <= ledger["metrics_version"]:
        for k in sorted(active.keys() | versions.keys()):
            if k not in active:
                violations.append(f"{k}: added without a metrics_version bump")
            elif k not in versions:
                violations.append(f"{k}: removed without a metrics_version bump")
            elif active[k] != versions[k]:
                old, new = active[k], versions[k]
                violations.append(f"{k}: version {old} -> {new} without a metrics_version bump")
    for k, entry in sorted(current["metrics"].items()):
        if k not in history:
            continue
        n = entry["version"]
        previous = history[k].get(str(n))
        if previous is not None and entry["tolerance"] != previous["tolerance"]:
            violations.append(f"{k} v{n}: tolerance changed without a version bump")
        if previous is not None and any(
            entry[field] != previous[field] for field in ("statistic", "direction", "axis")
        ):
            violations.append(
                f"{k} v{n}: headline statistic, direction or axis changed without a version bump"
            )
        if n < max(int(v) for v in history[k]):
            violations.append(f"{k}: version decreased")
    return violations


def extend_ledger(ledger: Mapping | None, current: Mapping) -> dict:
    """Append release identities without losing retired shapes (design §5.3, constraint 11).

    Refuse breaches before recording a release; keep prior entries unchanged so every
    earlier document remains covered. String version keys survive JSON round trips (#412).
    """
    if ledger is not None:
        violations = identity_violations(ledger, current)
        if violations:
            raise ValueError("; ".join(violations))
    history = deepcopy(ledger["history"]) if ledger is not None else {}
    for k, entry in sorted(current["metrics"].items()):
        history.setdefault(k, {}).setdefault(str(entry["version"]), {
            field: entry[field] for field in ("tolerance", "statistic", "direction", "axis")
        })
    return {
        "metrics_version": current["metrics_version"],
        "active": sorted(current["metrics"]),
        "history": history,
    }


def no_document_reason(cfg: EvalConfig) -> str | None:
    """Explain excluded legacy runs before any output is written (#412, design §6.6)."""
    if cfg.version == "v1":
        return 'version="v1" writes no evaluation_results.json (deprecated)'
    if cfg.de.backend == "deseq2":
        return 'de.backend="deseq2" writes no evaluation_results.json (deprecated)'
    return None


def _direct_url() -> str | None:
    """Share optional distribution metadata handling with settings (#412, design §6.2)."""
    from .settings import _direct_url as distribution_url
    return distribution_url("cell-eval2")


def suite_commit() -> str | None:
    """Use the engine record's immutable-commit rule for the suite (#412, design §6.2)."""
    from .settings import vcs_commit
    return vcs_commit(_direct_url())


def _document_value(value, path):
    """Represent undefined results as null, but refuse infinities (#412, design §6.5)."""
    if value is None or math.isnan(value):
        return None
    if not math.isfinite(value):
        raise ValueError(f"{path}: expected a finite number")
    return float(value)


def build_document(df: pl.DataFrame, facts: RunFacts, *, metrics: Sequence[str]) -> dict:
    """Join run facts and catalog shapes without moving CSV values (#412, design §6).

    Imports stay lazy as in `baseline._version_lazy`: cellstream imports this module
    before the package assigns its version, and settings itself imports run.
    """
    import polars as pl

    from . import __version__
    from .catalog_text import AXES, FACETS, STATISTICS, de_method, metric_text
    from .run import _derived_value, _wide_metric_names, metric_agg
    from .settings import build_settings

    reason = no_document_reason(facts.config)
    if reason is not None:
        raise ValueError(reason)
    labels = list(facts.labels)
    _check_labels(labels, "facts.labels")
    label_set = set(labels)
    rows = {}
    for perturbation, name, value in df.select("perturbation", "metric", "value").iter_rows():
        if perturbation not in label_set:
            raise ValueError(f"{name}.perturbation: {perturbation!r} is not a label")
        if name not in CATALOG:
            raise ValueError(f"metrics.{name}: unknown metric")
        key = (perturbation, name)
        if key in rows:
            raise ValueError(f"metrics.{name}.values: duplicate row for {perturbation!r}")
        rows[key] = _document_value(value, f"metrics.{name}.values[{perturbation!r}]")
    names = _wide_metric_names(df, metrics, caller="evaluation_results")
    for name in names:
        if name not in CATALOG:
            raise ValueError(f"metrics.{name}: unknown metric")
    ordered = dict.fromkeys([*metrics, *names])
    entries = {}
    for name in ordered:
        spec = CATALOG[name]
        text = metric_text(name)
        headline = {"statistic": headline_statistic(name), "direction": spec.scoring.direction}
        for bound in ("min", "max"):
            if getattr(text, bound) is not None:
                headline[bound] = getattr(text, bound)
        entry = {"version": spec.version, "headline": headline,
                 "display_name": text.display_name, "description": text.description}
        aliases = ([spec.v1_name] if spec.v1_name else []) + list(spec.aliases)
        if aliases:
            entry["aliases"] = aliases
        entry["facets"] = {"family": text.family, "kind": spec.kind}
        method = de_method(name)
        if method is not None:
            entry["facets"]["de_method"] = method
        if spec.derived is not None:
            entry["statistics"] = {"ratio_of_sums": _document_value(
                _derived_value(df, spec, require_components=True),
                f"metrics.{name}.statistics.ratio_of_sums")}
        elif name in SCALAR_METRICS:
            values = {value for (_, metric), value in rows.items()
                      if metric == name and value is not None}
            if len(values) > 1:
                raise ValueError(f"metrics.{name}.statistics.value: scalar rows disagree")
            entry["statistics"] = {"value": next(iter(values), None)}
        else:
            values = [rows.get((label, name)) for label in labels]
            nn = df.filter(pl.col("metric") == name)["value"].drop_nans()
            # Mirror `aggregate_metrics_wide` expression by expression, in frame order;
            # reordering by labels can move floating-point reductions (#412, design §6.5).
            med = nn.median()
            centre = med if metric_agg(name) == "median" else nn.mean()
            entry["statistics"] = {"count": len(labels), "null_count": values.count(None)}
            for stat, value in zip(("mean", "std", "min", "max", "median"),
                                   (centre, nn.std(), nn.min(), nn.max(), med)):
                entry["statistics"][stat] = _document_value(value, f"metrics.{name}.statistics.{stat}")
            entry["axis"] = metric_axis(name)
            entry["values"] = values
        entries[name] = entry
    doc = {
        "schema_version": SCHEMA_VERSION,
        "suite": {"name": SUITE_NAME, "version": __version__, "commit": suite_commit(),
                  "settings": build_settings(facts)},
        "statistics": deepcopy(STATISTICS),
        "axes": {"perturbation": {**AXES["perturbation"], "labels": labels}}
                if any("axis" in entry for entry in entries.values()) else {},
        "facets": deepcopy(FACETS),
        "metrics": entries,
    }
    check_document(doc)
    return doc


def _check_labels(labels, path):
    """Keep axis positions unambiguous even on a scalar-only run (#412, design §6.4)."""
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path}: expected a nonempty label array")
    seen = set()
    for i, label in enumerate(labels):
        if not isinstance(label, str) or not label:
            raise ValueError(f"{path}[{i}]: expected a nonempty string label")
        if label in seen:
            raise ValueError(f"{path}[{i}]: repeated label {label!r}")
        seen.add(label)


def check_document(doc: Mapping) -> None:
    """Enforce schema data rules and document constraints without jsonschema at runtime.

    The schema fixture tests this independent implementation (#412, contract §13–14).
    Accept Mapping objects so callers and disk-loaded documents share the same checks.
    """
    def fail(path, message):
        raise ValueError(f"{path}: {message}")

    def json_value(value, path):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    fail(f"{path}[{key!r}]", "object key must be a string")
                json_value(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                json_value(item, f"{path}[{i}]")
        elif isinstance(value, float):
            if not math.isfinite(value):
                fail(path, "expected a finite number")
        elif value is not None and not isinstance(value, (str, int, bool)):
            fail(path, "expected a JSON value")

    def obj(value, path, required=(), optional=()):
        if not isinstance(value, Mapping):
            fail(path, "expected an object")
        for key in required:
            if key not in value:
                fail(f"{path}.{key}", "required field missing")
        for key in value:
            if key not in (*required, *optional):
                fail(f"{path}.{key}", "unexpected field")

    def text(value, path):
        if not isinstance(value, str) or not value:
            fail(path, "expected nonempty text")

    def slug(value, path):
        if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
            fail(path, "expected a slug")

    def number(value, path, *, nullable=True):
        if value is None and nullable:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(path, "expected a number" + (" or null" if nullable else ""))
        if isinstance(value, float) and not math.isfinite(value):
            fail(path, "expected a finite number")

    def named(value, path, *, nonempty=False):
        if not isinstance(value, Mapping):
            fail(path, "expected an object")
        if nonempty and not value:
            fail(path, "expected a nonempty object")
        for name in value:
            slug(name, f"{path}.{name}")

    def declaration(value, path, *, required=(), optional=()):
        obj(value, path, ("display_name", "description", *required), optional)
        for key in ("display_name", "description"):
            text(value[key], f"{path}.{key}")

    json_value(doc, "document")
    obj(doc, "document", ("schema_version", "suite", "statistics", "axes", "metrics"), ("facets",))
    if type(doc["schema_version"]) is not int or doc["schema_version"] != SCHEMA_VERSION:
        fail("schema_version", "expected 1")
    suite = doc["suite"]
    obj(suite, "suite", ("name", "version", "commit", "settings"))
    slug(suite["name"], "suite.name")
    text(suite["version"], "suite.version")
    commit = suite["commit"]
    if commit is not None and (not isinstance(commit, str) or
                               re.fullmatch(r"[0-9a-f]{40}", commit) is None):
        fail("suite.commit", "expected null or 40 lowercase hex digits")
    if not isinstance(suite["settings"], Mapping):
        fail("suite.settings", "expected an object")

    statistics, axes, facets, metrics = (doc["statistics"], doc["axes"],
                                         doc.get("facets", {}), doc["metrics"])
    for group, entries in (("statistics", statistics), ("axes", axes),
                           ("facets", facets), ("metrics", metrics)):
        named(entries, group, nonempty=group in ("statistics", "metrics"))
    for name, entry in statistics.items():
        path = f"statistics.{name}"
        declaration(entry, path, optional=("combine",))
        if "combine" in entry and entry["combine"] not in ("mean", "sum", "none"):
            fail(f"{path}.combine", "expected mean, sum or none")
    for name, entry in axes.items():
        path = f"axes.{name}"
        declaration(entry, path, required=("labels",))
        _check_labels(entry["labels"], f"{path}.labels")
    for name, entry in facets.items():
        declaration(entry, f"facets.{name}")

    aliases, used_axes = set(), set()
    for name, entry in metrics.items():
        path = f"metrics.{name}"
        declaration(entry, path, required=("version", "headline", "statistics"),
                    optional=("axis", "values", "aliases", "facets"))
        version = entry["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            fail(f"{path}.version", "expected an int >= 1 (not bool)")
        headline = entry["headline"]
        obj(headline, f"{path}.headline", ("statistic", "direction"), ("min", "max"))
        slug(headline["statistic"], f"{path}.headline.statistic")
        if headline["direction"] not in ("higher", "lower", None):
            fail(f"{path}.headline.direction", "expected higher, lower or null")
        for bound in ("min", "max"):
            if bound in headline:
                number(headline[bound], f"{path}.headline.{bound}", nullable=False)
        if "min" in headline and "max" in headline and headline["min"] >= headline["max"]:
            fail(f"{path}.headline.min", "min must be less than max")
        stats = entry["statistics"]
        named(stats, f"{path}.statistics", nonempty=True)
        for stat, value in stats.items():
            if stat not in statistics:
                fail(f"{path}.statistics.{stat}", "undeclared statistic")
            number(value, f"{path}.statistics.{stat}")
        stat = headline["statistic"]
        if stat not in statistics or stat not in stats:
            fail(f"{path}.headline.statistic", f"statistic {stat!r} must be declared and present")
        value = stats[stat]
        if value is not None and (("min" in headline and value < headline["min"]) or
                                  ("max" in headline and value > headline["max"])):
            fail(f"{path}.statistics.{stat}", "headline statistic is outside its bounds")
        if "aliases" in entry:
            if not isinstance(entry["aliases"], list):
                fail(f"{path}.aliases", "expected an alias array")
            for i, alias in enumerate(entry["aliases"]):
                where = f"{path}.aliases[{i}]"
                if not isinstance(alias, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", alias) is None:
                    fail(where, "expected a label")
                if alias in aliases or alias in metrics:
                    fail(where, f"alias {alias!r} is repeated or equals a metric key")
                aliases.add(alias)
        named(entry.get("facets", {}), f"{path}.facets")
        for facet, value in entry.get("facets", {}).items():
            if facet not in facets:
                fail(f"{path}.facets.{facet}", "undeclared facet")
            text(value, f"{path}.facets.{facet}")
        if ("axis" in entry) != ("values" in entry):
            fail(f"{path}.axis/values", "axis and values must be present together")
        if "axis" in entry:
            axis = entry["axis"]
            slug(axis, f"{path}.axis")
            if axis not in axes:
                fail(f"{path}.axis", f"undeclared axis {axis!r}")
            used_axes.add(axis)
            values = entry["values"]
            if not isinstance(values, list) or len(values) != len(axes[axis]["labels"]):
                fail(f"{path}.values", "values must be an array with one value per axis label")
            for i, value in enumerate(values):
                number(value, f"{path}.values[{i}]")
    for axis in axes:
        if axis not in used_axes:
            fail(f"axes.{axis}", "axis is unused")


def write_document(doc: Mapping, path) -> None:
    """Validate before replacing a document, then emit strict UTF-8 JSON (#412, §6.1)."""
    check_document(doc)
    serialized = json.dumps(doc, allow_nan=False, indent=2) + "\n"
    destination = Path(path)
    candidate = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary = None
    try:
        # A sibling keeps os.replace atomic; open uses normal umask permissions so
        # readers retain access to published outputs (#412, design §6.1).
        with open(candidate, "x", encoding="utf-8") as stream:
            temporary = candidate
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_document(path) -> dict:
    """Reject ambiguous JSON before replay or validation (#412, contract §13.1)."""
    def constant(token):
        raise ValueError(f"non-strict JSON token {token}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"repeated JSON key {key!r}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=constant, object_pairs_hook=pairs)


def remove_stale_document(outdir) -> bool:
    """Prevent an earlier run's document from describing a failed rerun (#412, §6.6)."""
    path = Path(outdir) / FILENAME
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    logger.info("Removed earlier evaluation document %s", path)
    return True
