"""cell_eval2 — clean reimplementation of cell-eval metrics."""

from importlib.metadata import PackageNotFoundError, version as _version

from .ceiling import compute_ceiling
from .lfc_nmae_ref import compute_lfc_nmae_reference
from .cellstream import cell_archive_input_type, score_cellstream
from .config import DiscriminationParams, EvalConfig, FilterParams
from .run import aggregate_metrics, aggregate_metrics_wide, compute_metrics, precompute_cache
from .baseline import BaselineResult, build_generic_baseline
from .score import score_metrics
from .h5ad_manifest import MemBudget, ScoreResult, score_h5ad_manifest
from .rowstore import read_rowstore_plan, score_rowstore

try:
    # Single source of truth: the version is declared only in pyproject.toml, and read here
    # from the INSTALLED package metadata rather than parsed out of the file.
    # ⚠️ That means it cannot drift from the metadata -- NOT that it cannot drift from
    # pyproject.toml. An editable install freezes this at install time and goes on reporting
    # the frozen value as the checkout moves, which is exactly how a build can stamp artifacts
    # with the PREVIOUS release's identity. Reinstall after a version bump, before building
    # anything whose provenance matters.
    __version__ = _version("cell-eval2")
except PackageNotFoundError:  # imported from a source tree without an install
    __version__ = "0.0.0+unknown"
__all__ = [
    "BaselineResult",
    "DiscriminationParams",
    "EvalConfig",
    "FilterParams",
    "MemBudget",
    "ScoreResult",
    "compute_metrics",
    "aggregate_metrics",
    "compute_ceiling",
    "compute_lfc_nmae_reference",
    "aggregate_metrics_wide",
    "build_generic_baseline",
    "cell_archive_input_type",
    "precompute_cache",
    "score_metrics",
    "score_cellstream",
    "score_h5ad_manifest",
    "read_rowstore_plan",
    "score_rowstore",
]
