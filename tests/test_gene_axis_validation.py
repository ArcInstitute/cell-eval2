"""The streaming entry points must reject a (pred, real) pair whose gene axes disagree.

They adopted the REAL side's var_names for both sides and never compared pred's, so a pred
archive with the same gene COUNT in a different ORDER was scored gene-position-wise and produced
plausible finite numbers (ultrareview 2026-07-25). The in-memory path (io.validate_pair) and
score_streaming_cell (cell_source.validate_cell_pair) already raised; these did not.
"""
import numpy as np
import pytest

from cell_eval2 import de_compute
from cell_eval2.io import validate_gene_axis

# The gate is gpudge availability, NOT resolve_device("auto") == "cuda": the latter probes CUPY,
# so a host with the [gpu] extra but no gpudge would RUN this test and fail inside
# build_reference's compute_de, long before reaching the validation being tested.
_HAS_GPUDGE = de_compute._available("gpudge")


# --- the helper itself: CPU-only, no optional deps, so it always runs in CI ---

def test_validate_gene_axis_accepts_an_identical_axis():
    genes = np.array(["a", "b", "c"])
    validate_gene_axis(genes, genes)          # must not raise


def test_validate_gene_axis_rejects_a_permutation():
    with pytest.raises(ValueError, match="gene names/order differ between pred and real"):
        validate_gene_axis(np.array(["b", "a", "c"]), np.array(["a", "b", "c"]))


def test_validate_gene_axis_rejects_a_count_mismatch():
    with pytest.raises(ValueError, match="gene dimension mismatch"):
        validate_gene_axis(np.array(["a", "b"]), np.array(["a", "b", "c"]))


def test_validate_gene_axis_accepts_list_input():
    validate_gene_axis(["a", "b"], ["a", "b"])


# --- score_streaming: needs cellstream but NOT a GPU (metrics=["mae"] dispatches no DE) ---

def _permute_var(adata, seed=0):
    order = np.random.default_rng(seed).permutation(adata.n_vars)
    return adata[:, order].copy()


def test_score_streaming_rejects_a_permuted_pred_gene_axis(tmp_path, synthetic_counts_pair):
    pytest.importorskip("cellstream")
    from cellstream import write_sharded

    from cell_eval2 import EvalConfig
    from cell_eval2.scale import score_streaming

    pred, real = synthetic_counts_pair
    real_p, pred_p = tmp_path / "real.shad", tmp_path / "pred.shad"
    # The conftest fixtures use pert_col="target" (tests/conftest.py:38), matching EvalConfig's
    # default -- NOT "perturbation".
    write_sharded(real, str(real_p), group_by="target", overwrite=True)
    write_sharded(_permute_var(pred), str(pred_p), group_by="target", overwrite=True)

    cfg = EvalConfig(metrics=["mae"], input_type="counts")
    with pytest.raises(ValueError, match="gene names/order differ between pred and real"):
        score_streaming(str(pred_p), str(real_p), config=cfg)


def test_score_streaming_accepts_a_matching_pair(tmp_path, synthetic_counts_pair):
    pytest.importorskip("cellstream")
    from cellstream import write_sharded

    from cell_eval2 import EvalConfig
    from cell_eval2.scale import score_streaming

    pred, real = synthetic_counts_pair
    real_p, pred_p = tmp_path / "real.shad", tmp_path / "pred.shad"
    write_sharded(real, str(real_p), group_by="target", overwrite=True)
    write_sharded(pred, str(pred_p), group_by="target", overwrite=True)

    cfg = EvalConfig(metrics=["mae"], input_type="counts")
    out = score_streaming(str(pred_p), str(real_p), config=cfg)
    assert out.height > 0, "a matching pair must still score"


# --- score_piece: gpudge-gated. build_reference routes through _require_partition_config and
# --- unconditionally runs compute_de (partition_inmem.py:168,191), so it cannot run without the
# --- gpudge DE engine at all.

@pytest.mark.skipif(not _HAS_GPUDGE, reason="score_piece needs the gpudge DE backend")
def test_score_piece_rejects_a_piece_whose_genes_disagree_with_the_reference(
    tmp_path, synthetic_counts_pair
):
    from cell_eval2 import EvalConfig
    from cell_eval2.partition_inmem import build_reference, score_piece

    pred, real = synthetic_counts_pair
    cfg = EvalConfig(metrics=["mae"], input_type="counts")
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    # `comparator=` is REQUIRED here since #264 PR2 moved `expr_mae` (v1 `mae`) onto the
    # expression comparator: `build_reference` is one-sided, so it cannot resolve the
    # comparator itself and refuses to guess (`partition_inmem.py:213`). Both calls must be
    # given the same value, or the reference and the piece land in different spaces. This
    # test is gpudge-gated, so CPU CI never saw it -- it is the 15th COMPARATOR-REQ site,
    # one the plan's enumeration of 14 missed.
    build_reference(real, config=cfg, cache_dir=str(ref_dir), control_format="h5ad",
                    comparator="bulk_lognorm")

    piece = pred[pred.obs[cfg.pert_col] != cfg.control].copy()
    with pytest.raises(ValueError, match="gene names/order differ between pred and real"):
        score_piece(_permute_var(piece), str(ref_dir), config=cfg, piece_id="p0",
                    comparator="bulk_lognorm")
