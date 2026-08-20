import numpy as np
import pytest

from cell_eval2.gpu import resolve_device
from cell_eval2.gpu.distances import (
    _discrimination_ranks_xp,
    _match_ranks_xp,
    discrimination_ranks,
)
from cell_eval2.metrics.discrimination import discrimination_score

# A usable CUDA GPU? (cupy importable AND a device visible). The venv is shared across
# the CPU and GPU nodes, so cupy may import on a GPU-free node — gate on a real device.
_HAS_GPU = resolve_device("auto") == "cuda"


def _make(P, G, seed):
    """P-1 perturbations + a control, over a panel of P-1 target genes plus G non-target ones.

    Every perturbation's own gene IS measured, so the exclusion fires for all of them under
    either scope. The G filler genes are what makes the panel realistic (the competition panel
    is 300 targets in 18,533 genes) and are load-bearing since #343: naming every gene after a
    perturbation, as this fixture used to, leaves `exclusion_scope="panel"` with an EMPTY
    feature space -- which `panel_reduced` raises on rather than scoring 0.5 for everything.
    """
    rng = np.random.default_rng(seed)
    names = np.array([f"G{i}" for i in range(P - 1)] + ["ctrl"])
    genes = np.array([f"G{i}" for i in range(P - 1)] + [f"F{i}" for i in range(G)])
    n_g = genes.size
    real = (names, rng.normal(size=(P, n_g)))
    pred = (names, rng.normal(size=(P, n_g)))
    return real, pred, genes


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
@pytest.mark.parametrize("excl", [True, False])
@pytest.mark.parametrize("scope", ["panel", "row"])
@pytest.mark.parametrize("csrc", ["pred", "real"])
@pytest.mark.parametrize("pert_chunk", [4, 16, 1000])
def test_xp_numpy_kernel_matches_cpu_reference(metric, excl, scope, csrc, pert_chunk):
    # The xp-kernel run on numpy is the EXACT code cupy runs; it must reproduce the CPU
    # reference discrimination_score across metric x exclusion x SCOPE x control_source, and
    # over chunk boundaries (pert_chunk smaller than, equal-ish to, and larger than n).
    # Both scopes, deliberately: the two kernels are separately implemented -- that is how
    # #248 came to be wrong in both -- and #343 added a whole second branch to each.
    real, pred, genes = _make(23, 17, seed=5)
    kw = dict(control="ctrl", distance=metric, rank_denominator="n", tie_policy="midrank",
              control_source=csrc, exclude_target_gene=excl, exclusion_scope=scope, genes=genes)
    cpu = discrimination_score(pred_bulk=pred, real_bulk=real, **kw)
    got = _discrimination_ranks_xp(
        np, real, pred, genes=genes, metric=metric, exclude_target_gene=excl,
        exclusion_scope=scope, rank_denominator="n",
        tie_policy="midrank", pert_chunk=pert_chunk, control="ctrl",
        control_source=csrc,
    )
    assert got.keys() == cpu.keys()
    for k in cpu:
        assert got[k] == pytest.approx(cpu[k], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("tie_policy", ["midrank", "position"])
@pytest.mark.parametrize("pert_chunk", [1, 4, 1000])
def test_xp_kernel_matches_cpu_reference_on_an_ALL_TIED_row(tie_policy, pert_chunk):
    """Issue #282. The old module docstring justified the double-argsort with "exact ties
    don't occur", so no test ever fed it one -- and cupy's sort is not numpy's introsort,
    so a tied row could have ranked differently per device for the same input.

    Here every predicted effect is exactly the zero vector, which under cosine ties the
    ENTIRE distance matrix at 1.0. The kernel must agree with the CPU reference under both
    policies, and must not depend on where the chunk boundaries fall.
    """
    names = np.array([chr(ord("a") + i) * 2 for i in range(8)] + ["ctrl"])
    genes = np.array(["x0", "x1"])
    real = (names, np.array([[i + 1.0, 8.0 - i] for i in range(8)] + [[0.0, 0.0]]))
    pred = (names, np.full((9, 2), 7.0))          # pred effect == 0 for every target
    kw = dict(control="ctrl", distance="cosine", rank_denominator="n-1",
              tie_policy=tie_policy, control_source="pred", exclude_target_gene=False, exclusion_scope="panel")
    cpu = discrimination_score(pred_bulk=pred, real_bulk=real, genes=genes, **kw)
    got = _discrimination_ranks_xp(
        np, real, pred, genes=genes, metric="cosine", exclude_target_gene=False, exclusion_scope="panel",
        rank_denominator="n-1", tie_policy=tie_policy, pert_chunk=pert_chunk,
        control="ctrl", control_source="pred",
    )
    assert got == pytest.approx(cpu)
    if tie_policy == "midrank":
        # and the value itself is the no-information point, not an index artifact
        assert got == pytest.approx({n: 0.5 for n in names[:-1]})
    else:
        assert got == pytest.approx(
            dict(zip(names[:-1], [1.0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7, 0.0]))
        )


class _FakeDeviceArray:
    """Pretends to be a cupy array: exposes ``.get()`` and records that it was called."""

    def __init__(self, arr, log):
        self._a = np.asarray(arr)
        self._log = log

    def get(self):
        self._log.append("get")
        return self._a


class _NoSortXP:
    """A numpy proxy whose ``argsort`` explodes."""

    def __getattr__(self, name):
        if name == "argsort":
            raise AssertionError(
                "the 'position' branch must not sort through xp: under cupy that is a "
                "different algorithm from numpy's introsort, and upstream cell-eval -- the "
                "only reason this branch exists -- is numpy. Rank on the host instead."
            )
        return getattr(np, name)


def test_position_branch_ranks_on_the_host_and_never_sorts_through_xp():
    """Pins the host-offload itself, not just its result (issue #282, Codex re-review).

    ``test_xp_kernel_matches_cpu_reference_on_an_ALL_TIED_row`` injects ``xp=np``, so
    reverting this branch to ``xp.argsort`` would still be numpy-vs-numpy and would still
    pass. This test cannot be satisfied that way: ``xp.argsort`` raises, and the fake device
    arrays record whether the transfer to host actually happened.
    """
    log = []
    block = np.ones((3, 5))                       # all tied, the #282 shape
    match_cols = np.array([0, 2, 4])
    got = _match_ranks_xp(_NoSortXP(), _FakeDeviceArray(block, log),
                          _FakeDeviceArray(match_cols, log), "position")
    assert log == ["get", "get"], (
        f"expected both the distance block and the match columns to be pulled to host, "
        f"saw {log}"
    )
    # numpy's argsort on an all-equal row is the identity permutation -> rank == column
    assert list(np.asarray(got)) == [0, 2, 4]


class _NoTransferArray(np.ndarray):
    """Behaves as an ordinary array, but explodes on anything that would leave the device.

    Two escapes, and the second is the one that actually shipped: an explicit ``.get()``,
    and a reduction to a Python scalar. ``bool(arr.any())`` on a cupy array forces a
    device->host transfer and synchronizes the pipeline, so it leaves the device just as
    surely as ``.get()`` does while reading like ordinary control flow. Both Copilot and
    Gemini caught that on the #282 PR -- the earlier version of this class did not, because
    it only guarded ``.get()``. This is what stops it coming back.
    """

    def get(self):
        raise AssertionError(
            "the 'midrank' branch must not transfer the block to host: it is arithmetic "
            "and belongs on device. Only the legacy 'position' branch may transfer."
        )

    def any(self, *args, **kwargs):
        raise AssertionError(
            "the 'midrank' branch must not reduce to a host scalar: `bool(x.any())` "
            "synchronizes the cupy pipeline on EVERY block, including the common one with "
            "no NaN. Handle the NaN case branchlessly with xp.where instead."
        )


def test_midrank_branch_stays_on_device_and_never_sorts():
    """The complement, and it has to prove BOTH halves of its own name.

    Plain numpy arrays would make the "stays on device" half vacuous -- a host-conversion
    regression would pass, since numpy is already the host (Codex, round 3). The block is a
    subclass whose ``.get()`` raises, so a transfer fails loudly, and ``xp.argsort`` raises,
    so a sort fails loudly.
    """
    block = np.ones((3, 5)).view(_NoTransferArray)
    got = _match_ranks_xp(_NoSortXP(), block, np.array([0, 2, 4]), "midrank")
    assert list(np.asarray(got)) == [2.0, 2.0, 2.0]      # (5 - 1) / 2, whatever the column


def test_cpu_device_delegates_exactly():
    # device="cpu" must reproduce discrimination_score bit-for-bit (it IS that call).
    real, pred, genes = _make(15, 9, seed=6)
    cpu = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="cosine", rank_denominator="n-1", tie_policy="midrank",
                               control_source="real", exclude_target_gene=True, exclusion_scope="panel",
                               genes=genes)
    got = discrimination_ranks(real, pred, perts=None, genes=genes, metric="cosine",
                               exclude_target_gene=True, exclusion_scope="panel", rank_denominator="n-1", tie_policy="midrank",
                               pert_chunk=8, device="cpu", control="ctrl",
                               control_source="real")
    assert got == cpu


def test_supplied_perts_mismatch_raises():
    real, pred, genes = _make(10, 6, seed=8)
    with pytest.raises(ValueError, match="perts"):
        _discrimination_ranks_xp(
            np, real, pred, genes=genes, metric="l2", exclude_target_gene=False, exclusion_scope="panel",
            rank_denominator="n",
            tie_policy="midrank", pert_chunk=4, control="ctrl",
            control_source="pred", perts=np.array(["not", "the", "right", "labels"]),
        )


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_gpu_ranks_match_cpu(metric):
    real, pred, genes = _make(40, 32, seed=3)
    kw = dict(control="ctrl", distance=metric, rank_denominator="n", tie_policy="midrank",
              control_source="pred", exclude_target_gene=True, exclusion_scope="panel", genes=genes)
    cpu = discrimination_score(pred_bulk=pred, real_bulk=real, **kw)
    gpu = discrimination_ranks(real, pred, perts=None, genes=genes, metric=metric,
                               exclude_target_gene=True, exclusion_scope="panel", rank_denominator="n", tie_policy="midrank",
                               pert_chunk=16, device="cuda", control="ctrl",
                               control_source="pred")
    for k in cpu:
        assert gpu[k] == pytest.approx(cpu[k], rel=1e-4, abs=1e-6)


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
@pytest.mark.parametrize("scope", ["panel", "row"])
@pytest.mark.parametrize("csrc", ["pred", "real"])
def test_gpu_matches_cpu_chunk_boundaries(metric, scope, csrc):
    # larger panel + small pert_chunk so the GPU path crosses several chunk boundaries.
    # BOTH scopes on real CUDA: "row" is the only case that runs `correct_excluded_gene`
    # inside the cupy kernel, per chunk, so pinning only "panel" here would leave the legacy
    # device correction unexercised on a GPU box (cross-provider review of #343).
    real, pred, genes = _make(50, 40, seed=9)
    kw = dict(control="ctrl", distance=metric, rank_denominator="n-1", tie_policy="midrank",
              control_source=csrc, exclude_target_gene=True, exclusion_scope=scope, genes=genes)
    cpu = discrimination_score(pred_bulk=pred, real_bulk=real, **kw)
    gpu = discrimination_ranks(real, pred, perts=None, genes=genes, metric=metric,
                               exclude_target_gene=True, exclusion_scope=scope,
                               rank_denominator="n-1", tie_policy="midrank",
                               pert_chunk=7, device="cuda", control="ctrl",
                               control_source=csrc)
    for k in cpu:
        assert gpu[k] == pytest.approx(cpu[k], rel=1e-4, abs=1e-6)


def test_discrimination_cuda_frees_pool_before_cublas(monkeypatch):
    # F10.1: on the cuda path, discrimination_ranks must free cupy's caching pool BEFORE the cuBLAS
    # einsum/matmul ops -- the preceding in-memory pseudobulk can leave VRAM parked in cupy's pool,
    # causing CUBLAS_STATUS_ALLOC_FAILED at scale. Verify the ordering without a GPU: force the cuda
    # path, stub the kernel, and spy the pool-free.
    import cell_eval2.gpu as gpu_pkg
    import cell_eval2.gpu.distances as gd

    order = []
    monkeypatch.setattr(gd, "resolve_device", lambda d: "cuda")
    monkeypatch.setattr(gd, "xp_for", lambda d: None)
    monkeypatch.setattr(gpu_pkg, "_release_gpu_pool", lambda: order.append("free"))
    monkeypatch.setattr(gd, "_discrimination_ranks_xp", lambda *a, **k: order.append("kernel") or {})
    gd.discrimination_ranks((["A"], None), (["A"], None), genes=None, metric="l2",
                            exclude_target_gene=True, exclusion_scope="panel", rank_denominator="n-1", tie_policy="midrank", pert_chunk=1,
                            device="cuda", control="non-targeting", control_source="real")
    assert order == ["free", "kernel"], f"pool must be freed before the cuBLAS kernel; got {order}"


# ---------------------------------------------------------------------------
# Issue #248 on the GPU path. This is the branch that runs BY DEFAULT on a CUDA box,
# and it had its own separately-written copy of the label->column lookup -- which is
# exactly how the bug survived in two places at once. These tests run the xp kernel on
# numpy, which is the same code cupy executes (see the module docstring), so they pin
# the GPU path on a GPU-free node too.
# ---------------------------------------------------------------------------

def _guide_panel():
    """Guide-level panel (SYMBOL-N labels, bare-symbol genes) -- the #248 shape.

    Same construction as tests/test_discrimination.py::_guide_panel: A-1's rank-0 result
    on the full vector is bought entirely by its own transcript at gene A.
    """
    real = (np.array(["A-1", "B-1", "ctrl"]),
            np.array([[-9.0, 1.0, 0.0], [0.0, 5.0, 5.0], [0.0, 0.0, 0.0]]))
    pred = (np.array(["A-1", "B-1", "ctrl"]),
            np.array([[-9.0, 4.0, 4.0], [0.0, 5.0, 5.0], [0.0, 0.0, 0.0]]))
    genes = np.array(["A", "B", "C"])
    return real, pred, genes, {"A-1": "A", "B-1": "B"}


@pytest.mark.parametrize("pert_chunk", [1, 2, 64])
def test_xp_kernel_drops_the_mapped_column_for_guide_labels(pert_chunk):
    # The GPU kernel keys its exclusion columns by GLOBAL pred row index while streaming
    # in pert_chunk blocks -- pert_chunk=1 forces every row into its own block, which is
    # where an off-by-chunk-offset bug in that keying would show up.
    real, pred, genes, tgm = _guide_panel()
    got = _discrimination_ranks_xp(
        np, real, pred, genes=genes, metric="l1", exclude_target_gene=True, exclusion_scope="panel",
        rank_denominator="n",
        tie_policy="midrank", pert_chunk=pert_chunk, control="ctrl",
        control_source="pred", target_gene_map=tgm,
    )
    assert got == pytest.approx({"A-1": 0.5, "B-1": 1.0})


def test_xp_kernel_matches_cpu_reference_on_guide_labels():
    # The two backends must agree on the MAPPED result, not just the unmapped one --
    # they are separate implementations and this is the value that moved.
    real, pred, genes, tgm = _guide_panel()
    kw = dict(genes=genes, exclude_target_gene=True, exclusion_scope="panel", rank_denominator="n-1", tie_policy="midrank",
              control="ctrl", control_source="real", target_gene_map=tgm)
    cpu = discrimination_score(pred_bulk=pred, real_bulk=real, distance="cosine", **kw)
    got = _discrimination_ranks_xp(np, real, pred, metric="cosine", pert_chunk=2, **kw)
    assert got.keys() == cpu.keys()
    for k in cpu:
        assert got[k] == pytest.approx(cpu[k], rel=1e-9, abs=1e-12)


def test_xp_kernel_raises_on_zero_resolution():
    # The GPU twin of the CPU gate: guide labels, no map, exclusion on -> raise rather
    # than return a plausible number computed with nothing excluded.
    real, pred, genes, _ = _guide_panel()
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        _discrimination_ranks_xp(
            np, real, pred, genes=genes, metric="l1", exclude_target_gene=True, exclusion_scope="panel",
            rank_denominator="n",
            tie_policy="midrank", pert_chunk=8, control="ctrl", control_source="pred",
        )


def test_discrimination_ranks_forwards_target_gene_map_through_cpu_delegation():
    # device="cpu" delegates to discrimination_score; the map has to survive that hop,
    # or a CPU-resolved run would raise while the CUDA run scored fine.
    real, pred, genes, tgm = _guide_panel()
    got = discrimination_ranks(
        real, pred, perts=None, genes=genes, metric="l1", exclude_target_gene=True, exclusion_scope="panel",
        rank_denominator="n",
        tie_policy="midrank", pert_chunk=8, device="cpu", control="ctrl",
        control_source="pred", target_gene_map=tgm,
    )
    assert got == pytest.approx({"A-1": 0.5, "B-1": 1.0})

    # and without it, the delegation must raise rather than silently score
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        discrimination_ranks(
            real, pred, perts=None, genes=genes, metric="l1",
            exclude_target_gene=True, exclusion_scope="panel", rank_denominator="n", tie_policy="midrank", pert_chunk=8,
            device="cpu", control="ctrl", control_source="pred",
        )


@pytest.mark.skipif(not _HAS_GPU, reason="no CUDA device")
def test_cuda_kernel_drops_the_mapped_column_for_guide_labels():
    # Same assertion on real cupy, for a CUDA box. Skips on a GPU-free node, where the
    # numpy-xp tests above already cover the identical kernel source.
    real, pred, genes, tgm = _guide_panel()
    got = discrimination_ranks(
        real, pred, perts=None, genes=genes, metric="l1", exclude_target_gene=True, exclusion_scope="panel",
        rank_denominator="n",
        tie_policy="midrank", pert_chunk=2, device="cuda", control="ctrl",
        control_source="pred", target_gene_map=tgm,
    )
    assert got == pytest.approx({"A-1": 0.5, "B-1": 1.0})


def test_the_xp_kernel_rejects_an_unknown_exclusion_scope():
    """Without this guard any value but "panel" falls into the legacy `elif` and returns a
    plausible ROW-scope number for a typo -- on the branch that runs by default on a CUDA box.
    The CPU twin already raised; found by the cross-provider review of #343."""
    real, pred, genes = _make(8, 6, seed=11)
    with pytest.raises(ValueError, match="exclusion_scope must be"):
        _discrimination_ranks_xp(
            np, real, pred, genes=genes, metric="l2", exclude_target_gene=True,
            exclusion_scope="rows", rank_denominator="n-1", tie_policy="midrank",
            pert_chunk=4, control="ctrl", control_source="pred")
