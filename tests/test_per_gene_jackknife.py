"""The jackknife's PER-GENE decomposition, and the readers that can build it.

``jackknife_correction`` returns one scalar ``C_p`` per group: the sampling correction the
group-sum (``bulk_lognorm``) profile is owed. ``jackknife_per_gene`` returns the same quantity
before it is summed away, ``[P, G]``, plus the scalar computed by the scalar kernel's own
expression so that switching the per-gene pass on cannot move a number that was already published.

Nothing in ``src`` asks for the per-gene form: it is an OPT-IN (`per_gene_jk=True`) on the
in-memory, shard-streaming and cell-streaming readers, defaulting off everywhere, because it is
O(P*G) resident -- 44 MB at 300 x 18,533, 3 GB at 20,000 perturbations. The partitioned reader
cannot supply it at all (its ``_side_bulks`` calls request no moments).

Consumers are callers that need a correction restricted to a gene SUBSET
(``jackknife_correction(gene_mask=...)``, ``signal_fraction_mask``) -- per-gene noise scoring and
gene-level QC -- rather than the whole-panel scalar every metric already gets from
``jackknife_correction``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_eval2.moments import (GroupMoments, correction_per_gene_for, jackknife_correction,
                                jackknife_per_gene, signal_fraction_mask)
from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

G, NP, NC, DEPTH, TS = 200, 4, 40, 3000, 5e4
CONTROL = "non-targeting"
LABELS = [CONTROL] + [f"g{i}" for i in range(NP)]
GENES = np.array([f"g{i}" for i in range(G)])


@pytest.fixture(scope="module")
def panel():
    """A multinomially-sampled panel: real cell scatter, so the corrections are not degenerate.
    The control carries 6x a perturbation's cells, as a real panel does."""
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 1.0, G)
    base /= base.sum()
    profiles = [base] + [np.abs(base + 0.30 * rng.normal(0, base.std(), G)) for _ in range(NP)]
    profiles = [p / p.sum() for p in profiles]
    blocks, obs = [], []
    for lab, p in zip(LABELS, profiles):
        n = 6 * NC if lab == CONTROL else NC
        blocks.append(sp.csr_matrix(rng.multinomial(DEPTH, p, size=n).astype(np.int64)))
        obs += [lab] * n
    a = ad.AnnData(X=sp.vstack(blocks).tocsr(), obs=pd.DataFrame({"target": obs}),
                   var=pd.DataFrame(index=GENES))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    return a


def _codes(a):
    labels = a.obs["target"].to_numpy().astype(str)
    uniq, codes = np.unique(labels, return_inverse=True)
    return uniq, codes.astype(np.intp)


# --- the decomposition is exact ----------------------------------------------------------------

def test_per_gene_terms_sum_to_the_scalar_correction(panel):
    uniq, codes = _codes(panel)
    scalar = jackknife_correction(panel.X, codes, uniq.size, TS)
    per_gene, also_scalar = jackknife_per_gene(panel.X, codes, uniq.size, TS)
    assert per_gene.shape == (uniq.size, G)
    assert np.maximum(per_gene.sum(axis=1), 0.0) == pytest.approx(scalar, rel=1e-12, abs=1e-12)
    # ...and the scalar the per-gene kernel returns is BIT-identical, not merely close: summing
    # the per-gene terms is not, so `prep` must return this one or any already-published
    # whole-panel `expr_mse_unbiased` moves the day a caller asks for the terms.
    assert list(also_scalar) == list(scalar)


def test_the_scalar_jackknife_is_identical_whichever_prep_route_built_it(panel):
    _, _, off = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS,
                                                     per_gene_jk=False)
    _, _, on = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS,
                                                    per_gene_jk=True)
    assert list(off.jk) == list(on.jk)
    assert off.jk_gene is None and on.jk_gene is not None


def test_omitting_the_flag_is_the_same_as_passing_false(panel):
    _, _, default = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS)
    assert default.jk_gene is None


def test_a_masked_correction_matches_summing_the_masked_terms(panel):
    """`jackknife_correction(gene_mask=...)` is the whole point of keeping the terms: a caller
    can charge a correction over a gene SUBSET without re-deriving anything."""
    uniq, codes = _codes(panel)
    per_gene, _ = jackknife_per_gene(panel.X, codes, uniq.size, TS)
    rng = np.random.default_rng(3)
    mask = rng.random((uniq.size, G)) < 0.4
    masked = jackknife_correction(panel.X, codes, uniq.size, TS, gene_mask=mask)
    want = np.maximum(np.einsum("ij,ij->i", per_gene, mask.astype(float)), 0.0)
    assert masked == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_an_all_true_mask_reproduces_the_unmasked_correction(panel):
    uniq, codes = _codes(panel)
    unmasked = jackknife_correction(panel.X, codes, uniq.size, TS)
    allon = jackknife_correction(panel.X, codes, uniq.size, TS,
                                 gene_mask=np.ones((uniq.size, G), bool))
    assert allon == pytest.approx(unmasked, rel=1e-12, abs=1e-12)


def test_jk_gene_without_jk_is_refused():
    with pytest.raises(ValueError, match="without jk"):
        GroupMoments(perts=np.array(["a"]), counts=np.array([2.0]), sumsq=np.array([1.0]),
                     jk_gene=np.zeros((1, 3)))


# --- signal_fraction_mask ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.0, 1.5, -0.01, float("inf"), float("nan")])
def test_the_signal_fraction_floor_must_be_a_share(bad):
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        signal_fraction_mask(np.ones((2, 3)), np.zeros((2, 3)), 1, bad)


def test_the_mask_refuses_mismatched_shapes_and_a_control_row_off_the_end():
    with pytest.raises(ValueError, match="must be"):
        signal_fraction_mask(np.ones((2, 3)), np.zeros((2, 4)), 0, 0.5)
    with pytest.raises(ValueError, match="control_row"):
        signal_fraction_mask(np.ones((2, 3)), np.zeros((2, 3)), 2, 0.5)


def test_a_gene_that_did_not_move_is_never_scored():
    """`(1-eta) * 0 >= 0` passes the inequality, so a gene identically equal to the control
    across the whole panel would qualify on nothing at all -- and only inflate `|S_p|`."""
    means = np.array([[1.0, 1.0, 3.0], [1.0, 2.0, 3.0]])       # row 0 is the control
    jk = np.zeros((2, 3))
    keep = signal_fraction_mask(means, jk, 0, 0.5)
    assert not keep[:, 0].any() and not keep[:, 2].any()       # no change, either row
    assert keep[1, 1]                                          # moved, zero noise


def test_the_mask_is_the_inequality_it_documents(panel):
    """`keep <=> (1 - eta) * EF >= q_p + q_ctrl` with EF the squared change, evaluated against a
    straightforward recomputation rather than against itself."""
    _, means, moms = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS,
                                                          per_gene_jk=True)
    ctrl = int(np.flatnonzero(np.asarray(moms.perts).astype(str) == CONTROL)[0])
    for eta in (0.0, 0.5, 0.9):
        got = signal_fraction_mask(means, moms.jk_gene, ctrl, eta)
        ef = (means - means[ctrl][None, :]) ** 2
        want = (ef > 0.0) & ((1.0 - eta) * ef >= moms.jk_gene + moms.jk_gene[ctrl][None, :])
        np.testing.assert_array_equal(got, want)
    # and it is monotone: a higher floor never keeps a gene a lower one dropped
    lo = signal_fraction_mask(means, moms.jk_gene, ctrl, 0.5)
    hi = signal_fraction_mask(means, moms.jk_gene, ctrl, 0.9)
    assert not (hi & ~lo).any()


# --- correction_per_gene_for --------------------------------------------------------------------

def test_moments_without_the_terms_refuse_to_serve_a_per_gene_correction(panel):
    _, _, off = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS)
    with pytest.raises(ValueError, match="no per-gene jackknife"):
        correction_per_gene_for(off, np.asarray(off.perts), comparator="bulk_lognorm",
                                who="test")


def test_only_the_group_sum_comparator_has_per_gene_terms(panel):
    _, _, on = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS,
                                                    per_gene_jk=True)
    with pytest.raises(ValueError, match="bulk_lognorm"):
        correction_per_gene_for(on, np.asarray(on.perts), comparator="lognorm", who="test")


def test_the_per_gene_correction_is_aligned_by_label_not_by_position(panel):
    _, _, on = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS,
                                                    per_gene_jk=True)
    perts = np.asarray(on.perts).astype(str)
    shuffled = perts[::-1]
    got = correction_per_gene_for(on, shuffled, comparator="bulk_lognorm", who="test")
    np.testing.assert_array_equal(got, on.jk_gene[::-1])


# --- the readers ---------------------------------------------------------------------------------

def test_the_streaming_kernel_agrees_with_the_resident_one_per_gene():
    """The scalar twin of this lives in `test_jackknife.py`; a per-gene divergence would be
    invisible to it, because the terms can be wrong in ways that still sum correctly."""
    from cell_eval2.streaming_bulk import _streaming_jackknife

    rng = np.random.default_rng(4)
    n, g, P = 90, 60, 3
    Y = rng.poisson(4.0, size=(n, g)).astype(np.float64)
    Y[:, 0] += 150                                   # one dominant gene, as the scalar twin does
    X = sp.csr_matrix(Y)
    codes = np.repeat(np.arange(P), n // P).astype(np.intp)
    sums = np.stack([Y[codes == p].sum(axis=0) for p in range(P)])
    counts = np.array([float((codes == p).sum()) for p in range(P)])

    res_gene, res_scalar = jackknife_per_gene(X, codes, P, TS)
    for chunk in (7, 512):
        st_scalar, st_gene = _streaming_jackknife(lambda: iter([(X, codes)]), sums, counts, TS,
                                                  chunk=chunk, per_gene=True)
        assert st_scalar == pytest.approx(res_scalar, rel=0, abs=1e-15)
        assert st_gene == pytest.approx(res_gene, rel=0, abs=1e-15)
        assert np.maximum(st_gene.sum(axis=1), 0.0) == pytest.approx(st_scalar, rel=0, abs=1e-12)


def test_asking_the_streaming_kernel_for_terms_does_not_move_its_scalar():
    """`per_gene=False` must return a bare array of exactly the old values -- the flag appends,
    following that module's `with_moments`/`with_median_umi` convention."""
    from cell_eval2.streaming_bulk import _streaming_jackknife

    rng = np.random.default_rng(9)
    Y = rng.poisson(3.0, size=(40, 25)).astype(np.float64)
    X = sp.csr_matrix(Y)
    codes = np.zeros(40, dtype=np.intp)
    args = (lambda: iter([(X, codes)]), Y.sum(axis=0)[None, :], np.array([40.0]), TS)
    off = _streaming_jackknife(*args)
    on_scalar, _ = _streaming_jackknife(*args, per_gene=True)
    assert isinstance(off, np.ndarray) and off.ndim == 1
    np.testing.assert_array_equal(off, on_scalar)
    np.testing.assert_array_equal(off, jackknife_correction(X, codes, 1, TS))


def test_the_streaming_producer_carries_the_terms_only_when_asked(panel):
    from cell_eval2.streaming_bulk import _streaming_pseudobulk_cpu

    labels = panel.obs["target"].to_numpy().astype(str)
    perts = np.unique(labels)
    codes = np.searchsorted(perts, labels).astype(np.intp)
    X = panel.X.tocsr()

    # ⚠️ The two passes take DIFFERENT second elements: the first-pass `blocks` yields
    # (X, LABELS) and does its own searchsorted, while `make_blocks` -- the jackknife's fresh
    # factory -- yields (X, CODES) and indexes the group sums directly.
    common = dict(with_moments=True, make_blocks=lambda: iter([(X, codes)]), bulk_target_sum=TS)
    _, off = _streaming_pseudobulk_cpu(iter([(X, labels)]), perts, perts.size, G,
                                       ["bulk_lognorm"], None, **common)
    _, on = _streaming_pseudobulk_cpu(iter([(X, labels)]), perts, perts.size, G,
                                      ["bulk_lognorm"], None, per_gene_jk=True, **common)
    assert off["bulk_lognorm"].jk_gene is None
    assert on["bulk_lognorm"].jk_gene.shape == (perts.size, G)
    np.testing.assert_array_equal(off["bulk_lognorm"].jk, on["bulk_lognorm"].jk)


def test_the_accumulator_builds_per_gene_terms_on_the_numpy_backend(panel):
    """`GroupedMeanAccumulator.jackknife(per_gene=True)` is the branch the GPU and the GPU-style
    in-memory routes take. Exercised here on its NumPy backend, which is the same code path with
    `xp = numpy` -- the `[ok]` mask assignment, the `[:, None]` broadcast and the host hop all
    run. Without this, deleting the per-gene block would leave every other test green."""
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    common = dict(pert_col="target", norms=["bulk_lognorm"], target_sum=None, device="cpu",
                  with_moments=True, bulk_target_sum=TS)
    _, off = inmem_pseudobulk(panel, **common)
    _, on = inmem_pseudobulk(panel, per_gene_jk=True, **common)
    m_off, m_on = off["bulk_lognorm"], on["bulk_lognorm"]
    assert m_off.jk_gene is None
    assert m_on.jk_gene is not None and m_on.jk_gene.shape == (m_on.perts.size, G)
    np.testing.assert_array_equal(m_off.jk, m_on.jk)          # the scalar did not move
    assert np.maximum(m_on.jk_gene.sum(axis=1), 0.0) == pytest.approx(m_on.jk, rel=0, abs=1e-12)

    labels = panel.obs["target"].to_numpy().astype(str)
    codes = np.searchsorted(np.asarray(m_on.perts).astype(str), labels).astype(np.intp)
    res_gene, _ = jackknife_per_gene(panel.X.tocsr(), codes, m_on.perts.size, TS)
    assert m_on.jk_gene == pytest.approx(res_gene, rel=0, abs=1e-12), (
        "the accumulator's per-gene terms disagree with the resident kernel's"
    )


# --- the cache round-trip ------------------------------------------------------------------------

def test_the_moments_cache_round_trips_the_terms(panel, tmp_path):
    """`jk_gene` is an OPTIONAL npz key, so a floor-off artifact stays byte-compatible; a reader
    that needs the terms and finds them absent must see `None`, not a shape error."""
    from cell_eval2.cache import _dump_npz_moments, _load_npz_moments

    _, means, on = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS,
                                                        per_gene_jk=True)
    _, _, off = pseudobulk_bulk_lognorm_with_moments(panel, "target", bulk_target_sum=TS)
    for tag, moms, expect in (("on", on, True), ("off", off, False)):
        path = str(tmp_path / f"m_{tag}.npz")
        _dump_npz_moments(((np.asarray(moms.perts), means), moms), path)
        (perts2, means2), moms2 = _load_npz_moments(path)
        np.testing.assert_array_equal(np.asarray(perts2).astype(str),
                                      np.asarray(moms.perts).astype(str))
        np.testing.assert_array_equal(means2, means)
        np.testing.assert_array_equal(moms2.jk, moms.jk)
        if expect:
            np.testing.assert_array_equal(moms2.jk_gene, moms.jk_gene)
        else:
            assert moms2.jk_gene is None


# --- the guards: an unsatisfiable request must RAISE, not return None ------------------------------

@pytest.mark.parametrize("kwargs, msg", [
    (dict(with_moments=False, norms=["bulk_lognorm"]), "requires with_moments=True"),
    (dict(with_moments=True, norms=["lognorm"]), "requires 'bulk_lognorm' in norms"),
])
def test_per_gene_jk_refuses_a_request_it_cannot_honour(panel, kwargs, msg):
    """`per_gene_jk=True` used to be a SILENT no-op in both of these combinations.

    Both kernels return before any jackknife runs when moments are off, and the terms are only
    ever built for `bulk_lognorm`, so the caller got `jk_gene=None` back with no complaint --
    indistinguishable from asking for the terms and having them come out empty. The failure then
    surfaced much later, as a `None` reaching `signal_fraction_mask`.
    """
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    with pytest.raises(ValueError, match=msg):
        inmem_pseudobulk(panel, pert_col="target", target_sum=None, device="cpu",
                         bulk_target_sum=TS, per_gene_jk=True, **kwargs)


def test_per_gene_jk_is_still_a_no_op_request_when_not_asked_for(panel):
    """The guard must not fire on `per_gene_jk=False`, which is every existing caller."""
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    out = inmem_pseudobulk(panel, pert_col="target", target_sum=None, device="cpu",
                           norms=["bulk_lognorm"], with_moments=False, bulk_target_sum=TS)
    assert "bulk_lognorm" in out


def test_signal_fraction_mask_enforces_its_two_dimensional_contract():
    """A matching pair of 3-D inputs used to sail through the shape test and return a 3-D mask;
    a matching pair of 1-D inputs failed later inside the indexing, reporting neither argument."""
    for shape in ((2, 3, 4), (5,)):
        a = np.ones(shape)
        with pytest.raises(ValueError, match="must be 2-D"):
            signal_fraction_mask(a, a.copy(), 0, 0.5)


def test_signal_fraction_mask_requires_an_integral_control_row():
    """A float row index passed the range test and then failed during indexing, which read as a
    shape bug rather than a bad argument."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError, match="must be an integer row index"):
        signal_fraction_mask(m, np.zeros_like(m), 0.0, 0.5)
    # the integral neighbours still work, including numpy integer types
    for row in (0, 1, np.int64(0)):
        assert signal_fraction_mask(m, np.zeros_like(m), row, 0.5).shape == (2, 2)


def test_signal_fraction_mask_rejects_a_boolean_control_row():
    """`bool` subclasses `int`, so `isinstance(True, int)` is True and the obvious integer check
    lets a boolean through -- whereupon `means[True]` is BOOLEAN INDEXING, which adds an axis
    instead of selecting row 1. Caught by the codex round that reviewed the first fix."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    for bad in (True, False, np.bool_(True)):
        with pytest.raises(TypeError, match="boolean"):
            signal_fraction_mask(m, np.zeros_like(m), bad, 0.5)


@pytest.mark.parametrize("kernel", ["cpu", "gpu"])
def test_the_stream_kernels_refuse_an_unsatisfiable_per_gene_request(panel, kernel):
    """The guard has to be pinned on the KERNELS, not only on `inmem_pseudobulk`: the first version
    of this test exercised the in-memory reader alone, so deleting either kernel's call would have
    left it green."""
    import cell_eval2.streaming_bulk as sb

    fn = sb._streaming_pseudobulk_cpu if kernel == "cpu" else sb._streaming_pseudobulk_gpu
    perts = np.array(["a", "b"])
    with pytest.raises(ValueError, match="requires with_moments=True"):
        fn(iter([]), perts, 2, 3, ["bulk_lognorm"], None, with_moments=False, per_gene_jk=True)
    with pytest.raises(ValueError, match="requires 'bulk_lognorm' in norms"):
        fn(iter([]), perts, 2, 3, ["lognorm"], 1e4, with_moments=True, per_gene_jk=True)


def test_the_accumulator_refuses_per_gene_without_the_comparator():
    """`GroupedMeanAccumulator.jackknife` is public and directly callable, so the readers' guard
    does not cover it; it used to return `{}` and satisfy `per_gene=True` silently."""
    from cell_eval2.gpu.bulk import GroupedMeanAccumulator

    acc = GroupedMeanAccumulator(2, 3, normalizations=["lognorm"], target_sum=1e4,
                                 device="cpu", with_moments=True, bulk_target_sum=TS)
    with pytest.raises(ValueError, match="needs 'bulk_lognorm'"):
        acc.jackknife(lambda: iter([]), per_gene=True)
    acc.jackknife(lambda: iter([]), per_gene=False)      # the ordinary no-op still no-ops
