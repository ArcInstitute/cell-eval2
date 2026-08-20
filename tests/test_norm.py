import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cell_eval2 import norm


def _counts_adata():
    return ad.AnnData(X=np.array([[3.0, 0.0, 5.0], [1.0, 2.0, 0.0]], dtype=np.float32))


def _lognorm_adata():
    import scanpy as sc

    a = _counts_adata()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    return a


def _adata(X):
    return ad.AnnData(X=np.asarray(X, dtype=np.float64),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(len(X))]),
                      var=pd.DataFrame(index=[f"g{j}" for j in range(len(X[0]))]))


def _sparse_adata(X):
    X = np.asarray(X, dtype=np.float64)
    return ad.AnnData(X=csr_matrix(X),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
                      var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]))


def test_guess_is_lognorm_true_on_lognorm():
    from cell_eval2.norm import guess_is_lognorm

    assert guess_is_lognorm(_lognorm_adata()) is True


def test_guess_is_lognorm_false_on_counts():
    from cell_eval2.norm import guess_is_lognorm

    assert guess_is_lognorm(_counts_adata()) is False


def test_resolve_input_type_v1_guesses_counts():
    from cell_eval2.norm import resolve_input_type

    assert resolve_input_type(
        _counts_adata(), declared="lognorm", version="v1", allow_discrete=False
    ) == "counts"


def test_resolve_input_type_v2_returns_declared():
    from cell_eval2.norm import resolve_input_type

    assert resolve_input_type(
        _counts_adata(), declared="counts", version="v2", allow_discrete=False
    ) == "counts"


def test_resolve_input_type_v2_default_still_returns_declared():
    # autodetect omitted (default False): v2 must keep returning the declared type.
    from cell_eval2.norm import resolve_input_type

    assert resolve_input_type(
        _lognorm_adata(), declared="counts", version="v2", allow_discrete=False
    ) == "counts"


def test_resolve_input_type_autodetect_v2_fractional_is_lognorm():
    from cell_eval2.norm import resolve_input_type

    assert resolve_input_type(
        _lognorm_adata(), declared="counts", version="v2", allow_discrete=False, autodetect=True
    ) == "lognorm"


def test_resolve_input_type_autodetect_v2_counts_is_counts():
    from cell_eval2.norm import resolve_input_type

    assert resolve_input_type(
        _counts_adata(), declared="counts", version="v2", allow_discrete=False, autodetect=True
    ) == "counts"


def test_resolve_input_type_autodetect_honors_allow_discrete():
    # allow_discrete forces "counts" even on fractional data within the auto-detect path.
    from cell_eval2.norm import resolve_input_type

    assert resolve_input_type(
        _lognorm_adata(), declared="counts", version="v2", allow_discrete=True, autodetect=True
    ) == "counts"


def test_validate_counts_rejects_fractional():
    with pytest.raises(ValueError, match="fractional"):
        norm.validate_input_type(_adata([[1.0, 2.5], [3.0, 4.0]]), "counts")


def test_validate_counts_allows_fractional_when_flagged():
    # a constructed average/null predictor is legitimately fractional "counts"
    norm.validate_input_type(_adata([[1.0, 2.5], [3.0, 4.0]]), "counts", allow_fractional=True)


def test_validate_counts_still_rejects_negative_when_fractional_allowed():
    with pytest.raises(ValueError, match="negative"):
        norm.validate_input_type(_adata([[-1.0, 2.5]]), "counts", allow_fractional=True)


def test_validate_lognorm_rejects_all_integer():
    with pytest.raises(ValueError, match="integer"):
        norm.validate_input_type(_adata([[1.0, 2.0], [3.0, 4.0]]), "lognorm")


def test_validate_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        norm.validate_input_type(_adata([[-1.0, 2.0]]), "counts")


def test_scale_limit_rejects_large_lognorm():
    big = np.log1p(np.array([[2_000_000.0, 0.0]]))  # one gene over the budget
    with pytest.raises(ValueError, match="max_counts_per_cell"):
        norm.check_scale_limit(_adata(big), "lognorm", 1_000_000.0)


def test_lognorm_to_counts_raises():
    with pytest.raises(ValueError, match="cannot recover counts"):
        norm.to_normalization(_adata([[0.5, 1.5]]), "lognorm", "counts")


def test_validate_and_scale_limit_sparse_ok():
    # fractional, non-negative -> valid lognorm; small totals -> within limit
    a = _sparse_adata([[0.0, 1.5], [2.0, 0.0]])
    norm.validate_input_type(a, "lognorm")
    norm.check_scale_limit(a, "lognorm", 1_000_000.0)


def test_scale_limit_sparse_counts_rejects():
    a = _sparse_adata([[2_000_000.0, 0.0]])  # integer counts, per-cell total over budget
    with pytest.raises(ValueError, match="max_counts_per_cell"):
        norm.check_scale_limit(a, "counts", 1_000_000.0)


def test_validate_sparse_counts_rejects_fractional():
    with pytest.raises(ValueError, match="fractional"):
        norm.validate_input_type(_sparse_adata([[1.0, 2.5], [3.0, 0.0]]), "counts")


def test_counts_to_lognorm_matches_scanpy():
    import scanpy as sc
    a = _adata([[1.0, 0.0, 3.0], [0.0, 5.0, 2.0], [4.0, 1.0, 0.0]])
    out = norm.to_normalization(a, "counts", "lognorm")
    ref = a.copy()
    sc.pp.normalize_total(ref)
    sc.pp.log1p(ref)
    assert np.allclose(out.X, ref.X)


def test_validate_lognorm_accepts_all_zero_dense():
    norm.validate_input_type(_adata([[0.0, 0.0], [0.0, 0.0]]), "lognorm")  # no raise


def test_validate_lognorm_accepts_all_zero_sparse():
    norm.validate_input_type(_sparse_adata([[0.0, 0.0], [0.0, 0.0]]), "lognorm")  # no raise


def test_counts_to_normalized_matches_scanpy():
    import scanpy as sc
    a = _adata([[1.0, 0.0, 3.0], [0.0, 5.0, 2.0]])
    out = norm.to_normalization(a, "counts", "normalized")
    ref = a.copy()
    sc.pp.normalize_total(ref)
    assert np.allclose(out.X, ref.X)


def test_lognorm_to_normalized_is_expm1_dense():
    a = _adata([[0.5, 1.5], [2.0, 0.0]])
    out = norm.to_normalization(a, "lognorm", "normalized")
    assert np.allclose(np.asarray(out.X), np.expm1([[0.5, 1.5], [2.0, 0.0]]))


def test_lognorm_to_normalized_is_expm1_sparse():
    from scipy.sparse import issparse
    a = _sparse_adata([[0.5, 0.0], [0.0, 2.0]])
    out = norm.to_normalization(a, "lognorm", "normalized")
    assert issparse(out.X)
    assert np.allclose(out.X.toarray(), np.expm1([[0.5, 0.0], [0.0, 2.0]]))


def test_counts_to_counts_identity():
    a = _adata([[1.0, 2.0], [3.0, 4.0]])
    assert norm.to_normalization(a, "counts", "counts") is a


# --- chunked _is_all_integer equivalence vs the pre-optimization implementation (Task 2) ---
import scipy.sparse as _sp  # noqa: E402


def _old_is_all_integer(X):
    # The oracle takes `norm._INT_ATOL` rather than restating a literal: what it exists to check is
    # that CHUNKING does not change the answer, and an oracle carrying its own tolerance would
    # quietly become a second pin on the tolerance instead -- one that reads as green while
    # disagreeing with the gate. The tolerance itself is pinned by the section below.
    data = X.data if _sp.issparse(X) else np.asarray(X)
    return data.size == 0 or bool(np.allclose(data, np.rint(data), rtol=0,
                                             atol=norm._int_atol(data.dtype)))


@pytest.mark.parametrize("X", [
    np.array([[1.0, 2.0], [3.0, 4.0]]),                 # float-integer -> True
    np.array([[1.0, 2.5], [3.0, 4.0]]),                 # fractional -> False (short-circuits)
    np.array([[2, 0, 5], [1, 3, 0]], dtype=np.int32),   # integer dtype -> True (no scan)
    np.zeros((4, 4)),                                    # all-zero -> True
    np.empty((0, 3)),                                    # empty -> True
    np.array([[1.0, 2.0 + 1e-9], [3.0, 4.0]]),          # within atol of integer -> True
    np.array([[1.0, 2.01], [3.0, 4.0]]),                # outside tol -> False
])
def test_is_all_integer_matches_old_dense(X):
    a = ad.AnnData(X=np.asarray(X, dtype=X.dtype))
    assert norm._is_all_integer(a.X) == _old_is_all_integer(a.X)


def test_is_all_integer_matches_old_sparse():
    Xi = _sp.csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]]))
    Xf = _sp.csr_matrix(np.array([[1.0, 0.0, 3.5], [0.0, 2.0, 0.0]]))
    assert norm._is_all_integer(Xi) == _old_is_all_integer(Xi)
    assert norm._is_all_integer(Xi) is True
    assert norm._is_all_integer(Xf) == _old_is_all_integer(Xf)
    assert norm._is_all_integer(Xf) is False


def test_is_all_integer_chunk_boundary():
    n = norm._INT_CHUNK + 7
    flat = np.ones(n, dtype=np.float64)
    flat[-1] = 0.5                      # only non-integer in the last chunk
    assert norm._is_all_integer(flat.reshape(n, 1)) is False
    flat[-1] = 1.0
    assert norm._is_all_integer(flat.reshape(n, 1)) is True


# --- the counts integrality gate is an ABSOLUTE tolerance, not a relative one -------------------
#
# `np.allclose`'s default tolerance is RELATIVE (`atol + rtol*|b|`, rtol=1e-5, atol=1e-8), so the
# deviation it accepts as "still an integer" SCALES with the value: +-0.001 at 100, +-0.01 at
# 1,000, +-0.5 at 50,000 -- i.e. every possible fractional part -- and above 50,000 every value
# whatsoever passes. Nothing else bounded it: `check_scale_limit` caps the per-CELL total at 1e6,
# so a single entry at 50,000+ is legal. MEASURED on main `9f18ce3` (numpy 2.5.2): 5000.5 was
# rejected, and 50000.5 and 999999.5 were ACCEPTED as counts with allow_fractional=False.
#
# `docs/vcc2026_metrics/vcc2026-metrics.md` section 0 is the published rule MOTIVATING this change
# -- what lands is a proxy for it, integral to a nominal 1e-6 on the paths reaching
# `validate_input_type`, not the sentence itself: values
# "must be raw, untransformed counts: non-negative, integral, and with no cell total above 1e6. A
# matrix failing any of these is rejected rather than scored."

#: Accepted as counts BEFORE the tolerance was made absolute; rejected now. Annotated with the old
#: tolerance at that magnitude, which is what made each of them pass. That tolerance is
#: `atol + rtol*|b|` where `b` is `np.allclose`'s SECOND argument -- here `np.rint(v)`, not `v`, so
#: the formula is `1e-8 + 1e-5*|rint(v)|`.
_ACCEPTED_BEFORE_REJECTED_NOW = [
    100.001,        # old tol 1.0e-03 -- a thousandth of a count
    1000.001,       # old tol 1.0e-02 -- the value `tests/test_jackknife.py` pinned as ACCEPTED
    1000.01,        # old tol 1.0e-02 -- just inside it: error 9.999999999990905e-03 vs 1.000001e-02
    50000.5,        # old tol 5.0e-01 -- a HALF count: the worst case the relative rule reaches
    999999.5,       # old tol 1.0e+01 -- ten counts of slack, still inside the 1e6 per-cell cap
]

#: Must keep passing. The gate may only reject what is not an integer, and tightening a validation
#: gate is the one change here that can turn a previously-accepted input into a hard failure.
_STILL_ACCEPTED = [
    50000.0,            # an exact integer where the old rule allowed +-0.5 either side of it
    1_000_000.0,        # the per-cell cap itself
    999999.0 + 1e-7,    # float64 dust from an upstream transform (~1,000 ULP at this magnitude)
]


@pytest.mark.parametrize("value", _ACCEPTED_BEFORE_REJECTED_NOW)
@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_counts_gate_rejects_a_fractional_part_at_every_magnitude(mk, value):
    """The dense/sparse split is not a duplicate: it is the two `np.allclose` call sites.
    `_is_all_integer` sends a 2-D dense array through its own row-block loop and a sparse matrix
    through `_all_integer_1d` over `X.data`, so a tolerance fixed at one site and not the other
    would be a gate whose verdict depends on how the matrix happened to arrive."""
    with pytest.raises(ValueError, match="fractional"):
        norm.validate_input_type(mk([[value, 3.0], [1.0, 2.0]]), "counts")


@pytest.mark.parametrize("value", _STILL_ACCEPTED)
@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_counts_gate_still_accepts_exact_integers_and_float64_dust(mk, value):
    norm.validate_input_type(mk([[value, 3.0], [1.0, 2.0]]), "counts")   # must not raise


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_counts_gate_still_accepts_a_float32_large_integer(sparse):
    """float32 is EXACT for integers up to `2**24` = 16,777,216, sixteen times the 1e6 per-cell
    cap, so no integer-valued float32 submission can be lost to this tolerance."""
    X = np.array([[123456.0, 0.0], [0.0, 7.0]], dtype=np.float32)
    norm.validate_input_type(ad.AnnData(csr_matrix(X) if sparse else X), "counts")


def test_the_tolerance_no_longer_moves_with_the_value():
    """The property itself, stated without going through `validate_input_type`: the same deviation
    is judged the same way at every magnitude. Under the old relative rule 1e-4 was rejected at 1
    and accepted at 1,000 and above -- the same absolute error, two answers."""
    for v in (1.0, 1000.0, 50000.0, 999999.0):
        assert norm._is_all_integer(np.array([[v + 1e-7, 3.0]])) is True, v
        assert norm._is_all_integer(np.array([[v + 1e-4, 3.0]])) is False, v


def test_the_absolute_tolerance_boundary_is_the_constant_itself():
    """The band the tests above establish is wide -- roughly `1e-7 <= atol < 1e-4`: the acceptance
    tests require a 1e-07 deviation to pass and the rejection tests require 1e-04 to fail -- so the
    literal is pinned here rather than left implied (codex review).

    Behaviour first, the constant last, deliberately: on a tree without `_INT_ATOL` the name lookup
    raises before any behaviour is exercised, which would make this test red for the wrong reason.
    Only the REJECTION half discriminates from the old rule -- at v=1 the old tolerance was
    1.001e-05, so a 0.9e-06 deviation was accepted by both."""
    assert norm._is_all_integer(np.array([[1.0 + 0.9e-6, 3.0]])) is True
    assert norm._is_all_integer(np.array([[1.0 + 1.1e-6, 3.0]])) is False
    assert norm._INT_ATOL == 1e-6


def test_the_tolerance_is_a_noise_margin_and_not_exact_equality():
    """`atol=1e-6` is deliberate, not a rounded-down zero. Exact equality is defensible on the
    width argument alone -- float32 holds every integer under the cap -- but it would newly reject
    an otherwise-integral matrix carrying dust from an upstream float64 transform, and that is the
    one direction a tightened gate can do real damage in."""
    dusty = np.array([[999999.0 + 1e-7, 3.0]], dtype=np.float64)
    assert not (dusty == np.rint(dusty)).all(), "fixture must actually carry dust"
    assert norm._is_all_integer(dusty) is True


#: The OTHER direction, and the reason this change is not a pure tightening. `np.allclose`'s
#: tolerance is `atol + rtol*|b|` with `b = rint(v)`, so `rint(v) == 0` kills the relative term and
#: the OLD rule was `atol=1e-8` alone there -- 100x TIGHTER than the new absolute one. These values
#: were rejected as fractional counts and are now accepted.
_REJECTED_BEFORE_ACCEPTED_NOW = [5e-8, 1e-7, 5e-7, 9.9e-7]


@pytest.mark.parametrize("value", _REJECTED_BEFORE_ACCEPTED_NOW)
@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_the_zero_neighbourhood_LOOSENS_and_that_is_the_price_of_uniformity(mk, value):
    """Pinned because the first draft of this change described itself as a pure tightening, which is
    false (codex review). Accepted rather than special-cased: a uniform absolute tolerance is the
    coherent rule the relative one was not, and 1e-6 of a count on a value whose nearest integer is
    0 is what "dust" means. Reinstating 1e-8 below some threshold would buy monotonicity with a
    second, unmeasured constant."""
    norm.validate_input_type(mk([[value, 3.0], [1.0, 2.0]]), "counts")   # must not raise
    assert norm._is_all_integer(mk([[value, 3.0]]).X) is True


def _dtype_adata(X, dtype, *, sparse=False):
    """`_adata`/`_sparse_adata` both force float64, so neither can see a dtype effect."""
    arr = np.asarray(X, dtype=dtype)
    return ad.AnnData(X=csr_matrix(arr) if sparse else arr)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_the_tightening_holds_at_float32_and_float64(dtype, sparse):
    """`prep._grouped_sums`' exposure list accepts float16, float32 and float64 counts, and the two
    helpers above force float64 -- so without this the whole section tested one width (codex review).
    float16 is not here because it CANNOT reach this region at all; the test below proves that.
    `longdouble` is reachable and behaves identically (MEASURED: 50000.5 rejected), and is left out
    only because no reader will submit one -- the name says f32/f64 rather than "every reachable
    width", which would be false (codex review)."""
    a = _dtype_adata([[50000.5, 3.0], [1.0, 2.0]], dtype, sparse=sparse)
    with pytest.raises(ValueError, match="fractional"):
        norm.validate_input_type(a, "counts")
    b = _dtype_adata([[50000.0, 3.0], [1.0, 2.0]], dtype, sparse=sparse)
    norm.validate_input_type(b, "counts")                   # the exact integer still passes


def test_float16_has_NO_tightening_case_and_exactly_17_loosening_ones():
    """EXHAUSTIVE over every finite non-negative float16 -- all 31,744 of them, which is small enough
    to simply enumerate rather than argue about (codex review asked for dtype coverage; this is the
    whole domain).

    MEASURED: **0** values flip the tightening way and **17** flip the loosening way. The tightening
    is VACUOUS at this width, and the reason is the dtype, not the tolerance: the smallest relative
    deviation float16 can express away from zero is exactly `2**-11` = 4.8828125e-04 (exhaustively
    confirmed below, and NOT `spacing(n)/n`, which is 9.765625e-04 at n=1), always larger than the
    old `rtol=1e-5`, so the old rule never accepted a non-integer float16 value away from zero. Near
    zero it accepted none either, because `np.float16(1e-8)` underflows to 0.0. So float16's only
    verdict change is the loosening, and it is these 17 values -- from the smallest subnormal
    5.960464477539063e-08 up to 1.0132789611816406e-06.

    `np.isclose` is `np.allclose`'s own element-wise kernel, so this measures the predicate's
    arithmetic rather than a re-derivation of it; four values are cross-checked through
    `_is_all_integer` below to keep that equivalence honest."""
    vals = np.arange(0, 0x7C00, dtype=np.uint16).view(np.float16)   # 0 .. largest finite float16
    assert vals.size == 31_744
    target = np.rint(vals)
    old_ok = np.isclose(vals, target)                               # the DEFAULT relative tolerance
    # `np.float16(...)`, not `norm._INT_ATOL`: production passes `_int_atol(vals.dtype)`, and an
    # oracle left on the weak-scalar path would be free to disagree with the deliberately
    # stabilized gate under a future promotion change (codex review).
    new_ok = np.isclose(vals, target, rtol=0, atol=np.float16(norm._INT_ATOL))
    assert int((old_ok & ~new_ok).sum()) == 0, "float16 cannot reach the tightening region"
    # ...and WHY, measured rather than argued: away from zero the smallest relative deviation the
    # dtype can express is 2**-11, two orders above the old rtol, so the old rule had no near-integer
    # float16 value to accept in the first place.
    away = (target != 0) & (np.abs(vals - target) > 0)
    rel = np.abs(vals[away] - target[away]) / np.abs(target[away])
    assert float(rel.min()) == 2.0 ** -11 == 4.8828125e-04
    assert rel.min() > 1e-5, "the old rtol; anything above it cannot be hidden by the relative rule"
    loosened = vals[new_ok & ~old_ok]
    assert loosened.size == 17
    assert float(loosened.min()) == 5.960464477539063e-08
    assert float(loosened.max()) == 1.0132789611816406e-06
    for v in (loosened.min(), loosened.max(), np.float16(2048.0)):
        assert norm._is_all_integer(np.array([[v]], dtype=np.float16)) is True
    assert norm._is_all_integer(np.array([[512.5]], dtype=np.float16)) is False


def _f16_raw_csr(value):
    """A float16 CSR, built through the RAW-ARRAY constructor.

    `csr_matrix(dense_float16)` rejects float16 while raw-array construction accepts it -- the stable
    half of an asymmetry `norm._csr_row_block`'s docstring records, and the reason that helper takes
    no `dtype=`. Nothing is claimed here about the typed-EMPTY constructor: its float16 rejection
    arrived in scipy 1.18.0 and `pyproject` permits `scipy>=1.11`, so "the one way scipy allows" --
    what an earlier draft of this docstring said -- is false across the supported range (codex
    review).

    An even earlier draft read the dense rejection as "scipy.sparse has no float16 at all" and pinned
    THAT as a property. Also false, and worse: it removed the sparse integrality site's float16
    coverage while asserting the coverage was impossible."""
    data = np.array([value, 2.0], dtype=np.float16)
    return csr_matrix((data, np.array([0, 1], dtype=np.int32),
                       np.array([0, 2], dtype=np.int32)), shape=(1, 2))


@pytest.mark.parametrize("mk", [lambda v: np.array([[v, 2.0]], dtype=np.float16), _f16_raw_csr],
                         ids=["dense", "raw-csr"])
def test_float16_gets_atol_ROUNDED_TO_float16_which_is_not_1e_6(mk):
    """`norm._int_atol` rounds the tolerance to the ARRAY's dtype explicitly, so the effective
    tolerance is `atol` as representable at that width. MEASURED on numpy 2.5.2:
    `np.float16(1e-6)` is 1.0132789611816406e-06, so float16 input is judged against THAT, and the
    "bounded by 1e-6 per element" reading is false for it by 1.3%.

    Both routes, because both are reachable: the 2-D dense block loop, and `_all_integer_1d` over a
    raw float16 CSR's `.data`. Also why the zero-neighbourhood loosening is not "100x" at this width:
    `np.float16(1e-8)` underflows to 0.0, so the OLD rule admitted no near-zero float16 dust at
    all."""
    assert float(np.float16(1e-6)) == 1.0132789611816406e-06
    assert float(np.float16(1e-8)) == 0.0
    assert norm._is_all_integer(mk(1.0132789611816406e-06)) is True
    assert norm._is_all_integer(mk(1.0728836059570312e-06)) is False


def test_np_rint_does_NOT_raise_on_bool_so_the_loops_need_no_bool_short_circuit():
    """Refutes a HIGH-priority Gemini finding on PR #341: "`np.rint` is not supported for boolean
    dtypes in modern NumPy and will raise a `TypeError`", with a suggestion to short-circuit bool in
    both loops. MEASURED on numpy 2.5.2 -- the version this repo resolves -- it does not raise: bool
    resolves through `rint`'s first INEXACT loop, which is float16 here, exactly the resolution
    `norm._csr_row_block`'s docstring already records for `np.expm1` of bool/int8/uint8 input. All
    three routes reach the right answer. ⚠️ The assertion below is on the VALUES, not on the dtype:
    "any floating width" was a second draft and still pins an unstable property, since numpy could
    legitimately add a bool `rint` loop returning bool (codex review, refining Copilot's). The stable
    property is that it does not raise and returns the same 0/1s. float16 is what it happens to return
    on numpy 2.5.2 -- recorded as an observation, asserted as nothing.

    Pinned rather than argued so the suggestion is not re-applied later as a crash fix. A bool
    short-circuit would still be a small WIN -- bool values are 0/1 so the scan is foregone, and
    `np.issubdtype(np.bool_, np.integer)` is False so the existing integer short-circuit misses it --
    but that is a performance change, not the crash Gemini described, and it is not in this PR's
    scope."""
    b = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(np.rint(b), b)     # no raise, and the same 0/1s, at any dtype
    assert norm._is_all_integer(b) is True                      # 2-D dense block loop
    assert norm._is_all_integer(b.reshape(-1)) is True           # _all_integer_1d
    sparse_bool = csr_matrix(b)
    assert sparse_bool.dtype == np.bool_
    assert np.issubdtype(sparse_bool.dtype, np.integer) is False, "which is why it is not skipped"
    assert norm._is_all_integer(sparse_bool) is True             # _all_integer_1d over .data
    norm.validate_input_type(ad.AnnData(X=sparse_bool), "counts")


def test_the_dtype_rounded_atol_is_the_CODE_not_a_promotion_rule():
    """`_int_atol` exists because `pyproject` declares `numpy>=1.26`, a range that SPANS the NEP 50
    switch -- so "1e-6 as representable in the input's dtype" was otherwise a property of whichever
    numpy resolved rather than of `norm` (Copilot, PR #341). ⚠️ It pins an outcome both regimes
    already SHARE, not a measured divergence: under 1.26 the tolerance stays in-width because the
    legacy ufunc promotion of `atol + rtol*abs(y)` value-minimizes `1e-6` to the array's width -- NOT
    because of `result_type(y, 1.)`, which fixes `y`'s dtype and never sees `atol` at all (codex
    review corrected my first reading of this). What the cast buys is that a future promotion change
    cannot move the FLOATING-dtype tolerance value silently; bool and complex stay on the uncast
    path by design, and it freezes nothing about `rint`/`isclose` themselves.

    Two halves. It rounds for every FLOATING width, including `longdouble`; and it does NOT touch
    `bool` or `complex`, which reach the same two loops, because `np.bool_(1e-6)` is `True` and a cast
    there would invent a tolerance rather than pin one.

    Neither gets special handling, and an earlier draft said "complex compares on its magnitude",
    which is misleading (Copilot): neither integrality caller takes `abs(x)` before `rint`.
    `np.rint` is applied componentwise and `np.allclose` compares `abs(x - rint(x))`, the modulus of
    the DIFFERENCE, which is why the assertions below check the tolerance's shape rather than
    inventing a complex contract.

    ⚠️ The two dtypes are NOT in the same position, which a further draft got wrong by lumping them as
    "unsupported" (codex review). The published rule is VALUE-based, not dtype-based, so a bool 0/1
    matrix is legitimately integral counts and `validate_input_type` accepts it -- proved by
    `test_np_rint_does_NOT_raise_on_bool_...` -- it just misses the integer fast path, since
    `np.issubdtype(np.bool_, np.integer)` is False. COMPLEX is the one outside the real-count
    contract, and it stays a documented unclosed exposure."""
    for dt in (np.float16, np.float32, np.float64, np.longdouble):
        assert norm._int_atol(np.dtype(dt)) == np.dtype(dt).type(norm._INT_ATOL)
        # a raw type class and a string must work too: both raised `AttributeError` before the
        # `np.dtype(...)` coercion went in (Gemini)
        assert norm._int_atol(dt) == norm._int_atol(np.dtype(dt))
    assert norm._int_atol("float32") == np.float32(norm._INT_ATOL)
    assert norm._int_atol(np.dtype(np.float16)) == np.float16(1.0132789611816406e-06)
    for dt in (np.bool_, np.complex64, np.complex128):
        atol = norm._int_atol(np.dtype(dt))
        assert atol is norm._INT_ATOL and isinstance(atol, float), f"{dt} must keep the weak scalar"
    assert bool(np.bool_(norm._INT_ATOL)) is True, "which is why bool must not be cast"
    # The measured complex behaviour, so the corrected sentence above is provable rather than
    # asserted:
    # `rint` componentwise, `allclose` on the modulus of the difference, so a fractional IMAGINARY
    # part is caught exactly as a fractional real one is.
    assert norm._is_all_integer(np.array([[3 + 0j, 1 + 0j]])) is True
    assert norm._is_all_integer(np.array([[3 + 0.5j, 1 + 0j]])) is False
    assert norm._is_all_integer(np.array([[3.5 + 0j, 1 + 0j]])) is False
    assert norm._is_all_integer(np.array([[3 + 1e-7j, 1 + 0j]])) is True
    assert np.rint(np.array([[3 + 0.5j]]))[0, 0] == 3 + 0j, "rint is componentwise, not on modulus"


def test_the_explicit_cast_is_a_NO_OP_on_this_numpy():
    """The cast above was applied only after verifying it changes no verdict on the numpy in use --
    otherwise it would be a behaviour change smuggled in as a robustness fix. Both float16 subnormal
    boundary values and the three wider dtypes agree between the weak-scalar and cast forms."""
    for dt in (np.float16, np.float32, np.float64, np.longdouble):
        for v in (1.0132789611816406e-06, 1.0728836059570312e-06, 5e-7, 50000.5, 50000.0):
            a = np.array([[v, 2.0]], dtype=dt)
            weak = bool(np.allclose(a, np.rint(a), rtol=0, atol=1e-6))
            cast = bool(np.allclose(a, np.rint(a), rtol=0, atol=norm._int_atol(a.dtype)))
            assert weak == cast, f"{np.dtype(dt).name} {v!r}: {weak} vs {cast}"


def test_scipy_refuses_float16_from_a_DENSE_matrix_but_not_from_raw_arrays():
    """The asymmetry the helper above depends on, pinned so "float16 sparse is impossible" cannot be
    re-derived from the first half of it -- which is the mistake that produced the earlier draft.

    ⚠️ `ValueError` with NO message match, deliberately. `pyproject` permits `scipy>=1.11`, and the
    "does not support dtype float16" wording is 1.15+; 1.11-1.14 refuse the same conversion with
    "Output dtype not compatible with inputs." (codex review). The PROPERTY -- dense conversion
    refuses, raw arrays do not -- holds across that range; the sentence does not. For the same reason
    this says nothing about the typed-EMPTY constructor, whose float16 rejection also arrived later
    (`_csr_row_block`'s docstring dates it to scipy 1.18.0)."""
    with pytest.raises(ValueError):
        csr_matrix(np.zeros((2, 2), dtype=np.float16))
    assert _f16_raw_csr(3.0).dtype == np.float16


@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_the_sign_check_runs_BEFORE_the_loosened_zero_neighbourhood_on_a_NaN_FREE_matrix(mk):
    """The loosening above must not become a hole for NEGATIVE dust, and on a NaN-free matrix it does
    not -- the reason is ORDER, not tolerance. `validate_input_type` tests `_min_value(X) < 0` before
    it ever computes `_is_all_integer`, so -5e-07 is refused on non-negativity and never reaches the
    tolerance, under the old rule and the new one alike; `allow_fractional` does not help, since it
    gates only the fractional branch. Pinned because the loosening is what makes the interaction
    worth stating: a future reordering would open it silently.

    ⚠️ **NaN-free is load-bearing, and the qualifier is the correction** -- an earlier draft of this
    claimed the ordering protects negative dust unconditionally (codex review). It does not, and the
    hole is PRE-EXISTING rather than opened here: `_min_value` reduces with `np.min`, so one NaN makes
    the minimum NaN, `NaN < 0` is False, and the sign check does not fire. The mixed case is pinned
    below.

    ⚠️ Two more neighbours, both pre-existing and neither moved by the tolerance change. A NaN matrix
    reads as not-all-integer under both rules, so strict `counts` still refuses it as "fractional". A
    `+inf` matrix reads as ALL-INTEGER under both -- `np.allclose(inf, rint(inf))` is True -- so this
    gate accepts it; `check_scale_limit` is what refuses it, on a per-cell total of `inf`."""
    with pytest.raises(ValueError, match="negative"):
        norm.validate_input_type(mk([[-5e-7, 3.0]]), "counts")
    with pytest.raises(ValueError, match="negative"):
        norm.validate_input_type(mk([[-5e-7, 3.0]]), "counts", allow_fractional=True)


@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_a_NaN_POISONS_the_sign_check_so_negative_dust_survives_it(mk):
    """The exception to the test above, PRE-EXISTING and unmoved by the tolerance change, pinned so
    the "order protects it" claim is not read as unconditional (codex review).

    `_min_value` reduces with `np.min`, so a single NaN makes the minimum NaN and `NaN < 0` is False
    -- the negative value is never seen. Strict `counts` still refuses the matrix, but for the OTHER
    reason (NaN is not all-integer), and the two permissive routes accept it outright:
    `allow_fractional=True`, and `lognorm` -- which wants "not all-integer" and gets it. What refuses
    it downstream is `check_scale_limit`, not this gate. `norm.check_scale_limit`'s own docstring
    records the same NaN hole from the other side."""
    poisoned = mk([[np.nan, -5e-7], [2.0, 3.0]])
    assert np.isnan(norm._min_value(poisoned.X)), "fixture must poison the minimum"
    with pytest.raises(ValueError, match="fractional"):        # refused, but not for the sign
        norm.validate_input_type(poisoned, "counts")
    norm.validate_input_type(poisoned, "counts", allow_fractional=True)   # accepted
    norm.validate_input_type(poisoned, "lognorm")                        # accepted


@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_the_lognorm_mislabel_raise_can_NEWLY_fire_on_an_all_dust_matrix(mk):
    """The consequence of the line above on the other gate, and the reason "a stricter predicate
    makes the mislabel raise fire LESS" is true only AWAY from zero. A matrix of 5e-07 used to read
    as "not all-integer" -- so `input_type='lognorm'` accepted it -- and now reads as all-integer
    with a positive max, which is exactly what that raise tests for."""
    dust = mk([[5e-7, 0.0], [0.0, 5e-7]])
    assert norm._is_all_integer(dust.X) is True
    with pytest.raises(ValueError, match="all-integer"):
        norm.validate_input_type(dust, "lognorm")


@pytest.mark.parametrize("value", _ACCEPTED_BEFORE_REJECTED_NOW + _STILL_ACCEPTED)
@pytest.mark.parametrize("mk", [_adata, _sparse_adata], ids=["dense", "sparse"])
def test_allow_fractional_still_bypasses_the_tightened_gate(mk, value):
    """Load-bearing for every bundle build: `real_bundle._baseline_leg` and
    `baseline.build_generic_baseline` both flip `allow_fractional_counts=True` because a tiled
    baseline profile is a MEAN, hence fractional in any space. Tightening the predicate must not
    reach that path."""
    norm.validate_input_type(mk([[value, 3.0], [1.0, 2.0]]), "counts", allow_fractional=True)


def test_the_lognorm_mislabel_gate_is_unmoved_on_a_representative_CPM_lognorm_matrix():
    """`validate_input_type`'s second raise fires when `input_type='lognorm'` and the values are
    ALL-INTEGER, so AWAY FROM ZERO a stricter `_is_all_integer` makes it fire less. (Near zero it can
    newly fire -- see `test_the_lognorm_mislabel_raise_can_NEWLY_fire_on_an_all_dust_matrix`. Both
    directions are real; neither is the whole story.) `_is_all_integer` is an AND over elements, so
    the matrices whose verdict flips the "fires less" way are those where every value is inside the
    old relative tolerance of an integer AND at least one is outside the new absolute one -- not
    "every value outside", which an earlier draft of this docstring said and which is a strictly
    smaller set (codex review). A near-integer matrix like 1000.001 is the case: refused as
    "mislabeled raw counts" before, accepted now.

    What this test measures is that neither direction reaches a matrix that is actually lognorm --
    ONE representative fixture, not a proof over all lognorm data, and the fixture is built here
    rather than quoted from a scratch script so the number in this docstring is the number the test
    computes. A lognorm value is `log1p` of a CPM-scale quantity, so under the 1e6 per-cell cap it
    cannot exceed a MATHEMATICAL `log1p(1e6)` = 13.8155 (storage rounds -- float32 stores that as
    13.815511703491211, just above the float64 value -- which is why the assertion below is on the
    nearest integer, not on the bound), where the old tolerance was at most 1.4001e-04: `1e-8 +
    1e-5*rint(13.8155)` = `1e-8 + 1e-5*14`, since `np.allclose`'s relative term takes its SECOND
    operand, which is `np.rint(v)`, and 14 is the answer under either storage; and either rule
    needs EVERY value that close to an integer, not one. MEASURED on the 500 x 2,000 CPM-normalized
    Poisson matrix below: of its **777,297** nonzero entries, **0** are within either tolerance of an
    integer -- the only qualifying entries are the exact zeros, under both rules.

    Losing the near-integer case costs nothing real either: a lognorm matrix of 1000.001 would be
    `log1p` of `e**1000` counts."""
    import scanpy as sc

    # Deterministic: `default_rng(0)` fixes the Poisson draw, and `normalize_total`/`log1p` preserve
    # the zero pattern, so the nonzero COUNT does not depend on scanpy's version. `target_sum=1e6`
    # is what makes this CPM at the competition's own per-cell cap rather than at this matrix's
    # median row total of 3,001, which is what an argument-free `normalize_total` would use.
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.poisson(1.5, size=(500, 2000)).astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e6)
    sc.pp.log1p(a)
    X = np.asarray(a.X)
    nonzero = X != 0
    dev = np.abs(X - np.rint(X))
    # BOTH rules are spelled out as literals here, and the new one deliberately does NOT read
    # `norm._INT_ATOL`. What this test measures is a property of the DATA -- that neither rule finds
    # a near-integer nonzero in a real lognorm matrix -- so it must hold, and pass, on a tree from
    # either side of this change. Referencing the constant would make it fail on a pre-`_INT_ATOL`
    # tree by name lookup rather than by property, which is the "fails for the wrong reason" shape.
    # The constant itself is pinned by `test_the_absolute_tolerance_boundary_is_the_constant_itself`.
    old_tol = 1e-8 + 1e-5 * np.abs(np.rint(X))          # what `np.allclose`'s default computed
    n_old = int((nonzero & (dev <= old_tol)).sum())
    n_new = int((nonzero & (dev <= 1e-6)).sum())
    # The property that is actually needed, and NOT `X.max() <= np.log1p(1e6)`: `log1p(1e6)` = 13.8155
    # is the MATHEMATICAL ceiling, and storage rounds -- `np.log1p(np.float32(1e6))` is
    # 13.815511703491211, ABOVE the float64 value 13.815511557963774 -- so a stored float32 lognorm
    # value can sit above the float64 bound while being the same quantity (codex review). What matters
    # for the tolerance argument is the NEAREST INTEGER, since that is `np.allclose`'s reference
    # operand: it is 14 under either storage, which is where 1.4001e-04 comes from.
    assert np.rint(X).max() <= 14.0, "a lognorm value under the 1e6 cap rounds to at most 14"
    assert n_old == 0 and n_new == 0, (
        f"{n_old} nonzero entries were old-close and {n_new} are new-close; the claim is that both "
        "are 0, i.e. only the exact zeros qualify under either rule")
    assert norm._is_all_integer(X) is False
    norm.validate_input_type(a, "lognorm")                  # unchanged: still not "all-integer"
    # The exact count comes LAST, after the substantive property (codex review): it pins the number
    # quoted in the docstring, and if the fixture or the RNG output ever changes it must not be what
    # fails first -- an unrelated stream change would then mask whether the property still holds.
    assert int(nonzero.sum()) == 777_297, (
        f"the docstring quotes 777,297 nonzero entries and this fixture has {int(nonzero.sum())}; "
        "the fixture or the RNG output changed -- update the docstring, do not loosen this")

    # The matrix whose verdict DID flip the "fires less" way.
    near_integer = _adata([[1000.001, 1000.001], [1000.001, 1000.001]])
    assert norm._is_all_integer(near_integer.X) is False
    norm.validate_input_type(near_integer, "lognorm")       # no longer read as mislabeled counts


# --- chunked _expm1_row_totals equivalence + check_scale_limit (Task 3) ---
def _old_expm1_row_totals(X):
    if _sp.issparse(X):
        Xe = X.copy()
        Xe.data = np.expm1(Xe.data)
        return np.asarray(Xe.sum(axis=1)).ravel()
    return np.expm1(np.asarray(X)).sum(axis=1)


@pytest.mark.parametrize("X", [
    np.log1p(np.array([[10.0, 0.0, 5.0], [1.0, 2.0, 0.0]])),
    np.zeros((3, 4)),
])
def test_expm1_row_totals_matches_old_dense(X):
    np.testing.assert_allclose(norm._expm1_row_totals(X), _old_expm1_row_totals(X), rtol=1e-6, atol=1e-9)


def test_expm1_row_totals_matches_old_sparse():
    X = _sp.csr_matrix(np.log1p(np.array([[10.0, 0.0, 5.0], [0.0, 2.0, 0.0]])))
    np.testing.assert_allclose(norm._expm1_row_totals(X), _old_expm1_row_totals(X), rtol=1e-6, atol=1e-9)


def test_expm1_row_totals_chunk_boundary():
    n = norm._INT_CHUNK + 5     # n_cols=1 -> rows_per_chunk == _INT_CHUNK; spans >1 chunk
    X = np.log1p(np.arange(n, dtype=np.float64)).reshape(n, 1)
    np.testing.assert_allclose(norm._expm1_row_totals(X), _old_expm1_row_totals(X), rtol=1e-6, atol=1e-9)


def test_expm1_row_totals_sparse_empty_and_trailing_rows():
    # CSR chunked path: leading, interior, AND trailing empty rows must match the full-matrix expm1
    # sum exactly (byte-identical) and stay float64. Mirrors the _row_totals empty-rows guard.
    dense = np.log1p(np.array([[0, 0, 0], [3, 0, 5], [0, 0, 0], [1, 2, 4], [0, 0, 0]], dtype=np.float64))
    X = _sp.csr_matrix(dense)
    np.testing.assert_array_equal(norm._expm1_row_totals(X), _old_expm1_row_totals(X))
    assert norm._expm1_row_totals(X).dtype == np.float64


def test_expm1_row_totals_all_empty_sparse():
    X = _sp.csr_matrix((7, 5), dtype=np.float64)
    np.testing.assert_array_equal(norm._expm1_row_totals(X), np.zeros(7))


def test_expm1_row_totals_chunk_boundary_spans_multiple_blocks(monkeypatch):
    # tiny row-chunk so a modest lognorm matrix spans >1 block (exercises block stitching + tail)
    monkeypatch.setattr(norm, "_ROW_TOTALS_ROW_CHUNK", 4)
    rng = np.random.default_rng(7)
    dense = np.log1p((rng.random((21, 6)) < 0.5) * rng.integers(1, 50, (21, 6)))
    X = _sp.csr_matrix(dense)
    np.testing.assert_allclose(norm._expm1_row_totals(X), _old_expm1_row_totals(X), rtol=1e-12, atol=0)


def test_expm1_row_totals_mismatched_index_dtype():
    # int32 indices + int64 indptr (cellstream layout): the chunked path must NOT route through the
    # full-matrix constructor (which would upcast int32 indices -> int64 at nnz>2**31). expm1(log1p(v))
    # == v, so per-row totals are the raw row sums.
    X = _sp.csr_matrix((2, 5), dtype=np.float64)
    X.data = np.log1p(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    X.indices = np.tile(np.array([0, 2, 4]), 2).astype(np.int32)
    X.indptr = np.array([0, 3, 6], dtype=np.int64)
    assert X.indices.dtype == np.int32 and X.indptr.dtype == np.int64
    np.testing.assert_allclose(norm._expm1_row_totals(X), np.array([6.0, 15.0]), rtol=1e-12, atol=0)


def test_expm1_row_totals_noncanonical_coo_sums_duplicates():
    # A non-CSR format with DUPLICATE coordinates must be canonicalized (tocsr sums dups) BEFORE
    # expm1 -- expm1 is not additive, so expm1(3)+expm1(5) != expm1(3+5). Coordinate (0,1) appears
    # twice (3 and 5): the correct row-0 total uses expm1(8), not expm1(3)+expm1(5).
    Xcoo = _sp.coo_matrix((np.array([3.0, 5.0, 2.0]),
                           (np.array([0, 0, 1]), np.array([1, 1, 0]))), shape=(2, 3))
    dense = Xcoo.toarray()                              # [[0, 8, 0], [2, 0, 0]] (dups summed)
    np.testing.assert_allclose(
        norm._expm1_row_totals(Xcoo), np.expm1(dense).sum(axis=1), rtol=1e-12, atol=0)


@pytest.mark.parametrize("mk", [_sp.csc_matrix, _sp.lil_matrix, _sp.dok_matrix])
def test_expm1_row_totals_non_csr_formats(mk):
    # csc/lil/dok route through tocsr() then the chunked path (lil/dok have no flat numeric .data;
    # the old raw-.data path raised on them).
    dense = np.log1p(np.array([[0.0, 3.0, 0.0], [5.0, 0.0, 2.0]]))
    X = mk(dense)
    np.testing.assert_allclose(
        norm._expm1_row_totals(X), np.expm1(dense).sum(axis=1), rtol=1e-12, atol=0)


def test_check_scale_limit_lognorm_rejects_via_chunked_expm1(monkeypatch):
    # end-to-end: the chunked expm1 row-totals path triggers the lognorm scale-limit rejection across
    # a block boundary. Every stored value = log1p(30) stays UNDER the per-value cap (so the max-value
    # guard passes and we reach _expm1_row_totals); each row's expm1 sum = 60 is what over-scales.
    monkeypatch.setattr(norm, "_ROW_TOTALS_ROW_CHUNK", 2)
    dense = np.log1p(np.full((5, 2), 30.0))                       # per-row expm1 total 60
    a = ad.AnnData(X=_sp.csr_matrix(dense))
    norm.check_scale_limit(a, "lognorm", max_counts_per_cell=1000.0)   # ok (60 < 1000)
    with pytest.raises(ValueError, match="per-cell total"):            # 60 > 50, but log1p(30) < log1p(50)
        norm.check_scale_limit(a, "lognorm", max_counts_per_cell=50.0)


def test_check_scale_limit_still_rejects_overscale():
    a = ad.AnnData(X=np.array([[5.0, 5.0], [1.0, 1.0]]))   # counts; row totals 10 and 2
    norm.check_scale_limit(a, "counts", max_counts_per_cell=100.0)   # ok, no raise
    with pytest.raises(ValueError, match="exceeds"):
        norm.check_scale_limit(a, "counts", max_counts_per_cell=5.0)


# --- chunked _row_totals equivalence (avoids scipy X.sum(axis=1) full-matrix upcast) ---
def _old_row_totals(X):
    if _sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1)


@pytest.mark.parametrize("X", [
    np.array([[10.0, 0.0, 5.0], [1.0, 2.0, 0.0]]),
    np.zeros((3, 4)),
])
def test_row_totals_matches_old_dense(X):
    np.testing.assert_allclose(norm._row_totals(X), _old_row_totals(X), rtol=1e-9, atol=0)


def test_row_totals_matches_old_sparse_with_empty_and_trailing_rows():
    # leading, interior, AND trailing empty rows: the reduceat-style shortcuts trip on trailing
    # empties (out-of-bounds); the chunked per-block sum must match scipy exactly and stay float64.
    dense = np.array([[0, 0, 0], [3, 0, 5], [0, 0, 0], [1, 2, 4], [0, 0, 0]], dtype=np.uint16)
    X = _sp.csr_matrix(dense)
    np.testing.assert_array_equal(norm._row_totals(X), _old_row_totals(X))
    assert norm._row_totals(X).dtype == np.float64


def test_row_totals_chunk_boundary_spans_multiple_blocks(monkeypatch):
    # tiny row-chunk so a modest matrix spans >1 block (exercises block stitching + partial tail)
    monkeypatch.setattr(norm, "_ROW_TOTALS_ROW_CHUNK", 4)
    rng = np.random.default_rng(3)
    dense = ((rng.random((21, 6)) < 0.5) * rng.integers(1, 50, (21, 6))).astype(np.uint16)
    X = _sp.csr_matrix(dense)
    np.testing.assert_array_equal(norm._row_totals(X), _old_row_totals(X))


def test_row_totals_all_empty_sparse():
    X = _sp.csr_matrix((7, 5), dtype=np.uint16)
    np.testing.assert_array_equal(norm._row_totals(X), np.zeros(7))


def test_check_scale_limit_sparse_counts_rejects_via_chunked_totals(monkeypatch):
    # end-to-end: the chunked path still triggers the scale-limit rejection across a block boundary
    monkeypatch.setattr(norm, "_ROW_TOTALS_ROW_CHUNK", 2)
    dense = np.array([[1, 1], [1, 1], [1, 1], [90, 90]], dtype=np.uint16)  # last row total 180
    a = ad.AnnData(X=_sp.csr_matrix(dense))
    norm.check_scale_limit(a, "counts", max_counts_per_cell=1000.0)  # ok
    with pytest.raises(ValueError, match="exceeds"):
        norm.check_scale_limit(a, "counts", max_counts_per_cell=100.0)


@pytest.mark.parametrize("dense", [True, False])
def test_to_normalization_lognorm_to_normalized_bit_identical(dense):
    # expm1 result must be EXACT and the input must not be mutated.
    rng = np.random.default_rng(5)
    Xv = np.abs(rng.standard_normal((200, 30)))
    if not dense:
        Xv = Xv * (rng.random((200, 30)) < 0.4)
    X = Xv.copy() if dense else csr_matrix(Xv)
    obs = pd.DataFrame({"t": ["a"] * 200})
    adata = ad.AnnData(X=X, obs=obs)
    src_before = np.asarray(adata.X.copy()) if dense else adata.X.toarray()

    out = norm.to_normalization(adata, input_type="lognorm", target="normalized")

    got = np.asarray(out.X) if dense else out.X.toarray()
    ref = np.expm1(Xv if dense else csr_matrix(Xv).toarray())
    np.testing.assert_array_equal(got, ref)                      # exact, not allclose
    # input AnnData must be unchanged
    src_after = np.asarray(adata.X) if dense else adata.X.toarray()
    np.testing.assert_array_equal(src_after, src_before)



def test_to_normalization_lognorm_to_normalized_int_sparse_no_crash():
    # An integer sparse matrix declared lognorm must not crash on the in-place expm1 path
    # (regression guard: in-place expm1 cannot write float into an int .data array).
    from scipy.sparse import csr_matrix
    X = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64))
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"t": ["a", "b"]}))
    out = norm.to_normalization(adata, input_type="lognorm", target="normalized")
    np.testing.assert_array_equal(
        out.X.toarray(), np.expm1(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    )


# --- sparse min/max over .data (avoids scipy X.min()/X.max()'s int32->int64 index upcast) ---
def _mismatched_csr(data, *, canonical=True, n_cols=5):
    """A 2-row CSR with MISMATCHED index dtypes -- int32 column indices + int64 indptr -- built
    cellstream-style (attributes set directly, bypassing scipy's constructor dtype reconciliation).
    This is the 5.5M-cell submission's layout in miniature: nnz>2**31 forces an int64 indptr while
    the small column indices stay int32. ``canonical=False`` reverses the within-row index order so
    scipy would need to sort (where raw X.min()/X.max() then RAISE across the dtype mismatch)."""
    data = np.asarray(data)
    per_row = data.size // 2
    row = np.arange(0, per_row * 2, 2, dtype=np.int32)          # 0,2,4,... (< n_cols), sorted+unique
    if not canonical:
        row = row[::-1].copy()
    X = _sp.csr_matrix((2, n_cols), dtype=data.dtype)
    X.data = data
    X.indices = np.tile(row, 2).astype(np.int32)
    X.indptr = np.array([0, per_row, per_row * 2], dtype=np.int64)
    return X


def _old_minmax(X):
    return float(X.min()), float(X.max())


@pytest.mark.parametrize("dense", [
    np.array([[3.0, 0.0, 5.0], [1.0, 2.0, 0.0]]),      # implicit zeros
    np.array([[-2.0, 3.0], [5.0, -7.0]]),              # fully-dense negatives (no implicit zeros)
    np.array([[0.0, -4.0, 0.0], [5.0, 0.0, 0.0]]),     # negative + implicit zeros
    np.zeros((4, 4)),                                  # all-zero (nnz == 0)
    np.array([[1.0, 2.0], [3.0, 4.0]]),                # fully dense, positive
])
def test_min_max_value_match_scipy_canonical(dense):
    for X in (_sp.csr_matrix(dense), _sp.csc_matrix(dense)):
        exp_min, exp_max = _old_minmax(X)
        assert norm._min_value(X) == exp_min
        assert norm._max_value(X) == exp_max


def test_min_max_value_match_scipy_random_float_csr():
    X = _sp.random(300, 40, density=0.1, format="csr", dtype=np.float64, random_state=0)
    exp_min, exp_max = _old_minmax(X)
    assert norm._min_value(X) == exp_min
    assert norm._max_value(X) == exp_max


def test_min_max_value_mismatched_index_dtype_canonical():
    # int32 indices + int64 indptr: raw X.min()/X.max() would upcast indices to int64 (a full-nnz
    # temp at scale). data-based path returns the correct values with implicit zeros folded in.
    X = _mismatched_csr(np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16))
    assert X.indices.dtype == np.int32 and X.indptr.dtype == np.int64  # the layout under test
    assert norm._min_value(X) == 0.0    # implicit zeros present (nnz=6 < 2*5)
    assert norm._max_value(X) == 6.0


def test_min_max_value_mismatched_index_dtype_noncanonical():
    # The failing-test that pins the fix: raw X.min()/X.max() RAISE on a non-canonical
    # mismatched-dtype CSR (scipy cannot sort indices in place across the dtype gap); the
    # data-based path is unaffected and returns the correct values.
    X = _mismatched_csr(np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16), canonical=False)
    with pytest.raises(ValueError):
        X.min()                                        # documents the old-path failure
    assert norm._min_value(X) == 0.0
    assert norm._max_value(X) == 6.0
    # validate_input_type (counts) uses _min_value -> must not raise on this input
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["c0", "c1"]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]))
    norm.validate_input_type(adata, "counts")          # no raise


def test_validate_rejects_negative_mismatched_index_dtype():
    # A stored negative is still detected through the data-based min (the negative-value guard
    # holds on the mismatched-dtype layout).
    X = _mismatched_csr(np.array([1, 2, -3, 4, 5, 6], dtype=np.int32))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["c0", "c1"]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]))
    with pytest.raises(ValueError, match="negative"):
        norm.validate_input_type(adata, "counts")


def test_min_max_value_propagate_nan():
    # NaN must PROPAGATE (match scipy X.min()/X.max()), not be swallowed to 0.0 by the implicit-zero
    # fold. The fold always passes the reduced value first (`fold(m, 0.0)`), and Python min/max return
    # the first arg when the comparison is False -- which it always is for NaN -- so `min(nan, 0.0)` and
    # `max(nan, 0.0)` are nan, matching scipy. This locks that behaviour against a fold arg-order change.
    dense = np.array([[np.nan, 0.0, 2.0], [0.0, 3.0, 0.0]])  # NaN + implicit zeros (exercises the fold)
    for X in (_sp.csr_matrix(dense), _sp.csc_matrix(dense)):
        assert np.isnan(float(X.min())) and np.isnan(float(X.max()))   # scipy reference
        assert np.isnan(norm._min_value(X))
        assert np.isnan(norm._max_value(X))
    # mismatched-dtype canonical CSR (int32 indices + int64 indptr) with a stored NaN
    Xm = _mismatched_csr(np.array([np.nan, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64))
    assert np.isnan(norm._min_value(Xm))
    assert np.isnan(norm._max_value(Xm))


def test_min_max_value_never_call_scipy_reduction(monkeypatch):
    # Guard: _min_value/_max_value (and validate_input_type) must NOT route through scipy's
    # X.min()/X.max() -- that is the code path whose sum_duplicates() upcasts indices to int64.
    def _boom(*a, **k):
        raise AssertionError("scipy sparse .min()/.max() must not be called")
    monkeypatch.setattr(_sp.csr_matrix, "min", _boom)
    monkeypatch.setattr(_sp.csr_matrix, "max", _boom)
    X = _sp.csr_matrix(np.array([[3.0, 0.0, 5.0], [1.0, 2.0, 0.0]], dtype=np.uint16))
    assert norm._min_value(X) == 0.0
    assert norm._max_value(X) == 5.0
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["c0", "c1"]),
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    norm.validate_input_type(adata, "counts")          # exercises _min_value, no .min() call


# --- check_scale_limit reuses a precomputed per-cell max (skip _row_totals on counts) ---
def test_check_scale_limit_precomputed_skips_row_totals(monkeypatch):
    # counts + precomputed max <= cap: passes WITHOUT touching _row_totals.
    a = _sparse_adata([[3.0, 0.0], [0.0, 5.0]])  # true max row total = 5

    def _boom(X):
        raise AssertionError("_row_totals must not run when a precomputed max is supplied")

    monkeypatch.setattr(norm, "_row_totals", _boom)
    norm.check_scale_limit(a, "counts", 1_000_000.0, precomputed_row_total_max=5.0)


def test_check_scale_limit_precomputed_over_budget_rejects():
    a = _sparse_adata([[1.0, 0.0]])
    with pytest.raises(ValueError, match="max_counts_per_cell"):
        norm.check_scale_limit(a, "counts", 1_000_000.0, precomputed_row_total_max=2_000_000.0)


def test_check_scale_limit_counts_none_still_scans():
    # No precomputed max -> existing full _row_totals path (rejects the over-budget cell).
    a = _sparse_adata([[2_000_000.0, 0.0]])
    with pytest.raises(ValueError, match="max_counts_per_cell"):
        norm.check_scale_limit(a, "counts", 1_000_000.0)


def test_check_scale_limit_lognorm_ignores_precomputed(monkeypatch):
    # lognorm still runs _expm1_row_totals; the counts-only precomputed arg is ignored.
    calls = {"n": 0}
    real = norm._expm1_row_totals

    def _spy(X):
        calls["n"] += 1
        return real(X)

    monkeypatch.setattr(norm, "_expm1_row_totals", _spy)
    a = _sparse_adata([[0.0, 1.5], [2.0, 0.0]])
    norm.check_scale_limit(a, "lognorm", 1_000_000.0, precomputed_row_total_max=999.0)
    assert calls["n"] == 1


# --- _csr_row_block: the check-free row-block view (moved here from streaming_bulk, PR #73) ---
def test_csr_row_block_reexport_is_the_same_object():
    # PR #73's tests and gpu/bulk.py both import it from streaming_bulk; the move must not break them.
    from cell_eval2.streaming_bulk import _csr_row_block as reexported

    assert reexported is norm._csr_row_block


def test_csr_row_block_is_a_view_and_pins_the_prune_threshold():
    # The entire point of the helper. scipy's constructor runs check_format -> prune() ->
    # _prune_array, which copies a view satisfying `size < base.size // 2` (integer floor). One
    # nonzero per row makes the block/base size ratio exactly the row ratio, so the boundary is
    # testable exactly: 10% is pruned, 50% is NOT, 90% is not. If these `old`
    # assertions ever fail, scipy changed _prune_array and the spec's rationale needs re-checking.
    n, g = 200, 4
    rows = np.arange(n)
    X = _sp.csr_matrix((np.ones(n, np.float32), (rows, rows % g)), shape=(n, g))
    assert X.nnz == n and np.array_equal(X.indptr, np.arange(n + 1))

    def old_block(start, stop):
        lo, hi = int(X.indptr[start]), int(X.indptr[stop])
        return X.__class__(
            (X.data[lo:hi], X.indices[lo:hi], X.indptr[start:stop + 1] - lo),
            shape=(stop - start, X.shape[1]), copy=False,
        )

    assert not np.shares_memory(old_block(0, 20).data, X.data)    # 10% -> pruned into a copy
    assert np.shares_memory(old_block(0, 100).data, X.data)       # exactly 50% -> size < base//2 is False
    assert np.shares_memory(old_block(0, 180).data, X.data)       # 90% -> shares

    for start, stop in [(0, 20), (0, 100), (0, 180)]:
        blk = norm._csr_row_block(X, start, stop)
        assert np.shares_memory(blk.data, X.data)                 # the helper always shares
        assert np.shares_memory(blk.indices, X.indices)
        np.testing.assert_array_equal(blk.toarray(), old_block(start, stop).toarray())


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
def test_csr_row_block_preserves_parent_index_dtypes(cls_name):
    # The helper PRESERVES int32 indices + int64 indptr; the raw constructor reconciles them (on
    # csr_array it upcasts indices int32 -> int64, the full-nnz temporary the chunking exists to
    # avoid). assert_array_equal would not notice a dtype change, so assert dtypes explicitly.
    cls = getattr(_sp, cls_name)
    X = cls(np.array([[1., 0, 2.], [0, 3., 0], [4., 0, 0], [0, 5., 6.]]))
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int64)

    blk = norm._csr_row_block(X, 0, 4)
    assert blk.indices.dtype == np.int32 and blk.indptr.dtype == np.int64   # both preserved

    lo, hi = int(X.indptr[0]), int(X.indptr[4])
    ref = cls((X.data[lo:hi], X.indices[lo:hi], X.indptr[0:5] - lo), shape=(4, 3), copy=False)
    # Pin the divergence, not just our side of it: csr_matrix downcasts indptr to int32,
    # csr_array upcasts indices to int64.
    expected = {"csr_matrix": (np.int32, np.int32), "csr_array": (np.int64, np.int64)}[cls_name]
    assert (ref.indices.dtype, ref.indptr.dtype) == expected

    np.testing.assert_array_equal(
        np.asarray(blk.sum(axis=1)).ravel(), np.asarray(ref.sum(axis=1)).ravel()
    )


def test_csr_row_block_data_override_rejects_bad_shapes():
    X = _sp.random(50, 10, density=0.3, format="csr", dtype=np.float64, random_state=1)
    with pytest.raises(ValueError, match="length"):
        norm._csr_row_block(X, 0, 25, data=np.zeros(3))
    nnz = int(X.indptr[25]) - int(X.indptr[0])
    with pytest.raises(ValueError, match="1-D"):
        norm._csr_row_block(X, 0, 25, data=np.zeros((nnz, 1)))


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
@pytest.mark.parametrize("dt", [np.bool_, np.int8, np.uint8, np.int16, np.int32,
                                np.float32, np.float64])
def test_csr_row_block_expm1_override_matches_the_raw_constructor(cls_name, dt):
    # np.expm1 of bool/int8/uint8 is float16, which scipy's EMPTY-CSR constructor rejects even
    # though the raw-array constructor accepts it. The block must match the raw construction in
    # dtype and bytes across the promotion classes _expm1_row_totals can be handed (float16 from
    # bool/int8/uint8, float32 from int16, float64 from int32 -- plus the float inputs themselves).
    cls = getattr(_sp, cls_name)
    dense = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    X = cls(dense.astype(dt))
    lo, hi = int(X.indptr[0]), int(X.indptr[3])
    over = np.expm1(X.data[lo:hi])
    ref = cls((over, X.indices[lo:hi], X.indptr[0:4] - lo), shape=(3, 3), copy=False)
    blk = norm._csr_row_block(X, 0, 3, data=over)
    assert blk.dtype == ref.dtype
    np.testing.assert_array_equal(blk.data, ref.data)
    blk_sum, ref_sum = np.asarray(blk.sum(axis=1)).ravel(), np.asarray(ref.sum(axis=1)).ravel()
    assert blk_sum.dtype == ref_sum.dtype
    assert blk_sum.tobytes() == ref_sum.tobytes()


def _oracle_block(X, start, stop):
    """Today's raw-array construction, the oracle every layout test compares against."""
    lo, hi = int(X.indptr[start]), int(X.indptr[stop])
    return X.__class__(
        (X.data[lo:hi], X.indices[lo:hi], X.indptr[start:stop + 1] - lo),
        shape=(stop - start, X.shape[1]), copy=False,
    )


def _assert_block_matches(X, start, stop):
    """The contract: same .data dtype and values, and a BYTE-identical row-sum. Index dtypes are
    deliberately NOT compared -- the helper preserves the parent's where the raw constructor
    reconciles them (see test_csr_row_block_preserves_parent_index_dtypes)."""
    ref, blk = _oracle_block(X, start, stop), norm._csr_row_block(X, start, stop)
    assert blk.dtype == ref.dtype
    assert blk.data.dtype == ref.data.dtype
    np.testing.assert_array_equal(blk.data, ref.data)
    ref_sum = np.asarray(ref.sum(axis=1)).ravel()
    blk_sum = np.asarray(blk.sum(axis=1)).ravel()
    assert blk_sum.dtype == ref_sum.dtype
    assert blk_sum.tobytes() == ref_sum.tobytes()


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
def test_csr_row_block_matches_raw_constructor_on_noncanonical_csr(cls_name):
    # NOTE: coo.tocsr() SUMS duplicate coordinates, so it cannot build a non-canonical matrix.
    # Build straight from duplicate CSR arrays, and assert the premise before relying on it.
    cls = getattr(_sp, cls_name)
    X = cls((np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 2]), np.array([0, 2, 3, 4])),
            shape=(3, 3))
    assert X.nnz == 4 and not X.has_canonical_format, "fixture is not actually non-canonical"
    _assert_block_matches(X, 0, 3)


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
def test_csr_row_block_matches_raw_constructor_on_overallocated_buffer(cls_name):
    # _prune_array fires only when nnz < len(data) // 2, i.e. len(data) >= 2*nnz + 2. A short
    # tail does NOT prune and would test nothing: nnz=4 against len=9 gives 4 < 4, which is False.
    cls = getattr(_sp, cls_name)
    X = cls(np.array([[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 0]]))
    nnz = X.nnz
    X.data = np.concatenate([X.data, np.zeros(2 * nnz + 1)])
    X.indices = np.concatenate([X.indices, np.zeros(2 * nnz + 1, dtype=X.indices.dtype)])
    assert nnz < X.data.size // 2, "over-allocation too small to trigger pruning"
    _assert_block_matches(X, 0, 3)


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
def test_csr_row_block_matches_raw_constructor_on_offset_indptr(cls_name):
    cls = getattr(_sp, cls_name)
    X = cls(np.array([[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 0], [0, 5.0, 6.0]]))
    assert int(X.indptr[1]) > 0, "block must start at a nonzero indptr offset"
    _assert_block_matches(X, 1, 4)


def test_row_totals_mismatched_index_dtype():
    # int32 indices + int64 indptr (cellstream's layout). The block path must preserve both -- a
    # constructor that reconciles them would upcast indices int32 -> int64, the full-nnz temporary
    # the chunking exists to avoid. Mirrors test_expm1_row_totals_mismatched_index_dtype.
    X = _sp.csr_matrix((2, 5), dtype=np.float64)
    X.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    X.indices = np.tile(np.array([0, 2, 4]), 2).astype(np.int32)
    X.indptr = np.array([0, 3, 6], dtype=np.int64)
    assert X.indices.dtype == np.int32 and X.indptr.dtype == np.int64
    got = norm._row_totals(X)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, np.array([6.0, 15.0]), rtol=1e-12, atol=0)
    blk = norm._csr_row_block(X, 0, 2)
    assert blk.indices.dtype == np.int32 and blk.indptr.dtype == np.int64


def test_row_totals_float32_is_the_widened_float32_sum(monkeypatch):
    # The real matrices are float32. scipy's block .sum(axis=1) accumulates in the DATA dtype, so
    # _row_totals is the float32 sum WIDENED into its float64 accumulator -- not a float64
    # reduction, which would be more accurate but not identical and could flip a borderline
    # max_counts_per_cell comparison. _old_row_totals returns float32, so widen before comparing
    # bytes; a plain assert_array_equal here would be dtype-blind and prove less.
    monkeypatch.setattr(norm, "_ROW_TOTALS_ROW_CHUNK", 4)
    rng = np.random.default_rng(11)
    dense = ((rng.random((21, 6)) < 0.6) * rng.integers(1, 5000, (21, 6))).astype(np.float32)
    X = _sp.csr_matrix(dense)
    got, ref = norm._row_totals(X), _old_row_totals(X)
    assert got.dtype == np.float64 and ref.dtype == np.float32
    assert got.tobytes() == ref.astype(np.float64).tobytes()


@pytest.mark.parametrize("fn,build", [
    ("_row_totals", lambda d: _sp.csr_matrix(d.astype(np.float64))),
    ("_expm1_row_totals", lambda d: _sp.csr_matrix(np.log1p(d.astype(np.float64)))),
])
def test_row_totals_exactly_one_full_block_no_tail(monkeypatch, fn, build):
    # n == _ROW_TOTALS_ROW_CHUNK exactly: the loop runs one full block and no partial tail, the
    # boundary the 21-row multi-block tests skip over.
    monkeypatch.setattr(norm, "_ROW_TOTALS_ROW_CHUNK", 8)
    rng = np.random.default_rng(13)
    dense = (rng.random((8, 5)) < 0.6) * rng.integers(1, 40, (8, 5))
    X = build(dense)
    old = _old_row_totals if fn == "_row_totals" else _old_expm1_row_totals
    np.testing.assert_allclose(getattr(norm, fn)(X), old(X), rtol=1e-12, atol=0)
def test_resolve_target_sum_passthrough_is_identical():
    # A numeric target_sum is returned UNCHANGED (not float()-ed), so no existing config
    # object or config_hash shifts.
    #
    # Asserted by IDENTITY against an np.float32 token, not by equality: `float(1e6) == 1e6`
    # is True, so an `== 1e6` check passes even if the resolver casts -- it cannot tell
    # "returned unchanged" from "returned a new equal float", which is the whole claim.
    # np.float32 is also the concrete type that motivated the claim: EvalConfig accepts one
    # (math.isfinite does), and casting it would shift config_hash.
    a = _adata([[1, 2], [3, 4]])
    token = np.float32(1e6)
    assert norm.resolve_target_sum(a, input_type="counts", target_sum=token) is token
    assert norm.resolve_target_sum(a, input_type="lognorm", target_sum=token) is token
    assert norm.resolve_target_sum(a, input_type="counts", target_sum=1e6) == 1e6
    assert norm.resolve_target_sum(a, input_type="lognorm", target_sum=1e4) == 1e4


def test_resolve_target_sum_counts_is_nnz_median():
    # Library sizes 3, 7, 11 -> median 7. The zero-total cell is EXCLUDED (nnz median).
    a = _adata([[1, 2], [3, 4], [5, 6], [0, 0]])
    assert norm.resolve_target_sum(a, input_type="counts", target_sum=None) == 7.0


def test_resolve_target_sum_matches_scanpy_dense_branch():
    # Pins our rule against sc.pp.normalize_total's DENSE branch (_compute_nnz_median):
    # normalizing with our resolved number must equal normalizing with target_sum=None.
    import scanpy as sc

    rng = np.random.default_rng(0)
    X = rng.poisson(5, size=(50, 8)).astype(np.float64)
    X[3] = 0  # a zero-total cell -- the case the two scanpy branches disagree on
    a = _adata(X)
    ours = norm.resolve_target_sum(a, input_type="counts", target_sum=None)
    auto, explicit = a.copy(), a.copy()
    sc.pp.normalize_total(auto, target_sum=None)
    sc.pp.normalize_total(explicit, target_sum=ours)
    assert np.allclose(np.asarray(auto.X), np.asarray(explicit.X))


def test_resolve_target_sum_lognorm_stays_none():
    a = _adata([[1, 2], [3, 4]])
    assert norm.resolve_target_sum(a, input_type="lognorm", target_sum=None) is None


def test_resolve_target_sum_all_zero_control_raises():
    a = _adata([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="median library size"):
        norm.resolve_target_sum(a, input_type="counts", target_sum=None)


def test_resolve_target_sum_empty_control_raises():
    a = ad.AnnData(X=np.zeros((0, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="no control cells"):
        norm.resolve_target_sum(a, input_type="counts", target_sum=None)


def test_resolve_target_sum_bad_input_type_raises():
    a = _adata([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="input_type"):
        norm.resolve_target_sum(a, input_type="normalized", target_sum=None)


def test_resolve_target_sum_sparse_equals_dense():
    # Format-independence is the point: scanpy's own two branches do not agree here.
    rng = np.random.default_rng(1)
    X = rng.poisson(4, size=(40, 6)).astype(np.float64)
    X[7] = 0
    assert (norm.resolve_target_sum(_adata(X), input_type="counts", target_sum=None)
            == norm.resolve_target_sum(_sparse_adata(X), input_type="counts", target_sum=None))


# --- #287: the lognorm scale limit must not reject this library's own exact-cap output ---
#
# `to_normalization(counts -> lognorm)` stores float32 log1p values; `check_scale_limit`
# reconstructs them with expm1 and compares STRICTLY, so a row normalized to exactly the cap
# came back at 1,000,000.25 and was rejected. The round trip's error is now budgeted
# (`_SCALE_LIMIT_TOL_ULP`), sized from a measured ceiling of 8.05 ULP over caps 1e3..1e9 and
# G from 2 to 18,533.

@pytest.mark.parametrize("sparse", [False, True])
def test_scale_limit_accepts_the_librarys_own_exact_cap_normalization(sparse):
    """The issue's own repro, verbatim in shape. `[[1, 3]]` normalized to a target of exactly
    the default v2 cap reconstructs at 1,000,000.25 -- a quarter of one count above a budget of
    a million, entirely from float32 -- and used to raise."""
    X = np.array([[1, 3]], dtype=np.float32)
    counts = ad.AnnData(_sp.csr_matrix(X) if sparse else X)
    lognorm = norm.to_normalization(counts, "counts", "lognorm", target_sum=1_000_000.0)
    total = float(norm._expm1_row_totals(lognorm.X)[0])
    assert total > 1_000_000.0, (
        f"fixture no longer exercises the overshoot: reconstructed {total!r}")
    norm.check_scale_limit(lognorm, "lognorm", 1_000_000.0)   # must not raise


def test_scale_limit_tolerance_covers_the_measured_roundoff_ceiling():
    """Sized from a ceiling, not from the one case in the issue: asserts every reconstruction of
    this library's own normalization lands inside the budget, so a future change to
    `_expm1_row_totals` that makes the round trip noisier fails here rather than in a user's run.

    ⚠️ This sweep is a SUBSET of the offline measurement that set the 16-ULP constant. That one
    covered caps 1e3..1e9 and G from 2 to 18,533 across uniform / heavy-tailed / one-gene-spike
    compositions and peaked at 8.05 ULP; this one stops at G=2,000 to keep the suite fast. The
    docstring used to quote the offline range as if the test covered it (codex review)."""
    rng = np.random.default_rng(7)
    worst_ulp = 0.0
    for cap in (1e3, 1e4, 1e6, 1e9):
        for G in (2, 100, 2_000):
            X = np.maximum(np.rint(rng.lognormal(0.0, 4.0, size=(40, G))), 0.0)
            X[X.sum(axis=1) == 0, 0] = 1.0
            a = ad.AnnData(_sp.csr_matrix(X.astype(np.float32)))
            ln = norm.to_normalization(a, "counts", "lognorm", target_sum=float(cap))
            totals = norm._expm1_row_totals(ln.X)
            worst_ulp = max(worst_ulp, float(np.abs(totals - cap).max() / cap
                                             / np.finfo(np.float32).eps))
            norm.check_scale_limit(ln, "lognorm", cap)        # must not raise
    assert worst_ulp < norm._SCALE_LIMIT_TOL_ULP, (
        f"measured {worst_ulp:.2f} ULP against a budget of {norm._SCALE_LIMIT_TOL_ULP} -- the "
        "tolerance no longer covers the round trip it was sized for")


def test_scale_limit_still_rejects_a_resolvable_breach():
    """The tolerance buys ~1.9 counts at the v2 cap. Ten counts over is 84 ULP -- well outside
    it, and still rejected. Without this the widening could not be told from removing the gate.

    Split over two genes deliberately: a single gene carrying the whole row trips the per-VALUE
    guard first, and this test is about the row TOTAL."""
    X = np.log1p(np.array([[500_005.0, 500_005.0]], dtype=np.float32))   # total 1,000,010
    a = ad.AnnData(_sp.csr_matrix(X))
    with pytest.raises(norm.ScaleLimitError, match="per-cell total"):
        norm.check_scale_limit(a, "lognorm", 1_000_000.0)


def test_scale_limit_error_states_the_tolerance_it_allowed():
    """An operator reading the rejection has to be able to tell the budget from the slack."""
    X = np.log1p(np.array([[900_000.0, 900_000.0]], dtype=np.float32))   # total 1.8e6
    with pytest.raises(norm.ScaleLimitError) as exc:
        norm.check_scale_limit(ad.AnnData(X), "lognorm", 1_000_000.0)
    assert "ULP" in str(exc.value) and "max_counts_per_cell=1000000" in str(exc.value)


def test_scale_limit_per_value_guard_also_carries_the_tolerance():
    """The second site #287 did not name. A row whose whole budget sits in ONE gene stores
    `log1p(cap)` in float32, which can round UP past the float64 `np.log1p(cap)` -- measured at
    cap=1e3. Found by the ceiling sweep above, so it is pinned separately here."""
    for cap in (1e3, 1e4, 1e6):
        X = np.array([[cap]], dtype=np.float32)
        ln = norm.to_normalization(ad.AnnData(_sp.csr_matrix(X)), "counts", "lognorm",
                                   target_sum=float(cap))
        norm.check_scale_limit(ln, "lognorm", cap)        # must not raise
    # an order-of-magnitude breach still trips the per-value guard, tolerance or not
    with pytest.raises(norm.ScaleLimitError, match="overflow expm1"):
        norm.check_scale_limit(ad.AnnData(np.log1p(np.array([[2_000_000.0, 0.0]]))),
                               "lognorm", 1_000_000.0)


def test_scale_limit_tolerance_follows_the_STORED_dtype():
    """A float64 lognorm matrix has ~1e-9 of float32's round-trip error, so handing it float32's
    slack would loosen the gate by eight orders for nothing. Same overshoot, two dtypes, two
    answers: 5e-7 relative is inside float32's 16-ULP budget (1.9e-6) and far outside
    float64's (3.6e-15)."""
    over = 1_000_000.0 * (1.0 + 5e-7)
    assert norm._scale_limit_rtol(np.zeros(1, dtype=np.float32)) > 5e-7
    assert norm._scale_limit_rtol(np.zeros(1, dtype=np.float64)) < 5e-7
    a64 = ad.AnnData(np.log1p(np.array([[over]], dtype=np.float64)))
    with pytest.raises(norm.ScaleLimitError):
        norm.check_scale_limit(a64, "lognorm", 1_000_000.0)


def test_the_tolerance_is_CAPPED_at_float32_never_widened_by_a_narrow_dtype():
    """16 ULP was measured on float32/float64. Applied verbatim to float16 (eps 9.77e-04) it is a
    1.6% relative budget -- a reconstructed total of 1010.0 sails past a cap of 1000. A gate must
    fail CLOSED outside the domain its tolerance was measured on. Found by codex review; without
    the `min` in `_scale_limit_rtol` this test goes red."""
    eps32 = float(np.finfo(np.float32).eps)
    assert norm._scale_limit_rtol(np.zeros(1, dtype=np.float16)) <= norm._SCALE_LIMIT_TOL_ULP * eps32
    # the concrete hole: 0.95% over a cap of 1000, which 16 ULP of float16 would have allowed
    X = np.log1p(np.array([[504.75, 504.75]], dtype=np.float16))
    total = float(norm._expm1_row_totals(X)[0])
    assert total > 1000.0 * (1.0 + norm._SCALE_LIMIT_TOL_ULP * eps32), (
        f"fixture must overshoot float32's budget; reconstructed {total!r}")
    with pytest.raises(norm.ScaleLimitError):
        norm.check_scale_limit(ad.AnnData(X), "lognorm", 1000.0)


def test_a_dtype_WIDER_than_float64_does_not_claim_precision_it_cannot_keep():
    """`_expm1_row_totals` writes into a float64 array and `_max_value` returns a Python float, so
    longdouble precision is not preserved end to end. Claiming longdouble's eps would be a
    tolerance the pipeline cannot honour."""
    assert (norm._scale_limit_rtol(np.zeros(1, dtype=np.longdouble))
            == norm._SCALE_LIMIT_TOL_ULP * float(np.finfo(np.float64).eps))


def test_scale_limit_counts_branch_gets_NO_tolerance():
    """Deliberately asymmetric. A counts row is a sum of raw stored values with no round trip in
    it, and for non-negative integers below the cap every partial sum is exact -- so a tolerance
    there would only weaken the gate. One count over must still reject."""
    a = _sparse_adata([[1_000_001.0, 0.0]])
    with pytest.raises(norm.ScaleLimitError, match="max_counts_per_cell") as exc:
        norm.check_scale_limit(a, "counts", 1_000_000.0)
    assert "ULP" not in str(exc.value), "the counts rejection must not quote a lognorm budget"
    assert "rtol" not in str(exc.value)
    # the precomputed-max shortcut takes the same branch and must behave the same way
    with pytest.raises(norm.ScaleLimitError):
        norm.check_scale_limit(_sparse_adata([[1.0, 0.0]]), "counts", 1_000_000.0,
                               precomputed_row_total_max=1_000_001.0)


# --- the rejection messages have to name the tolerance they applied (Copilot review, PR #302) ----


def test_a_MARGINAL_rejection_reports_an_excess_on_rtol_s_own_scale():
    """The point of quoting the excess. A value just past the tolerated bound must read as barely
    over -- an excess on `rtol`'s order of magnitude, next to `rtol` itself -- so an operator can
    see it is a boundary case and not an arithmetic error.

    Measured against the CAP the sentence names, not against the tolerated bound: against the bound
    a marginal rejection prints ~0, which contradicts its own "exceeds <cap> by" prose (codex)."""
    cap_ln = float(np.log1p(1_000_000.0))
    rtol = norm._scale_limit_rtol(np.zeros(1, dtype=np.float32))
    # ONE float32 step past the tolerated bound -- the smallest rejectable value there is. At
    # 3*rtol past it the two formulas both land inside a loose band and the test cannot tell them
    # apart; here the against-the-bound excess collapses to ~7e-08, far below rtol.
    bound = np.nextafter(np.float32(cap_ln * (1.0 + rtol)), np.float32(np.inf))
    with pytest.raises(norm.ScaleLimitError) as exc:
        norm.check_scale_limit(ad.AnnData(np.array([[bound, 0.0]], dtype=np.float32)),
                               "lognorm", 1_000_000.0)
    msg = str(exc.value)
    assert f"rtol={rtol:.3g}" in msg, "the applied tolerance must be named, not just the cap"
    excess = float(msg.split(" by ")[1].split(" relative")[0])
    assert rtol <= excess < 10.0 * rtol, (
        f"a marginal rejection must read as marginal and on rtol's scale; got {excess:.3g} "
        f"against rtol {rtol:.3g} -- an excess far BELOW rtol means it is being measured against "
        f"the tolerated bound rather than the cap the message names")


def test_the_per_value_guard_is_LOOSER_in_counts_terms_and_the_row_total_still_binds():
    """The per-value guard applies `rtol` in LOG space and the row-total guard applies it on the
    COUNTS scale, so translated to counts the per-value slack is `cap`x looser -- 2.635e-05 against
    1.907e-06 at the v2 cap, a ratio of log1p(1e6) = 13.816. Read as a units bug in review
    (PR #302); it is not, because the two budget different errors and the ROW TOTAL binds on
    magnitude. A single gene stored just under the per-value bound must be rejected anyway."""
    CAP = 1_000_000.0
    cap_ln = float(np.log1p(CAP))
    rtol = norm._scale_limit_rtol(np.zeros(1, dtype=np.float32))
    per_value_slack = float(np.expm1(cap_ln * (1.0 + rtol))) / CAP - 1.0
    assert per_value_slack / rtol == pytest.approx(cap_ln, rel=1e-3), (
        "the per-value slack in counts terms should be cap x the row-total slack")

    # ... and that extra slack cannot leak: the row-total guard sees the breach and rejects it.
    just_inside = np.nextafter(np.float32(cap_ln * (1.0 + rtol)), np.float32(-np.inf))
    with pytest.raises(norm.ScaleLimitError, match="per-cell total") as exc:
        norm.check_scale_limit(ad.AnnData(np.array([[just_inside, 0.0]], dtype=np.float32)),
                               "lognorm", CAP)
    assert "lognorm value" not in str(exc.value), (
        "this must be the ROW-TOTAL rejection -- the point is that the per-value guard let it by")


def test_a_rejection_does_not_emit_a_numpy_warning_on_its_way_out():
    """The excess is a diagnostic, so it must not be able to make a RAISE noisy. At a degenerate
    `max_counts_per_cell` (subnormal, so `log1p` underflows to ~0) the ratio overflows; `errstate`
    keeps that from reaching the caller as a RuntimeWarning alongside the exception. `inf relative`
    is then the honest thing to print."""
    a = _sparse_adata([[1.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(norm.ScaleLimitError, match="inf relative"):
            norm.check_scale_limit(a, "lognorm", 5e-324)
        with pytest.raises(norm.ScaleLimitError) as exc:
            norm.check_scale_limit(a, "counts", 5e-324)
    # ... and the message must REPORT that degenerate cap rather than rounding it to 0.0. `.1f`
    # printed `max_counts_per_cell=0.0`, hiding the input in the one message written to make
    # pathological input diagnosable (Copilot review, raised twice).
    assert "max_counts_per_cell=4.94065646e-324" in str(exc.value)


# --- NaN must not pass a `>` gate by comparing False (Copilot review of PR #302) -----------------


@pytest.mark.parametrize("input_type", ["counts", "lognorm"])
def test_a_NaN_matrix_is_REJECTED_rather_than_passing_every_bound(input_type):
    """Every bound in `check_scale_limit` is a `>`, and NaN answers False to all of them -- so
    before this check a NaN matrix cleared the gate without being tested. MEASURED: both branches
    passed. `inf` never had the problem, because `inf > budget` is True."""
    a = _sparse_adata([[np.nan, 1.0], [2.0, 3.0]])
    with pytest.raises(norm.ScaleLimitError, match="NaN"):
        norm.check_scale_limit(a, input_type, 1_000_000.0)


@pytest.mark.parametrize("input_type", ["counts", "lognorm"])
def test_POSITIVE_inf_was_already_rejected_by_the_comparison_itself(input_type):
    """The other half of the Copilot finding, which measurement did not support: `+inf` compares
    True against any finite bound, so it raised on both branches before the NaN check existed.
    Asserting the BOUND's own message (not merely the absence of "NaN") is what makes this a
    discriminator -- a guard that lumped every non-finite value together would still say "NaN"
    is absent while having taken over a rejection the comparison was already making."""
    a = _sparse_adata([[np.inf, 1.0], [2.0, 3.0]])
    with pytest.raises(norm.ScaleLimitError, match="exceeds"):
        norm.check_scale_limit(a, input_type, 1_000_000.0)


@pytest.mark.parametrize("input_type", ["counts", "lognorm"])
def test_NEGATIVE_inf_still_passes_here_and_that_is_deliberate(input_type):
    """`-inf` evades both bounds -- the counts row total is `-inf` so the MAX ignores it, and
    `expm1(-inf)` is a finite `-1`. Not handled here on purpose: a sign rule does not belong in a
    magnitude budget, and `validate_input_type` already rejects negatives in every mode -- of the
    MODE, not of every matrix: a NaN poisons that sign check, see
    `test_a_NaN_POISONS_the_sign_check_so_negative_dust_survives_it`. Pinned
    both ways so the division of labour is a decision rather than an oversight."""
    a = _sparse_adata([[-np.inf, 1.0], [2.0, 3.0]])
    norm.check_scale_limit(a, input_type, 1_000_000.0)          # must NOT raise
    for allow_fractional in (False, True):
        with pytest.raises(ValueError, match="negative"):
            norm.validate_input_type(a, input_type, allow_fractional=allow_fractional)


def test_a_NaN_precomputed_row_total_is_rejected_but_the_MATRIX_is_not_inspected():
    """Exactly what the shortcut buys, and what it does not. Passing a NaN max rejects; but the
    shortcut skips `_row_totals` entirely, so a NaN MATRIX behind a finite max still passes -- the
    gate trusts its producer there. Recorded rather than fixed: carrying a non-finite flag
    alongside the max would change `run._check_scale_limit_once`'s signature, and no current
    producer can reach it (the GPU accumulator propagates NaN into its max)."""
    with pytest.raises(norm.ScaleLimitError, match="NaN"):
        norm.check_scale_limit(_sparse_adata([[1.0, 0.0]]), "counts", 1_000_000.0,
                               precomputed_row_total_max=float("nan"))
    norm.check_scale_limit(_sparse_adata([[np.nan, 1.0]]), "counts", 1_000_000.0,
                           precomputed_row_total_max=5.0)       # trusted producer: no raise


def test_the_integrality_gate_rejects_NaN_counts_only_when_fractional_is_DISALLOWED():
    """Scope of the fix, at function level. `run.full` calls `_validate_input_once` before
    `_check_scale_limit_once` (`run.py:503-506`), so the ordinary `compute_metrics` path already
    rejected a NaN counts matrix one gate earlier -- `_is_all_integer` is False on NaN. This pins
    the FUNCTION behaviour that makes that true, and the two cases where it does not: fractional
    allowed, and `lognorm` (where the integrality test runs in the opposite direction -- it
    rejects an all-integer matrix, and NaN reads as "not all-integer", the verdict lognorm wants).
    """
    Xn = csr_matrix(np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32))
    assert norm._is_all_integer(Xn) is False
    with pytest.raises(ValueError):
        norm.validate_input_type(ad.AnnData(Xn), "counts", allow_fractional=False)
    norm.validate_input_type(ad.AnnData(Xn), "counts", allow_fractional=True)   # no raise
    norm.validate_input_type(ad.AnnData(Xn), "lognorm")                         # no raise
    # the opposite direction, which is what "lognorm tests integrality" actually means
    with pytest.raises(ValueError, match="all-integer"):
        norm.validate_input_type(_sparse_adata([[1.0, 2.0], [3.0, 4.0]]), "lognorm")
