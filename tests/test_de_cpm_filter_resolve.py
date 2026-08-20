import pytest

from cell_eval2.de_compute import _resolve_cpm_filter


@pytest.mark.parametrize("filt,itype,backend,expected", [
    (5.0, "counts", "pdex", 5.0),        # counts: always applies
    (5.0, "counts", "gpudge", 5.0),
    (5.0, "lognorm", "gpudge", 5.0),     # THE FIX: gpudge gate is norm-invariant -> keep on lognorm
    (5.0, "lognorm", "pdex", None),      # unchanged: CPU _apply_cpm_filter is scale-dependent -> skip
    (5.0, "lognorm", "scanpy", None),
    (None, "lognorm", "gpudge", None),   # no filter requested -> None
    (None, "counts", "pdex", None),
])
def test_resolve_cpm_filter(filt, itype, backend, expected):
    assert _resolve_cpm_filter(filt, input_type=itype, resolved_backend=backend) == expected
