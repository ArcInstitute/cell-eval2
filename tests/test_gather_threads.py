"""Unit tests for the row-count-aware gather-thread policy (#149).

Deliberately cellstream-free: the resolver is pure arithmetic over (n_rows, gather_threads),
so these run in CI, which installs base+dev only and therefore has no cellstream.
"""
import os

import pytest

from cell_eval2 import _threads
from cell_eval2._threads import ROWS_PER_THREAD, cpu_allowance, resolve_gather_threads


@pytest.fixture
def cap12(monkeypatch):
    """A cgroup-limited eval node (12-16 allowed of 208 reported)."""
    monkeypatch.setattr(_threads, "cpu_allowance", lambda: 12)


@pytest.fixture
def cap144(monkeypatch):
    """A fat node like this one (144 affinity of 192 cpu_count)."""
    monkeypatch.setattr(_threads, "cpu_allowance", lambda: 144)


def test_auto_never_resolves_to_serial(cap12):
    """#149's central trap: cellstream clamps nw = max(1, min(n_threads, n_rows)), so handing
    EvalConfig's -1 straight through silently yields ONE thread. The resolver must never emit
    -1 or 0, and a large read must get real parallelism."""
    assert resolve_gather_threads(136_315, -1) == 12
    assert resolve_gather_threads(217_760, -1) == 12


def test_large_reads_take_the_whole_cap(cap144):
    """Policy assertion (NOT a measured optimum -- the published sweep stops at 12 threads):
    a read with far more rows than ROWS_PER_THREAD * cap is limited by the cap, not the ramp."""
    assert resolve_gather_threads(136_315, -1) == 144
    assert resolve_gather_threads(217_760, -1) == 144


def test_small_group_read_saturates_at_eight(cap144):
    """The ONE point the measurements pin: the median group (727 rows / 3.7 M nnz) peaks at 8
    threads (3.67-3.81x) and REGRESSES at 12 (3.47x). ROWS_PER_THREAD is chosen so 727 lands on
    exactly 8 under any cap. Any divisor in 91..103 would do this; 96 is the maintainer's pick."""
    assert resolve_gather_threads(727, -1) == 8


def test_ramp_is_ceiling_division():
    assert resolve_gather_threads(ROWS_PER_THREAD, 999) == 1
    assert resolve_gather_threads(ROWS_PER_THREAD + 1, 999) == 2
    assert resolve_gather_threads(2 * ROWS_PER_THREAD, 999) == 2


@pytest.mark.parametrize("n_rows", [0, 1, ROWS_PER_THREAD - 1])
def test_tiny_reads_are_serial(n_rows, cap144):
    assert resolve_gather_threads(n_rows, -1) == 1


def test_negative_row_count_is_floored(cap144):
    """Defensive: a bogus negative row count must still produce a legal thread count."""
    assert resolve_gather_threads(-5, -1) == 1


def test_explicit_value_is_a_cap_not_a_passthrough(cap144):
    """An explicitly-set positive gather_threads caps the auto value AND stays row-count aware,
    so a 727-row read never spawns 64 threads just because the user asked for 64."""
    assert resolve_gather_threads(136_315, 4) == 4
    assert resolve_gather_threads(727, 64) == 8
    assert resolve_gather_threads(727, 4) == 4


def test_explicit_value_does_not_consult_affinity(monkeypatch):
    """A positive gather_threads is honoured verbatim as the cap -- cpu_allowance() is not even
    called (silently re-capping a user's explicit request would be wrong)."""
    def boom():
        raise AssertionError("cpu_allowance() must not be called for an explicit value")

    monkeypatch.setattr(_threads, "cpu_allowance", boom)
    assert resolve_gather_threads(136_315, 8) == 8


def test_unknown_row_count_uses_the_small_read_default(cap144):
    """n_rows=None means 'not knowable' (an archive whose group records are unavailable). It
    must land on the measured small-read saturation point, never on the raw cap."""
    assert resolve_gather_threads(None, -1) == 8
    assert resolve_gather_threads(None, 4) == 4


@pytest.mark.parametrize("bad", [0, -2, -1.0, 1.5, True, False, None, "8"])
def test_invalid_gather_threads_raises(bad, cap144):
    """EvalConfig validates its own field, but the resolver is also a direct API. Every invalid
    value must RAISE -- silently coercing 0/-2/True to 1 would manufacture a NEW silent-serial
    path, which is precisely the bug this module exists to prevent."""
    with pytest.raises(ValueError):
        resolve_gather_threads(1000, bad)


def test_cpu_allowance_uses_affinity_not_cpu_count(monkeypatch):
    """#149: os.cpu_count() reports the machine's cores (208 on the eval nodes, 192 here)
    while the cgroup allows far fewer (12-16 / 144). A cpu_count-derived default would spawn 208
    threads per job across 8 concurrent jobs."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 208)
    assert cpu_allowance() == 12


def test_cpu_allowance_falls_back_to_cpu_count_without_affinity(monkeypatch):
    """sched_getaffinity is Linux-only; macOS/Windows must still get a sane positive value."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert cpu_allowance() == 8


def test_cpu_allowance_is_at_least_one(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert cpu_allowance() == 1
