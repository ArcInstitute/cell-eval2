"""Row-count-aware decode-thread policy for the cell-layout gather path (#149).

cellstream's ``CellStore`` gather methods (``gather_rows`` / ``gather_rows_adata`` /
``read_group`` / ``read_reference``) take an ``n_threads`` that parallelises the Rust
``decode_cells`` path across the selected rows. It defaults to ``1``, so every cell-layout
read cell_eval2 issued before this module decoded single-threaded, on any node, no matter how
many cores were free.

Two traps this module exists to avoid, both measured (#149):

1. **``-1`` must never be passed through.** cellstream clamps with
   ``nw = max(1, min(int(n_threads), int(row_ids.size)))``, so ``-1`` resolves to **1** -- a
   silent no-op that looks correctly wired and passes every test. For the same reason this
   module RAISES on ``0``/``-2``/``True`` rather than coercing them: a coercion would just
   manufacture another silent-serial path.
2. **``os.cpu_count()`` is the wrong source.** On the cgroup-limited eval nodes it reports 208
   against a cgroup allowance of 12-16 (here: 192 vs 144 affinity). A cpu_count-derived default
   under an 8-way run would spawn 208 threads per job across 8 concurrent jobs.

And one policy: the thread count must scale with the number of rows being decoded. Measured
speedup vs ``n_threads=1`` on a real 1.3 M-cell archive (each point bracketed by an
``n_threads=1`` baseline re-measured immediately before and after, since the node drifts;
every thread count returned a bit-identical matrix)::

    read pattern                rows / nnz         n=4        n=8         n=12
    batch (208 groups)      136,315 / 693 M       4.15x      7.94x     9.9-10.2x
    reference pool          217,760 / 1.107 G   4.19-4.68x  7.19-8.01x 10.1-11.3x
    single group (median)       727 / 3.7 M     2.65-2.77x  3.67-3.81x   3.47x  <- REGRESSES

Small per-group reads saturate at 8 and regress at 12; large batch/reference reads keep
scaling. cellstream's own clamp to the row count is NOT sufficient -- it would still hand a
727-row read the full cap (144 threads on this node).

SCOPE OF THE EVIDENCE, so the next reader does not over-trust these constants: the table pins
exactly ONE point for the ramp (727 rows -> 8 threads), and it stops at 12 threads. The auto
cap being the full affinity allowance is therefore an extrapolation beyond the measured range.
"""
from __future__ import annotations

import os

# Rows of decode work per thread. A MAINTAINER-SET HEURISTIC anchored on the single measured
# point above: it is chosen so the 727-row median group resolves to ceil(727/96) = 8, its
# measured saturation point. Any divisor in 91..103 would satisfy that constraint equally well
# -- 96 is not an empirically-identified optimum, and nothing else in the data distinguishes it.
ROWS_PER_THREAD = 96

# Row count assumed when the caller cannot know it (an archive whose group records are
# unavailable). 8 * ROWS_PER_THREAD -> 8 threads: the measured small-read saturation point, and
# never worse than the flat-8 default #149 originally proposed.
_UNKNOWN_ROWS = 8 * ROWS_PER_THREAD


def cpu_allowance() -> int:
    """CPUs this process may actually run on: the scheduling-affinity mask, NOT ``cpu_count()``.

    Under cgroup / SLURM / container CPU limits ``os.cpu_count()`` reports the machine's core
    count while the process is confined to a fraction of them (#149). ``sched_getaffinity`` is
    Linux-only, so fall back to ``cpu_count()`` elsewhere. Always >= 1.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return max(1, len(getaffinity(0)))
        except OSError:      # pragma: no cover - affinity query can fail in exotic sandboxes
            pass
    return max(1, os.cpu_count() or 1)


def resolve_gather_threads(n_rows: int | None, gather_threads: int) -> int:
    """Decode threads for a gather of ``n_rows`` rows under an ``EvalConfig.gather_threads``.

    ``gather_threads == -1`` (the default) means "auto": the cap is :func:`cpu_allowance`. Any
    positive int is used as the cap verbatim -- an explicit request is honoured, not silently
    re-capped. The cap is then reduced by the row-count ramp, so a small read never spawns more
    threads than it has work for::

        threads = max(1, min(cap, ceil(n_rows / ROWS_PER_THREAD)))

    ``n_rows=None`` means "not knowable" and is treated as ``_UNKNOWN_ROWS`` (-> 8 threads, or
    the cap if lower). The return value is ALWAYS >= 1 -- never ``-1`` or ``0``, which cellstream
    would silently turn back into a serial decode (see the module docstring).

    Raises:
        ValueError: if ``gather_threads`` is not ``-1`` or a positive non-bool ``int``.
            Deliberately strict -- ``EvalConfig`` validates the config-driven path, but this is
            also a direct API, and coercing a bad value would create a new silent-serial path.
    """
    if isinstance(gather_threads, bool) or not isinstance(gather_threads, int):
        raise ValueError(
            f"gather_threads must be -1 (auto) or a positive int, got {gather_threads!r}"
        )
    if gather_threads == 0 or gather_threads < -1:
        raise ValueError(
            f"gather_threads must be -1 (auto) or a positive int, got {gather_threads!r}"
        )
    cap = cpu_allowance() if gather_threads == -1 else gather_threads
    rows = _UNKNOWN_ROWS if n_rows is None else int(n_rows)
    want = -(-max(rows, 0) // ROWS_PER_THREAD)      # ceil division, negatives floored to 0
    return max(1, min(cap, want))
