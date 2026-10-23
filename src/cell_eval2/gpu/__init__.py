"""GPU submodule (extraction-ready). ``cupy`` is optional; its absence -> CPU fallback.

The discrimination / pseudobulk kernels here mirror the CPU reference in
``cell_eval2.distances`` / ``cell_eval2.streaming_bulk`` but run on whichever array
module ``xp_for`` returns (numpy on CPU, cupy on CUDA). ``device`` is resolved once
per call via :func:`resolve_device`; ``"auto"`` picks cuda iff cupy is importable and
a GPU is visible, else cpu — so default behavior is unchanged on a GPU-free node.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as _cp  # noqa: F401

    HAS_CUPY = True
except Exception:
    # Any import failure (no cupy, no CUDA runtime/driver, ABI mismatch) -> CPU only.
    HAS_CUPY = False


def resolve_device(device: str) -> str:
    """Resolve a ``device`` knob (``"auto"|"cuda"|"cpu"``) to ``"cuda"`` or ``"cpu"``.

    ``"cpu"`` is always honored. ``"cuda"`` requires cupy (raises ``RuntimeError`` if
    absent). ``"auto"`` picks ``"cuda"`` iff cupy is importable and at least one GPU is
    visible, otherwise ``"cpu"`` — preserving today's CPU behavior with no cupy.
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not HAS_CUPY:
            raise RuntimeError(
                "device='cuda' requires cupy (install the 'gpu' extra: "
                "pip install 'cell-eval2[gpu]')"
            )
        # Also require a visible GPU (the 'auto' branch does): cupy can import with no driver/
        # device, and returning 'cuda' then crashes with an opaque cudaErrorNoDevice deep in a
        # stage. Fail fast with a clear error instead (F10.2).
        import cupy as cp
        try:
            n_devices = cp.cuda.runtime.getDeviceCount()
        except Exception as e:  # noqa: BLE001 - unusable runtime -> clear message, not a raw CUDA error
            raise RuntimeError(
                f"device='cuda' but the CUDA runtime is unusable (no driver/device): {e}"
            ) from e
        if n_devices <= 0:
            raise RuntimeError("device='cuda' but no CUDA device is visible (getDeviceCount()==0)")
        return "cuda"
    if device == "auto":
        if not HAS_CUPY:
            return "cpu"
        try:
            import cupy as cp

            return "cuda" if cp.cuda.runtime.getDeviceCount() > 0 else "cpu"
        except Exception:
            # cupy imported but the runtime is unusable (no driver/device) -> CPU.
            return "cpu"
    raise ValueError(f"device must be auto|cuda|cpu, got {device!r}")


def xp_for(device: str):
    """Return the array module for ``device``: cupy for ``"cuda"``, numpy otherwise."""
    if device == "cuda":
        import cupy as cp

        return cp
    import numpy as np

    return np


#: Bytes of the first host-to-device copy in this process whose pinned staging failed (#404),
#: or ``None``. Once set, `_to_device` copies every host array through pageable memory.
_pinned_h2d_failed_nbytes = None

#: ``cudaErrorInvalidValue`` in the CUDA runtime's ``cudaError`` enum. cupy 14.1.1 has an
#: ``errorInvalidValue`` in runtime.pyx, but it is a C-level ``cdef`` that Python cannot see.
_CUDA_ERROR_INVALID_VALUE = 1


def _pin_failure_type():
    """``(CUDARuntimeError, cudaErrorInvalidValue)`` from cupy, or ``None`` without cupy.

    A function rather than two constants so the CPU-only tests can substitute a fake error type:
    CI has no cupy to raise the real one (tests/_cupy_like.py).
    """
    try:
        from cupy_backends.cuda.api import runtime
    except Exception:
        return None
    return runtime.CUDARuntimeError, _CUDA_ERROR_INVALID_VALUE


def _to_device(xp, a):
    """``xp.asarray(a)`` for a host array, through pageable memory if cupy cannot pin it (#404).

    ``cupy.asarray`` copies a pageable numpy array through a page-locked staging buffer from
    cupy's pinned pool, which rounds the size up to a power of two. On Linux 6.11, pinning N
    pages takes one physically contiguous N*8-byte kernel allocation (mm/gup.c; 2 MiB, order 9,
    for a 1 GiB buffer). That allocation fails on a host whose free memory is fragmented, even
    with hundreds of GB free, and CUDA reports ``cudaErrorInvalidValue``. cupy falls back to a
    pageable copy only on ``cudaErrorMemoryAllocation``, so the error reaches us. The error is
    not sticky: cupy's ``check_status`` clears it, and nothing was enqueued.

    On that error this copy, and every later one through this function in the process, goes
    through ``ndarray.set`` from pageable memory. The host array ``set`` reads is made here,
    contiguous and little-endian, and held until the stream has finished, so it is never freed
    while the copy may still read it. For an array of one or more dimensions the result has the
    dtype, shape, layout and values ``cupy.asarray`` gives. Logs one warning naming the size
    that failed. Any other error is re-raised.

    The test is on the error, not on the call that raised it: a ``cudaErrorInvalidValue`` from
    anywhere in ``cupy.asarray`` is taken for a failed pin. One from the copy itself (a bad
    argument) recurs in the pageable copy and propagates. One from a call the pageable copy does
    not repeat (cupy's pointer query, or its event calls) would be absorbed by a successful retry.
    """
    global _pinned_h2d_failed_nbytes
    if xp is np:
        return np.asarray(a)
    if _pinned_h2d_failed_nbytes is None:
        try:
            return xp.asarray(a)
        except Exception as e:
            pin = _pin_failure_type()
            if pin is None or not isinstance(e, pin[0]) or e.status != pin[1]:
                raise
            host = np.asarray(a)
            _pinned_h2d_failed_nbytes = host.nbytes
            logger.warning(
                "cupy could not pin a %s-byte host buffer for a host-to-device copy (%s). This "
                "copy and every later pseudobulk copy in this process use pageable memory "
                "instead. The usual cause is fragmented host memory: Linux 6.11 needs one "
                "physically contiguous kernel allocation to pin a buffer (cell_eval2#404).",
                f"{host.nbytes:,}", e,
            )
    else:
        host = np.asarray(a)
    # cupy.asarray's layout rule, read off the input before any conversion: F only for an
    # F-but-not-C array. Make the array `set` will read here, contiguous in that order and
    # little-endian as cupy.asarray makes it, so this frame holds it through the wait below.
    # Left to `set`, a strided array's contiguous copy would be freed when `set` returns.
    order = "F" if host.flags.fnc else "C"
    host = np.asarray(host, dtype=host.dtype.newbyteorder("<"), order=order)
    out = xp.empty(host.shape, dtype=host.dtype, order=order)
    stream = xp.cuda.get_current_stream()
    out.set(host, stream=stream)
    # CUDA documents an async pageable H2D copy only as one that "might be synchronous" with the
    # host, so wait before `host` can be freed: it may be a temporary (`blk.toarray()`, or the
    # copy made above). Measured cost of the latched path, on `update`: ~1%.
    stream.synchronize()
    return out


def _release_gpu_pool() -> None:
    """Return cupy's cached device blocks (and pinned host blocks) to the CUDA driver.

    gpudge's in-memory external-ref DE sizes its gene-chunk to fill most of VRAM and, on
    return, leaves those bytes in cupy's *caching pool* rather than the driver (gpudge_arc#76).
    A subsequent GPU phase in the SAME process that allocates OUTSIDE cupy's pool -- notably
    cell_eval2's discrimination (``cupy.einsum``/``matmul`` -> a cuBLAS handle + workspace via
    ``cudaMalloc``) -- then fails with ``CUBLAS_STATUS_ALLOC_FAILED`` at 5.5M-cell scale, even
    though the bytes are "free" inside the pool. Freeing the pool here hands them back to the
    driver so external allocators can reuse them. No-op when cupy is absent or has no device
    (the CPU DE backends), and never raises. Lives in this lightweight GPU module (not de_compute)
    so low-level callers -- gpu.distances -- need not import the heavy DE stack (Gemini #119).
    """
    try:
        import cupy
    except Exception:
        return
    try:
        import gc

        gc.collect()  # reclaim any unreachable cupy arrays into the pool first, so free_all_blocks
        # actually returns them to the driver (a block is pooled only once its array refcount hits 0)
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
