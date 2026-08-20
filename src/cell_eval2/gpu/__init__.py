"""GPU submodule (extraction-ready). ``cupy`` is optional; its absence -> CPU fallback.

The discrimination / pseudobulk kernels here mirror the CPU reference in
``cell_eval2.distances`` / ``cell_eval2.streaming_bulk`` but run on whichever array
module ``xp_for`` returns (numpy on CPU, cupy on CUDA). ``device`` is resolved once
per call via :func:`resolve_device`; ``"auto"`` picks cuda iff cupy is importable and
a GPU is visible, else cpu — so default behavior is unchanged on a GPU-free node.
"""

from __future__ import annotations

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
