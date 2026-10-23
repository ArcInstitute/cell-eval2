"""A numpy stand-in for cupy whose host-to-device copies cannot pin (#404).

CPU CI has no cupy, so it cannot raise the real ``CUDARuntimeError``. `CupyLike` is numpy in
every attribute except the ones a host-to-device copy touches: ``asarray`` raises the way cupy's
pinned staging does on a fragmented host, ``empty`` returns arrays that carry cupy's
``ndarray.set``, and the current stream is a `FakeStream`. The arithmetic is therefore the CPU
path's, bit for bit; only the route a host array takes to the "device" differs. Use it with a
fixture that points ``cell_eval2.gpu._pin_failure_type`` at `FakeCudaError`.
"""
import types
import weakref

import numpy as np

INVALID_VALUE = 1  # cudaErrorInvalidValue


class FakeCudaError(RuntimeError):
    """Stands in for ``cupy_backends.cuda.api.runtime.CUDARuntimeError``."""

    def __init__(self, status):
        super().__init__(f"fake CUDA error {status}")
        self.status = status


class FakeStream:
    """A stream whose copies read their host source only at ``synchronize()``.

    That is the latest an async pageable copy may read it: CUDA documents one only as "might be
    synchronous" with the host. The source is held weakly, as the driver does not keep it alive
    either, so a source freed before the wait fails the copy instead of passing by luck.
    """

    def __init__(self):
        self.syncs = 0
        self._pending = []

    def copy(self, dst, src):
        self._pending.append((dst, weakref.ref(src)))

    def synchronize(self):
        self.syncs += 1
        pending, self._pending = self._pending, []
        for dst, ref in pending:
            src = ref()
            if src is None:
                raise AssertionError("a host source was freed before the stream finished its copy")
            dst[...] = src


class DeviceArray(np.ndarray):
    """A numpy array with cupy's ``ndarray.set(arr, stream=None)``."""

    def set(self, arr, stream=None):
        # As cupy's `set` (core.pyx:2163): same dtype and shape, a contiguous copy of `arr` in this
        # array's layout, then an async copy on `stream`. Without a stream this fake copies at
        # once; cupy would use the current stream (`_to_device` always passes one).
        if self.dtype != arr.dtype:
            raise TypeError(f"{arr.dtype} array cannot be set to {self.dtype} array")
        if self.shape != arr.shape:
            raise ValueError(f"shape mismatch: {self.shape} vs {arr.shape}")
        src = np.ascontiguousarray(arr) if self.flags.c_contiguous else np.asfortranarray(arr)
        if stream is None:
            self[...] = src
        else:
            # A new view of the caller's buffer (what a 0-d input gets) lives as long as that
            # buffer, so track the caller's array; only a real copy is the temporary to watch.
            stream.copy(self, arr if np.may_share_memory(src, arr) else src)


class _BrokenDeviceArray(DeviceArray):
    def set(self, arr, stream=None):
        raise FakeCudaError(INVALID_VALUE)


class CupyLike:
    """numpy, except that ``asarray`` fails to pin, ``empty`` returns `DeviceArray`, and the
    current stream is a `FakeStream`."""

    def __init__(self, status=INVALID_VALUE, *, broken_set=False):
        self.status = status
        self.asarray_calls = 0
        self._array_type = _BrokenDeviceArray if broken_set else DeviceArray
        self.stream = FakeStream()
        self.cuda = types.SimpleNamespace(get_current_stream=lambda: self.stream)

    @property
    def syncs(self):
        return self.stream.syncs

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, a, *args, **kwargs):
        self.asarray_calls += 1
        raise FakeCudaError(self.status)

    def empty(self, shape, dtype=float, order="C"):
        return np.empty(shape, dtype=dtype, order=order).view(self._array_type)
