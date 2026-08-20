import pytest

from cell_eval2.gpu import HAS_CUPY, resolve_device, xp_for


def test_resolve_device_cpu_always():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_without_cupy(monkeypatch):
    import cell_eval2.gpu as g

    monkeypatch.setattr(g, "HAS_CUPY", False)
    assert g.resolve_device("auto") == "cpu"


def test_resolve_device_cuda_requires_cupy(monkeypatch):
    import cell_eval2.gpu as g

    monkeypatch.setattr(g, "HAS_CUPY", False)
    with pytest.raises(RuntimeError, match="cupy"):
        g.resolve_device("cuda")


def test_resolve_device_invalid_raises():
    with pytest.raises(ValueError, match="auto|cuda|cpu"):
        resolve_device("gpu")


def test_xp_for_cpu_is_numpy():
    import numpy as np

    assert xp_for("cpu") is np


def test_has_cupy_is_bool():
    assert isinstance(HAS_CUPY, bool)


@pytest.mark.skipif(resolve_device("auto") != "cuda", reason="no usable CUDA GPU")
def test_resolve_device_cuda_with_cupy():
    # On a node with cupy AND a visible GPU, "cuda" resolves to "cuda" and xp_for returns cupy.
    # Gate on a usable GPU via resolve_device (not just HAS_CUPY): explicit "cuda" now requires a
    # visible device (F10.2), so a cupy-installed-but-no-device node must SKIP, not assert "cuda".
    import cupy as cp

    assert resolve_device("cuda") == "cuda"
    assert xp_for("cuda") is cp


@pytest.mark.skipif(resolve_device("auto") != "cuda", reason="no usable CUDA GPU")
def test_resolve_device_auto_with_cupy_and_gpu():
    # cupy present + at least one visible GPU -> auto picks cuda. Gate on a usable GPU via
    # resolve_device (not HAS_CUPY): a cupy-installed-but-no-device node then SKIPS cleanly
    # instead of raising cudaErrorNoDevice from cp.cuda.runtime.getDeviceCount().
    assert resolve_device("auto") == "cuda"


def test_resolve_device_cuda_no_visible_device_raises(monkeypatch):
    # F10.2: explicit device="cuda" with cupy importable but NO visible GPU must raise a clear
    # RuntimeError (mirroring the 'auto' branch's getDeviceCount check), not return "cuda" and later
    # crash with an opaque cudaErrorNoDevice deep in a stage. Inject a fake cupy so this runs the
    # same with or without a real cupy install.
    import sys
    import types

    import cell_eval2.gpu as g

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 0))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(g, "HAS_CUPY", True)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        g.resolve_device("cuda")
