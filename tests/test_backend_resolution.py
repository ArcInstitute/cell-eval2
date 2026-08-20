"""The tiered de.backend='auto' policy (2026-07-25 ultrareview, Tier 1 change 1).

'auto' used to walk gpudge -> pdex -> scanpy and, because scanpy is a HARD dependency, always
succeeded -- so a GPU host without gpudge silently produced scanpy DE numbers. Now: gpudge when
available; RAISE when a CUDA device is present but gpudge is not; warn-and-fall-back only on a
host with no GPU, where there is no better backend to pick.
"""
import logging
import sys

import pytest

from cell_eval2 import de_compute


@pytest.fixture(autouse=True)
def _clear_warn_guard():
    de_compute._reset_auto_backend_warnings()
    yield
    de_compute._reset_auto_backend_warnings()


def _patch(monkeypatch, *, avail, cuda):
    monkeypatch.setattr(de_compute, "_available", lambda b: avail[b])
    monkeypatch.setattr(de_compute, "_cuda_device_present", lambda: cuda)


def test_auto_returns_gpudge_silently_when_available(monkeypatch, caplog):
    _patch(monkeypatch, avail={"gpudge": True, "pdex": True, "scanpy": True}, cuda=True)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de_compute"):
        assert de_compute._resolve_backend("auto") == "gpudge"
    assert caplog.records == [], "the happy path must not warn"


def test_auto_raises_on_a_gpu_host_without_gpudge(monkeypatch):
    _patch(monkeypatch, avail={"gpudge": False, "pdex": True, "scanpy": True}, cuda=True)
    with pytest.raises(RuntimeError, match="CUDA device but the gpudge DE backend is unavailable"):
        de_compute._resolve_backend("auto")


def test_auto_warns_and_uses_pdex_without_a_gpu(monkeypatch, caplog):
    _patch(monkeypatch, avail={"gpudge": False, "pdex": True, "scanpy": True}, cuda=False)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de_compute"):
        assert de_compute._resolve_backend("auto") == "pdex"
    assert len(caplog.records) == 1
    assert "using pdex" in caplog.records[0].message


def test_auto_warns_about_scanpy_slowness_when_pdex_is_missing(monkeypatch, caplog):
    _patch(monkeypatch, avail={"gpudge": False, "pdex": False, "scanpy": True}, cuda=False)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de_compute"):
        assert de_compute._resolve_backend("auto") == "scanpy"
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "pdex is not installed" in msg and "slower" in msg


def test_auto_warning_is_emitted_once_per_process(monkeypatch, caplog):
    _patch(monkeypatch, avail={"gpudge": False, "pdex": True, "scanpy": True}, cuda=False)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de_compute"):
        for _ in range(3):
            assert de_compute._resolve_backend("auto") == "pdex"
    assert len(caplog.records) == 1, "_resolve_backend is called many times per run"


def test_the_once_guard_does_not_suppress_a_different_backend_warning(monkeypatch, caplog):
    avail = {"gpudge": False, "pdex": True, "scanpy": True}
    monkeypatch.setattr(de_compute, "_available", lambda b: avail[b])
    monkeypatch.setattr(de_compute, "_cuda_device_present", lambda: False)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de_compute"):
        assert de_compute._resolve_backend("auto") == "pdex"
        avail["pdex"] = False
        assert de_compute._resolve_backend("auto") == "scanpy"
    assert len(caplog.records) == 2, "the guard is per-backend, not a global mute"


def test_auto_still_raises_when_nothing_is_available(monkeypatch):
    _patch(monkeypatch, avail={"gpudge": False, "pdex": False, "scanpy": False}, cuda=False)
    with pytest.raises(RuntimeError, match="no DE backend available"):
        de_compute._resolve_backend("auto")


def test_explicit_backends_are_unaffected_by_the_new_tier(monkeypatch):
    """Only 'auto' changed: an explicit backend still resolves (or raises) as before."""
    _patch(monkeypatch, avail={"gpudge": False, "pdex": True, "scanpy": True}, cuda=True)
    assert de_compute._resolve_backend("pdex") == "pdex"
    with pytest.raises(RuntimeError, match="not available"):
        de_compute._resolve_backend("gpudge")


# --- _cuda_device_present itself. The policy tests above monkeypatch it, so without these its
# --- true branches are never exercised anywhere [codex F8].

def test_cuda_device_present_true_when_cupy_reports_cuda(monkeypatch):
    import cell_eval2.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "resolve_device", lambda _d: "cuda")
    assert de_compute._cuda_device_present() is True


def test_cuda_device_present_falls_back_to_torch_when_cupy_says_cpu(monkeypatch):
    import types

    import cell_eval2.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "resolve_device", lambda _d: "cpu")
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert de_compute._cuda_device_present() is True


def test_cuda_device_present_false_when_both_probes_say_no(monkeypatch):
    import types

    import cell_eval2.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "resolve_device", lambda _d: "cpu")
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert de_compute._cuda_device_present() is False


def test_cuda_device_present_degrades_to_false_and_never_raises(monkeypatch):
    """Detection gates a HARD error, so it must never be the thing that fails a run."""
    import cell_eval2.gpu as gpu_mod

    def _boom(_device):
        raise RuntimeError("cupy runtime exploded")

    monkeypatch.setattr(gpu_mod, "resolve_device", _boom)
    # A None entry in sys.modules makes `import torch` raise ModuleNotFoundError (an
    # ImportError subclass): "import of torch halted; None in sys.modules". It does NOT
    # bind torch to None -- verified on this interpreter.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert de_compute._cuda_device_present() is False


# --- The policy error must not be swallowed before expensive work (Checkpoint-2 codex).
# --- _use_inmem_external_ref used to catch every Exception from _resolve_backend and return
# --- False, which routed a CUDA-host-without-gpudge run into _pred_de_input's CPU concat --
# --- documented as OOM-prone at CCL_2 scale -- so the user could get an OOM instead of the
# --- actionable policy error.

def test_the_gpu_policy_error_reaches_the_caller_from_use_inmem_external_ref(monkeypatch):
    from dataclasses import replace as _replace

    from cell_eval2 import EvalConfig
    from cell_eval2.run import _use_inmem_external_ref

    _patch(monkeypatch, avail={"gpudge": False, "pdex": True, "scanpy": True}, cuda=True)
    cfg = _replace(EvalConfig(), control_source="real")
    with pytest.raises(RuntimeError, match="CUDA device but the gpudge DE backend is unavailable"):
        _use_inmem_external_ref(cfg)


def test_pred_de_input_raises_before_materializing_anything(monkeypatch):
    """The raise must land BEFORE _pred_de_input materializes the real control pool."""
    import cell_eval2.run as run_mod
    from dataclasses import replace as _replace

    from cell_eval2 import EvalConfig

    _patch(monkeypatch, avail={"gpudge": False, "pdex": True, "scanpy": True}, cuda=True)

    def _explode(_source):
        raise AssertionError("_materialize must not run: the policy error should precede it")

    monkeypatch.setattr(run_mod, "_materialize", _explode)
    cfg = _replace(EvalConfig(), control_source="real")
    with pytest.raises(RuntimeError, match="CUDA device but the gpudge DE backend is unavailable"):
        run_mod._pred_de_input(object(), object(), cfg=cfg)


def test_explicit_backends_do_not_trigger_availability_resolution(monkeypatch):
    """An explicit backend must NOT be resolved for a layout decision (Checkpoint-2 codex, r2).

    A warm DE-table cache hit is served by _compute_de_side's store without ever calling
    compute_de, so the backend need not be installed at all -- which is exactly why _cache_backend
    resolves only "auto" too. Resolving an explicit backend here made an uninstalled one fail a
    run that previously completed from cache, and this PR's own _DE_RESULT_SEMANTICS bump makes
    that shape (warm DE table, cold result) more likely by invalidating prior results.
    """
    from dataclasses import replace as _replace

    from cell_eval2 import EvalConfig
    from cell_eval2.run import _use_inmem_external_ref

    # Nothing installed and a CUDA device visible: resolving ANY of these would raise.
    _patch(monkeypatch, avail={"gpudge": False, "pdex": False, "scanpy": False}, cuda=True)
    base = _replace(EvalConfig(), control_source="real")
    for backend, expected in (("pdex", False), ("scanpy", False), ("gpudge", True)):
        cfg = _replace(base, de=_replace(base.de, backend=backend))
        assert _use_inmem_external_ref(cfg) is expected, backend   # must not raise
