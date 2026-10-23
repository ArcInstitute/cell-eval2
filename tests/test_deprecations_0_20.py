import warnings

import pytest

from cell_eval2 import run
from cell_eval2.config import DEParams, EvalConfig


@pytest.mark.parametrize("name", ["v1", "legacy", "cell-eval-0.7.6", "cell_eval_0_7_6"])
def test_the_v1_presets_warn(name):
    with pytest.warns(FutureWarning, match="v1 presets will be removed"):
        EvalConfig.from_preset(name)


@pytest.mark.parametrize("name", ["v2", "corrected", "vcc2026", "vcc-2026", "vcc_2026"])
def test_the_other_presets_do_not(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        EvalConfig.from_preset(name)


def test_the_deseq2_backend_warns_once_per_scoring_call():
    with pytest.warns(FutureWarning, match="DESeq2 backend will be removed") as rec:
        run._resolve_config(EvalConfig(de=DEParams(backend="deseq2")), {})
    assert sum(issubclass(w.category, FutureWarning) for w in rec) == 1


def test_other_backends_do_not():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        run._resolve_config(EvalConfig(de=DEParams(backend="pdex")), {})
