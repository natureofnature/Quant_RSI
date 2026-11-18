"""One fit and one close-to-close target for every research fit (#159 Phase 3).

Research-study attempts, latest refits, walk-forward folds, model and financial
comparisons, and rolling research share fit_estimator and log_close_ratio_labels. Row
selection stays with each caller. study_qlib fits through upstream Qlib, and the frozen
forecast workflow keeps its event-sourced labels and open-to-open targets under
explicit contracts.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.core.model_parameters import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research import study_estimators
from quant_rsi.research.study_estimators import (
    StandardizedRidge,
    estimator_state,
    fit_estimator,
    log_close_ratio_labels,
    predict_estimator,
)

RESEARCH = Path(study_estimators.__file__).parent
SHARED = {"study_estimators.py", "study_qlib.py"}


def direct_fits(path):
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        owner = getattr(func.value, "id", None) if isinstance(func, ast.Attribute) else None
        if (owner == "lgb" and name == "train") or name == "StandardizedRidge":
            found.append(node.lineno)
    return found


@pytest.mark.parametrize(
    "path", sorted(p for p in RESEARCH.glob("*.py") if p.name not in SHARED), ids=lambda p: p.name
)
def test_research_modules_fit_only_through_the_shared_helper(path):
    assert direct_fits(path) == []


def test_labels_are_close_to_close_log_ratios():
    closes = np.array([[100.0], [110.0], [121.0], [133.1]])
    assert np.allclose(log_close_ratio_labels(closes, [0, 1], 2), np.log([1.21, 1.21]))


def test_shared_fit_reproduces_the_fixed_configurations():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(60, 3)), rng.normal(size=60)
    ridge = fit_estimator("ridge", x, y, alpha=2.0)
    assert isinstance(ridge, StandardizedRidge)
    assert np.allclose(predict_estimator(ridge, x[:5]), StandardizedRidge(x, y, 2.0).predict(x[:5]))
    assert estimator_state(ridge) == StandardizedRidge(x, y, 2.0).state()
    import lightgbm as lgb

    booster = fit_estimator("lightgbm", x, y)
    reference = lgb.train(
        MODEL_PARAMETERS | {"seed": 7}, lgb.Dataset(x, label=y), num_boost_round=BOOST_ROUNDS
    )
    assert np.array_equal(predict_estimator(booster, x), reference.predict(x, num_threads=1))
    assert estimator_state(booster) == {"model_string": reference.model_to_string()}
