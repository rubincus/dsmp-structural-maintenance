"""Regression tests for deterministic probability recalibration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from dsmp.validation_metrics import apply_isotonic_regression, fit_isotonic_regression


def test_isotonic_fit_is_invariant_to_the_order_of_tied_predictions():
    pred = np.asarray([0.10, 0.10, 0.25, 0.25, 0.25, 0.70, 0.70, 0.90])
    observed = np.asarray([0, 1, 1, 0, 1, 0, 1, 1])
    permutation = np.asarray([7, 2, 5, 0, 6, 3, 1, 4])

    x1, y1 = fit_isotonic_regression(pred, observed)
    x2, y2 = fit_isotonic_regression(pred[permutation], observed[permutation])

    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)
    assert np.all(np.diff(y1) >= 0.0)
    assert np.allclose(
        apply_isotonic_regression(pred, x1, y1),
        apply_isotonic_regression(pred, x2, y2),
    )
