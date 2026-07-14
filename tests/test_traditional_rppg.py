from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.baselines import traditional_rppg
from src.product import adult_hr_mvp


def test_ica_convergence_failure_is_not_relabelled_as_an_independent_route(monkeypatch: pytest.MonkeyPatch) -> None:
    def nonconverged_fit_transform(self, values):  # noqa: ANN001, ANN202
        warnings.warn("did not converge", ConvergenceWarning)
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(traditional_rppg.FastICA, "fit_transform", nonconverged_fit_transform)
    rgb = np.tile(np.asarray([[90.0, 110.0, 80.0]]), (64, 1))

    with pytest.raises(RuntimeError, match="converged route"):
        traditional_rppg.ica(rgb)


def test_candidate_route_failure_is_retained_in_runtime_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_route(rgb):  # noqa: ANN001, ANN202
        raise RuntimeError("route did not converge")

    monkeypatch.setattr(adult_hr_mvp, "METHODS", {"ICA": failed_route})
    window = pd.DataFrame(
        {
            "region": ["forehead"] * 64,
            "frame_index": list(range(64)),
            "mean_r": np.linspace(90.0, 100.0, 64),
            "mean_g": np.linspace(100.0, 110.0, 64),
            "mean_b": np.linspace(80.0, 90.0, 64),
        }
    )
    failures = []

    rows = adult_hr_mvp._candidate_rows_for_window(
        window,
        sample_id="fixture_w000",
        fps=8.0,
        window_id=0,
        start_sec=0.0,
        end_sec=8.0,
        route_failures=failures,
    )

    assert rows == []
    assert failures == [
        {
            "window_id": 0,
            "region": "forehead",
            "method": "ICA",
            "error_type": "RuntimeError",
            "error_message": "route did not converge",
        }
    ]
