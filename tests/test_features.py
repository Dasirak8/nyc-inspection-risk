import numpy as np
import pandas as pd
import pandas.testing as pdt

from inspection_risk.features import FEATURES, make_features


def fixture():
    history = pd.DataFrame({
        "camis": ["A", "A", "A", "B"],
        "date": pd.to_datetime(["2024-12-23", "2024-12-24", "2025-01-06", "2024-01-01"]),
        "closed": [1, 0, 0, 0], "score": [42.0, 2.0, 5.0, 8.0],
        "critical_codes": [2, 1, 0, 0],
    })
    targets = pd.DataFrame({
        "camis": ["A", "A", "B", "C"],
        "date": pd.to_datetime(["2025-01-10", "2025-01-17", "2025-01-10", "2025-01-10"]),
        "closed": [0, 0, 0, 1], "borough": ["QUEENS"] * 4,
    })
    return history, targets


def test_history_frozen_monday_includes_exact_lag_boundary():
    history, targets = fixture()
    result = make_features(history, targets)
    a = result.loc[result.camis.eq("A")].reset_index(drop=True)
    assert a.prior_days.tolist() == [1, 2]
    assert a.last_score.tolist() == [42.0, 2.0]
    assert a.prior_closures.tolist() == [1, 1]
    assert a.days_since_last.tolist() == [14, 20]
    assert a.last_critical_codes.tolist() == [2, 1]


def test_window_and_cold_start_have_distinct_meanings():
    history, targets = fixture()
    result = make_features(history, targets).set_index("camis")
    assert result.loc["B", "prior_days"] == 1
    assert result.loc["B", "prior_days_365"] == 0
    assert result.loc["C", "prior_days"] == 0
    assert np.isnan(result.loc["C", "last_score"])


def test_current_labels_metadata_and_unavailable_history_cannot_change_features():
    history, targets = fixture()
    original = make_features(history, targets)
    targets["closed"] = 1 - targets.closed
    targets["borough"] = "BRONX"
    history.loc[history.date.eq("2025-01-06"), ["score", "closed", "critical_codes"]] = [499, 1, 40]
    updated = make_features(history, targets)
    pdt.assert_frame_equal(original[list(FEATURES)], updated[list(FEATURES)])
