import pandas as pd

from inspection_risk.features import FEATURES
from inspection_risk.modeling import FOLDS, HOLDOUT, backtest


def test_end_to_end_temporal_model_comparison_with_synthetic_labels():
    rows = []
    for week in pd.date_range("2024-07-01", HOLDOUT[1], freq="W-MON", inclusive="left"):
        for i in range(10):
            rows.append({**dict.fromkeys(FEATURES, float(i)), "date": week, "week": week,
                         "camis": str(i), "closed": int(i == 9)})
    validation, holdout, selection = backtest(pd.DataFrame(rows))
    assert validation.start.nunique() == len(FOLDS)
    assert holdout.week.min() == pd.Timestamp(HOLDOUT[0])
    assert holdout.week.max() < pd.Timestamp(HOLDOUT[1])
    assert selection["selected"] in ["history_rule", *selection["family_winners"].values()]
    assert validation.groupby("start").slots.nunique().eq(1).all()
    last = validation.loc[validation.start.eq(FOLDS[-1][0])]
    # Monday Dec 22 is at the assumed availability boundary, but its full queue isn't.
    assert last.n.eq(110).all()
    assert last.end_exclusive.eq("2025-12-22").all()
