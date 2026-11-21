import numpy as np
import pandas as pd

from inspection_risk.evaluation import choose, fit_rows, rank_weekly, summarize


def test_budget_is_week_by_week_and_ties_are_independent_of_labels_and_row_order():
    q = pd.DataFrame({"week": ["w1"] * 4 + ["w2"] * 2, "tie": [4, 3, 2, 1, 6, 5],
                      "closed": [1, 0, 1, 0, 0, 1], "score": [1, 1, 1, 1, 0, 1]})
    rank = rank_weekly(q, "score", 0.25)
    assert rank["selected"].sum() == 2
    assert rank.loc[rank.selected, "tie"].tolist() == [1, 5]
    shuffled = q.sample(frac=1, random_state=11).assign(closed=0)
    assert rank_weekly(shuffled, "score", 0.25).loc[lambda f: f.selected, "tie"].tolist() == [1, 5]
    report = summarize(rank, "score")
    assert report["captured"] == 1
    assert report["precision"] == 0.5
    np.testing.assert_allclose(report["recall"], 1 / 3)


def test_training_label_must_already_be_available():
    q = pd.DataFrame({"date": pd.to_datetime(["2025-04-06", "2025-04-07", "2025-04-08",
                                             "2025-04-21"]),
                      "week": pd.to_datetime(["2025-03-31", "2025-04-07", "2025-04-07",
                                              "2025-04-21"])})
    fit = fit_rows(q, "2025-04-21")
    assert fit.index.tolist() == [0, 1]


def test_model_selection_uses_pooled_captures_then_average_precision():
    runs = pd.DataFrame({"model": ["logit", "logit", "tree", "tree"],
                         "captured": [15, 15, 28, 1], "ap": [.1, .1, .2, .5]})
    assert choose(runs) == "logit"
    runs.loc[3, "captured"] = 2
    assert choose(runs) == "tree"
