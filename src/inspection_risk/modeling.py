"""Compare transparent rules, regularized logistic regression and gradient boosting."""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluation import choose, fit_rows, rank_weekly, summarize
from .features import FEATURES, LAG

FIT_START = "2024-07-01"
FOLDS = (("2025-04-07", "2025-07-07"), ("2025-07-07", "2025-10-06"),
         ("2025-10-06", "2026-01-05"))
HOLDOUT = ("2026-01-05", "2026-06-29")
BUDGET = 0.20
BUDGETS = (0.05, 0.10, BUDGET, 0.30, 0.40, 0.50)
RULES = ("random", "history_rule")


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict

    def build(self):
        if self.family == "logit":
            return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 StandardScaler(), LogisticRegression(max_iter=500, **self.params))
        return HistGradientBoostingClassifier(max_iter=160, learning_rate=.06, early_stopping=False,
                                             min_samples_leaf=60, random_state=42, **self.params)


GRID = (
    *(Candidate(f"logit_C{c}", "logit", {"C": c}) for c in (0.01, 0.1, 1.0)),
    *(Candidate(f"boost_L{leaves}_R{penalty}", "boost", {
        "max_leaf_nodes": leaves, "l2_regularization": penalty,
    }) for leaves in (7, 15) for penalty in (1.0, 10.0)),
)


def add_rules(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["tie"] = [int.from_bytes(hashlib.blake2b(f"{c}|{d.date()}".encode(), digest_size=8,
                                                 person=b"nyc-tie").digest(), "big")
                    for c, d in zip(frame.camis, frame.date, strict=True)]
    frame["random"] = -frame.tie.astype(float)
    # Prior closure first; higher most-recent score next. Unavailable score ranks last in tier.
    frame["history_rule"] = frame.prior_closures.gt(0).astype(int) * 1000 + frame.last_score.fillna(-1)
    return frame


def _score(spec: Candidate, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    with threadpool_limits(limits=1):
        model = spec.build().fit(train[list(FEATURES)], train.closed)
        return model.predict_proba(test[list(FEATURES)])[:, 1]


def _dates(frame, start, stop):
    return frame.loc[frame.week.ge(start) & frame.week.lt(stop)].copy()


def backtest(frame: pd.DataFrame, lag: int = LAG):
    """Lock a model on development; fit family champions and evaluate once on holdout."""
    frame = add_rules(frame.loc[frame.week.ge(FIT_START)])
    records = []
    # Only select on whole queues whose Sunday outcomes could be known at lock time.
    last_available = pd.Timestamp(HOLDOUT[0]) - pd.Timedelta(days=lag)
    mature_end = (last_available + pd.Timedelta(days=1)).to_period("W-SUN").start_time
    for start, scheduled_end in FOLDS:
        end = min(pd.Timestamp(scheduled_end), mature_end).date().isoformat()
        train = fit_rows(frame, start, lag)
        queue = _dates(frame, start, end)
        if train.closed.nunique() < 2 or queue.closed.nunique() < 2:
            raise ValueError(f"Fold {start} lacks both target classes. Check source coverage.")
        for spec in GRID:
            queue[spec.name] = _score(spec, train, queue)
        for name in (*RULES, *(c.name for c in GRID)):
            ranked = rank_weekly(queue, name, BUDGET)
            records.append(dict(model=name, start=start, end_exclusive=end, train_n=len(train),
                                train_positives=int(train.closed.sum()),
                                train_last_date=str(train.date.max().date()),
                                **summarize(ranked, name)))
    validation = pd.DataFrame(records)
    by_family = {family: choose(validation.loc[validation.model.isin(
        [c.name for c in GRID if c.family == family])]) for family in ("logit", "boost")}
    choice = choose(validation.loc[validation.model.isin(["history_rule", *by_family.values()])])
    train = fit_rows(frame, HOLDOUT[0], lag)
    holdout = _dates(frame, *HOLDOUT)
    if holdout.closed.nunique() < 2:
        raise ValueError("Holdout lacks both target classes. Reuse the archived input snapshot.")
    for spec in GRID:
        if spec.name in by_family.values():
            holdout[spec.name] = _score(spec, train, holdout)
    selection = {"selected": choice, "family_winners": by_family,
                 "train_n": len(train), "train_last_date": str(train.date.max().date()),
                 "feature_lag_days": lag, "budget": BUDGET, "fit_start": FIT_START,
                 "holdout": list(HOLDOUT), "candidates": [vars(c) for c in GRID]}
    return validation, holdout, selection
