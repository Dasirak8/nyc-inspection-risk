"""Causal ordering by event date with an explicit, assumed reporting delay."""

import pandas as pd

LAG = 14
FEATURES = (
    "prior_days", "prior_days_365", "prior_closures", "prior_closures_365",
    "days_since_last", "last_closed", "last_score", "last_critical_codes",
)


def make_features(history: pd.DataFrame, targets: pd.DataFrame, lag: int = LAG) -> pd.DataFrame:
    """For a Monday queue, consume only events dated at most Monday minus lag."""
    if not 1 <= lag <= 365:
        raise ValueError("lag must be between 1 and 365 days")
    if history.duplicated(["camis", "date"]).any() or targets.duplicated(["camis", "date"]).any():
        raise ValueError("History and target inputs must be unique by CAMIS/date.")
    past = history.sort_values(["date", "camis"]).rename(columns={
        "date": "last_date", "closed": "last_closed", "score": "last_score",
        "critical_codes": "last_critical_codes",
    })
    g = past.groupby("camis", sort=False)
    past["prior_days"] = g.cumcount() + 1
    past["prior_closures"] = g.last_closed.cumsum()
    queue = targets.sort_values(["date", "camis"]).reset_index(drop=True).copy()
    queue["week"] = queue.date - pd.to_timedelta(queue.date.dt.weekday, unit="D")
    queue["cutoff"] = queue.week - pd.Timedelta(days=lag)
    prefix = pd.merge_asof(queue, past, left_on="cutoff", right_on="last_date", by="camis")
    prefix["before_window"] = prefix.week - pd.Timedelta(days=366)
    counts = ["prior_days", "prior_closures"]
    old = pd.merge_asof(prefix[["camis", "before_window"]], past[["camis", "last_date", *counts]],
                        left_on="before_window", right_on="last_date", by="camis")
    for col in counts:
        prefix[col] = prefix[col].fillna(0).astype(int)
        prefix[f"{col}_365"] = (prefix[col] - old[col].fillna(0)).astype(int)
    prefix["days_since_last"] = (prefix.week - prefix.last_date).dt.days
    for col in ("last_score", "last_closed", "last_critical_codes"):
        prefix[col] = prefix[col].astype(float)
    return prefix.drop(columns=["before_window"])
