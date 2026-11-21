"""Weekly capacity policy, temporal fit boundary, and summary metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def rank_weekly(queue: pd.DataFrame, score: str, budget: float) -> pd.DataFrame:
    if not 0 < budget <= 1:
        raise ValueError("budget must lie in (0, 1]")
    if not np.isfinite(queue[score]).all():
        raise ValueError(f"Scores from {score} must be finite.")
    ranked = queue.sort_values(["week", score, "tie"], ascending=[True, False, True]).copy()
    groups = ranked.groupby("week", sort=False)
    ranked["selected"] = groups.cumcount() < np.ceil(budget * groups[score].transform("size"))
    return ranked


def summarize(ranked: pd.DataFrame, score: str) -> dict:
    labels = ranked.closed
    total, positives, slots = len(labels), int(labels.sum()), int(ranked.selected.sum())
    found = int(ranked.loc[ranked.selected, "closed"].sum())
    prevalence = positives / total if total else None
    precision = found / slots if slots else None
    both = labels.nunique() == 2
    return {
        "n": total, "positives": positives, "slots": slots, "captured": found,
        "false_positives": slots - found, "missed": positives - found, "prevalence": prevalence,
        "precision": precision, "recall": found / positives if positives else None,
        "lift": precision / prevalence if prevalence and precision is not None else None,
        "ap": float(average_precision_score(labels, ranked[score])) if both else None,
        "roc_auc": float(roc_auc_score(labels, ranked[score])) if both else None,
    }


def fit_rows(frame: pd.DataFrame, as_of: str, lag: int = 14) -> pd.DataFrame:
    boundary = pd.Timestamp(as_of)
    return frame.loc[frame.week.lt(boundary) & frame.date.le(boundary - pd.Timedelta(days=lag))]


def choose(metrics: pd.DataFrame) -> str:
    """Same folds and weekly denominator: maximize total captures, then mean fold AP."""
    scores = metrics.groupby("model", as_index=False).agg(captured=("captured", "sum"),
                                                         ap=("ap", "mean"))
    return str(scores.sort_values(["captured", "ap", "model"], ascending=[False, False, True])
               .iloc[0].model)
