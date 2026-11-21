"""Aggregate EDA and holdout policy diagnostics; never export restaurant IDs here."""

import numpy as np
import pandas as pd

from .evaluation import rank_weekly, summarize
from .modeling import BUDGET, BUDGETS, FIT_START, HOLDOUT, RULES


def _counts(table, key):
    result = table.groupby(key, dropna=False, observed=True).agg(
        n=("closed", "size"), positives=("closed", "sum"),
    ).reset_index()
    return result.assign(prevalence=result.positives / result.n)


def development_eda(queue):
    dev = queue.loc[queue.week.ge(FIT_START) & queue.week.lt(HOLDOUT[0])].copy()
    dev["quarter"] = dev.week.dt.to_period("Q").astype(str)
    dev["prior_closure"] = np.where(dev.prior_closures.gt(0), "Prior closure", "No prior closure")
    dev["history"] = np.where(dev.prior_days.gt(0), "Observed history", "No observed history")
    return {"eda_quarter": _counts(dev, "quarter"), "eda_closure": _counts(dev, "prior_closure"),
            "eda_history": _counts(dev, "history")}


def paired_blocks(weekly, model, baseline="history_rule", repeats=2000, block=4):
    """Percentile interval for total capture difference across equally many holdout weeks."""
    wide = weekly.pivot(index="week", columns="model", values="captured").sort_index()
    delta = (wide[model] - wide[baseline]).to_numpy(dtype=int)
    n = len(delta)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n, size=(repeats, (n + block - 1) // block))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(repeats, -1)[:, :n]
    low, high = np.quantile(delta[index].sum(axis=1), [.025, .975])
    return {"versus": baseline, "difference": int(delta.sum()), "low": float(low),
            "high": float(high), "weeks": n, "block_weeks": block, "replicates": repeats}


def holdout_diagnostics(queue, selection):
    models = (*RULES, *selection["family_winners"].values())
    policy, weekly, slices = [], [], []
    saved = queue[["camis", "date", "week", "tie", "closed", "borough", "prior_days", *models]].copy()
    for name in models:
        for budget in BUDGETS:
            ranked = rank_weekly(queue, name, budget)
            policy.append(dict(model=name, budget=budget, **summarize(ranked, name)))
            if budget != BUDGET:
                continue
            saved[f"selected_{name}"] = ranked.selected
            ranked["history"] = np.where(ranked.prior_days.gt(0), "Observed history", "No observed history")
            ranked["captured"] = ranked.closed * ranked.selected
            for week, subset in ranked.groupby("week"):
                weekly.append(dict(week=week, model=name, n=len(subset), positives=int(subset.closed.sum()),
                                   slots=int(subset.selected.sum()), captured=int(subset.captured.sum())))
            for dimension in ("borough", "history"):
                for level, subset in ranked.groupby(dimension):
                    # Use the original citywide selection; do not re-rank per subgroup.
                    slices.append(dict(model=name, dimension=dimension, level=level,
                                       **summarize(subset, name)))
    by_week = pd.DataFrame(weekly)
    tables = {"holdout_policy": pd.DataFrame(policy), "holdout_weekly": by_week,
              "holdout_slices": pd.DataFrame(slices)}
    learned = [name for name in models if name not in RULES]
    return tables, saved, [dict(model=name, **paired_blocks(by_week, name)) for name in learned]
