"""Independently reconcile the 20% holdout policy using SQLite window functions."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

# Model names are bound as values. The long-form table lets SQL rank all policies at once.
QUERY = """
WITH ranked AS (
    SELECT model, week, closed,
           ROW_NUMBER() OVER (PARTITION BY model, week ORDER BY score DESC, tie ASC) AS rank,
           COUNT(*) OVER (PARTITION BY model, week) AS weekly_n
    FROM scores
), flagged AS (
    SELECT *, rank <= (weekly_n + 4) / 5 AS selected
    FROM ranked
)
SELECT model, COUNT(*) AS n, SUM(closed) AS positives, SUM(selected) AS slots,
       SUM(closed * selected) AS captured
FROM flagged
GROUP BY model
ORDER BY model;
"""


def reconcile(directory):
    policy = pd.read_csv(directory / "holdout_policy.csv").query("budget == .20")
    columns = ["model", "n", "positives", "slots", "captured"]
    expected = policy[columns].sort_values("model").reset_index(drop=True)
    predictions = pd.read_csv(directory / "local_only/holdout_predictions.csv", dtype={"tie": str})
    # Zero-pad unsigned 64-bit ties so SQLite TEXT ordering matches Python unsigned ordering.
    predictions["tie"] = predictions.tie.str.zfill(20)
    scores = predictions.melt(id_vars=["week", "closed", "tie"], value_vars=expected.model,
                              var_name="model", value_name="score")
    with closing(sqlite3.connect(":memory:")) as db:
        scores.to_sql("scores", db, index=False)
        actual = pd.read_sql_query(QUERY, db)
    pd.testing.assert_frame_equal(actual, expected)
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("runs/reference"))
    args = parser.parse_args()
    actual = reconcile(args.directory)
    print(json.dumps(actual.to_dict(orient="records"), indent=2))
