"""Render a decision brief and chart assets from saved aggregate tables only."""

import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

from .modeling import BUDGET

NAVY, BLUE, TEAL, GRAY = "#142842", "#215CC4", "#117C71", "#646F80"
COLORS = [GRAY, "#E08037", BLUE, TEAL]


def _label(name):
    if name.startswith("logit"):
        return "Logistic regression"
    if name.startswith("boost"):
        return "Gradient boosting"
    return {"random": "Random ranking", "history_rule": "History rule"}[name]


def _page(title, kicker, subtitle):
    fig = plt.figure(figsize=(9, 6), facecolor="#F6F8FC")
    fig.text(.09, .935, kicker, color=BLUE, weight="bold", size=11)
    fig.text(.09, .857, title, color=NAVY, weight="bold", size=23)
    fig.text(.09, .805, subtitle, color=GRAY, size=10)
    fig.text(.09, .035, "NYC INSPECTION RISK  /  RETROSPECTIVE PORTFOLIO STUDY", color=GRAY, size=8)
    return fig


def _table(fig, rows, headers, box):
    ax = fig.add_axes(box, frame_on=False)
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", bbox=[0, 0, 1, 1],
                     colWidths=[.27, .16, .17, .17, .23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE5F0")
        cell.set_facecolor(NAVY if i == 0 else ("white" if i % 2 else "#EAF0F8"))
        if i == 0:
            cell.set_text_props(color="white", weight="bold")
        if j == 0 and i:
            cell.set_text_props(color=NAVY, weight="bold")


def _note(fig, y, text):
    fig.text(.09, y, fill(text, 104), size=11, color=NAVY, va="top", linespacing=1.45)


def render(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    load = lambda name: pd.read_csv(source / f"{name}.csv")  # noqa: E731
    experiment = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    policy = load("holdout_policy")
    metrics = policy.loc[policy.budget.eq(BUDGET)]
    winner = experiment["selected"]
    primary = metrics.set_index("model").loc[winner]
    baseline = metrics.set_index("model").loc["history_rule"]
    delta = int(primary.captured - baseline.captured)
    edition = f"{experiment['holdout'][0]} through {experiment['holdout'][1]} (exclusive)"
    report = _page("Which inspections should be prioritized?", "01  /  DECISION BRIEF", edition)
    report.text(.09, .681, f"{int(primary.captured)} of {int(primary.positives)} closures captured",
                color=TEAL, weight="bold", size=19)
    report.text(.09, .628,
                f"{_label(winner)} selected on validation  |  {int(primary.slots):,} of {int(primary.n):,} visits flagged",
                color=GRAY, size=10)
    _table(report, [[_label(r.model), str(int(r.captured)), f"{r.precision:.1%}", f"{r.recall:.1%}",
                     f"{r.ap:.3f}"] for r in metrics.itertuples()],
           ["Policy", "Captured", "Precision", "Recall", "Avg precision"], [.09, .33, .82, .24])
    bounds = next((r for r in experiment["capture_intervals"] if r["model"] == winner), None)
    change = (f"Difference versus the history rule: {delta:+d} captures. Approximate paired "
              f"95% 4-week-block interval: [{bounds['low']:.0f}, {bounds['high']:.0f}]. " if bounds else "")
    if winner == "history_rule":
        change = "Neither learned model beat the history rule at this capacity. "
    _note(report, .276, change + "This ranks visits that actually occurred, not a real advance schedule. "
          "Validate prospectively before considering use in enforcement.")
    _note(report, .136, f"Outcome: recorded DOHMH closure at a routine initial inspection. "
          f"Monday features assume a {experiment['feature_lag_days']}-day reporting lag.")

    eda = _page("Check what the history can tell us", "02  /  DEVELOPMENT EDA",
                "Before the holdout | coverage and associations, not causal effects")
    trend, rates = load("eda_quarter"), load("eda_closure")
    ax = eda.add_axes([.13, .40, .40, .32], facecolor="white")
    ax.plot(trend.quarter, trend.prevalence, marker="o", color=BLUE, lw=2)
    ax.set_title("Routine closure rate by decision quarter", loc="left", color=NAVY, fontsize=10)
    ax.tick_params(axis="x", rotation=38, labelsize=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(axis="y", alpha=.22)
    ax = eda.add_axes([.64, .40, .25, .32], facecolor="white")
    ax.bar(["None seen", "Prior closure"], rates.set_index("prior_closure").loc[
        ["No prior closure", "Prior closure"], "prevalence"], color=[GRAY, TEAL], width=.55)
    ax.set_title("Closure history vs. outcome", loc="left", color=NAVY, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", alpha=.22)
    depth = load("eda_history").set_index("history").loc["No observed history"]
    historical = rates.set_index("prior_closure")
    _note(eda, .235, f"{int(depth.n):,} development visits have no available inspection history. "
          f"Closure rate is {historical.loc['Prior closure', 'prevalence']:.1%} with "
          f"a prior closure vs. {historical.loc['No prior closure', 'prevalence']:.1%} without one. "
          "No observed history is not proof of a new restaurant: the public extract truncates history.")

    capacity = _page("Capacity changes the answer", "03  /  HOLDOUT TRADE-OFF",
                     "Citywide weekly rank | fraction of visits eligible for priority review")
    ax = capacity.add_axes([.14, .315, .72, .415], facecolor="white")
    for (name, group), color in zip(policy.groupby("model", sort=False), COLORS, strict=True):
        ax.plot(group.budget, group.recall, marker="o", lw=2, label=_label(name), color=color)
    ax.axvline(BUDGET, linestyle="--", color=GRAY, lw=1, alpha=.65)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Nominal weekly capacity")
    ax.set_ylabel("Share of recorded closures captured")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    details = load("holdout_slices").query("model == @winner and dimension == 'history'")
    cold = details.set_index("level").loc["No observed history"]
    _note(capacity, .175, f"At the 20% budget, {int(primary.false_positives):,} flagged visits do not "
          f"end in closure; {int(primary.missed)} closures fall below the cutoff. The no-history group "
          f"accounts for {int(cold.missed)} of those misses ({int(cold.positives)} closures in that group).")

    with PdfPages(output / "findings.pdf", metadata={"Title": "NYC Inspection Risk",
                                                     "CreationDate": None}) as pdf:
        for name, fig in (("decision", report), ("development", eda), ("capacity", capacity)):
            fig.savefig(output / f"{name}.png", dpi=155)
            pdf.savefig(fig)
            plt.close(fig)
