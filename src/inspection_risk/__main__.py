"""CLI: acquire/cache a public extract, run a backtest, and render a saved run."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .data import acquire, build_cohort, load_csv
from .diagnostics import development_eda, holdout_diagnostics
from .features import LAG, make_features
from .modeling import backtest


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(input_path, output, lag):
    source = acquire(input_path)
    logging.info("Read source snapshot: %s (%d rows)", source["sha256"], source["rows"])
    history, targets, audit = build_cohort(load_csv(input_path))
    if not len(history) or not len(targets):
        raise ValueError("Extract does not contain usable historical and routine inspections.")
    queue = make_features(history, targets, lag)
    logging.info("Backtesting %d reconstructed routine inspection-days", len(queue))
    validation, holdout, selection = backtest(queue, lag)
    holdout_tables, predictions, intervals = holdout_diagnostics(holdout, selection)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {"validation": validation, **development_eda(queue), **holdout_tables}.items():
        table.to_csv(output / f"{name}.csv", index=False)
    private = output / "local_only"
    private.mkdir(exist_ok=True)
    predictions.to_csv(private / "holdout_predictions.csv", index=False)
    _write(output / "snapshot.json", source)
    _write(output / "data_audit.json", audit)
    _write(output / "experiment.json", selection | {"capture_intervals": intervals})
    logging.info("Validation selected %s. Results written to %s", selection["selected"], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_csv = Path("data/raw/inspections.csv")
    fetch = sub.add_parser("fetch", help="Download or verify the cached extract")
    fetch.add_argument("--input", type=Path, default=default_csv)
    fetch.add_argument("--refresh", action="store_true")
    bench = sub.add_parser("run", help="Reconstruct data and run the temporal benchmark")
    bench.add_argument("--input", type=Path, default=default_csv)
    bench.add_argument("--output", type=Path, default=Path("runs/reference"))
    bench.add_argument("--lag", type=int, default=LAG, help="Assumed event publication lag, days")
    report = sub.add_parser("report", help="Build figures and PDF from aggregate result files")
    report.add_argument("--input", type=Path, default=Path("runs/reference"))
    report.add_argument("--output", type=Path, default=Path("runs/reference"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "fetch":
            print(json.dumps(acquire(args.input, refresh=args.refresh), indent=2))
        elif args.command == "run":
            run(args.input, args.output, args.lag)
        else:
            from .report import render

            render(args.input, args.output)
    except (ValueError, OSError) as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
