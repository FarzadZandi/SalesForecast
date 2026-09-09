"""Command-line interface for Sales Forecast."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_STORES, DEFAULT_WINDOWS, ProjectPaths
from .data import load_weekly_data, write_quality_report
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sales-forecast")
    parser.add_argument(
        "command",
        choices=["audit", "run"],
        help="Audit input data or run backtesting, selection, and forecasting.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--stores", type=int, default=DEFAULT_STORES)
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["statistical", "rf", "neural"],
        default=["statistical", "rf", "neural"],
    )
    parser.add_argument("--neural-max-steps", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    paths = ProjectPaths(args.project_root.resolve())
    if args.command == "audit":
        _, _, report = load_weekly_data(paths, store_count=args.stores)
        paths.outputs.mkdir(parents=True, exist_ok=True)
        destination = paths.outputs / "data_quality.json"
        write_quality_report(report, destination)
        print(f"Data audit passed. Report: {destination}")
        return

    outputs = run_pipeline(
        paths,
        models=args.models,
        store_count=args.stores,
        n_windows=args.windows,
        neural_max_steps=args.neural_max_steps,
    )
    print(f"Forecast complete: {len(outputs['sales_forecast'])} store-week rows")
    print(outputs["model_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
