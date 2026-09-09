"""End-to-end backtesting, model selection, and forecasting."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import DEFAULT_STORES, DEFAULT_WINDOWS, FORECAST_HORIZON, ProjectPaths
from .data import load_weekly_data, write_quality_report
from .metrics import evaluate_models, select_models_by_store
from .models import (
    neural_backtest,
    neural_forecast,
    random_forest_backtest,
    random_forest_forecast,
    statistical_backtest,
    statistical_forecast,
)

SUPPORTED_MODELS = {"statistical", "rf", "neural"}


def _merge_backtests(frames: list[pd.DataFrame]) -> pd.DataFrame:
    keys = ["unique_id", "ds", "cutoff", "y"]
    result = frames[0]
    for frame in frames[1:]:
        model_columns = [column for column in frame.columns if column not in keys]
        result = result.merge(
            frame[[*keys, *model_columns]], on=keys, how="inner", validate="one_to_one"
        )
    return result.sort_values(["unique_id", "cutoff", "ds"]).reset_index(drop=True)


def _merge_forecasts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    keys = ["unique_id", "ds"]
    result = frames[0]
    for frame in frames[1:]:
        model_columns = [column for column in frame.columns if column not in keys]
        result = result.merge(
            frame[[*keys, *model_columns]], on=keys, how="inner", validate="one_to_one"
        )
    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _selected_forecast(forecasts: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    choices = selection.set_index("unique_id")["model"].to_dict()
    rows: list[dict[str, object]] = []
    for row in forecasts.itertuples(index=False):
        model = choices[row.unique_id]
        value = max(0.0, float(getattr(row, model)))
        rows.append(
            {
                "store_id": row.unique_id,
                "week_start": row.ds,
                "forecast_sales": value,
                "selected_model": model,
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(
    paths: ProjectPaths,
    models: Iterable[str] = ("statistical", "rf", "neural"),
    store_count: int = DEFAULT_STORES,
    n_windows: int = DEFAULT_WINDOWS,
    neural_max_steps: int = 100,
) -> dict[str, pd.DataFrame]:
    requested = list(dict.fromkeys(models))
    unknown = set(requested) - SUPPORTED_MODELS
    if unknown:
        raise ValueError(f"Unsupported model groups: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one model group is required")

    history, future, quality = load_weekly_data(paths, store_count=store_count)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    write_quality_report(quality, paths.outputs / "data_quality.json")

    backtests: list[pd.DataFrame] = []
    forecasts: list[pd.DataFrame] = []
    if "statistical" in requested:
        backtests.append(statistical_backtest(history, n_windows=n_windows))
        forecasts.append(statistical_forecast(history))
    if "rf" in requested:
        backtests.append(random_forest_backtest(history, n_windows=n_windows))
        forecasts.append(random_forest_forecast(history, future))
    if "neural" in requested:
        backtests.append(neural_backtest(history, n_windows=n_windows, max_steps=neural_max_steps))
        forecasts.append(neural_forecast(history, max_steps=neural_max_steps))

    backtest = _merge_backtests(backtests)
    model_metrics = evaluate_models(backtest)
    selection = select_models_by_store(backtest)
    all_forecasts = _merge_forecasts(forecasts)
    selected = _selected_forecast(all_forecasts, selection)
    summary = (
        selected.assign(horizon=selected.groupby("store_id").cumcount() + 1)
        .groupby("store_id", as_index=False)
        .agg(
            selected_model=("selected_model", "first"),
            weeks_1_3_sales=(
                "forecast_sales",
                lambda values: float(values.iloc[:3].sum()),
            ),
            weeks_1_8_sales=("forecast_sales", "sum"),
        )
    )
    if not selected.groupby("store_id").size().eq(FORECAST_HORIZON).all():
        raise RuntimeError("Final output does not contain eight forecasts per store")

    outputs = {
        "backtest_predictions": backtest,
        "model_metrics": model_metrics,
        "store_model_selection": selection,
        "all_model_forecasts": all_forecasts,
        "sales_forecast": selected,
        "forecast_summary": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(paths.outputs / f"{name}.csv", index=False)
    return outputs
