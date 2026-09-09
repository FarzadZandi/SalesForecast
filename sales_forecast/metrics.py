"""Business-oriented metrics and model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FORECAST_HORIZON, SHORT_HORIZON

KEY_COLUMNS = {"unique_id", "ds", "cutoff", "y", "horizon"}


def add_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["ds"] = pd.to_datetime(result["ds"])
    result["cutoff"] = pd.to_datetime(result["cutoff"])
    result["horizon"] = ((result["ds"] - result["cutoff"]).dt.days // 7).astype(int)
    return result


def _scores(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    error = predicted_values - actual_values
    denominator = np.abs(actual_values).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "bias": float(error.sum() / denominator) if denominator else np.nan,
    }


def evaluate_models(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate weekly, short-horizon, and eight-week-total accuracy."""

    frame = add_horizon(predictions) if "horizon" not in predictions else predictions.copy()
    model_columns = [column for column in frame.columns if column not in KEY_COLUMNS]
    rows: list[dict[str, object]] = []
    scopes = {
        "weeks_1_3": frame[frame["horizon"].between(1, SHORT_HORIZON)],
        "weeks_1_8": frame[frame["horizon"].between(1, FORECAST_HORIZON)],
    }
    for scope, subset in scopes.items():
        for model in model_columns:
            valid = subset[["y", model]].dropna()
            rows.append({"scope": scope, "model": model, **_scores(valid["y"], valid[model])})

    totals = (
        frame[frame["horizon"].between(1, FORECAST_HORIZON)]
        .groupby(["unique_id", "cutoff"], as_index=False)[["y", *model_columns]]
        .sum()
    )
    for model in model_columns:
        valid = totals[["y", model]].dropna()
        rows.append(
            {"scope": "eight_week_total", "model": model, **_scores(valid["y"], valid[model])}
        )
    return pd.DataFrame(rows).sort_values(["scope", "wape", "rmse"]).reset_index(drop=True)


def select_models_by_store(predictions: pd.DataFrame) -> pd.DataFrame:
    """Balance COO short-horizon WAPE and CFO eight-week-total percentage error."""

    frame = add_horizon(predictions) if "horizon" not in predictions else predictions.copy()
    model_columns = [column for column in frame.columns if column not in KEY_COLUMNS]
    rows: list[dict[str, object]] = []
    for store_id, store in frame.groupby("unique_id", sort=False):
        short = store[store["horizon"].between(1, SHORT_HORIZON)]
        full = store[store["horizon"].between(1, FORECAST_HORIZON)]
        actual_short_denominator = short["y"].abs().sum()
        actual_totals = full.groupby("cutoff")["y"].sum()
        for model in model_columns:
            short_wape = (
                (short[model] - short["y"]).abs().sum() / actual_short_denominator
                if actual_short_denominator
                else np.nan
            )
            predicted_totals = full.groupby("cutoff")[model].sum()
            total_ape = ((predicted_totals - actual_totals).abs() / actual_totals.abs()).replace(
                [np.inf, -np.inf], np.nan
            )
            total_mape = float(total_ape.mean())
            score = 0.6 * float(short_wape) + 0.4 * total_mape
            rows.append(
                {
                    "unique_id": store_id,
                    "model": model,
                    "short_wape": float(short_wape),
                    "eight_week_total_mape": total_mape,
                    "selection_score": score,
                }
            )
    scores = pd.DataFrame(rows)
    selected = scores.loc[scores.groupby("unique_id")["selection_score"].idxmin()].copy()
    return selected.sort_values("unique_id").reset_index(drop=True)
