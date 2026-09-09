"""Data loading, validation, and weekly aggregation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import (
    DYNAMIC_FEATURES,
    FORECAST_HORIZON,
    STATIC_FEATURES,
    ProjectPaths,
)

SALES_COLUMNS = {
    "store_id",
    "date",
    "sales",
    "customers",
    "open",
    "promo",
    "state_holiday",
    "school_holiday",
}
FUTURE_COLUMNS = SALES_COLUMNS - {"sales"}
METADATA_COLUMNS = {"store_id", "store_type", "assortment", "competition_distance"}


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _store_number(value: str) -> int:
    match = re.fullmatch(r"store_(\d+)", str(value))
    if match is None:
        raise ValueError(f"Invalid store identifier: {value!r}")
    return int(match.group(1))


def _selected_store_ids(frame: pd.DataFrame, count: int) -> list[str]:
    ids = sorted(frame["store_id"].astype(str).unique(), key=_store_number)
    if len(ids) < count:
        raise ValueError(f"Requested {count} stores, but only {len(ids)} are available")
    return ids[:count]


def _holiday_indicator(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    return (~normalized.isin({"0", "0.0", "none", "nan", ""})).astype("int8")


def _week_start(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def _prepare_metadata(metadata: pd.DataFrame, store_ids: list[str]) -> pd.DataFrame:
    _require_columns(metadata, METADATA_COLUMNS, "metadata.csv")
    metadata = metadata[metadata["store_id"].isin(store_ids)].copy()
    if metadata["store_id"].duplicated().any():
        raise ValueError("metadata.csv contains duplicate store identifiers")
    if set(store_ids) != set(metadata["store_id"]):
        missing = sorted(set(store_ids) - set(metadata["store_id"]), key=_store_number)
        raise ValueError(f"Metadata is missing selected stores: {missing}")

    distance = pd.to_numeric(metadata["competition_distance"], errors="coerce")
    metadata["competition_distance"] = distance.fillna(distance.median()).astype(float)
    for category in "abcd":
        metadata[f"store_type_{category}"] = (
            metadata["store_type"].astype(str).str.lower() == category
        ).astype("int8")
    for category in "abc":
        metadata[f"assortment_{category}"] = (
            metadata["assortment"].astype(str).str.lower() == category
        ).astype("int8")
    return metadata[["store_id", *STATIC_FEATURES]].rename(columns={"store_id": "unique_id"})


def _aggregate_weekly(frame: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame["ds"] = _week_start(frame["date"])
    frame["state_holiday"] = _holiday_indicator(frame["state_holiday"])
    binary = ["open", "promo", "state_holiday", "school_holiday"]
    for column in binary:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    aggregations: dict[str, str] = {
        "open": "sum",
        "promo": "sum",
        "state_holiday": "sum",
        "school_holiday": "sum",
    }
    if include_target:
        frame["sales"] = pd.to_numeric(frame["sales"], errors="raise")
        aggregations["sales"] = "sum"

    weekly = (
        frame.groupby(["store_id", "ds"], as_index=False)
        .agg(aggregations)
        .rename(
            columns={
                "store_id": "unique_id",
                "sales": "y",
                "open": "open_days",
                "promo": "promo_days",
                "state_holiday": "state_holiday_days",
                "school_holiday": "school_holiday_days",
            }
        )
    )
    return weekly.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def load_weekly_data(
    paths: ProjectPaths,
    store_count: int = 100,
    horizon: int = FORECAST_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load selected stores and return aligned historical and future weekly frames."""

    sales = pd.read_csv(paths.sales, low_memory=False)
    future = pd.read_csv(paths.future, low_memory=False)
    metadata = pd.read_csv(paths.metadata)
    _require_columns(sales, SALES_COLUMNS, "sales_data.csv")
    _require_columns(future, FUTURE_COLUMNS, "future_values.csv")

    store_ids = _selected_store_ids(sales, store_count)
    sales = sales[sales["store_id"].isin(store_ids)].copy()
    future = future[future["store_id"].isin(store_ids)].copy()

    duplicate_sales = int(sales.duplicated(["store_id", "date"]).sum())
    duplicate_future = int(future.duplicated(["store_id", "date"]).sum())
    if duplicate_sales or duplicate_future:
        raise ValueError(
            f"Duplicate store-date rows: sales={duplicate_sales}, future={duplicate_future}"
        )
    if sales["sales"].isna().any() or (pd.to_numeric(sales["sales"]) < 0).any():
        raise ValueError("Historical sales must be present and non-negative")
    if future["open"].isna().any():
        missing = future.loc[future["open"].isna(), "store_id"].unique().tolist()
        raise ValueError(f"Future open status is missing for selected stores: {missing}")

    history = _aggregate_weekly(sales, include_target=True)
    future_weekly = _aggregate_weekly(future, include_target=False)
    static = _prepare_metadata(metadata, store_ids)
    history = history.merge(static, on="unique_id", how="left", validate="many_to_one")
    future_weekly = future_weekly.merge(static, on="unique_id", how="left", validate="many_to_one")

    last_history = history["ds"].max()
    expected_dates = pd.date_range(
        last_history + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON"
    )
    future_weekly = future_weekly[future_weekly["ds"].isin(expected_dates)].copy()
    counts = future_weekly.groupby("unique_id").size()
    if len(counts) != store_count or not counts.eq(horizon).all():
        raise ValueError(
            "Future data must contain every selected store for all eight forecast weeks"
        )
    if set(future_weekly["ds"].unique()) != set(expected_dates):
        raise ValueError("Future weekly dates do not immediately follow the historical data")

    history_counts = history.groupby("unique_id").size()
    report: dict[str, object] = {
        "selected_stores": store_count,
        "store_range": [store_ids[0], store_ids[-1]],
        "historical_daily_rows": int(len(sales)),
        "future_daily_rows": int(len(future)),
        "historical_weeks_per_store": {
            "min": int(history_counts.min()),
            "max": int(history_counts.max()),
        },
        "history_start": history["ds"].min().date().isoformat(),
        "history_end": last_history.date().isoformat(),
        "forecast_start": expected_dates.min().date().isoformat(),
        "forecast_end": expected_dates.max().date().isoformat(),
        "history_duplicate_store_dates": duplicate_sales,
        "future_duplicate_store_dates": duplicate_future,
        "future_customers_missing": int(future["customers"].isna().sum()),
        "future_customers_excluded": True,
        "history_target_missing": int(history["y"].isna().sum()),
        "history_target_negative": int((history["y"] < 0).sum()),
        "dynamic_features": DYNAMIC_FEATURES,
        "static_features": STATIC_FEATURES,
    }
    return history, future_weekly, report


def write_quality_report(report: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
