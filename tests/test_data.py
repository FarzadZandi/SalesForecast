from pathlib import Path

import pandas as pd

from sales_forecast.config import ProjectPaths
from sales_forecast.data import load_weekly_data


def _write_fixture(root: Path) -> ProjectPaths:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    history_dates = pd.date_range("2022-01-03", periods=56 * 7, freq="D")
    future_dates = pd.date_range(
        history_dates.max() + pd.Timedelta(days=1), periods=8 * 7, freq="D"
    )
    sales_rows = []
    future_rows = []
    for store in ["store_1", "store_2"]:
        for index, date in enumerate(history_dates):
            sales_rows.append(
                {
                    "store_id": store,
                    "date": date,
                    "sales": 100 + index % 7,
                    "customers": 10,
                    "open": 1,
                    "promo": index % 2,
                    "state_holiday": "0",
                    "school_holiday": 0,
                }
            )
        for date in future_dates:
            future_rows.append(
                {
                    "store_id": store,
                    "date": date,
                    "customers": None,
                    "open": 1,
                    "promo": 0,
                    "state_holiday": 0,
                    "school_holiday": 0,
                }
            )
    pd.DataFrame(sales_rows).to_csv(raw / "sales_data.csv", index=False)
    pd.DataFrame(future_rows).to_csv(raw / "future_values.csv", index=False)
    pd.DataFrame(
        {
            "store_id": ["store_1", "store_2"],
            "store_type": ["a", "b"],
            "assortment": ["a", "c"],
            "competition_distance": [100.0, None],
        }
    ).to_csv(raw / "metadata.csv", index=False)
    return ProjectPaths(root)


def test_load_weekly_data_aligns_monday_weeks(tmp_path: Path) -> None:
    history, future, report = load_weekly_data(_write_fixture(tmp_path), store_count=2)

    assert history.groupby("unique_id").size().eq(56).all()
    assert future.groupby("unique_id").size().eq(8).all()
    assert history["ds"].dt.weekday.eq(0).all()
    assert future["ds"].min() == history["ds"].max() + pd.Timedelta(weeks=1)
    assert history["y"].iloc[0] == sum(100 + index % 7 for index in range(7))
    assert report["future_customers_excluded"] is True
    assert history["competition_distance"].isna().sum() == 0
