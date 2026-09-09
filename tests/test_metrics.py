import pandas as pd

from sales_forecast.metrics import evaluate_models, select_models_by_store


def _predictions() -> pd.DataFrame:
    rows = []
    for store in ["store_1", "store_2"]:
        cutoff = pd.Timestamp("2024-01-01")
        for horizon in range(1, 9):
            actual = 100.0
            rows.append(
                {
                    "unique_id": store,
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(weeks=horizon),
                    "y": actual,
                    "ShortModel": actual if horizon <= 3 else 80.0,
                    "TotalModel": 110.0 if horizon <= 3 else 94.0,
                }
            )
    return pd.DataFrame(rows)


def test_metrics_keep_business_horizons_separate() -> None:
    metrics = evaluate_models(_predictions())
    short = metrics.query("scope == 'weeks_1_3' and model == 'ShortModel'").iloc[0]
    full = metrics.query("scope == 'weeks_1_8' and model == 'ShortModel'").iloc[0]
    assert short["wape"] == 0.0
    assert full["wape"] > short["wape"]


def test_selection_uses_short_and_total_objectives() -> None:
    selected = select_models_by_store(_predictions())
    assert set(selected["model"]) <= {"ShortModel", "TotalModel"}
    assert selected["selection_score"].notna().all()
