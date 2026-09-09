"""Shared forecasting configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FORECAST_HORIZON = 8
SHORT_HORIZON = 3
SEASON_LENGTH = 52
DEFAULT_WINDOWS = 4
DEFAULT_STORES = 100
FREQUENCY = "W-MON"

DYNAMIC_FEATURES = [
    "open_days",
    "promo_days",
    "state_holiday_days",
    "school_holiday_days",
]
STATIC_FEATURES = [
    "competition_distance",
    "store_type_a",
    "store_type_b",
    "store_type_c",
    "store_type_d",
    "assortment_a",
    "assortment_b",
    "assortment_c",
]


@dataclass(frozen=True)
class ProjectPaths:
    """Locations used by the pipeline."""

    root: Path

    @property
    def raw_data(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def sales(self) -> Path:
        return self.raw_data / "sales_data.csv"

    @property
    def future(self) -> Path:
        return self.raw_data / "future_values.csv"

    @property
    def metadata(self) -> Path:
        return self.raw_data / "metadata.csv"
