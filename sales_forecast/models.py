"""Statistical, random-forest, and neural forecasting adapters."""

from __future__ import annotations

import pandas as pd
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from .config import (
    DEFAULT_WINDOWS,
    DYNAMIC_FEATURES,
    FORECAST_HORIZON,
    FREQUENCY,
    SEASON_LENGTH,
    STATIC_FEATURES,
)


def _model_frame(history: pd.DataFrame) -> pd.DataFrame:
    return history[["unique_id", "ds", "y", *DYNAMIC_FEATURES, *STATIC_FEATURES]].copy()


def statistical_backtest(
    history: pd.DataFrame,
    n_windows: int = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    models = [
        SeasonalNaive(season_length=SEASON_LENGTH, alias="SeasonalNaive"),
        AutoETS(season_length=SEASON_LENGTH, model="ZZZ", alias="AutoETS"),
    ]
    engine = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
    return engine.cross_validation(
        df=history[["unique_id", "ds", "y"]],
        h=FORECAST_HORIZON,
        n_windows=n_windows,
        step_size=FORECAST_HORIZON,
        refit=True,
    )


def statistical_forecast(history: pd.DataFrame) -> pd.DataFrame:
    models = [
        SeasonalNaive(season_length=SEASON_LENGTH, alias="SeasonalNaive"),
        AutoETS(season_length=SEASON_LENGTH, model="ZZZ", alias="AutoETS"),
    ]
    engine = StatsForecast(models=models, freq=FREQUENCY, n_jobs=-1)
    return engine.forecast(df=history[["unique_id", "ds", "y"]], h=FORECAST_HORIZON)


def random_forest() -> MLForecast:
    estimator = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        max_features=0.8,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    return MLForecast(
        models={"RandomForest": estimator},
        freq=FREQUENCY,
        lags=[1, 2, 3, 4, 8, 13, 26, 52],
        date_features=["week", "month"],
    )


def random_forest_backtest(
    history: pd.DataFrame,
    n_windows: int = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    return random_forest().cross_validation(
        df=_model_frame(history),
        n_windows=n_windows,
        h=FORECAST_HORIZON,
        step_size=FORECAST_HORIZON,
        refit=True,
        static_features=STATIC_FEATURES,
    )


def random_forest_forecast(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    engine = random_forest()
    engine.fit(_model_frame(history), static_features=STATIC_FEATURES)
    return engine.predict(
        h=FORECAST_HORIZON,
        X_df=future[["unique_id", "ds", *DYNAMIC_FEATURES]],
    )


def neural_backtest(
    history: pd.DataFrame,
    n_windows: int = DEFAULT_WINDOWS,
    max_steps: int = 100,
) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import LSTM

    model = LSTM(
        h=FORECAST_HORIZON,
        input_size=2 * SEASON_LENGTH,
        encoder_hidden_size=128,
        encoder_n_layers=2,
        decoder_hidden_size=128,
        decoder_layers=2,
        max_steps=max_steps,
        val_check_steps=min(max_steps, max(10, max_steps // 2)),
        scaler_type="robust",
        loss=MAE(),
        random_seed=42,
        alias="LSTM",
        start_padding_enabled=True,
        enable_progress_bar=False,
        logger=False,
    )
    engine = NeuralForecast(models=[model], freq=FREQUENCY)
    return engine.cross_validation(
        df=history[["unique_id", "ds", "y"]],
        n_windows=n_windows,
        step_size=FORECAST_HORIZON,
        refit=True,
        verbose=0,
    )


def neural_forecast(history: pd.DataFrame, max_steps: int = 100) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import LSTM

    model = LSTM(
        h=FORECAST_HORIZON,
        input_size=2 * SEASON_LENGTH,
        encoder_hidden_size=128,
        encoder_n_layers=2,
        decoder_hidden_size=128,
        decoder_layers=2,
        max_steps=max_steps,
        val_check_steps=min(max_steps, max(10, max_steps // 2)),
        scaler_type="robust",
        loss=MAE(),
        random_seed=42,
        alias="LSTM",
        start_padding_enabled=True,
        enable_progress_bar=False,
        logger=False,
    )
    engine = NeuralForecast(models=[model], freq=FREQUENCY)
    engine.fit(df=history[["unique_id", "ds", "y"]], verbose=0)
    return engine.predict(verbose=0)
