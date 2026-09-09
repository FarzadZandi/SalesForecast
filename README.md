# Sales Forecast

An eight-week weekly sales forecasting pipeline for a multi-store European retailer. The project compares statistical benchmarks, a global random forest, and a global LSTM, then selects the best validated model for each store.

## Business objective

The forecast serves two planning needs:

- Operations: accurate weekly forecasts for weeks 1-3.
- Finance: an accurate sales total across weeks 1-8.

The default run forecasts the first 100 stores in numeric order (`store_1` through `store_100`). Each store-level model score assigns 60% weight to short-horizon WAPE and 40% to the mean absolute percentage error of the eight-week total.

## What changed from the original submission

The original work was a valuable experiment log, but it was not safe to run as a production forecast. The cleaned implementation fixes the following issues:

- Selects 100 stores instead of stopping at `store_99`.
- Aggregates complete Monday-to-Sunday weeks and verifies that the future horizon starts immediately after history.
- Excludes `customers`, whose future values are entirely missing. Historical customer counts are not allowed to leak into a forecast that cannot receive them at prediction time.
- Keeps lag and validation calculations within each store series.
- Uses non-overlapping, expanding-window backtests with the same eight-week horizon as production.
- Reports weeks 1-3, weeks 1-8, and the eight-week total as separate business metrics.
- Writes deterministic forecast and validation files instead of ending with notebook-only objects.
- Keeps private data, course documents, the original presentation, old notebooks, and training logs outside Git history.

## Models

`SeasonalNaive` repeats the observation from 52 weeks earlier and provides a transparent annual benchmark. `AutoETS` represents a maintainable statistical model for weekly level, trend, and seasonality.

`RandomForest` is a global machine-learning model trained across stores. It uses sales lags at 1, 2, 3, 4, 8, 13, 26, and 52 weeks, calendar features, known future opening and promotion schedules, holiday indicators, and static store attributes. The model never uses future customer counts.

`LSTM` is a global neural benchmark with a two-year input window. It is intentionally univariate because the supplied experiment does not establish reliable future customer values. Neural training is reproducible but computationally more expensive, so a quick run can omit it.

## Validated benchmark

The cleaned pipeline was run on all 100 stores with four non-overlapping eight-week validation windows and 100 LSTM training steps per fit. The values below are aggregate WAPE across the validation predictions.

| Model | Weeks 1-3 | Weeks 1-8 | Eight-week totals |
| --- | ---: | ---: | ---: |
| Random Forest | **7.88%** | **7.34%** | **4.11%** |
| Seasonal Naive | 9.06% | 10.46% | 5.83% |
| LSTM | 14.83% | 15.52% | 4.24% |
| AutoETS | 17.51% | 16.31% | 4.33% |

Random Forest gives the strongest portfolio-level result for both business objectives. Store-level validation still matters: the final selector chose Random Forest for 55 stores, Seasonal Naive for 42, AutoETS for two, and LSTM for one. This keeps a simple annual baseline when it is more reliable for a specific store and limits the neural model to the single series where its combined score justified the cost.

## Project layout

```text
sales_forecast/                  Forecasting package and CLI
tests/                           Data and metric regression tests
notebooks/sales_forecast_analysis.ipynb
                                 Small reviewable companion notebook
data/raw/                        Private input CSV files, ignored by Git
outputs/                         Generated forecasts and diagnostics, ignored by Git
archive/                         Original notebooks and documents, ignored by Git
```

## Setup

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[neural,dev]"
```

The core statistical and machine-learning workflow can be installed without PyTorch:

```powershell
python -m pip install -e ".[dev]"
```

For the exact direct package versions used in the validated Windows/Python 3.12 run, install `requirements-tested.txt`. PyTorch wheels are platform-specific, so use the official PyTorch installer if the pinned wheel is unavailable for your operating system or accelerator.

Copy the three private inputs into `data/raw/` as described in [data/README.md](data/README.md).

## Usage

Audit the source files and weekly alignment:

```powershell
python -m sales_forecast audit
```

Run the full benchmark and forecast:

```powershell
python -m sales_forecast run
```

Run a faster CPU-only check without the LSTM:

```powershell
python -m sales_forecast run --models statistical rf --stores 5 --windows 1
```

Reduce neural training only for a smoke test:

```powershell
python -m sales_forecast run --stores 5 --windows 1 --neural-max-steps 2
```

## Outputs

The pipeline writes the following private files to `outputs/`:

- `data_quality.json`: source coverage, dates, missing values, and feature policy.
- `backtest_predictions.csv`: rolling-origin forecasts and actuals.
- `model_metrics.csv`: MAE, RMSE, WAPE, and bias for each business horizon.
- `store_model_selection.csv`: model scores and the selected model per store.
- `all_model_forecasts.csv`: forecasts from every fitted candidate.
- `sales_forecast.csv`: the selected weekly forecast for each store.
- `forecast_summary.csv`: weeks 1-3 and weeks 1-8 totals for planning.

WAPE is calculated as `sum(abs(forecast - actual)) / sum(abs(actual))`. Bias is signed error divided by total absolute actual sales. The eight-week-total metric first sums each validation window, then evaluates the aggregate. This prevents weekly volatility from being confused with the CFO's cash-flow planning target.

## Validation and maintenance

Run the automated checks before committing changes:

```powershell
python -m pytest
ruff check .
```

Retrain when a new complete sales week arrives. Re-run the full four-window benchmark monthly or after material changes to promotions, store operations, or the feature schema. Before extending beyond the first 100 stores, run the data audit for all stores and compare accuracy by store type and sales volume. Do not promote a neural model merely because it is more complex. It must improve the same rolling business metrics used by the final selector.

## Technical references

- [MLForecast end-to-end workflow](https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/end_to_end_walkthrough.html)
- [StatsForecast cross-validation API](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html)
- [NeuralForecast cross-validation](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html)
- [NeuralForecast exogenous-variable guidance](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)

## Privacy

This repository does not publish the retailer-provided CSV files, assignment PDFs, presentation, personal identifiers, old notebooks, or generated forecasts. They remain available locally under ignored directories. Check `git status` and `git ls-files` before every push.
