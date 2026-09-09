# Data files

Place the three assignment datasets in `data/raw/`:

- `sales_data.csv`
- `future_values.csv`
- `metadata.csv`

The raw files are intentionally excluded from Git because they contain company-provided data. The pipeline validates their schema, selects the first 100 numerically ordered store IDs, and aggregates Monday-to-Sunday weeks.

`customers` is excluded from the forecasting features because its future values are missing. The model uses only information available when a forecast is created: scheduled opening, promotions, holidays, and static store metadata.

