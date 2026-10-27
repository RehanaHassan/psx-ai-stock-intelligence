# PSX AI Stock Intelligence

A small Streamlit research dashboard for experimenting with daily Pakistan Stock Exchange (PSX) OHLCV data. It is free educational software, not financial advice.

## Features

- Accepts a local CSV containing `Date, Open, High, Low, Close, Volume`.
- Validates dates, finite numeric values, positive prices, non-negative volume, and OHLC relationships.
- Displays historical close and volume charts.
- Builds return, volatility, moving-average, volume, and daily-range features.
- Compares a Random Forest classifier with a simple majority-class baseline.
- Uses a chronological train/test split with a gap equal to the forecast horizon to reduce future-data leakage.
- Reports historical holdout metrics and an experimental latest probability estimate.

## Limitations and safety

- Predictions are experimental estimates, never guaranteed profits or investment advice.
- The app does not fetch live market data, use paid APIs, or require API keys.
- Results do not account for transaction costs, slippage, taxes, dividends, corporate actions, or changing market regimes.
- Historical metrics can be unstable, especially with short or unrepresentative datasets.
- Use only datasets you are permitted to use. Do not redistribute PSX market data unless its license permits redistribution.

## Windows setup

Python 3.13 is supported by the current dependency set. From Command Prompt in this project folder:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the dashboard:

```bat
streamlit run app.py
```

Upload a CSV with at least 100 valid daily rows. The app sorts dates and removes duplicate dates by keeping the last row. Invalid dates, non-finite values, invalid OHLC relationships, and insufficient data are rejected with an explanation.

## Tests

Run the focused test suite:

```bat
python -m pytest -q
```

The tests cover CSV validation, feature creation, and chronological splitting. Pytest is included in `requirements.txt` so the documented setup installs the test runner too.

## Data and licensing

PSX publishes historical data pages, but commercial use or redistribution may require rights, a license, or prior approval. Before building a public or paid product, confirm permitted access and use with the data provider.

- Official PSX historical data: <https://dps.psx.com.pk/historical>
- PSX data services: <https://www.psx.com.pk/psx/product-and-services/data-services-vending>

## Development scope

This project intentionally remains a simple educational prototype. Future work may include walk-forward evaluation, transaction-cost assumptions, corporate-action handling, and additional explainable indicators, without implying guaranteed performance.
