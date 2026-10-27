# Synthetic OHLCV fixture

`sample_psx_synthetic.csv` is a deterministic **SYNTHETIC/DEMO** dataset for testing the upload and analysis workflow. It was generated with Python's `random.Random` seed `20261027` and contains 260 weekday-dated rows with internally consistent OHLCV values.

All values are invented for software testing. They are not PSX prices, do not represent any real security or issuer, and must not be used for investment or market analysis.

The CSV intentionally contains only the six application columns: `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.
