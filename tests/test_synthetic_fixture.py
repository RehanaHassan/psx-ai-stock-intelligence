from pathlib import Path

import pandas as pd


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_psx_synthetic.csv"
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def test_synthetic_ohlcv_fixture_is_valid_and_chronological():
    frame = pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])

    assert frame.columns.tolist() == REQUIRED_COLUMNS
    assert len(frame) >= 250
    assert frame["Date"].notna().all()
    assert frame["Date"].is_monotonic_increasing
    assert frame["Date"].is_unique
    assert (frame[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (frame["Volume"] >= 0).all()
    assert (frame["High"] >= frame[["Open", "Close"]].max(axis=1)).all()
    assert (frame["Low"] <= frame[["Open", "Close"]].min(axis=1)).all()
