import numpy as np
import pandas as pd
import pytest

from stock_analysis import (
    DataValidationError,
    build_features,
    chronological_split,
    evaluate_random_forest,
    prepare_labeled_data,
    validate_and_prepare_data,
)


def make_ohlcv(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = 100 + np.sin(np.arange(rows) / 3) * 10 + np.arange(rows) * 0.05
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(rows, 1000.0),
        }
    )


def test_validation_normalizes_and_sorts_valid_data():
    frame = make_ohlcv()
    shuffled = frame.iloc[::-1].copy()
    result = validate_and_prepare_data(shuffled)
    assert result["Date"].is_monotonic_increasing
    assert len(result) == len(frame)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda df: df.drop(columns=["Volume"]), "Missing required columns"),
        (lambda df: df.assign(Date="not-a-date"), "invalid"),
        (lambda df: df.assign(Close=np.inf), "finite"),
        (lambda df: df.assign(High=99), "OHLC"),
    ],
)
def test_validation_rejects_invalid_csv_data(change, message):
    with pytest.raises(DataValidationError, match=message):
        validate_and_prepare_data(change(make_ohlcv()))


def test_validation_rejects_empty_data():
    with pytest.raises(DataValidationError, match="empty"):
        validate_and_prepare_data(pd.DataFrame())


def test_build_features_uses_historical_values_and_future_target():
    frame = make_ohlcv()
    result = build_features(frame, horizon=5)
    assert {"Return_1d", "MA_Ratio_20", "Volume_Change", "Target"}.issubset(result)
    assert result["Target"].iloc[0] == 1
    assert pd.isna(result["Target"].iloc[-1])


def test_chronological_split_has_ordered_gap_and_two_training_classes():
    frame = make_ohlcv(150)
    labeled = build_features(frame).dropna().copy()
    train, test = chronological_split(labeled, horizon=5)
    assert train["Date"].max() < test["Date"].min()
    assert len(labeled.loc[train.index.max() + 1 : test.index.min() - 1]) == 5
    assert train["Target"].nunique() == 2


def test_chronological_split_rejects_single_class_training_data():
    frame = make_ohlcv(150)
    labeled = build_features(frame).dropna().copy()
    split_index = int(len(labeled) * 0.8) - 5
    labeled.loc[labeled.index[:split_index], "Target"] = 1
    with pytest.raises(DataValidationError, match="only one target class"):
        chronological_split(labeled, horizon=5)


def test_validation_rejects_non_finite_and_negative_market_values():
    with pytest.raises(DataValidationError, match="greater than zero"):
        validate_and_prepare_data(make_ohlcv().assign(Close=0))
    with pytest.raises(DataValidationError, match="negative"):
        validate_and_prepare_data(make_ohlcv().assign(Volume=-1))


def test_validation_deduplicates_dates_after_sorting():
    frame = pd.concat([make_ohlcv(), make_ohlcv(1)], ignore_index=True)
    frame.loc[len(frame) - 1, "Date"] = frame.loc[0, "Date"]
    result = validate_and_prepare_data(frame)
    assert result["Date"].is_unique
    assert len(result) == 120


def test_model_evaluation_returns_metrics_and_predictions_without_leakage():
    labeled = prepare_labeled_data(make_ohlcv(180))
    train, test = chronological_split(labeled, horizon=5)
    result = evaluate_random_forest(train, test)
    assert len(result.predictions) == len(test)
    assert set(result.metrics) == {
        "Balanced accuracy",
        "Accuracy",
        "Precision (up)",
        "Recall (up)",
    }
    assert set(result.baseline_metrics) == set(result.metrics)
    assert train["Date"].max() < test["Date"].min()


def test_chronological_split_rejects_insufficient_rows():
    labeled = prepare_labeled_data(make_ohlcv(120))
    with pytest.raises(DataValidationError, match="Not enough rows"):
        chronological_split(labeled, minimum_train_rows=200)
