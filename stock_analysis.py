"""Reusable validation, feature, and chronological evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
FEATURE_COLUMNS = [
    "Return_1d",
    "Return_5d",
    "Return_10d",
    "Volatility_10d",
    "MA_Ratio_5",
    "MA_Ratio_20",
    "Volume_Change",
    "Range_Pct",
]


class DataValidationError(ValueError):
    """Raised when an uploaded OHLCV table cannot be analyzed safely."""


def validate_and_prepare_data(
    df: pd.DataFrame,
    *,
    minimum_rows: int = 100,
) -> pd.DataFrame:
    """Validate and normalize an uploaded daily OHLCV table."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataValidationError("The uploaded CSV is empty.")

    prepared = df.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in prepared.columns]
    if missing:
        raise DataValidationError(
            "Missing required columns: " + ", ".join(missing) + "."
        )

    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    if prepared["Date"].isna().any():
        raise DataValidationError("Date contains invalid or blank values.")

    for column in REQUIRED_COLUMNS[1:]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    numeric = prepared[REQUIRED_COLUMNS[1:]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataValidationError(
            "Price and volume columns must contain only finite numeric values."
        )

    if (prepared[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise DataValidationError("Open, High, Low, and Close must all be greater than zero.")
    if (prepared["Volume"] < 0).any():
        raise DataValidationError("Volume cannot be negative.")

    invalid_ohlc = (
        (prepared["High"] < prepared[["Open", "Close"]].max(axis=1))
        | (prepared["Low"] > prepared[["Open", "Close"]].min(axis=1))
        | (prepared["High"] < prepared["Low"])
    )
    if invalid_ohlc.any():
        raise DataValidationError(
            f"Found {int(invalid_ohlc.sum())} row(s) with invalid OHLC relationships."
        )

    prepared = (
        prepared.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if len(prepared) < minimum_rows:
        raise DataValidationError(
            f"At least {minimum_rows} valid daily rows are required; "
            f"received {len(prepared)}."
        )
    return prepared


def build_features(df: pd.DataFrame, *, horizon: int = 5) -> pd.DataFrame:
    """Create historical-only predictors and a future-direction target."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    x = df.copy()
    x["Return_1d"] = x["Close"].pct_change(1)
    x["Return_5d"] = x["Close"].pct_change(5)
    x["Return_10d"] = x["Close"].pct_change(10)
    x["Volatility_10d"] = x["Return_1d"].rolling(10).std()
    x["MA_Ratio_5"] = x["Close"] / x["Close"].rolling(5).mean() - 1
    x["MA_Ratio_20"] = x["Close"] / x["Close"].rolling(20).mean() - 1
    x["Volume_Change"] = x["Volume"].pct_change()
    x["Range_Pct"] = (x["High"] - x["Low"]) / x["Close"]
    future_close = x["Close"].shift(-horizon)
    x["Target"] = (future_close > x["Close"]).where(future_close.notna()).astype("Int64")
    return x.replace([np.inf, -np.inf], np.nan)


def prepare_labeled_data(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build features and remove rows without enough historical/future context."""
    data = build_features(df, horizon=horizon)
    labeled = data.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    if labeled.empty or labeled["Target"].nunique() < 2:
        raise DataValidationError(
            "Not enough usable rows or only one target class after feature creation."
        )
    labeled["Target"] = labeled["Target"].astype(int)
    return labeled


def chronological_split(
    labeled: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    horizon: int = 5,
    minimum_train_rows: int = 30,
    minimum_test_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered rows while leaving a horizon-sized gap before the test set."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    split_index = int(len(labeled) * (1 - test_fraction))
    train_end = split_index - horizon
    train = labeled.iloc[:train_end].copy()
    test = labeled.iloc[split_index:].copy()
    if len(train) < minimum_train_rows or len(test) < minimum_test_rows:
        raise DataValidationError(
            "Not enough rows for the requested chronological train/test split."
        )
    if train["Target"].nunique() < 2:
        raise DataValidationError(
            "The chronological training period contains only one target class."
        )
    return train, test


@dataclass(frozen=True)
class EvaluationResult:
    model: RandomForestClassifier
    predictions: np.ndarray
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]


def evaluate_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> EvaluationResult:
    """Fit on the earlier period and evaluate on the later period only."""
    if train["Target"].nunique() < 2:
        raise DataValidationError(
            "The chronological training period contains only one target class."
        )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["Target"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    baseline_predictions = np.full(len(test), train["Target"].mode().iloc[0])

    def metrics_for(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
        return {
            "Balanced accuracy": balanced_accuracy_score(actual, predicted),
            "Accuracy": accuracy_score(actual, predicted),
            "Precision (up)": precision_score(actual, predicted, zero_division=0),
            "Recall (up)": recall_score(actual, predicted, zero_division=0),
        }

    return EvaluationResult(
        model=model,
        predictions=predictions,
        metrics=metrics_for(test["Target"], predictions),
        baseline_metrics=metrics_for(test["Target"], baseline_predictions),
    )
