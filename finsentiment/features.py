"""Turn the raw price/news table into model features and target labels."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SENTIMENT_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def label_trigger(r: float, threshold: float) -> int:
    """Up (1) / down (-1) / flat (0) for a return, ignoring small moves."""
    if r > threshold:
        return 1
    if r < -threshold:
        return -1
    return 0


def label_triggers(df: pd.DataFrame, *, threshold: float, horizons: Iterable[int]) -> pd.DataFrame:
    """Add the target label for each horizon based on the future return.

    We look h days ahead to decide the label. The return_{h}d columns written
    here are forward returns; add_price_features later overwrites them with
    trailing returns, which is what the model actually sees.
    """
    df = df.copy()
    close = df["Close"].astype(float)
    for h in horizons:
        nxt = close.shift(-h)
        ret = (nxt - close) / close
        df[f"next_close_{h}d"] = nxt
        df[f"return_{h}d"] = ret
        df[f"trigger_{h}d"] = ret.apply(lambda x: label_trigger(x, threshold))
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the technical indicators used as features.

    Everything here looks backward only (rolling/trailing windows), so a row for
    day t never uses information from after t.
    """
    df = df.sort_values("aligned_date").reset_index(drop=True).copy()
    close = df["Close"].astype(float)

    # trailing returns (overwrites the forward returns from label_triggers)
    for h in (1, 3, 5):
        df[f"return_{h}d"] = close.pct_change(periods=h)

    df["SMA_5"] = close.rolling(window=5).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()

    # 14-day RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # small constant avoids divide-by-zero
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def add_lagged_features(df: pd.DataFrame, features: list[str], *, lags: int = 5) -> pd.DataFrame:
    """Add lagged copies (t-1 .. t-lags) of the given feature columns."""
    df = df.sort_values("aligned_date").copy()
    for feat in features:
        if feat in df.columns:
            for lag in range(1, lags + 1):
                df[f"{feat}_lag{lag}"] = df[feat].shift(lag)
    return df


def prepare_features(
    df: pd.DataFrame,
    *,
    numeric_features: list[str],
    blocklist: list[str],
    impute_missing_sentiment: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the model matrix and return (df, feature_cols).

    Encodes sentiment to a number, fills gaps, and drops the blocklist columns
    (returns/ids that would leak the target). aligned_date is kept around for the
    time-based and grouped splits but isn't used as a feature.
    """
    df = df.copy()
    if "aligned_date" in df.columns:
        df["aligned_date"] = pd.to_datetime(df["aligned_date"], errors="coerce")

    if "sentiment_num" not in df.columns and "sentiment" in df.columns:
        df["sentiment_num"] = df["sentiment"].map(SENTIMENT_MAP)

    if impute_missing_sentiment:
        # treat a missing label as neutral and a missing confidence as 0.5
        if "sentiment_num" in df.columns:
            df["sentiment_num"] = df["sentiment_num"].fillna(0).astype(int)
        if "sentiment_confidence" in df.columns:
            df["sentiment_confidence"] = df["sentiment_confidence"].fillna(0.5)
    else:
        df = df.dropna(subset=[c for c in ("sentiment_num", "sentiment_confidence") if c in df])

    feature_cols = [c for c in numeric_features if c in df.columns]
    drop_cols = [c for c in blocklist if c in df.columns and c != "aligned_date"]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df, feature_cols
