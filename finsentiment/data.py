"""Load the prices and news, clean them up, and align them by date.

Steps:
  - load the S&P 500 price table (or download fresh OHLCV from yfinance),
  - load and clean the news corpus,
  - attach each article to the trading day it could first act on,
  - label each row up/down/flat at the t+1, t+3 and t+5 horizons.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import get_logger
from .features import add_price_features, label_triggers

log = get_logger(__name__)


# Cleaning & validation

class DataQualityError(ValueError):
    """A dataset is missing something we need to continue."""


_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")


def clean_text(text) -> str:
    """Strip HTML tags and URLs, then squeeze whitespace.

    We keep the original casing and wording — the LLMs handle raw text well, and
    words like "Fed" or "earnings" matter, so there's no lowercasing or stopword
    removal here.
    """
    if text is None or isinstance(text, float):
        return ""
    s = _HTML_TAG.sub(" ", str(text))
    s = _URL.sub(" ", s)
    return _WS.sub(" ", s).strip()


def check_columns(df: pd.DataFrame, required, name: str = "frame") -> None:
    """Fail early if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(f"{name}: missing required columns {missing}")


def quality_report(df: pd.DataFrame, name: str = "frame", *, max_null: float = 0.5) -> dict:
    """Log row/column counts and call out any mostly-empty columns."""
    nulls = df.isna().mean()
    high = nulls[nulls > max_null].round(3).to_dict()
    log.info("%s: %d rows x %d cols | high-null(>%.0f%%): %s",
             name, len(df), df.shape[1], max_null * 100, high or "none")
    return {"rows": len(df), "cols": df.shape[1], "high_null": high}


# Prices

def download_prices(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from yfinance with the date as a normal column."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval=interval)
    df = df.reset_index().rename(columns={"Date": "aligned_date"})
    return df


def build_price_table(raw_prices: pd.DataFrame, *, threshold: float, horizons) -> pd.DataFrame:
    """Build the price table: OHLCV + indicators + trigger labels.

    This is the price_df that the news gets joined against later.
    """
    df = raw_prices.copy()
    df["aligned_date"] = pd.to_datetime(df["aligned_date"])
    df = df.sort_values("aligned_date").reset_index(drop=True)

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    df = label_triggers(df, threshold=threshold, horizons=horizons)
    df = add_price_features(df)
    df = df.bfill()  # fill the leading NaNs from the rolling windows
    return df


def load_price_table(path: str | Path) -> pd.DataFrame:
    """Load the saved price table and parse the date column."""
    df = pd.read_csv(path)
    df["aligned_date"] = pd.to_datetime(df["aligned_date"])
    return df.sort_values("aligned_date").reset_index(drop=True)


# News

def load_news(path: str | Path, *, start: str, end: str) -> pd.DataFrame:
    """Load the news corpus, drop empty rows, and keep the date window."""
    df = pd.read_csv(path)
    check_columns(df, ["Title", "Text", "Publishdate"], name="news")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Publishdate"] = pd.to_datetime(df["Publishdate"], errors="coerce")
    df = df.dropna(subset=["Text", "Publishdate"])
    mask = (df["Publishdate"] >= pd.Timestamp(start)) & (df["Publishdate"] <= pd.Timestamp(end))
    df = df.loc[mask].sort_values("Publishdate").reset_index(drop=True)
    quality_report(df, name="news (filtered)")
    return df


# Alignment + text prep

def align_news_to_prices(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join each article to the trading day it could first be acted on.

    merge_asof(direction="backward") picks the latest trading day at or before
    the article's timestamp, so we never attach future prices to an article.
    """
    news = news.sort_values("Publishdate")
    prices = prices.sort_values("aligned_date")
    merged = pd.merge_asof(
        news, prices, left_on="Publishdate", right_on="aligned_date", direction="backward"
    )
    return merged


def add_text_columns(df: pd.DataFrame, *, max_words: int, clean: bool = True) -> pd.DataFrame:
    """Build the text the model reads: gpt_input, short_text and lengths.

    short_text is the first max_words words and is what we send to the LLMs.
    """
    df = df.copy()
    title = df["Title"].map(clean_text) if clean else df["Title"].fillna("")
    text = df["Text"].map(clean_text) if clean else df["Text"].fillna("")
    df["gpt_input"] = (title + ". " + text).str.slice(0, 1500)
    df["title_length"] = title.str.len()
    df["text_length"] = text.str.len()
    df["short_text"] = df["gpt_input"].apply(lambda x: " ".join(str(x).split()[:max_words]))
    return df


def build_merged_table(cfg) -> pd.DataFrame:
    """Run the whole news+price merge and return the master table.

    This is what gets saved as data/interim/merged_final.csv. Trigger columns go
    last so the targets sit at the end of the table.
    """
    prices = load_price_table(cfg.path("raw_prices"))
    news = load_news(
        cfg.path("raw_news"), start=cfg.market["start_date"], end=cfg.market["end_date"]
    )
    merged = align_news_to_prices(news, prices)
    merged = merged.drop(columns=["next_close_1d", "next_close_3d", "next_close_5d",
                                  "links", "symbol", "company"], errors="ignore")
    merged = add_text_columns(merged, max_words=cfg.sentiment["text_max_words"])

    triggers = [f"trigger_{h}d" for h in cfg.triggers["horizons"]]
    check_columns(merged, triggers, name="merged")
    ordered = [c for c in merged.columns if c not in triggers] + \
              [c for c in triggers if c in merged.columns]
    merged = merged[ordered]
    quality_report(merged, name="merged_final")
    return merged


def load_merged_table(cfg) -> pd.DataFrame:
    """Load the prebuilt master table (picks the sample file when use_sample)."""
    df = pd.read_csv(cfg.path("merged"))
    if "aligned_date" in df.columns:
        df["aligned_date"] = pd.to_datetime(df["aligned_date"], errors="coerce")
    return df
