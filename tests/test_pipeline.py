"""Smoke + unit tests for the FinSentiment pipeline (run on sample data).

    pytest -q

Covers each stage: cleaning, feature engineering, label remap round-trip,
the positional sentiment join, and an end-to-end train+evaluate on the
committed sample slice (no API keys or large data required).
"""
import pandas as pd
import pytest

from finsentiment.config import load_config
from finsentiment.data import clean_text, check_columns, DataQualityError
from finsentiment.features import label_trigger, add_price_features, prepare_features
from finsentiment.modeling import LABEL_MAP, INV_LABEL_MAP
from finsentiment.pipeline import assemble_dataset, run_experiment


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["use_sample"] = True          # operate on data/sample/*
    c["split"]["strategy"] = "random"
    return c


# --- cleaning -------------------------------------------------------------- #
def test_clean_text_strips_html_urls_and_whitespace():
    assert clean_text("<p>Fed  cuts</p>  http://x.io rates") == "Fed cuts rates"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_check_columns_raises_on_missing():
    with pytest.raises(DataQualityError):
        check_columns(pd.DataFrame({"a": [1]}), ["a", "b"], name="t")


# --- feature engineering --------------------------------------------------- #
def test_label_trigger_deadband():
    assert label_trigger(0.01, 0.005) == 1
    assert label_trigger(-0.01, 0.005) == -1
    assert label_trigger(0.001, 0.005) == 0


def test_add_price_features_columns():
    df = pd.DataFrame({
        "aligned_date": pd.date_range("2020-01-01", periods=30, freq="D"),
        "Close": pd.Series(range(100, 130), dtype=float),
    })
    out = add_price_features(df)
    for col in ["SMA_5", "EMA_10", "volatility_5d", "RSI_14", "return_1d"]:
        assert col in out.columns


# --- modeling contract ----------------------------------------------------- #
def test_label_map_roundtrip():
    for k in (-1, 0, 1):
        assert INV_LABEL_MAP[LABEL_MAP[k]] == k


# --- integration ----------------------------------------------------------- #
def test_assemble_dataset_aligns(cfg):
    df = assemble_dataset(cfg, "gpt4o_mini")
    assert {"sentiment", "sentiment_num", "sentiment_confidence"} <= set(df.columns)
    assert len(df) > 0


def test_prepare_features_excludes_blocklist(cfg):
    df = assemble_dataset(cfg, "gpt4o_mini")
    prepared, feats = prepare_features(
        df, numeric_features=list(cfg.features["numeric"]),
        blocklist=list(cfg.features["blocklist"]),
    )
    assert "return_1d" not in prepared.columns        # leakage cols dropped
    assert "aligned_date" in prepared.columns          # retained for splits
    assert all(f in prepared.columns for f in feats)


def test_end_to_end_experiment(cfg):
    results = run_experiment(cfg, "gpt4o_mini", "random_forest", verbose=False)
    assert results, "expected at least one horizon result"
    for r in results:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["macro_f1"] <= 1.0
