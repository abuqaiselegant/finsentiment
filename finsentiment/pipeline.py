"""High-level orchestration: assemble dataset -> train -> evaluate.

Ties the modules together so a whole experiment grid (sentiment model x
classifier x horizon) can be run from one call or the ``train`` script.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data as data_mod
from . import evaluation, get_logger, modeling
from .features import add_lagged_features, prepare_features
from .sentiment import load_sentiment_file

log = get_logger(__name__)


def assemble_dataset(cfg, model_name: str) -> pd.DataFrame:
    """Merge the interim master table with a model's cached sentiment labels.

    Sentiment files align to ``merged_final`` by row order, so they are attached
    positionally then trimmed to a common length.
    """
    merged = data_mod.load_merged_table(cfg)

    sent_dir = cfg.path("sentiment_dir")
    suffix = ".sample.csv" if cfg.get("use_sample") else ".csv"
    sent_path = Path(sent_dir) / f"{model_name}_sentiments{suffix}"
    sentiment = load_sentiment_file(sent_path)

    if len(merged) != len(sentiment):
        log.warning("row-count mismatch on positional join: merged=%d, %s sentiment=%d "
                    "-> trimming to %d", len(merged), model_name, len(sentiment),
                    min(len(merged), len(sentiment)))
    n = min(len(merged), len(sentiment))
    df = merged.iloc[:n].reset_index(drop=True)
    df[["sentiment", "sentiment_confidence", "sentiment_num"]] = \
        sentiment.iloc[:n].reset_index(drop=True)
    return df


def run_experiment(cfg, model_name: str, classifier: str, *, with_lags: bool = False,
                   verbose: bool = True) -> list[dict]:
    """Train + evaluate one (sentiment model, classifier) across all horizons."""
    df = assemble_dataset(cfg, model_name)

    if with_lags:
        lag_feats = list(cfg.features["numeric"])
        df = add_lagged_features(df, lag_feats, lags=cfg.features["lags"])

    df, feature_cols = prepare_features(
        df,
        numeric_features=list(cfg.features["numeric"]),
        blocklist=list(cfg.features["blocklist"]),
    )
    if with_lags:
        feature_cols = feature_cols + [c for c in df.columns if "_lag" in c]

    results = []
    for h in cfg.triggers["horizons"]:
        target = f"trigger_{h}d"
        if target not in df.columns:
            continue
        Xtr, Xte, ytr, yte = modeling.split_train_test(
            df, feature_cols, target,
            strategy=cfg.split["strategy"], test_size=cfg.split["test_size"],
            random_state=cfg.split["random_state"], embargo_days=cfg.split["embargo_days"],
        )
        model = modeling.train(classifier, Xtr, ytr, cfg)
        metrics = evaluation.evaluate(
            model, Xte, yte, feature_names=feature_cols,
            target_name=f"{model_name} | {classifier} | {target}", verbose=verbose,
        )
        metrics.update({"model": model_name, "classifier": classifier, "horizon": h})
        results.append(metrics)
    return results


def run_grid(cfg, model_names: list[str], classifiers: list[str], *,
             verbose: bool = False) -> pd.DataFrame:
    """Run the full grid and return a tidy results table."""
    rows = []
    for m in model_names:
        for clf in classifiers:
            rows.extend(run_experiment(cfg, m, clf, verbose=verbose))
    return pd.DataFrame(rows)
