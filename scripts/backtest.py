#!/usr/bin/env python
"""Backtest a trained signal and report Sharpe ratio + returns per horizon.

    python scripts/backtest.py --model gpt4o_mini --classifier xgboost

Trains on the configured split, then evaluates a long/short strategy driven by
the predicted triggers. Returns are un-annualised and exclude transaction costs.
"""
import argparse

from finsentiment.config import load_config
from finsentiment import modeling
from finsentiment.evaluation import backtest_sharpe
from finsentiment.features import prepare_features
from finsentiment.pipeline import assemble_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt4o_mini")
    ap.add_argument("--classifier", default="xgboost")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.sample:
        cfg["use_sample"] = True

    # Keep the raw frame (with return_* columns) for the backtest; the prepared
    # frame (returns dropped as leakage) is used only for fitting the model.
    raw = assemble_dataset(cfg, args.model)
    prepared, feature_cols = prepare_features(
        raw, numeric_features=list(cfg.features["numeric"]),
        blocklist=list(cfg.features["blocklist"]),
    )

    for h in cfg.triggers["horizons"]:
        target, ret_col = f"trigger_{h}d", f"return_{h}d"
        if target not in prepared.columns or ret_col not in raw.columns:
            continue
        Xtr, Xte, ytr, yte = modeling.split_train_test(
            prepared, feature_cols, target, strategy=cfg.split["strategy"],
            test_size=cfg.split["test_size"], random_state=cfg.split["random_state"],
            embargo_days=cfg.split["embargo_days"],
        )
        model = modeling.train(args.classifier, Xtr, ytr, cfg)
        backtest_sharpe(model, Xte, yte, raw, return_col=ret_col, target_name=f"t+{h}")


if __name__ == "__main__":
    main()
