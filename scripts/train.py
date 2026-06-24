#!/usr/bin/env python
"""Train + evaluate the model grid and write a tidy results table.

    # full grid on all cached models / classifiers
    python scripts/train.py

    # a single combination, with the full per-class report
    python scripts/train.py --models gpt4o_mini --classifiers random_forest --verbose

    # quick smoke test on the committed sample data
    python scripts/train.py --sample --models gpt4o_mini --classifiers random_forest
"""
import argparse

from finsentiment.config import load_config
from finsentiment.pipeline import run_grid

DEFAULT_MODELS = ["gpt4o_mini", "gpt41_mini", "gpt5", "deepseek"]
DEFAULT_CLASSIFIERS = ["random_forest", "xgboost", "lightgbm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    ap.add_argument("--sample", action="store_true", help="use committed sample data")
    ap.add_argument("--split", default=None, help="override split strategy: random|time|grouped")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None, help="results CSV (default: results/metrics.csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.sample:
        cfg["use_sample"] = True
    if args.split:
        cfg["split"]["strategy"] = args.split

    results = run_grid(cfg, args.models, args.classifiers, verbose=args.verbose)

    out = args.out or (cfg.path("results_dir") / "metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True) if hasattr(out, "parent") else None
    results.to_csv(out, index=False)
    print("\n=== Summary (accuracy / macro_f1) ===")
    print(results.to_string(index=False))
    print(f"\nWrote results -> {out}")


if __name__ == "__main__":
    main()
