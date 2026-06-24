#!/usr/bin/env python
"""Extract sentiment labels for a model and cache them to data/processed/sentiment.

    python scripts/run_sentiment.py --model gpt4o_mini
    python scripts/run_sentiment.py --model finbert --limit 100

Requires the relevant API key in the environment for LLM providers
(see .env.example). FinBERT runs locally. Writes ``<model>_sentiments.csv``.
"""
import argparse
from pathlib import Path

from finsentiment.config import load_config
from finsentiment.data import load_merged_table
from finsentiment.sentiment import get_provider


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="catalog name, e.g. gpt4o_mini / finbert")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only classify the first N rows")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_merged_table(cfg)
    texts = df["short_text"]
    if args.limit:
        texts = texts.head(args.limit)

    out_dir = Path(cfg.path("sentiment_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_sentiments.csv"

    provider = get_provider(args.model, cfg)
    result = provider.classify_frame(texts, checkpoint_path=out_path)
    print(f"\nClassified {len(result):,} articles -> {out_path}")
    print(result["sentiment"].value_counts())


if __name__ == "__main__":
    main()
