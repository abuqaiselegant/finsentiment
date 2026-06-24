#!/usr/bin/env python
"""Build the interim master table (news aligned to prices + text columns).

    python scripts/build_dataset.py [--config config.yaml] [--out PATH]

Reads the raw news + engineered price table from config and writes the
``merged_final`` table used by every sentiment pipeline.
"""
import argparse

from finsentiment.config import load_config
from finsentiment.data import build_merged_table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output CSV (default: config paths.merged)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = build_merged_table(cfg)
    out = args.out or cfg.path("merged")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
