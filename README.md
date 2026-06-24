# FinSentiment

### Exploring Large Language Models for Sentiment-Driven Trading Using Financial News Insights

[![CI](https://github.com/abuqaiselegant/finsentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/abuqaiselegant/finsentiment/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> MSc Computer Science (Artificial Intelligence) thesis — University of Nottingham.
> **Author:** Abu Qais · **Supervisor:** Dr Damian Eke.

A research study on whether sentiment extracted from financial news by **Large
Language Models** improves short-horizon (t+1 / t+3 / t+5) prediction of **S&P 500**
index direction when fused with technical indicators — and how general-purpose LLMs
(**GPT-4o-mini, GPT-4.1-mini, GPT-5, DeepSeek-V3**) compare against the
finance-specific transformer **FinBERT**, across Random Forest, XGBoost and LightGBM
classifiers, evaluated with both predictive metrics and a Sharpe-ratio backtest.

---

## Abstract

Financial markets are shaped by both technical trading patterns and investor
sentiment, with financial news central to shaping expectations. Traditional and
domain-specific sentiment models (e.g. FinBERT) capture part of this signal but are
limited in adaptability and forward-looking reasoning. This study investigates
whether LLM-derived sentiment signals enhance short-horizon market prediction when
combined with technical indicators. Five sentiment sources are evaluated across three
horizons for S&P 500 movements; sentiment, confidence and market indicators (RSI,
SMA, EMA, volatility) are integrated into tree-based classifiers, trained under
time-aware splits to limit leakage, and assessed economically via simulated returns
and Sharpe ratios.

## Research questions

- **RQ1** — How do GPT-4o-mini, GPT-4.1-mini, GPT-5, DeepSeek and FinBERT compare at
  extracting context-aware sentiment from financial news?
- **RQ2** — To what extent does integrating sentiment (and embeddings) with technical
  indicators improve predictive performance over sentiment-only or technical-only models?
- **RQ3** — How effective are zero-shot prompting strategies for producing
  machine-readable sentiment suitable for trading signals?
- **RQ4** — Do predictive gains translate into economically meaningful, risk-adjusted
  returns (Sharpe ratio) across horizons?

---

## Results at a glance

Random Forest accuracy with *sentiment + technical* features:

| Sentiment source | t+1 | t+3 | t+5 |
|---|---|---|---|
| GPT-4o-mini  | 0.849 | **0.893** | 0.889 |
| GPT-4.1-mini | 0.849 | 0.885 | **0.894** |
| GPT-5        | 0.845 | 0.888 | 0.882 |
| DeepSeek-V3  | **0.854** | 0.888 | 0.891 |
| FinBERT      | 0.773 | 0.789 | 0.794 |

Key findings: fusing LLM **sentiment labels + confidence** with technical indicators
clearly outperforms FinBERT and the technical-only baseline; **embeddings alone sit
near chance** (~0.50); Random Forest is the most reliable classifier; and economic
value (Sharpe ratio) is strongest at the longer t+5 horizon.

---

## Repository structure

```
finsentiment/        # research pipeline (one config-driven code path)
├── config.py        # YAML config + .env-based key loading + logging
├── data.py          # extract, clean, validate, align (merge_asof), triggers
├── features.py      # technicals, lags, trigger labels, model matrices
├── sentiment/       # pluggable sentiment providers
│   ├── base.py      #   shared prompt, parsing, checkpointing
│   ├── finbert.py   #   FinBERT labels + 768-d [CLS] embeddings
│   └── llm.py       #   OpenAI / DeepSeek (chat + embeddings)
├── modeling.py      # splits (random/time/grouped+embargo) + RF/XGB/LGBM
├── evaluation.py    # accuracy / macro-F1 / report + Sharpe backtest
└── pipeline.py      # assemble -> train -> evaluate orchestration

scripts/             # thin CLIs over the package
├── build_dataset.py # raw news + prices -> data/interim/merged_final.csv
├── run_sentiment.py # extract & cache sentiment for one model
├── train.py         # train + evaluate the model grid -> results/metrics.csv
└── backtest.py      # Sharpe ratio / returns per horizon

tests/               # pytest smoke + unit tests (run on the sample data)
data/                # see "Data" below
docs/                # methodology & architecture notes
Makefile             # make setup/dataset/sentiment/train/demo/test
config.yaml          # paths, features, split, hyper-parameters
```

Each stage applies light **data validation** (schema + null checks) and structured
**logging**; `make demo` runs the whole pipeline on the committed sample and
`make test` runs the suite — both without API keys or large data.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                                # core (pandas, sklearn, xgboost, lightgbm)
pip install -e ".[sentiment,data,analysis]"     # + transformers/openai/yfinance/plots

cp .env.example .env                            # add OPENAI_API_KEY / DEEPSEEK_API_KEY
```

API keys are read from the environment (via `.env`); none are stored in the repository.

## Reproducing the study

A small committed sample lets you run end-to-end immediately — no API keys or large
downloads required:

```bash
python scripts/train.py --sample --models gpt4o_mini --classifiers random_forest --verbose
python scripts/backtest.py --sample --model gpt4o_mini --classifier random_forest
```

Full study (with the complete datasets placed under `data/` — see below):

```bash
# 1. build the aligned news↔price master table
python scripts/build_dataset.py

# 2. extract sentiment per model (LLMs need the API key; FinBERT runs locally)
python scripts/run_sentiment.py --model gpt4o_mini
python scripts/run_sentiment.py --model finbert

# 3. train + evaluate the full grid  -> results/metrics.csv
python scripts/train.py

# 4. economic evaluation
python scripts/backtest.py --model gpt4o_mini --classifier xgboost
```

Any `config.yaml` setting can be overridden on the CLI, e.g. `--split grouped`,
`--models gpt5 deepseek`, `--classifiers xgboost`.

---

## Data

The datasets are **not** fully committed (size + news licensing). Layout:

```
data/
├── raw/                 # inputs        (git-ignored)
│   ├── stock_data_articles.csv      # CNBC/HuggingFace financial news
│   └── price_df.csv                 # engineered S&P 500 OHLCV + technicals + triggers
├── interim/             # generated     (git-ignored)
│   └── merged_final.csv             # article × price master table
├── processed/           # generated     (git-ignored)
│   ├── sentiment/<model>_sentiments.csv
│   └── embeddings/<model>_embeddings.csv   # large
└── sample/              # small committed slices (runnable demo)
```

Place the full files in `data/raw|interim|processed` (the directory structure is
kept via `.gitkeep`). Only `data/sample/` is tracked. Sentiment files align to
`merged_final` by row order.

---

## Methodology

- **Universe & period** — S&P 500 index (`^GSPC`), daily, 2014-01-01 → 2024-04-20.
- **Targets** — ternary `trigger_{1,3,5}d` ∈ {Down, Flat, Up} via a ±0.5 % dead-band
  on forward returns.
- **Features** — SMA-5, EMA-10, RSI-14, 5-day volatility, returns; LLM/FinBERT
  sentiment label + confidence; optional embeddings; optional t-1…t-5 lags.
- **News→price alignment** — each article is mapped to the latest trading day at/before
  publication (`merge_asof`, backward) for causal integrity.
- **Models** — Random Forest, XGBoost (labels remapped to {0,1,2}), LightGBM.
- **Validation** — random / time-based / grouped-by-date with embargo (time-based default).
- **Evaluation** — accuracy, macro-F1, per-class report; long/short Sharpe-ratio backtest.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full architecture and
methodological notes, including leakage considerations and split design.

## Limitations & future work

- Tree-based models do not explicitly capture temporal dependencies; sequence models
  (LSTM/Transformer) are a natural extension.
- LLM input is capped at the first 500 words per article; long disclosures may carry
  later sentiment that truncation misses.
- Backtests exclude transaction costs, slippage and execution latency.
- Hybrid pipelines that combine FinBERT's domain calibration with LLM contextual depth,
  and more extensive embargo/rolling-window validation, are promising next steps.

## Citation

```bibtex
@mastersthesis{qais2025finsentiment,
  title  = {Exploring Large Language Models for Sentiment-Driven Trading Using Financial News Insights},
  author = {Abu Qais},
  school = {University of Nottingham},
  year   = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
