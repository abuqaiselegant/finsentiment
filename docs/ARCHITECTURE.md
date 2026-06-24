# FinSentiment — Architecture & Methodology

This document describes the design of the FinSentiment research pipeline: how
financial news and market data are ingested, cleaned, aligned and turned into
features; how sentiment is extracted from FinBERT and several LLMs; and how the
resulting models are trained, evaluated and backtested.

---

## 1. Overview

The system predicts the short-horizon direction of the **S&P 500** index from a
combination of **financial-news sentiment** and **technical indicators**. For each
news article it derives a sentiment signal (label + confidence) from a chosen model
(FinBERT, GPT-4o-mini, GPT-4.1-mini, GPT-5, or DeepSeek-V3), aligns it with
market features for the relevant trading day, and trains tree-based classifiers to
predict a ternary movement label (Down / Flat / Up) over three horizons
(t+1, t+3, t+5). Models are evaluated with both predictive metrics and a
Sharpe-ratio backtest.

The pipeline is a single, configuration-driven code path: the sentiment model,
feature set, train/test split strategy and classifier hyper-parameters are all
selected through `config.yaml`, so every experiment is one consistent flow rather
than a per-model variant.

---

## 2. Data sources

| Source | Description |
|---|---|
| **Market prices** | S&P 500 index (`^GSPC`) daily OHLCV via `yfinance`, 2014-01-01 → 2024-04-20, engineered into the `price_df` table (OHLCV + technicals + trigger labels). |
| **Financial news** | A corpus of S&P-500-relevant articles (CNBC / HuggingFace), with title, body text, publication date and metadata. |

Articles vary widely in length (from a headline to long-form reports); the pipeline
caps LLM input at the first 500 words to focus on the most salient content.

---

## 3. System architecture

```
                 ┌────────────────────────┐
   yfinance ───► │  price_df              │  OHLCV + engineered features
   (^GSPC)       │  • return 1/3/5d       │
                 │  • SMA_5, EMA_10        │
                 │  • RSI_14, volatility   │
                 │  • trigger_1d/3d/5d     │  (ternary targets)
                 └───────────┬────────────┘
                             │  merge_asof (backward) on date
   news corpus ──────────────┤  → each article aligned to last trading day
   (Title/Text/date)         │
                             ▼
                 ┌────────────────────────┐
                 │  merged_final          │  article × price master table
                 │  short_text (≤500 wds) │  (+ cleaning, validation)
                 └───────────┬────────────┘
                             │
                ┌────────────┴─────────────┐
                ▼                           ▼
        ┌──────────────┐            ┌────────────────┐
        │ Sentiment    │            │  Embeddings     │
        │ {label,conf} │            │ FinBERT CLS 768 │
        │ FinBERT/LLM  │            │ OpenAI 1536     │
        └──────┬───────┘            └───────┬────────┘
               └──────────────┬─────────────┘
                              ▼
                 ┌────────────────────────┐
                 │  Feature assembly       │
                 │  + optional t-1..t-5    │
                 │    lagged features      │
                 └───────────┬─────────────┘
                             ▼
                 ┌────────────────────────┐
                 │  Classifiers            │  RandomForest / XGBoost / LightGBM
                 │  Split: random / time / │  (XGB remaps {-1,0,1}→{0,1,2})
                 │         grouped+embargo │
                 └───────────┬─────────────┘
                             ▼
                 ┌────────────────────────┐
                 │  Evaluation             │  accuracy, macro-F1, report,
                 │                         │  confusion matrix,
                 │                         │  Sharpe-ratio backtest
                 └────────────────────────┘
```

---

## 4. Pipeline stages

### 4.1 Extraction — `finsentiment/data.py`
- `download_prices` pulls raw OHLCV from yfinance; `build_price_table` engineers
  technical indicators and trigger labels into the `price_df` schema.
- `load_news` reads the news corpus; `load_price_table` / `load_merged_table` load
  the engineered tables (sample-aware via config).

### 4.2 Cleaning & validation — `finsentiment/data.py`
- `clean_text` performs LLM-friendly normalisation: strip HTML tags and URLs and
  collapse whitespace. Case and finance terms are preserved (no lowercasing or
  stopword removal), since modern LLMs read raw natural text best.
- `check_columns` enforces required-column schemas; `quality_report` logs row/column
  counts and flags high-null columns. Both run at each load/build stage.

### 4.3 News–price alignment — `align_news_to_prices`
Each article is matched to the latest trading day at or before its publication
timestamp using `pd.merge_asof(direction="backward")`. This preserves causal
integrity — an article is only ever associated with market information available
up to its trading day. News outside trading hours maps to the next session.

### 4.4 Feature engineering — `finsentiment/features.py`
- **Targets** — `label_triggers` computes forward returns and applies a ±0.5 %
  dead-band to produce ternary `trigger_{1,3,5}d` ∈ {-1, 0, 1}.
- **Technical indicators** — `add_price_features` computes trailing
  `return_{1,3,5}d`, `SMA_5`, `EMA_10`, `volatility_5d` (5-day rolling std of
  returns) and `RSI_14`, all backward-looking.
- **Lagged features** — `add_lagged_features` optionally appends t-1…t-5 shifts of
  the numeric/sentiment features.
- **Model matrix** — `prepare_features` encodes sentiment, imputes missing values,
  drops leakage/identifier columns (the blocklist), and returns the feature list.
  `aligned_date` is retained for time-aware splits but never used as a feature.

### 4.5 Sentiment extraction — `finsentiment/sentiment/`
A common `SentimentProvider` interface (`base.py`) defines the shared prompt,
strict-format response parsing, and resumable checkpointing (`classify_frame`):

- **FinBERT** (`finbert.py`) — `yiyanghkust/finbert-tone` for tone labels and a
  768-dimensional `[CLS]` embedding from the final hidden state.
- **LLMs** (`llm.py`) — OpenAI (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5`) and DeepSeek
  (`deepseek-chat`, via the OpenAI-compatible endpoint). Zero-shot prompt returns a
  sentiment label and a confidence in [0, 1]; an `embed` method exposes OpenAI
  `text-embedding-3-small` (1536-d).

All providers emit `sentiment`, `sentiment_confidence` and a numeric `sentiment_num`
(`{Negative:-1, Neutral:0, Positive:1}`), cached to `data/processed/sentiment/`.

### 4.6 Modeling — `finsentiment/modeling.py`
- **Splits** — `split_train_test` supports `random` (stratified), `time`
  (chronological train/test) and `grouped` (random over dates with an embargo gap).
- **Classifiers** — Random Forest, XGBoost and LightGBM. XGBoost requires
  non-negative labels, so triggers `{-1,0,1}` are remapped to `{0,1,2}` for training
  and mapped back for reporting; `predict` always returns trigger-space labels.

### 4.7 Evaluation & backtest — `finsentiment/evaluation.py`
- `evaluate` reports accuracy, macro-F1, a per-class classification report and the
  confusion matrix, plus feature importances where available.
- `backtest_sharpe` runs a long/short strategy from predicted signals
  (`position = predicted trigger`, `period return = position × realised return`),
  reporting the Sharpe ratio, cumulative return, mean return and volatility.

### Orchestration — `finsentiment/pipeline.py`
`assemble_dataset` joins the master table with a model's cached sentiment;
`run_experiment` / `run_grid` execute the full (sentiment model × classifier ×
horizon) grid and return a tidy results table.

---

## 5. Package layout

```
finsentiment/
├── __init__.py       version + get_logger (structured logging)
├── config.py         YAML config + .env key loading
├── data.py           extract · clean_text · validation · align · triggers
├── features.py       technicals · trigger labels · lags · prepare_features
├── sentiment/
│   ├── base.py       prompt · parsing · checkpointing · provider interface
│   ├── finbert.py    FinBERT labels + 768-d [CLS] embeddings
│   └── llm.py        OpenAI / DeepSeek chat + embeddings
├── modeling.py       splits + RF/XGB/LGBM
├── evaluation.py     metrics + Sharpe backtest
└── pipeline.py       assemble → train → evaluate

scripts/   build_dataset.py · run_sentiment.py · train.py · backtest.py
tests/     test_pipeline.py
data/      raw · interim · processed · sample
```

| Concern | Module |
|---|---|
| Price/news extraction, cleaning, validation, alignment, triggers | `data.py` |
| Technical indicators, lags, model matrix | `features.py` |
| Sentiment labels + embeddings (FinBERT / LLM) | `sentiment/` |
| Train/test splits, classifiers | `modeling.py` |
| Metrics + economic backtest | `evaluation.py` |
| End-to-end grid orchestration | `pipeline.py` |

---

## 6. Configuration

`config.yaml` is the single source of truth for paths, market window, the trigger
threshold, feature lists (`numeric`, `blocklist`, `lags`), the sentiment model
catalog, the split strategy, and per-classifier hyper-parameters. `use_sample: true`
transparently redirects paths to the committed sample slices. Secrets are read from
the environment (`.env`) via `finsentiment.config` — never from tracked files.

---

## 7. Data-engineering practices

- **Validation** — schema and null-quality checks at each load/build stage; the
  positional sentiment join warns on any row-count mismatch.
- **Logging** — structured, timestamped logs (`finsentiment.get_logger`) across the
  data path.
- **Testing** — `tests/test_pipeline.py` covers cleaning, feature engineering, the
  XGBoost label-remap round-trip, the sentiment join and an end-to-end
  train+evaluate on the sample slice (`pytest -q`).
- **Orchestration** — a `Makefile` exposes `setup · dataset · sentiment · train ·
  backtest · demo · test · clean`; `make demo` runs end-to-end on the sample with no
  API keys or large data.

---

## 8. Reproducibility & sample data

Real datasets are kept local (large files + news licensing); only small slices under
`data/sample/` are committed so the pipeline runs on a fresh clone. The structure is
preserved with `.gitkeep` files. Regenerate the sample from the full local datasets:

```python
import pandas as pd, os, glob
N = 300
pd.read_csv("data/interim/merged_final.csv").head(N).to_csv("data/sample/merged_final.sample.csv", index=False)
pd.read_csv("data/raw/price_df.csv").head(N).to_csv("data/sample/price_df.sample.csv", index=False)
for f in glob.glob("data/processed/sentiment/*.csv"):
    name = os.path.basename(f).replace(".csv", ".sample.csv")
    pd.read_csv(f).head(N).to_csv(f"data/sample/sentiment/{name}", index=False)
pd.read_csv("data/processed/embeddings/gpt4o_mini_embeddings.csv", nrows=25)\
  .to_csv("data/sample/gpt4o_mini_embeddings.sample.csv", index=False)
```

---

## 9. Design considerations

- **Causal integrity / leakage.** Predictors for day *t* use only information
  available up to *t*: technical indicators are trailing, and news is aligned
  backward to the prior trading session. For time-series robustness the pipeline
  defaults to the **time-based split**; the **grouped split with an embargo** further
  reduces near-day leakage by separating train/test date ranges, and is available via
  `--split grouped`. A random stratified split is also provided for baseline
  comparison, but is optimistic for sequential data.
- **High-dimensional embeddings.** Raw embeddings carry far more dimensions than the
  technical/sentiment features; used alone they tend toward majority-class predictions.
  The primary feature set therefore fuses sentiment **labels + confidence** with
  technical indicators, with embeddings available as an optional variant.
- **Class imbalance.** Neutral days dominate, so **macro-F1** is reported alongside
  accuracy, and class weighting is used where supported (LightGBM, XGBoost).
- **Economic validity.** Statistical accuracy is complemented by a Sharpe-ratio
  backtest; reported returns are un-annualised and exclude transaction costs, so they
  measure signal quality rather than a deployable strategy.
