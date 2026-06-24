"""FinSentiment — LLM-driven sentiment features for short-horizon S&P 500 prediction.

Research pipeline for "Exploring Large Language Models for Sentiment-Driven
Trading Using Financial News Insights". See docs/ARCHITECTURE.md.

Pipeline stages
---------------
1. data      : download/load prices, load+clean news, align, build triggers.
2. features  : technical indicators, lagged features, model-ready matrices.
3. sentiment : FinBERT / LLM sentiment labels + (optional) embeddings.
4. modeling  : train/test splits and RF / XGBoost / LightGBM classifiers.
5. evaluation: accuracy / macro-F1 / reports + Sharpe-ratio backtest.
"""

__version__ = "1.0.0"

import logging


def get_logger(name: str = "finsentiment") -> logging.Logger:
    """Get a logger that prints to the console. Safe to call more than once."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # only add the handler the first time
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["__version__", "get_logger"]
