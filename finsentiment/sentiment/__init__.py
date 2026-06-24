"""Sentiment providers: a common interface over FinBERT and LLM backends.

    from finsentiment.sentiment import get_provider
    provider = get_provider("gpt4o_mini", cfg)
    labels_df = provider.classify_frame(df["short_text"])

All providers return a DataFrame with ``sentiment`` (Positive/Neutral/Negative)
and ``sentiment_confidence`` columns, plus a numeric ``sentiment_num``.
"""
from __future__ import annotations

from .base import SentimentProvider, load_sentiment_file


def get_provider(name: str, cfg) -> SentimentProvider:
    """Instantiate the provider registered under ``name`` in config.yaml."""
    spec = cfg.sentiment["models"].get(name)
    if spec is None:
        raise KeyError(f"Unknown sentiment model '{name}'. "
                       f"Known: {list(cfg.sentiment['models'])}")
    provider = spec["provider"]
    if provider == "finbert":
        from .finbert import FinBERTProvider
        return FinBERTProvider(name=name, model=spec["model"])
    if provider in ("openai", "deepseek"):
        from .llm import LLMProvider
        return LLMProvider(name=name, model=spec["model"], provider=provider,
                           max_words=cfg.sentiment["text_max_words"])
    raise ValueError(f"Unsupported provider '{provider}' for model '{name}'.")


__all__ = ["SentimentProvider", "get_provider", "load_sentiment_file"]
