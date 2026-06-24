"""FinBERT sentiment: tone labels, plus the [CLS] vector if you want embeddings.

Uses yiyanghkust/finbert-tone for the label and, optionally, the 768-number
[CLS] vector from the last layer.
"""
from __future__ import annotations

import numpy as np

from .base import SentimentProvider


class FinBERTProvider(SentimentProvider):
    """Sentiment from FinBERT, a BERT model trained on financial text."""

    def __init__(self, *, name: str = "finbert", model: str = "yiyanghkust/finbert-tone",
                 max_length: int = 512):
        super().__init__(name)
        import torch
        from transformers import AutoModel, AutoTokenizer, pipeline

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()
        self.pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=model,
            truncation=True, max_length=max_length,
        )

    def classify_one(self, text: str):
        try:
            out = self.pipeline(text, truncation=True, max_length=self.max_length)[0]
            return out["label"].capitalize(), float(out.get("score", float("nan")))
        except Exception as exc:
            print(f"[{self.name}] error: {exc}")
            return None, None

    def embed(self, text: str) -> np.ndarray:
        """Return the 768-number [CLS] vector from the last layer."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length",
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]  # the [CLS] token sums up the text
        return cls.squeeze().numpy()
