"""FinBERT sentiment provider: tone labels + optional [CLS] embeddings.

Uses ``yiyanghkust/finbert-tone`` for both the sentiment pipeline and the
768-dimensional [CLS] embedding extracted from the final hidden state.
"""
from __future__ import annotations

import numpy as np

from .base import SentimentProvider


class FinBERTProvider(SentimentProvider):
    """Domain-specific transformer sentiment + embedding extractor."""

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
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] error: {exc}")
            return None, None

    def embed(self, text: str) -> np.ndarray:
        """Return the 768-d [CLS] embedding from the final hidden layer."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length",
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls.squeeze().numpy()
