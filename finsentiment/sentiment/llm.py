"""Sentiment from OpenAI and DeepSeek models.

Covers gpt-4o-mini, gpt-4.1-mini and gpt-5 (OpenAI) plus deepseek-chat. DeepSeek
speaks the OpenAI API, so the same client handles both — only the base URL and
API key differ. Keys come from the environment.
"""
from __future__ import annotations

import time

from ..config import require_key
from .base import SentimentProvider, build_prompt, parse_output

# provider -> (env var holding the key, base_url or None for OpenAI's default)
_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", None),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}


class LLMProvider(SentimentProvider):
    """Label one article at a time with a chat model."""

    def __init__(self, *, name: str, model: str, provider: str = "openai",
                 max_words: int = 500, temperature: float | None = 0.3,
                 throttle_s: float = 1.0):
        super().__init__(name)
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider}")
        from openai import OpenAI  # import here so openai is only needed if used

        env_var, base_url = _PROVIDERS[provider]
        self.model = model
        self.max_words = max_words
        self.temperature = temperature
        self.throttle_s = throttle_s
        self._embedding_provider = provider
        self.client = OpenAI(api_key=require_key(env_var), base_url=base_url)

    def classify_one(self, text: str):
        text = " ".join(text.split()[: self.max_words])
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": build_prompt(text)},
            ],
        }
        # gpt-5 only takes the default temperature, so we leave it off there
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
            out = resp.choices[0].message.content.strip()
        except Exception as exc:  # one bad article shouldn't kill the whole run
            print(f"[{self.name}] error: {exc}")
            return None, None
        finally:
            # be nice to the rate limit
            if self.throttle_s:
                time.sleep(self.throttle_s)
        return parse_output(out)

    def embed(self, text: str, model: str = "text-embedding-3-small"):
        """Get an embedding vector for the text (OpenAI text-embedding-3-small)."""
        try:
            resp = self.client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except Exception as exc:
            print(f"[{self.name}] embedding error: {exc}")
            return None
