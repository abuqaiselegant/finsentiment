"""LLM sentiment provider for OpenAI and DeepSeek (OpenAI-compatible) models.

Covers gpt-4o-mini, gpt-4.1-mini, gpt-5 (provider ``openai``) and
deepseek-chat (provider ``deepseek``). API keys are read from the environment;
they are never hard-coded.
"""
from __future__ import annotations

import time

from ..config import require_key
from .base import SentimentProvider, build_prompt, parse_output

# provider -> (env var, base_url)
_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", None),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}


class LLMProvider(SentimentProvider):
    """Zero-shot sentiment classification via a chat-completions endpoint."""

    def __init__(self, *, name: str, model: str, provider: str = "openai",
                 max_words: int = 500, temperature: float | None = 0.3,
                 throttle_s: float = 1.0):
        super().__init__(name)
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider}")
        from openai import OpenAI  # imported lazily so the dep is optional

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
        # gpt-5 is called without an explicit temperature.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
            out = resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 - surfaced, then skipped
            print(f"[{self.name}] error: {exc}")
            return None, None
        finally:
            if self.throttle_s:
                time.sleep(self.throttle_s)
        return parse_output(out)

    def embed(self, text: str, model: str = "text-embedding-3-small"):
        """Return an embedding vector (OpenAI ``text-embedding-3-small``)."""
        try:
            resp = self.client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] embedding error: {exc}")
            return None
