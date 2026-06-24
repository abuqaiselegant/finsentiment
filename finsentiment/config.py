"""Loads config.yaml and reads API keys from the environment.

Secrets come from environment variables (or a local .env file) and never from
a file that gets committed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; fall back to a no-op
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

# repo root is one level up from this package
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class Config(dict):
    """A dict you can also read with dot notation.

    So cfg.market.ticker works as well as cfg["market"]["ticker"]. Nested dicts
    are wrapped on access so the dot style keeps working all the way down.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Return an absolute path for a configured key.

        If use_sample is on and we have a sample file for this key, point at the
        sample instead of the full dataset.
        """
        if self.get("use_sample") and key in self.get("sample_paths", {}):
            rel = self["sample_paths"][key]
        else:
            rel = self["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Read the YAML config and load any local .env into the environment."""
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)


def require_key(env_var: str) -> str:
    """Read an API key from the environment, or raise if it isn't set."""
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            f"Copy .env.example to .env and add your key (see README)."
        )
    return key
