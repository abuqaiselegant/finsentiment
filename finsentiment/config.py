"""Configuration loading and API-key resolution.

Loads ``config.yaml`` into a lightweight attribute-accessible mapping and
reads secrets from the environment (optionally via a local ``.env`` file).
No secret ever lives in a tracked file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:  # optional, but recommended for local dev
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

# Repository root = parent of the finsentiment package directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class Config(dict):
    """Dict that also supports attribute access and nested resolution.

    ``cfg.market["ticker"]`` and ``cfg["market"]["ticker"]`` both work, and
    nested dicts are wrapped so ``cfg.split.strategy`` works too.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve a configured path against the repo root.

        Honours ``use_sample``: when enabled, keys present in ``sample_paths``
        transparently point at the committed sample slices.
        """
        if self.get("use_sample") and key in self.get("sample_paths", {}):
            rel = self["sample_paths"][key]
        else:
            rel = self["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load YAML config and pull a local ``.env`` into the environment."""
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)


def require_key(env_var: str) -> str:
    """Fetch a required secret from the environment or fail loudly."""
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            f"Copy .env.example to .env and add your key (see README)."
        )
    return key
