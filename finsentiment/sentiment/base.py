"""Bits every provider shares: the prompt, reply parsing, and the run loop."""
from __future__ import annotations

import abc
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..features import SENTIMENT_MAP

PROMPT_TEMPLATE = '''You are a financial news sentiment analysis assistant.

Classify the sentiment of the following news article as **Positive**, **Negative**, or **Neutral**.
Also provide a confidence score between 0 and 1 that reflects how confident you are about the classification.

News article:
"""{text}"""

Respond strictly in this format:
Sentiment: <Positive/Negative/Neutral>
Confidence: <score between 0 and 1>'''


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def parse_output(output: str | None) -> tuple[str | None, float | None]:
    """Pull (sentiment, confidence) out of the model's two-line reply.

    Returns (None, None) if the reply doesn't look like we asked.
    """
    if not output:
        return None, None
    try:
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        sent_line = next((l for l in lines if l.lower().startswith("sentiment:")), None)
        conf_line = next((l for l in lines if l.lower().startswith("confidence:")), None)
        sentiment = sent_line.split(":", 1)[1].strip() if sent_line else None
        confidence = float(conf_line.split(":", 1)[1].strip()) if conf_line else None
        return sentiment, confidence
    except (ValueError, IndexError):
        return None, None


class SentimentProvider(abc.ABC):
    """Common interface; each backend just implements classify_one."""

    #: used to name the output columns, e.g. gpt4o_mini_sentiment
    name: str

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def classify_one(self, text: str) -> tuple[str | None, float | None]:
        """One article in, (label, confidence) out."""

    def classify_frame(
        self,
        texts: pd.Series,
        *,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 50,
    ) -> pd.DataFrame:
        """Run over a column of articles, label by label.

        With a checkpoint_path it saves progress to CSV every checkpoint_every
        rows, so a long (and, for LLMs, paid) run can pick up where it stopped.
        """
        sent_col, conf_col = f"{self.name}_sentiment", f"{self.name}_confidence"
        results: list[tuple[str | None, float | None]] = []
        for i, text in tqdm(enumerate(texts), total=len(texts), desc=self.name):
            results.append(self.classify_one(str(text)))
            if checkpoint_path and i % checkpoint_every == 0:
                pd.DataFrame(results, columns=[sent_col, conf_col]).to_csv(checkpoint_path, index=False)
        df = pd.DataFrame(results, columns=[sent_col, conf_col])
        if checkpoint_path:
            df.to_csv(checkpoint_path, index=False)
        return _standardise(df, sent_col, conf_col)


def load_sentiment_file(path: str | Path) -> pd.DataFrame:
    """Read a cached *_sentiments.csv and rename its columns to the standard ones."""
    df = pd.read_csv(path)
    sent_col = next(c for c in df.columns if c.endswith("_sentiment"))
    conf_col = next(c for c in df.columns if c.endswith("_confidence"))
    return _standardise(df, sent_col, conf_col)


def _standardise(df: pd.DataFrame, sent_col: str, conf_col: str) -> pd.DataFrame:
    """Rename to sentiment/sentiment_confidence and add the numeric column."""
    out = pd.DataFrame({
        "sentiment": df[sent_col],
        "sentiment_confidence": df[conf_col],
    })
    out["sentiment_num"] = out["sentiment"].map(SENTIMENT_MAP)
    return out
