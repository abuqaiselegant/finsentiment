"""Train/test splitting strategies and the RF / XGBoost / LightGBM models.

XGBoost requires non-negative class labels, so triggers ``{-1,0,1}`` are
remapped to ``{0,1,2}`` for training and mapped back for reporting. The
remapping is encapsulated here so callers always work in trigger space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_train_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    strategy: str = "time",
    test_size: float = 0.2,
    random_state: int = 42,
    embargo_days: int = 1,
):
    """Split into (X_train, X_test, y_train, y_test) by the chosen strategy.

    strategy:
      - ``random``  : stratified random split (note: optimistic for time series).
      - ``time``    : chronological — earliest (1-test_size) train, latest test.
      - ``grouped`` : random split over *dates*, with an embargo gap between the
                      train/test date ranges to limit near-day leakage.
    """
    data = df.dropna(subset=[target_col]).copy()
    data[target_col] = data[target_col].astype(int)
    X, y = data[feature_cols].copy(), data[target_col].copy()

    if strategy == "random":
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)

    if strategy == "time":
        if "aligned_date" not in data.columns:
            raise ValueError("aligned_date column required for the time-based split.")
        order = data.sort_values("aligned_date").index
        split = int((1 - test_size) * len(order))
        tr, te = order[:split], order[split:]
        return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]

    if strategy == "grouped":
        return _grouped_split(data, X, y, test_size=test_size,
                              random_state=random_state, embargo_days=embargo_days)

    raise ValueError(f"Unknown split strategy: {strategy}")


def _grouped_split(data, X, y, *, test_size, random_state, embargo_days):
    if "aligned_date" not in data.columns:
        raise ValueError("aligned_date column required for the grouped split.")
    dates = data["aligned_date"]
    unique = np.array(sorted(dates.dropna().unique()))
    rng = np.random.default_rng(random_state)
    n_test = int(test_size * len(unique))
    test_dates = set(rng.choice(unique, size=n_test, replace=False))

    if embargo_days:
        embargo = pd.Timedelta(days=embargo_days)
        test_arr = np.array(sorted(test_dates))
        embargoed = {d for d in unique
                     if any(abs((d - td) / np.timedelta64(1, "D")) <= embargo_days
                            for td in test_arr) and d not in test_dates}
    else:
        embargoed = set()

    test_mask = dates.isin(test_dates)
    train_mask = ~test_mask & ~dates.isin(embargoed)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def train_random_forest(X, y, **params):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def train_xgboost(X, y, **params):
    """Train XGBoost in remapped label space; stores ``label_map`` on the model."""
    from xgboost import XGBClassifier
    y_mapped = y.map(LABEL_MAP).astype(int)
    model = XGBClassifier(
        objective="multi:softmax", num_class=3, eval_metric="mlogloss",
        n_jobs=-1, **params,
    )
    model.fit(X, y_mapped)
    model.label_map = LABEL_MAP
    return model


def train_lightgbm(X, y, **params):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_jobs=-1, **params)
    model.fit(X, y)
    return model


TRAINERS = {
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
}


def train(name: str, X, y, cfg):
    """Train the named classifier using hyper-parameters from config."""
    params = dict(cfg.models.get(name, {}))
    return TRAINERS[name](X, y, **params)


def predict(model, X) -> np.ndarray:
    """Predict in trigger space ``{-1,0,1}`` regardless of the backend."""
    y_pred = model.predict(X)
    if getattr(model, "label_map", None):  # XGBoost trained in mapped space
        y_pred = pd.Series(y_pred).map(INV_LABEL_MAP).to_numpy()
    return y_pred
