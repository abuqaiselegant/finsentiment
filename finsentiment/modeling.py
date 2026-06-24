"""Train/test splits and the three classifiers (RF, XGBoost, LightGBM).

Our labels are -1/0/1, but XGBoost only accepts 0..n labels, so we shift them to
0/1/2 to train and shift back to report. That bookkeeping is kept here so the
rest of the code only ever deals with -1/0/1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}


# Splits

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
    """Return X_train, X_test, y_train, y_test for the chosen split.

    random   - plain stratified split. Easy, but leaks across time so it tends
               to look better than it should.
    time     - oldest rows train, newest rows test. The honest default.
    grouped  - split on whole dates, with an embargo gap so days right next to a
               test day don't sneak into training.
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

    # drop training days that sit within embargo_days of any test day
    if embargo_days:
        test_arr = np.array(sorted(test_dates))
        embargoed = {d for d in unique
                     if any(abs((d - td) / np.timedelta64(1, "D")) <= embargo_days
                            for td in test_arr) and d not in test_dates}
    else:
        embargoed = set()

    test_mask = dates.isin(test_dates)
    train_mask = ~test_mask & ~dates.isin(embargoed)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# Models

def train_random_forest(X, y, **params):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def train_xgboost(X, y, **params):
    """Train XGBoost on the shifted 0/1/2 labels and remember the mapping."""
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
    """Train one classifier by name, with hyper-parameters from config."""
    params = dict(cfg.models.get(name, {}))
    return TRAINERS[name](X, y, **params)


def predict(model, X) -> np.ndarray:
    """Predict and always hand back -1/0/1, even for XGBoost."""
    y_pred = model.predict(X)
    if getattr(model, "label_map", None):  # XGBoost predicts in 0/1/2 space
        y_pred = pd.Series(y_pred).map(INV_LABEL_MAP).to_numpy()
    return y_pred
