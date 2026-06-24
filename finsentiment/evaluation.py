"""Predictive metrics, reporting, and the Sharpe-ratio backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modeling import predict


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test, *, feature_names=None, target_name="", verbose=True) -> dict:
    """Compute accuracy + macro-F1 (and optionally print a full report)."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    y_pred = predict(model, X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    if verbose:
        print(f"\nResults for {target_name}:")
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1m:.4f}")
        print("\nClassification report:\n",
              classification_report(y_test, y_pred, digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]))
        if feature_names is not None and hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            print("\nTop features:\n", imp.sort_values(ascending=False).head(15))

    return {"target": target_name, "accuracy": acc, "macro_f1": f1m}


# --------------------------------------------------------------------------- #
# Economic evaluation
# --------------------------------------------------------------------------- #
def backtest_sharpe(model, X_test, y_test, df_test, *, return_col="return_1d",
                    target_name="") -> dict:
    """Long/short backtest from predicted signals; returns Sharpe + summary.

    Strategy: position = predicted trigger in {-1,0,1}; period return =
    position * realised ``return_col``. Sharpe is the per-period mean/std of
    strategy returns (un-annualised, no transaction costs — see docs).
    """
    y_pred = predict(model, X_test)
    res = df_test.loc[y_test.index].copy()
    res["y_pred"] = y_pred
    res["strategy_return"] = res["y_pred"] * res[return_col]
    res["cumulative_return"] = (1 + res["strategy_return"]).cumprod()

    r = res["strategy_return"].dropna()
    sharpe = r.mean() / r.std() if r.std() else float("nan")
    summary = {
        "target": target_name,
        "sharpe": float(sharpe),
        "total_return_pct": float((res["cumulative_return"].iloc[-1] - 1) * 100) if len(res) else float("nan"),
        "avg_daily_return": float(r.mean()),
        "volatility": float(r.std()),
    }
    print(f"[{target_name}] Sharpe={summary['sharpe']:.4f} "
          f"TotalReturn={summary['total_return_pct']:.2f}%")
    return summary
