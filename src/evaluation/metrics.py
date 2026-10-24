"""Shared evaluation metrics for Solar-Pulse (Phase 2 & Phase 3).

Standard scoring logic:
- RMSE and MAE computed over all hours and daylight hours (is_daylight == 1).
- Strictly evaluates real actuals only (ac_imputed == 0).
- Predictions are clipped at 0.0 before scoring.
- Skill score: 1 - RMSE / RMSE_seasonal_naive at the corresponding horizon.
"""

import numpy as np
import pandas as pd


def score(actual: pd.Series, pred: pd.Series, mask: pd.Series | np.ndarray) -> tuple[float, float]:
    """Computes RMSE and MAE between actual and pred over masked rows.
    
    Negative predictions should be clipped before passing or are evaluated directly on mask.
    """
    actual_sub = actual[mask]
    pred_sub = pred[mask]
    e = actual_sub - pred_sub
    if len(e) == 0:
        return 0.0, 0.0
    rmse = float(np.sqrt((e ** 2).mean()))
    mae = float(e.abs().mean())
    return rmse, mae


def metrics_table(
    df: pd.DataFrame,
    eval_idx: pd.DatetimeIndex,
    forecasts: dict[str, tuple[pd.Series | None, pd.Series | None]],
    target_col: str = "ac_power_kw",
) -> pd.DataFrame:
    """Builds a standardized metrics table across models and horizons.
    
    forecasts: dict mapping model_name -> (forecast_1h, forecast_24h)
    """
    actual = df.loc[eval_idx, target_col]
    real = df.loc[eval_idx, "ac_imputed"] == 0
    day = real & (df.loc[eval_idx, "is_daylight"] == 1)

    rows = []
    for name, (h1, h24) in forecasts.items():
        for horizon, pred in (("1h", h1), ("24h", h24)):
            if pred is None:
                continue
            pred = pred.clip(lower=0.0)
            rmse, mae = score(actual, pred, real)
            rmse_d, mae_d = score(actual, pred, day)
            rows.append({
                "model": name,
                "horizon": horizon,
                "rmse_all": rmse,
                "mae_all": mae,
                "rmse_daylight": rmse_d,
                "mae_daylight": mae_d,
            })

    t = pd.DataFrame(rows)
    if t.empty:
        return t

    ref = t[t["model"] == "Seasonal naive (t-24)"].set_index("horizon")
    if not ref.empty:
        t["skill_vs_seasonal_naive"] = 1 - t["rmse_all"] / t["horizon"].map(ref["rmse_all"])
    else:
        t["skill_vs_seasonal_naive"] = np.nan

    return t.round(3)
