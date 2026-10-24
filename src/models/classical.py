"""Phase 2: baselines and classical time-series models.

Reads data/features/features_plantX.csv and uses the split column already in it:
train 15 May to 7 June (fit), val 8 to 10 June (choose settings), test 11 to 17 June (used once).

Two horizons, both evaluated on the same hours:
  1h  : one-step-ahead. Parameters stay fixed; the model state is updated with each new actual hour.
  24h : day-ahead. From each midnight, one forecast of the next 24 hours using data up to 23:00 the day before.

Models: persistence (t-1), seasonal naive (t-24), Holt-Winters (24 h additive season),
SARIMA (p,d,q)(P,D,Q)24, SARIMAX with irradiation and module temperature as inputs.
SARIMAX uses measured weather at the forecast hour, so its scores are an upper bound
(in real use these inputs would come from a weather forecast).

Metrics: RMSE and MAE in kW over all hours and daylight hours, plus skill against seasonal naive.
Only hours with a real (not imputed) actual value are scored.

Outputs: reports/phase2/plantX/*.csv and summary.md, figures/phase2/plantX/*.png
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", InterpolationWarning)
warnings.filterwarnings("ignore", message=".*(frequency|Non-invertible|Non-stationary|Maximum Likelihood).*")

ROOT = Path(__file__).resolve().parents[2]
TARGET = "ac_power_kw"
EXOG = ["irradiation", "module_temperature"]
S = 24  # daily season in hours

# Candidate settings, chosen on the validation days (mean of 1h and 24h RMSE)
HW_GRID = [
    {"trend": None, "damped_trend": False},
    {"trend": "add", "damped_trend": True},
]
SARIMA_GRID = [
    ((1, 0, 1), (0, 1, 1)),
    ((2, 0, 1), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((2, 0, 0), (0, 1, 1)),
]


# ---------------------------------------------------------------- data
def load(plant):
    path = ROOT / "data" / "features" / f"features_plant{plant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/features/build.py first.")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True).asfreq("h")
    assert len(df) == 816 and df[TARGET].notna().all()
    return df


def period(df, name):
    return df.index[df["split"] == name]


# ---------------------------------------------------------------- forecasters
# Each returns (one_step, day_ahead): two Series indexed by the evaluation hours.
# fit_end is the last hour the parameters may see; eval_idx starts the hour after it.

def naive_forecasts(y, eval_idx):
    return {
        "Persistence (t-1)": (y.shift(1).loc[eval_idx], None),
        "Seasonal naive (t-24)": (y.shift(S).loc[eval_idx], y.shift(S).loc[eval_idx]),
    }


def _hw_known(y, cfg, p):
    """Holt-Winters on y with parameters and initial states fixed to a previous fit."""
    trend = cfg["trend"]
    model = ExponentialSmoothing(
        y, trend=trend, damped_trend=cfg["damped_trend"], seasonal="add", seasonal_periods=S,
        initialization_method="known", initial_level=p["initial_level"],
        initial_trend=p["initial_trend"] if trend else None, initial_seasonal=p["initial_seasons"],
    )
    return model.fit(
        smoothing_level=p["smoothing_level"],
        smoothing_trend=p["smoothing_trend"] if trend else None,
        smoothing_seasonal=p["smoothing_seasonal"],
        damping_trend=p["damping_trend"] if cfg["damped_trend"] else None,
        optimized=False,
    )


def holt_winters(y, fit_end, eval_idx, cfg):
    res = ExponentialSmoothing(
        y.loc[:fit_end], trend=cfg["trend"], damped_trend=cfg["damped_trend"],
        seasonal="add", seasonal_periods=S, initialization_method="estimated",
    ).fit()
    p = res.params
    # Fitted values of a fixed-parameter filter are one-step-ahead forecasts
    one_step = _hw_known(y.loc[:eval_idx[-1]], cfg, p).fittedvalues.loc[eval_idx]
    day_ahead = []
    for origin in eval_idx[eval_idx.hour == 0]:
        hist = y.loc[:origin - pd.Timedelta(hours=1)]
        fc = _hw_known(hist, cfg, p).forecast(S)
        fc.index = pd.date_range(origin, periods=S, freq="h")
        day_ahead.append(fc)
    return one_step, pd.concat(day_ahead).loc[eval_idx], res


def sarimax(y, X, fit_end, eval_idx, order, sorder):
    res = SARIMAX(
        y.loc[:fit_end], exog=None if X is None else X.loc[:fit_end],
        order=order, seasonal_order=sorder + (S,),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    # Extend the state with the evaluation hours, parameters unchanged
    full = res.append(y.loc[eval_idx], exog=None if X is None else X.loc[eval_idx], refit=False)
    one_step = full.get_prediction(start=eval_idx[0], end=eval_idx[-1]).predicted_mean
    day_ahead = []
    for origin in eval_idx[eval_idx.hour == 0]:
        end = origin + pd.Timedelta(hours=S - 1)
        day_ahead.append(full.get_prediction(start=origin, end=end, dynamic=True).predicted_mean)
    return one_step, pd.concat(day_ahead).loc[eval_idx], res


# ---------------------------------------------------------------- metrics
def score(actual, pred, mask):
    e = (actual - pred)[mask]
    return float(np.sqrt((e ** 2).mean())), float(e.abs().mean())


def metrics_table(df, eval_idx, forecasts):
    actual = df.loc[eval_idx, TARGET]
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
            rows.append({"model": name, "horizon": horizon, "rmse_all": rmse, "mae_all": mae,
                         "rmse_daylight": rmse_d, "mae_daylight": mae_d})
    t = pd.DataFrame(rows)
    ref = t[t["model"] == "Seasonal naive (t-24)"].set_index("horizon")
    t["skill_vs_seasonal_naive"] = 1 - t["rmse_all"] / t["horizon"].map(ref["rmse_all"])
    return t.round(3)


# ---------------------------------------------------------------- diagnostics
def stationarity(y_train):
    rows = []
    for label, s in (("raw", y_train), ("seasonal diff (lag 24)", y_train.diff(S).dropna())):
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
        rows.append({
            "series": label, "adf_stat": adf_stat, "adf_p": adf_p, "kpss_stat": kpss_stat, "kpss_p": kpss_p,
            "verdict": "stationary" if adf_p < 0.05 and kpss_p > 0.05 else
                       "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else "mixed (tests disagree)",
        })
    return pd.DataFrame(rows).round(4)


def plot_acf_pacf(y_train, path):
    d = y_train.diff(S).dropna()
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    plot_acf(y_train, lags=72, ax=ax[0, 0], title="ACF, raw")
    plot_pacf(y_train, lags=72, ax=ax[0, 1], title="PACF, raw", method="ywm")
    plot_acf(d, lags=72, ax=ax[1, 0], title="ACF, seasonal diff (lag 24)")
    plot_pacf(d, lags=72, ax=ax[1, 1], title="PACF, seasonal diff (lag 24)", method="ywm")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ljung_box(fitted):
    rows = []
    for name, res in fitted.items():
        resid = pd.Series(res.resid).iloc[2 * S:]  # skip the start-up period
        lb = acorr_ljungbox(resid, lags=[S, 2 * S])
        for lag, r in lb.iterrows():
            rows.append({"model": name, "lag": lag, "lb_stat": r["lb_stat"], "lb_p": r["lb_pvalue"],
                         "white_noise": bool(r["lb_pvalue"] > 0.05)})
    return pd.DataFrame(rows).round(4)


def plot_forecasts(df, eval_idx, forecasts, horizon, path):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(eval_idx, df.loc[eval_idx, TARGET], color="black", lw=2, label="Actual")
    for name, (h1, h24) in forecasts.items():
        pred = h1 if horizon == "1h" else h24
        if pred is not None:
            ax.plot(eval_idx, pred.clip(lower=0.0), lw=1, alpha=0.85, label=name)
    ax.set_title(f"Test week, {horizon} ahead forecasts vs actual")
    ax.set_ylabel("AC power per inverter (kW)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- pipeline
def select(name, grid, run, y, df, train_end, val_idx):
    """Fit each candidate on train, score on validation, return the best setting and the table."""
    rows = []
    for cfg in grid:
        t0 = time.time()
        h1, h24, _ = run(train_end, val_idx, cfg)
        m = metrics_table(df, val_idx, {"c": (h1, h24)})
        r1, r24 = m.set_index("horizon").loc[["1h", "24h"], "rmse_all"]
        rows.append({"model": name, "setting": str(cfg), "val_rmse_1h": r1, "val_rmse_24h": r24,
                     "val_rmse_mean": (r1 + r24) / 2, "seconds": round(time.time() - t0, 1)})
        print(f"  {name} {cfg}: val RMSE 1h {r1:.1f}, 24h {r24:.1f}")
    table = pd.DataFrame(rows)
    best = grid[int(table["val_rmse_mean"].idxmin())]
    return best, table


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plant", type=int, default=1, choices=[1, 2])
    plant = ap.parse_args().plant

    df = load(plant)
    y = df[TARGET]
    X = df[EXOG]
    train_idx, val_idx, test_idx = period(df, "train"), period(df, "val"), period(df, "test")
    train_end, val_end = train_idx[-1], val_idx[-1]

    rep_dir = ROOT / "reports" / "phase2" / f"plant{plant}"
    fig_dir = ROOT / "figures" / "phase2" / f"plant{plant}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plant {plant}: train {len(train_idx)} h, val {len(val_idx)} h, test {len(test_idx)} h")

    # 1. Diagnostics on the training period only
    stat = stationarity(y.loc[train_idx])
    stat.to_csv(rep_dir / "stationarity.csv", index=False)
    plot_acf_pacf(y.loc[train_idx], fig_dir / "acf_pacf.png")
    print(stat.to_string(index=False))

    # 2. Model selection on the validation days
    runners = {
        "Holt-Winters": (HW_GRID, lambda end, idx, cfg: holt_winters(y, end, idx, cfg)),
        "SARIMA": (SARIMA_GRID, lambda end, idx, cfg: sarimax(y, None, end, idx, *cfg)),
        "SARIMAX (irradiation, module temp)": (SARIMA_GRID, lambda end, idx, cfg: sarimax(y, X, end, idx, *cfg)),
    }
    best, sel_tables = {}, []
    for name, (grid, run) in runners.items():
        best[name], t = select(name, grid, run, y, df, train_end, val_idx)
        sel_tables.append(t)
    selection = pd.concat(sel_tables, ignore_index=True).round(3)
    selection.to_csv(rep_dir / "model_selection_val.csv", index=False)

    # 3. Refit chosen settings on train + val, forecast the test week once
    forecasts = naive_forecasts(y, test_idx)
    fitted = {}
    for name, (_, run) in runners.items():
        h1, h24, res = run(val_end, test_idx, best[name])
        forecasts[f"{name} {best[name]}"] = (h1, h24)
        fitted[name] = res

    metrics = metrics_table(df, test_idx, forecasts)
    metrics.to_csv(rep_dir / "metrics_test.csv", index=False)
    lb = ljung_box(fitted)
    lb.to_csv(rep_dir / "ljung_box.csv", index=False)

    out = pd.DataFrame({"actual": y.loc[test_idx], "is_daylight": df.loc[test_idx, "is_daylight"]})
    for name, (h1, h24) in forecasts.items():
        out[f"{name} | 1h"] = h1.clip(lower=0.0)
        if h24 is not None:
            out[f"{name} | 24h"] = h24.clip(lower=0.0)
    out.round(3).to_csv(rep_dir / "forecasts_test.csv")

    for horizon in ("1h", "24h"):
        plot_forecasts(df, test_idx, forecasts, horizon, fig_dir / f"test_forecasts_{horizon}.png")

    write_summary(rep_dir / "summary.md", plant, stat, selection, best, metrics, lb)
    print("\nTest metrics (kW):")
    print(metrics.to_string(index=False))
    print(f"\nWritten: {rep_dir.relative_to(ROOT)} and {fig_dir.relative_to(ROOT)}")


def md_table(t):
    head = "| " + " | ".join(map(str, t.columns)) + " |"
    sep = "|" + "---|" * len(t.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write_summary(path, plant, stat, selection, best, metrics, lb):
    L = [f"# Phase 2 results, Plant {plant}", "",
         "Train 15 May to 7 June (576 h), settings chosen on 8 to 10 June (72 h), scored once on the test week "
         "11 to 17 June (168 h). RMSE and MAE in kW per reporting inverter. Skill = 1 - RMSE / RMSE of seasonal naive "
         "at the same horizon (above 0 means better than the baseline).", "",
         "SARIMAX uses measured irradiation and module temperature at the forecast hour, so its scores are an "
         "upper bound; in operation these inputs would come from a weather forecast.", "",
         "## Test metrics", "", md_table(metrics), "",
         "## Stationarity (training period)", "", md_table(stat), "",
         "## Settings chosen on validation", ""]
    L += [f"- {k}: {v}" for k, v in best.items()]
    L += ["", md_table(selection), "",
          "## Ljung-Box on in-sample residuals (lags 24 and 48)", "", md_table(lb), ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
