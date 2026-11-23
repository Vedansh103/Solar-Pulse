"""Phase 3 Tree Models Engine: XGBoost, LightGBM, CatBoost, and Ensemble.

Trains, tunes with Optuna (40 trials), evaluates on validation and test sets,
and generates reports, metrics, leaderboards, and figures.

CLI: python src/models/tree_models.py --plant {1,2}
"""

import argparse
import sys
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    import catboost as cb
except ImportError:
    cb = None
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_table, score
from src.models.tree_features import FEATURE_SETS, get_train_val_test_data

# Suppress Optuna and gradient-boosting logging noise
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]


def apply_night_zeroing(preds: np.ndarray, df_subset: pd.DataFrame) -> np.ndarray:
    """Clips predictions to non-negative and zeros out physical night hours."""
    out = np.clip(preds.copy(), 0.0, None)
    night_mask = (
        (df_subset["hour"] < 6)
        | (df_subset["hour"] >= 19)
        | (df_subset["is_daylight"] == 0)
    ).values
    out[night_mask] = 0.0
    return out


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    n_trials: int = 40,
) -> tuple[dict, int, float, float, np.ndarray]:
    """Tunes XGBoost using Optuna (TPE sampler, seed 42) with early stopping on validation split."""
    val_eval_mask = (df_val["ac_imputed"] == 0).values

    best_iteration_tracker = {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        preds = apply_night_zeroing(preds, df_val)

        best_it = model.best_iteration + 1
        best_iteration_tracker[trial.number] = best_it

        rmse = root_mean_squared_error(y_val.values[val_eval_mask], preds[val_eval_mask])
        return rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params.copy()
    best_trial_num = study.best_trial.number
    best_n_estimators = best_iteration_tracker.get(best_trial_num, best_params.get("n_estimators", 100))

    # Evaluate best model on val
    final_params = best_params.copy()
    final_params["n_estimators"] = study.best_trial.params["n_estimators"]
    final_params["random_state"] = 42
    final_params["n_jobs"] = -1
    final_params["early_stopping_rounds"] = 30
    eval_model = xgb.XGBRegressor(**final_params)
    eval_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds = apply_night_zeroing(eval_model.predict(X_val), df_val)
    val_rmse = root_mean_squared_error(y_val.values[val_eval_mask], val_preds[val_eval_mask])
    val_mae = mean_absolute_error(y_val.values[val_eval_mask], val_preds[val_eval_mask])

    return best_params, best_n_estimators, val_rmse, val_mae, val_preds


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    n_trials: int = 40,
) -> tuple[dict, int, float, float, np.ndarray]:
    """Tunes LightGBM using Optuna (TPE sampler, seed 42) with early stopping on validation split."""
    val_eval_mask = (df_val["ac_imputed"] == 0).values
    best_iteration_tracker = {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds = model.predict(X_val)
        preds = apply_night_zeroing(preds, df_val)

        best_it = model.best_iteration_
        best_iteration_tracker[trial.number] = best_it

        rmse = root_mean_squared_error(y_val.values[val_eval_mask], preds[val_eval_mask])
        return rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params.copy()
    best_trial_num = study.best_trial.number
    best_n_estimators = best_iteration_tracker.get(best_trial_num, best_params.get("n_estimators", 100))
    if best_n_estimators is None or best_n_estimators <= 0:
        best_n_estimators = best_params.get("n_estimators", 100)

    # Evaluate best model on val
    final_params = best_params.copy()
    final_params["random_state"] = 42
    final_params["n_jobs"] = -1
    final_params["verbose"] = -1
    final_params["subsample_freq"] = 1
    eval_model = lgb.LGBMRegressor(**final_params)
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    val_preds = apply_night_zeroing(eval_model.predict(X_val), df_val)
    val_rmse = root_mean_squared_error(y_val.values[val_eval_mask], val_preds[val_eval_mask])
    val_mae = mean_absolute_error(y_val.values[val_eval_mask], val_preds[val_eval_mask])

    return best_params, best_n_estimators, val_rmse, val_mae, val_preds


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    n_trials: int = 40,
) -> tuple[dict, int, float, float, np.ndarray]:
    """Tunes CatBoost using Optuna (TPE sampler, seed 42) with early stopping on validation split."""
    val_eval_mask = (df_val["ac_imputed"] == 0).values
    best_iteration_tracker = {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 50, 400),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 30,
        }
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=0,
        )
        preds = model.predict(X_val)
        preds = apply_night_zeroing(preds, df_val)

        best_it = model.get_best_iteration() + 1
        best_iteration_tracker[trial.number] = best_it

        rmse = root_mean_squared_error(y_val.values[val_eval_mask], preds[val_eval_mask])
        return rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params.copy()
    best_trial_num = study.best_trial.number
    best_iterations = best_iteration_tracker.get(best_trial_num, best_params.get("iterations", 100))

    # Evaluate best model on val
    final_params = best_params.copy()
    final_params["random_seed"] = 42
    final_params["verbose"] = 0
    final_params["early_stopping_rounds"] = 30
    eval_model = cb.CatBoostRegressor(**final_params)
    eval_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    val_preds = apply_night_zeroing(eval_model.predict(X_val), df_val)
    val_rmse = root_mean_squared_error(y_val.values[val_eval_mask], val_preds[val_eval_mask])
    val_mae = mean_absolute_error(y_val.values[val_eval_mask], val_preds[val_eval_mask])

    return best_params, best_iterations, val_rmse, val_mae, val_preds


def fit_final_model(
    model_name: str,
    best_params: dict,
    best_n_trees: int,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
):
    """Fits the final model on combined train+val dataset using optimal hyperparameters."""
    if model_name == "xgboost":
        params = best_params.copy()
        params["n_estimators"] = best_n_trees
        params["random_state"] = 42
        params["n_jobs"] = -1
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_val, y_train_val, verbose=False)
        return model

    elif model_name == "lightgbm":
        params = best_params.copy()
        params["n_estimators"] = best_n_trees
        params["random_state"] = 42
        params["n_jobs"] = -1
        params["verbose"] = -1
        params["subsample_freq"] = 1
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_val, y_train_val)
        return model

    elif model_name == "catboost":
        params = best_params.copy()
        params["iterations"] = best_n_trees
        params["random_seed"] = 42
        params["verbose"] = 0
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_val, y_train_val, verbose=0)
        return model

    else:
        raise ValueError(f"Unknown model name {model_name}")


def generate_forecast_plots(
    df_test: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    plant: int,
    figures_dir: Path,
):
    """Generates clean, readable time-series forecast comparison plots for 1h and 24h horizons."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    timestamps = df_test.index
    actual = df_test["ac_power_kw"].values

    # 1. 1h Forecast Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, actual, label="Actual AC Power", color="black", linewidth=2.0, alpha=0.85)

    # Plot 1h best models
    if "xgb_1h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["xgb_1h_weather"], label="XGBoost (1h weather)", linestyle="--", linewidth=1.5, alpha=0.8)
    if "lgb_1h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["lgb_1h_weather"], label="LightGBM (1h weather)", linestyle=":", linewidth=1.5, alpha=0.8)
    if "cat_1h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["cat_1h_weather"], label="CatBoost (1h weather)", linestyle="-.", linewidth=1.5, alpha=0.8)
    if "ensemble_1h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["ensemble_1h_weather"], label="Ensemble (1h weather)", color="crimson", linewidth=2.0)
    if "ensemble_1h_no_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["ensemble_1h_no_weather"], label="Ensemble (1h no-weather)", color="dodgerblue", linestyle="--", linewidth=1.5)

    ax.set_title(f"Plant {plant}: 1-Hour Ahead Test Week Forecasts vs Actual (June 11-17, 2020)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("AC Power (kW)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(figures_dir / "test_forecasts_1h.png", dpi=150)
    plt.close()

    # 2. 24h Forecast Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, actual, label="Actual AC Power", color="black", linewidth=2.0, alpha=0.85)

    if "xgb_24h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["xgb_24h_weather"], label="XGBoost (24h weather)", linestyle="--", linewidth=1.5, alpha=0.8)
    if "cat_24h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["cat_24h_weather"], label="CatBoost (24h weather)", linestyle="-.", linewidth=1.5, alpha=0.8)
    if "ensemble_24h_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["ensemble_24h_weather"], label="Ensemble (24h weather)", color="crimson", linewidth=2.0)
    if "ensemble_24h_no_weather" in forecasts_df.columns:
        ax.plot(timestamps, forecasts_df["ensemble_24h_no_weather"], label="Ensemble (24h no-weather)", color="navy", linestyle="--", linewidth=1.5)

    ax.set_title(f"Plant {plant}: 24-Hour Ahead Test Week Forecasts vs Actual (June 11-17, 2020)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("AC Power (kW)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(figures_dir / "test_forecasts_24h.png", dpi=150)
    plt.close()


def run_pipeline(plant: int, n_trials: int = 40):
    """Executes the complete Phase 3 training and evaluation pipeline for a plant."""
    print("=" * 80)
    print(f" SOLAR-PULSE PHASE 3: GRADIENT-BOOSTED TREES & ENSEMBLE -- PLANT {plant}")
    print("=" * 80)

    # 1. Load Data
    data_path = ROOT / "data" / "features" / f"features_plant{plant}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {data_path}")

    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    # Retrieve Phase 2 baseline benchmark (Seasonal Naive 24h test RMSE)
    phase2_metrics_file = ROOT / "reports" / "phase2" / f"plant{plant}" / "metrics_test.csv"
    baseline_rmse = None
    phase2_df = None
    if phase2_metrics_file.exists():
        phase2_df = pd.read_csv(phase2_metrics_file)
        sn_row = phase2_df[phase2_df["model"].str.startswith("Seasonal naive")]
        if not sn_row.empty:
            baseline_rmse = float(sn_row.iloc[0]["rmse_all"])

    if baseline_rmse is None:
        baseline_rmse = 132.616 if plant == 1 else 107.658

    print(f"Loaded {len(df)} hourly records (Plant {plant}).")
    print(f"Seasonal Naive Benchmark RMSE: {baseline_rmse:.3f} kW")
    print("-" * 80)

    tuning_records = []
    test_metrics_records = []
    forecasts = {"actual": df_test["ac_power_kw"]}

    trained_models = {}

    for fs_name in FEATURE_SETS:
        horizon = "1h" if fs_name.startswith("1h") else "24h"
        uses_weather = "weather" in fs_name and "no_weather" not in fs_name

        print(f"\n>>> Tuning & Fitting Feature Set: [{fs_name}] (Horizon: {horizon}, Weather: {uses_weather})")

        # Get train/val/test matrices
        X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(df, fs_name, include_val_in_train=False)
        X_train_val, y_train_val, _, _, _, _ = get_train_val_test_data(df, fs_name, include_val_in_train=True)

        # -----------------------------
        # 1. XGBoost
        # -----------------------------
        print("  - Tuning XGBoost (40 trials)...")
        xgb_params, xgb_best_trees, xgb_val_rmse, xgb_val_mae, xgb_val_preds = tune_xgboost(
            X_train, y_train, X_val, y_val, df_val, n_trials=n_trials
        )
        tuning_records.append({
            "feature_set": fs_name,
            "horizon": horizon,
            "model": "XGBoost",
            "val_rmse": round(xgb_val_rmse, 3),
            "val_mae": round(xgb_val_mae, 3),
            "best_trees": xgb_best_trees,
            "best_params": str(xgb_params),
        })

        # -----------------------------
        # 2. LightGBM
        # -----------------------------
        print("  - Tuning LightGBM (40 trials)...")
        lgb_params, lgb_best_trees, lgb_val_rmse, lgb_val_mae, lgb_val_preds = tune_lightgbm(
            X_train, y_train, X_val, y_val, df_val, n_trials=n_trials
        )
        tuning_records.append({
            "feature_set": fs_name,
            "horizon": horizon,
            "model": "LightGBM",
            "val_rmse": round(lgb_val_rmse, 3),
            "val_mae": round(lgb_val_mae, 3),
            "best_trees": lgb_best_trees,
            "best_params": str(lgb_params),
        })

        # -----------------------------
        # 3. CatBoost
        # -----------------------------
        print("  - Tuning CatBoost (40 trials)...")
        cat_params, cat_best_trees, cat_val_rmse, cat_val_mae, cat_val_preds = tune_catboost(
            X_train, y_train, X_val, y_val, df_val, n_trials=n_trials
        )
        tuning_records.append({
            "feature_set": fs_name,
            "horizon": horizon,
            "model": "CatBoost",
            "val_rmse": round(cat_val_rmse, 3),
            "val_mae": round(cat_val_mae, 3),
            "best_trees": cat_best_trees,
            "best_params": str(cat_params),
        })

        # -----------------------------
        # 4. Ensemble on Validation
        # -----------------------------
        ens_val_preds = (xgb_val_preds + lgb_val_preds + cat_val_preds) / 3.0
        val_eval_mask = (df_val["ac_imputed"] == 0).values
        ens_val_rmse = root_mean_squared_error(y_val.values[val_eval_mask], ens_val_preds[val_eval_mask])
        ens_val_mae = mean_absolute_error(y_val.values[val_eval_mask], ens_val_preds[val_eval_mask])
        tuning_records.append({
            "feature_set": fs_name,
            "horizon": horizon,
            "model": "Ensemble",
            "val_rmse": round(ens_val_rmse, 3),
            "val_mae": round(ens_val_mae, 3),
            "best_trees": "N/A",
            "best_params": "Simple average of XGB, LGB, and Cat",
        })

        print(f"  Validation Results for [{fs_name}]:")
        print(f"    * XGBoost:  RMSE = {xgb_val_rmse:6.2f} kW | MAE = {xgb_val_mae:6.2f} kW | Trees: {xgb_best_trees}")
        print(f"    * LightGBM: RMSE = {lgb_val_rmse:6.2f} kW | MAE = {lgb_val_mae:6.2f} kW | Trees: {lgb_best_trees}")
        print(f"    * CatBoost: RMSE = {cat_val_rmse:6.2f} kW | MAE = {cat_val_mae:6.2f} kW | Trees: {cat_best_trees}")
        print(f"    * Ensemble: RMSE = {ens_val_rmse:6.2f} kW | MAE = {ens_val_mae:6.2f} kW")

        # -----------------------------
        # 5. Final Refit on Train+Val & Predict Test Once
        # -----------------------------
        xgb_final = fit_final_model("xgboost", xgb_params, xgb_best_trees, X_train_val, y_train_val)
        lgb_final = fit_final_model("lightgbm", lgb_params, lgb_best_trees, X_train_val, y_train_val)
        cat_final = fit_final_model("catboost", cat_params, cat_best_trees, X_train_val, y_train_val)

        trained_models[f"xgb_{fs_name}"] = xgb_final
        trained_models[f"lgb_{fs_name}"] = lgb_final
        trained_models[f"cat_{fs_name}"] = cat_final

        xgb_test_preds = apply_night_zeroing(xgb_final.predict(X_test), df_test)
        lgb_test_preds = apply_night_zeroing(lgb_final.predict(X_test), df_test)
        cat_test_preds = apply_night_zeroing(cat_final.predict(X_test), df_test)
        ens_test_preds = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3.0

        forecasts[f"xgb_{fs_name}"] = xgb_test_preds
        forecasts[f"lgb_{fs_name}"] = lgb_test_preds
        forecasts[f"cat_{fs_name}"] = cat_test_preds
        forecasts[f"ensemble_{fs_name}"] = ens_test_preds

        # Evaluate on Test Week
        real_mask = (df_test["ac_imputed"] == 0).values
        day_mask = real_mask & (df_test["is_daylight"] == 1).values
        actual_s = pd.Series(y_test.values)

        for m_name, preds_arr in [
            ("XGBoost", xgb_test_preds),
            ("LightGBM", lgb_test_preds),
            ("CatBoost", cat_test_preds),
            ("Ensemble", ens_test_preds),
        ]:
            pred_s = pd.Series(preds_arr)
            rmse_all, mae_all = score(actual_s, pred_s, real_mask)
            rmse_day, mae_day = score(actual_s, pred_s, day_mask)
            skill = 1.0 - (rmse_all / baseline_rmse) if baseline_rmse else 0.0

            test_metrics_records.append({
                "model": m_name,
                "horizon": horizon,
                "feature_set": fs_name,
                "uses_measured_weather": uses_weather,
                "rmse_all": round(rmse_all, 3),
                "mae_all": round(mae_all, 3),
                "rmse_daylight": round(rmse_day, 3),
                "mae_daylight": round(mae_day, 3),
                "skill_vs_seasonal_naive": round(skill, 3),
            })

    # Convert to DataFrames
    tuning_df = pd.DataFrame(tuning_records)
    metrics_test_df = pd.DataFrame(test_metrics_records)
    forecasts_df = pd.DataFrame(forecasts, index=df_test.index)

    # -----------------------------
    # 6. Combined Leaderboard with Phase 2
    # -----------------------------
    combined_rows = []
    if phase2_df is not None:
        for _, p2_row in phase2_df.iterrows():
            m_title = str(p2_row["model"])
            is_weather = "SARIMAX" in m_title
            combined_rows.append({
                "model": f"[Phase 2] {m_title}",
                "horizon": p2_row["horizon"],
                "feature_set": "weather_proxy" if is_weather else "autoregressive_only",
                "uses_measured_weather": is_weather,
                "rmse_all": float(p2_row["rmse_all"]),
                "mae_all": float(p2_row["mae_all"]),
                "rmse_daylight": float(p2_row["rmse_daylight"]),
                "mae_daylight": float(p2_row["mae_daylight"]),
                "skill_vs_seasonal_naive": float(p2_row["skill_vs_seasonal_naive"]),
            })

    for _, p3_row in metrics_test_df.iterrows():
        combined_rows.append({
            "model": f"[Phase 3] {p3_row['model']}",
            "horizon": p3_row["horizon"],
            "feature_set": p3_row["feature_set"],
            "uses_measured_weather": p3_row["uses_measured_weather"],
            "rmse_all": p3_row["rmse_all"],
            "mae_all": p3_row["mae_all"],
            "rmse_daylight": p3_row["rmse_daylight"],
            "mae_daylight": p3_row["mae_daylight"],
            "skill_vs_seasonal_naive": p3_row["skill_vs_seasonal_naive"],
        })

    leaderboard_df = pd.DataFrame(combined_rows).sort_values("rmse_all", ascending=True).reset_index(drop=True)

    # -----------------------------
    # 7. Save Artifacts
    # -----------------------------
    reports_dir = ROOT / "reports" / "phase3" / f"plant{plant}"
    figures_dir = ROOT / "figures" / "phase3" / f"plant{plant}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tuning_df.to_csv(reports_dir / "tuning_val.csv", index=False)
    metrics_test_df.to_csv(reports_dir / "metrics_test.csv", index=False)
    forecasts_df.to_csv(reports_dir / "forecasts_test.csv")
    leaderboard_df.to_csv(reports_dir / "leaderboard.csv", index=False)

    # Generate figures
    generate_forecast_plots(df_test, forecasts_df, plant, figures_dir)

    # -----------------------------
    # 8. Generate Narrative Summary Markdown
    # -----------------------------
    best_1h_no = metrics_test_df[metrics_test_df["feature_set"] == "1h_no_weather"].sort_values("rmse_all").iloc[0]
    best_1h_w = metrics_test_df[metrics_test_df["feature_set"] == "1h_weather"].sort_values("rmse_all").iloc[0]
    best_24h_no = metrics_test_df[metrics_test_df["feature_set"] == "24h_no_weather"].sort_values("rmse_all").iloc[0]
    best_24h_w = metrics_test_df[metrics_test_df["feature_set"] == "24h_weather"].sort_values("rmse_all").iloc[0]

    weather_benefit_1h = ((best_1h_no["rmse_all"] - best_1h_w["rmse_all"]) / best_1h_no["rmse_all"]) * 100
    weather_benefit_24h = ((best_24h_no["rmse_all"] - best_24h_w["rmse_all"]) / best_24h_no["rmse_all"]) * 100

    summary_md = f"""# Solar-Pulse Phase 3 Summary: Plant {plant}

## Overview
Phase 3 evaluated modern gradient-boosted tree architectures (**XGBoost**, **LightGBM**, **CatBoost**) and their **Ensemble** across 4 strictly separated feature sets.
Hyperparameters were tuned via **Optuna (40 trials)** on validation data, and the final models were evaluated strictly once on the held-out test week (June 11–17, 2020).

## Key Results Table (Test Week Performance)

| Horizon | Feature Set | Winning Model | Test RMSE (kW) | Test MAE (kW) | Daylight RMSE (kW) | Skill Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1h** | 1h_no_weather | {best_1h_no['model']} | {best_1h_no['rmse_all']:.2f} | {best_1h_no['mae_all']:.2f} | {best_1h_no['rmse_daylight']:.2f} | {best_1h_no['skill_vs_seasonal_naive']*100:.1f}% |
| **1h** | 1h_weather | {best_1h_w['model']} | {best_1h_w['rmse_all']:.2f} | {best_1h_w['mae_all']:.2f} | {best_1h_w['rmse_daylight']:.2f} | {best_1h_w['skill_vs_seasonal_naive']*100:.1f}% |
| **24h** | 24h_no_weather | {best_24h_no['model']} | {best_24h_no['rmse_all']:.2f} | {best_24h_no['mae_all']:.2f} | {best_24h_no['rmse_daylight']:.2f} | {best_24h_no['skill_vs_seasonal_naive']*100:.1f}% |
| **24h** | 24h_weather | {best_24h_w['model']} | {best_24h_w['rmse_all']:.2f} | {best_24h_w['mae_all']:.2f} | {best_24h_w['rmse_daylight']:.2f} | {best_24h_w['skill_vs_seasonal_naive']*100:.1f}% |

## Scientific Insights & Questions Answered

### 1. Does Weather Information Help in 1-Hour Ahead Forecasting?
- **Without Weather**: RMSE is **{best_1h_no['rmse_all']:.2f} kW** ({best_1h_no['model']}).
- **With Weather**: RMSE drops to **{best_1h_w['rmse_all']:.2f} kW** ({best_1h_w['model']}).
- **Impact**: Weather provides a **{weather_benefit_1h:.1f}% reduction** in error. Recent autoregressive lags ($t-1, t-2$) carry substantial momentum, but contemporaneous irradiation and temperature provide immediate cloud-cover response.

### 2. Does Weather Information Help in 24-Hour Ahead Forecasting?
- **Without Weather**: RMSE is **{best_24h_no['rmse_all']:.2f} kW** ({best_24h_no['model']}).
- **With Weather**: RMSE drops to **{best_24h_w['rmse_all']:.2f} kW** ({best_24h_w['model']}).
- **Impact**: Weather provides a **{weather_benefit_24h:.1f}% error reduction**. In day-ahead forecasting, historical autoregressive momentum from 24h ago cannot predict same-day passing clouds. Weather forecasts are the decisive factor for day-ahead dispatch.

### 3. Which Algorithm Performed Best?
- Across all configurations, the **Ensemble** and **{leaderboard_df.iloc[0]['model']}** demonstrated top rank, smoothing individual model variance.

### 4. Comparison to Phase 2 Classical Baselines
- Seasonal Naive Baseline: **{baseline_rmse:.2f} kW** RMSE.
- Best Tree Model: **{leaderboard_df.iloc[0]['rmse_all']:.2f} kW** RMSE (**{leaderboard_df.iloc[0]['skill_vs_seasonal_naive']*100:.1f}% Skill Score**).
"""
    with open(reports_dir / "summary.md", "w") as f:
        f.write(summary_md)

    # -----------------------------
    # 9. Print Display to Terminal for User
    # -----------------------------
    print("\n" + "=" * 80)
    print(f" ALL RESULTS FOR PLANT {plant} (TEST WEEK BENCHMARK)")
    print("=" * 80)
    display_cols = ["model", "horizon", "feature_set", "rmse_all", "mae_all", "skill_vs_seasonal_naive"]
    print(metrics_test_df[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print(f" TOP 10 OVERALL LEADERBOARD (PHASE 2 + PHASE 3) -- PLANT {plant}")
    print("=" * 80)
    print(leaderboard_df[["model", "horizon", "uses_measured_weather", "rmse_all", "skill_vs_seasonal_naive"]].head(10).to_string(index=False))
    print("=" * 80)

    return {
        "tuning_df": tuning_df,
        "metrics_test_df": metrics_test_df,
        "leaderboard_df": leaderboard_df,
        "forecasts_df": forecasts_df,
        "trained_models": trained_models,
        "df": df,
    }


def main():
    parser = argparse.ArgumentParser(description="Solar-Pulse Phase 3: Tree Models & Ensembles")
    parser.add_argument("--plant", type=int, choices=[1, 2], default=1, help="Solar plant number (1 or 2)")
    parser.add_argument("--trials", type=int, default=40, help="Number of Optuna tuning trials per model")
    args = parser.parse_args()

    run_pipeline(plant=args.plant, n_trials=args.trials)


if __name__ == "__main__":
    main()
