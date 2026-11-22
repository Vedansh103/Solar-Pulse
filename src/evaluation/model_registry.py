"""Phase 7: Official Model Registry & Production Deployment Engine.

Serializes the production champion model (LightGBM with weather) for Plant 1 and Plant 2,
stores versioned artifacts in models/registry/, generates model cards, and provides a unified
inference API for downstream consumers (LLM Assistant in P8-P10 and Dashboard in P11).

CLI: python src/evaluation/model_registry.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tree_features import FEATURE_SETS, get_train_val_test_data, add_extended_lags
from src.evaluation.metrics import score

warnings.filterwarnings("ignore")

REGISTRY_DIR = ROOT / "models/registry"


def apply_night_zeroing(preds: np.ndarray, df_subset: pd.DataFrame) -> np.ndarray:
    """Enforces physical solar constraint: power is exactly 0.0 at night."""
    out = np.clip(preds.copy(), 0.0, None)
    if "hour" in df_subset.columns:
        night_mask = (df_subset["hour"] < 6) | (df_subset["hour"] >= 19)
        if "is_daylight" in df_subset.columns:
            night_mask = night_mask | (df_subset["is_daylight"] == 0)
        out[night_mask.values] = 0.0
    return out


# Best hyperparameters identified via Optuna 40-trial validation tuning in Phase 3
CHAMPION_HYPERPARAMS = {
    1: {
        "1h": {
            "n_estimators": 296,
            "max_depth": 3,
            "num_leaves": 31,
            "learning_rate": 0.02897,
            "subsample": 0.6047,
            "colsample_bytree": 0.9839,
            "reg_alpha": 0.00282,
            "reg_lambda": 2.9079,
            "min_child_samples": 9,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        },
        "24h": {
            "n_estimators": 269,
            "max_depth": 8,
            "num_leaves": 8,
            "learning_rate": 0.02605,
            "subsample": 0.9626,
            "colsample_bytree": 0.9774,
            "reg_alpha": 0.00384,
            "reg_lambda": 0.8772,
            "min_child_samples": 22,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        },
    },
    2: {
        "1h": {
            "n_estimators": 222,
            "max_depth": 7,
            "num_leaves": 56,
            "learning_rate": 0.1693,
            "subsample": 0.8606,
            "colsample_bytree": 0.6503,
            "reg_alpha": 0.0207,
            "reg_lambda": 0.00527,
            "min_child_samples": 24,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        },
        "24h": {
            "n_estimators": 108,
            "max_depth": 8,
            "num_leaves": 30,
            "learning_rate": 0.0384,
            "subsample": 0.7581,
            "colsample_bytree": 0.7153,
            "reg_alpha": 0.0463,
            "reg_lambda": 0.00429,
            "min_child_samples": 17,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        },
    },
}


def train_and_register_champion_models() -> dict:
    """Trains production LightGBM models on full training data (Train + Val) and serializes them."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    registry_manifest = {}

    for plant_id in [1, 2]:
        raw_feat_path = ROOT / f"data/features/features_plant{plant_id}.csv"
        df_raw = pd.read_csv(raw_feat_path, index_col="timestamp", parse_dates=True)

        for horizon in ["1h", "24h"]:
            feat_set_key = f"{horizon}_weather"
            params = CHAMPION_HYPERPARAMS[plant_id][horizon]

            # Fit on combined train + val split (648 hours continuous)
            X_tr, y_tr, _, _, X_te, y_te = get_train_val_test_data(
                df_raw, feat_set_key, include_val_in_train=True
            )

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)

            # Test evaluation check
            raw_preds = model.predict(X_te)
            test_sub = df_raw.loc[X_te.index]
            final_preds = apply_night_zeroing(raw_preds, test_sub)

            eval_mask = (test_sub["ac_imputed"] == 0).values
            rmse_test, mae_test = score(y_te, pd.Series(final_preds, index=X_te.index), eval_mask)

            # Baseline reference for skill score
            baseline_val = 132.616 if plant_id == 1 else 107.658
            skill = 1.0 - (rmse_test / baseline_val)

            # Artifact paths
            model_filename = f"lightgbm_{horizon}_plant{plant_id}.joblib"
            model_path = REGISTRY_DIR / model_filename
            joblib.dump(model, model_path)

            key = f"plant{plant_id}_{horizon}"
            registry_manifest[key] = {
                "plant": plant_id,
                "horizon": horizon,
                "model_family": "LightGBM",
                "artifact_file": model_filename,
                "feature_set": feat_set_key,
                "feature_names": FEATURE_SETS[feat_set_key],
                "num_features": len(FEATURE_SETS[feat_set_key]),
                "training_samples": len(X_tr),
                "test_rmse_kw": round(float(rmse_test), 3),
                "test_mae_kw": round(float(mae_test), 3),
                "skill_score": round(float(skill), 3),
                "hyperparameters": params,
            }

            print(f"Registered: {model_filename} | Test RMSE: {rmse_test:.2f} kW | Skill: {skill:.3f}")

    # Generate Model Card
    model_card = {
        "model_name": "Solar-Pulse Production Forecaster",
        "version": "1.0.0",
        "registry_status": "production_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_champion_architecture": "LightGBM (Gradient Boosted Trees)",
        "probabilistic_companion_architecture": "Quantile LSTM (P10/P50/P90)",
        "target_variable": "ac_power_kw",
        "target_description": "Mean AC power output per reporting inverter in kW",
        "capacity_scaling_ratio_p1_to_p2": 0.802,
        "physical_postprocessing": {
            "night_zeroing": True,
            "night_definition": "hour < 6 or hour >= 19 or is_daylight == 0",
            "lower_bound_clip": 0.0,
        },
        "registered_models": registry_manifest,
    }

    card_path = REGISTRY_DIR / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)

    print(f"--> Saved Model Card to: {card_path}")
    return model_card


def load_registered_model(plant: int = 1, horizon: str = "1h") -> tuple[lgb.LGBMRegressor, dict]:
    """Loads a serialized model and its metadata from the registry."""
    card_path = REGISTRY_DIR / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found at {card_path}. Run model_registry.py first.")

    with open(card_path, "r") as f:
        card = json.load(f)

    key = f"plant{plant}_{horizon}"
    if key not in card["registered_models"]:
        raise KeyError(f"Model key '{key}' not in registered models: {list(card['registered_models'].keys())}")

    meta = card["registered_models"][key]
    model_file = REGISTRY_DIR / meta["artifact_file"]
    model = joblib.load(model_file)

    return model, meta


def predict_operational(
    model: lgb.LGBMRegressor,
    features_df: pd.DataFrame,
    feature_names: list[str] = None,
) -> np.ndarray:
    """Generates physically constrained predictions given input features.
    
    Automatically computes extended lags if needed and enforces physical night zeroing.
    """
    df = features_df.copy()
    if feature_names is not None:
        missing = [c for c in feature_names if c not in df.columns]
        if missing and "ac_power_kw" in df.columns:
            df = add_extended_lags(df)
        X = df[feature_names].astype(float)
    else:
        X = df.astype(float)

    raw_preds = model.predict(X)
    return apply_night_zeroing(raw_preds, df)


if __name__ == "__main__":
    train_and_register_champion_models()
