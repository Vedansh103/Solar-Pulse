"""SHAP Explainability and Physical Attribution Analysis (Phase 3).

Computes TreeExplainer SHAP values on test set for winning validation models,
generates beeswarm plots, bar plots, and grouped attribution tables.

CLI: python src/models/shap_analysis.py --plant {1,2}
"""

import argparse
import ast
import sys
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tree_features import FEATURE_SETS, get_train_val_test_data
from src.models.tree_models import fit_final_model

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

FEATURE_GROUPS = {
    "Sunlight": [
        "irradiation",
        "ghi_clear",
        "clearness_index",
        "irradiation_lag_1",
        "irradiation_lag_24",
        "irradiation_lag_48",
        "irradiation_roll_3h_std",
    ],
    "Heat": [
        "module_temperature",
        "ambient_temperature",
        "delta_T",
    ],
    "History": [
        "ac_power_lag_1",
        "ac_power_lag_2",
        "ac_power_lag_24",
        "ac_power_lag_48",
        "ac_power_lag_72",
        "ac_power_roll_3h_mean",
        "same_hour_mean_3d",
    ],
    "Time": [
        "hour_sin",
        "hour_cos",
    ],
}


def compute_group_attribution(
    shap_values: np.ndarray,
    feature_names: list[str],
    horizon: str,
) -> pd.DataFrame:
    """Aggregates mean absolute SHAP values into physical domain groups summing to 100%."""
    mean_abs_per_feat = np.mean(np.abs(shap_values), axis=0)
    feat_to_mean = dict(zip(feature_names, mean_abs_per_feat))

    group_sums = {}
    for group_name, group_cols in FEATURE_GROUPS.items():
        total = sum(feat_to_mean.get(col, 0.0) for col in group_cols)
        group_sums[group_name] = total

    total_importance = sum(group_sums.values())
    rows = []
    for group_name, val in group_sums.items():
        pct = (val / total_importance * 100.0) if total_importance > 0 else 0.0
        rows.append({
            "horizon": horizon,
            "feature_group": group_name,
            "mean_abs_shap": round(val, 4),
            "pct_importance": round(pct, 2),
        })

    return pd.DataFrame(rows)


def run_shap_analysis(plant: int):
    """Loads tuning results, fits winning tree models, and computes SHAP explanations."""
    print("=" * 80)
    print(f" SOLAR-PULSE PHASE 3: SHAP EXPLAINABILITY ANALYSIS -- PLANT {plant}")
    print("=" * 80)

    reports_dir = ROOT / "reports" / "phase3" / f"plant{plant}"
    figures_dir = ROOT / "figures" / "phase3" / f"plant{plant}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tuning_file = reports_dir / "tuning_val.csv"
    if not tuning_file.exists():
        raise FileNotFoundError(f"Tuning file not found at {tuning_file}. Run tree_models.py first!")

    tuning_df = pd.read_csv(tuning_file)

    # Filter out 'Ensemble' since TreeExplainer inspects individual tree architectures
    tree_tuning = tuning_df[tuning_df["model"].isin(["XGBoost", "LightGBM", "CatBoost"])].copy()

    # Load plant data
    data_path = ROOT / "data" / "features" / f"features_plant{plant}.csv"
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)

    shap_group_dfs = []

    for horizon in ["1h", "24h"]:
        # Find winning tree model configuration for this horizon on validation split
        h_df = tree_tuning[tree_tuning["horizon"] == horizon].sort_values("val_rmse", ascending=True)
        winner_row = h_df.iloc[0]

        best_model_name = str(winner_row["model"]).lower()
        best_feature_set = str(winner_row["feature_set"])
        best_val_rmse = float(winner_row["val_rmse"])
        best_trees = int(winner_row["best_trees"]) if str(winner_row["best_trees"]).isdigit() else 100
        best_params = ast.literal_eval(str(winner_row["best_params"]))

        print(f"\nWinning Validation Tree Model for [{horizon}]: {winner_row['model']} on ({best_feature_set})")
        print(f"  Val RMSE: {best_val_rmse:.3f} kW | Trees: {best_trees}")

        # Prepare train+val data for refit and test data for explanation
        X_train_val, y_train_val, _, _, X_test, y_test = get_train_val_test_data(
            df, best_feature_set, include_val_in_train=True
        )

        model = fit_final_model(best_model_name, best_params, best_trees, X_train_val, y_train_val)

        # Run TreeExplainer
        print(f"  Computing SHAP TreeExplainer on {len(X_test)} test week rows...")
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X_test)
        shap_values = shap_explanation.values

        # 1. Beeswarm Plot
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_explanation, show=False, max_display=12)
        plt.title(f"Plant {plant} - {horizon.upper()} Horizon: SHAP Beeswarm ({winner_row['model']})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / f"shap_summary_{horizon}.png", dpi=150)
        plt.close()

        # 2. Bar Plot of Feature Importance
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_explanation, show=False, max_display=12)
        plt.title(f"Plant {plant} - {horizon.upper()} Horizon: Mean |SHAP| Importance", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / f"shap_bar_{horizon}.png", dpi=150)
        plt.close()

        # 3. Grouped Attribution Table
        group_df = compute_group_attribution(shap_values, list(X_test.columns), horizon)
        shap_group_dfs.append(group_df)

    # Save grouped table
    all_groups_df = pd.concat(shap_group_dfs, ignore_index=True)
    all_groups_df.to_csv(reports_dir / "shap_groups.csv", index=False)

    print("\n" + "=" * 80)
    print(f" PHYSICAL FEATURE GROUP ATTRIBUTIONS -- PLANT {plant}")
    print("=" * 80)
    print(all_groups_df.to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Solar-Pulse Phase 3: SHAP Explainability")
    parser.add_argument("--plant", type=int, choices=[1, 2], default=1, help="Solar plant number (1 or 2)")
    args = parser.parse_args()

    run_shap_analysis(plant=args.plant)


if __name__ == "__main__":
    main()
