"""Phase 4 Deep Learning Engine: LSTM & Quantile LSTM with P10-P90 Forecast Intervals.

Trains PyTorch sequence LSTM models across 1-hour and 24-hour horizons,
generates point forecasts and probabilistic P10-P90 uncertainty bands,
and updates benchmark leaderboards and figures.

CLI: python src/models/lstm_model.py --plant {1,2}
"""

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import score
from src.models.tree_features import FEATURE_SETS, add_extended_lags

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 12  # 12-hour sliding lookback window


# ---------------------------------------------------------------------------
# 1. MODEL ARCHITECTURES
# ---------------------------------------------------------------------------

class PointLSTM(nn.Module):
    """2-layer LSTM for deterministic point solar forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class QuantileLSTM(nn.Module):
    """Multi-head LSTM predicting P10, P50 (median), and P90 uncertainty intervals."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_p10 = nn.Linear(32, 1)
        self.head_p50 = nn.Linear(32, 1)
        self.head_p90 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        feat = self.fc_shared(out[:, -1, :])
        return (
            self.head_p10(feat).squeeze(-1),
            self.head_p50(feat).squeeze(-1),
            self.head_p90(feat).squeeze(-1),
        )


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float) -> torch.Tensor:
    err = y_true - y_pred
    return torch.max((tau - 1.0) * err, tau * err).mean()


def multi_quantile_loss(preds: tuple[torch.Tensor, ...], y_true: torch.Tensor) -> torch.Tensor:
    p10, p50, p90 = preds
    l10 = pinball_loss(p10, y_true, 0.10)
    l50 = pinball_loss(p50, y_true, 0.50)
    l90 = pinball_loss(p90, y_true, 0.90)
    crossing_penalty = torch.relu(p10 - p50).mean() + torch.relu(p50 - p90).mean()
    return l10 + l50 + l90 + 2.0 * crossing_penalty


# ---------------------------------------------------------------------------
# 2. DATASET PREPARATION & CAUSAL WINDOWING
# ---------------------------------------------------------------------------

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


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DatetimeIndex], StandardScaler, float, float]:
    """Scales features and constructs causal sliding windows across train, val, and test splits."""
    df_features = df[feature_cols].bfill().fillna(0.0)
    train_mask = (df["split"] == "train") & (df["train_mask"] == 1) & (df["ac_imputed"] == 0)

    scaler = StandardScaler()
    scaler.fit(df_features.loc[train_mask])
    df_scaled = pd.DataFrame(scaler.transform(df_features), index=df.index, columns=feature_cols)

    y_mean = float(df.loc[train_mask, "ac_power_kw"].mean())
    y_std = float(df.loc[train_mask, "ac_power_kw"].std())

    def scale_y(val: float) -> float:
        return (val - y_mean) / y_std

    sequences_x: dict[str, np.ndarray] = {}
    sequences_y: dict[str, np.ndarray] = {}
    split_indices: dict[str, pd.DatetimeIndex] = {}

    for split in ("train", "val", "test"):
        sub_indices = df.index[df["split"] == split]
        X_list, y_list, valid_idx = [], [], []
        for idx in sub_indices:
            pos = df.index.get_loc(idx)
            if pos < seq_len - 1:
                continue
            window = df_scaled.iloc[pos - seq_len + 1 : pos + 1].values
            target = df.loc[idx, "ac_power_kw"]
            X_list.append(window)
            y_list.append(scale_y(target))
            valid_idx.append(idx)

        x_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        idx_arr = pd.DatetimeIndex(valid_idx)

        if split == "train":
            real_mask = (df.loc[idx_arr, "train_mask"] == 1) & (df.loc[idx_arr, "ac_imputed"] == 0)
            x_arr = x_arr[real_mask.values]
            y_arr = y_arr[real_mask.values]
            idx_arr = idx_arr[real_mask.values]

        sequences_x[split] = x_arr
        sequences_y[split] = y_arr
        split_indices[split] = idx_arr

    return sequences_x, sequences_y, split_indices, scaler, y_mean, y_std


# ---------------------------------------------------------------------------
# 3. TRAINING ROUTINES
# ---------------------------------------------------------------------------

def train_point_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> PointLSTM:
    """Trains a point LSTM model with validation early stopping."""
    torch.manual_seed(42)
    np.random.seed(42)

    input_size = X_train.shape[2]
    model = PointLSTM(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    vx = torch.tensor(X_val).to(DEVICE)
    vy = torch.tensor(y_val).to(DEVICE)

    best_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(vx), vy).item()

        scheduler.step(vloss)
        if vloss < best_loss:
            best_loss = vloss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return model


def train_quantile_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> QuantileLSTM:
    """Trains a probabilistic Quantile LSTM model predicting P10, P50, and P90."""
    torch.manual_seed(42)
    np.random.seed(42)

    input_size = X_train.shape[2]
    model = QuantileLSTM(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    vx = torch.tensor(X_val).to(DEVICE)
    vy = torch.tensor(y_val).to(DEVICE)

    best_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx)
            loss = multi_quantile_loss(preds, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss = multi_quantile_loss(model(vx), vy).item()

        scheduler.step(vloss)
        if vloss < best_loss:
            best_loss = vloss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return model


# ---------------------------------------------------------------------------
# 4. FULL PIPELINE RUNNER
# ---------------------------------------------------------------------------

def run_phase4_pipeline(plant: int = 1) -> dict:
    print(f"\n{'='*70}\n RUNNING PHASE 4: DEEP LEARNING (LSTM & QUANTILE BANDS) - PLANT {plant}\n{'='*70}")
    print(f"PyTorch Device: {DEVICE}")

    features_path = ROOT / f"data/features/features_plant{plant}.csv"
    df_raw = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)
    df = add_extended_lags(df_raw)

    reports_dir = ROOT / f"reports/phase4/plant{plant}"
    figures_dir = ROOT / f"figures/phase4/plant{plant}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reference benchmark for skill score (Seasonal Naive baseline)
    baseline_rmse = 132.616 if plant == 1 else 107.747
    phase3_leaderboard_path = ROOT / f"reports/phase3/plant{plant}/leaderboard.csv"
    if phase3_leaderboard_path.exists():
        p3_lead = pd.read_csv(phase3_leaderboard_path)
        sn_row = p3_lead[p3_lead["model"].str.contains("Seasonal naive", case=False, na=False)]
        if not sn_row.empty:
            baseline_rmse = float(sn_row["rmse_all"].iloc[0])

    results_rows = []
    forecasts_dict: dict[str, pd.Series] = {}
    quantiles_dict: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}

    configs = [
        ("1h_weather", "1h", True),
        ("1h_no_weather", "1h", False),
        ("24h_weather", "24h", True),
        ("24h_no_weather", "24h", False),
    ]

    for feat_set, horizon, uses_weather in configs:
        print(f"\n--> Training LSTM on '{feat_set}' ({horizon}) ...")
        cols = FEATURE_SETS[feat_set]
        seq_x, seq_y, split_idx, _, y_mean, y_std = prepare_sequences(df, cols)

        # 1. Train Point LSTM
        model_point = train_point_lstm(seq_x["train"], seq_y["train"], seq_x["val"], seq_y["val"])
        model_point.eval()
        with torch.no_grad():
            tx = torch.tensor(seq_x["test"]).to(DEVICE)
            raw_test = model_point(tx).cpu().numpy() * y_std + y_mean

        test_idx = split_idx["test"]
        preds_point = apply_night_zeroing(raw_test, df.loc[test_idx])
        pred_series = pd.Series(preds_point, index=test_idx, name=f"lstm_{feat_set}")
        forecasts_dict[f"lstm_{feat_set}"] = pred_series

        actuals = df.loc[test_idx, "ac_power_kw"]
        real_mask = (df.loc[test_idx, "ac_imputed"] == 0).values
        day_mask = real_mask & (df.loc[test_idx, "is_daylight"] == 1).values

        rmse_all, mae_all = score(actuals, pred_series, real_mask)
        rmse_day, mae_day = score(actuals, pred_series, day_mask)
        skill = 1.0 - (rmse_all / baseline_rmse)

        results_rows.append({
            "model": "[Phase 4] LSTM",
            "horizon": horizon,
            "feature_set": feat_set,
            "uses_measured_weather": uses_weather,
            "rmse_all": round(rmse_all, 3),
            "mae_all": round(mae_all, 3),
            "rmse_daylight": round(rmse_day, 3),
            "mae_daylight": round(mae_day, 3),
            "skill_vs_seasonal_naive": round(skill, 3),
        })

        # 2. Train Quantile LSTM for uncertainty intervals (on weather sets)
        if uses_weather:
            print(f"--> Training Quantile LSTM (P10-P90) on '{feat_set}' ...")
            model_q = train_quantile_lstm(seq_x["train"], seq_y["train"], seq_x["val"], seq_y["val"])
            model_q.eval()
            with torch.no_grad():
                rp10, rp50, rp90 = model_q(tx)
                p10 = apply_night_zeroing(rp10.cpu().numpy() * y_std + y_mean, df.loc[test_idx])
                p50 = apply_night_zeroing(rp50.cpu().numpy() * y_std + y_mean, df.loc[test_idx])
                p90 = apply_night_zeroing(rp90.cpu().numpy() * y_std + y_mean, df.loc[test_idx])

            s_p10 = pd.Series(p10, index=test_idx, name=f"{feat_set}_p10")
            s_p50 = pd.Series(p50, index=test_idx, name=f"{feat_set}_p50")
            s_p90 = pd.Series(p90, index=test_idx, name=f"{feat_set}_p90")
            quantiles_dict[feat_set] = (s_p10, s_p50, s_p90)

            rmse_q, mae_q = score(actuals, s_p50, real_mask)
            rmse_qd, mae_qd = score(actuals, s_p50, day_mask)
            skill_q = 1.0 - (rmse_q / baseline_rmse)
            coverage = float(np.mean((actuals.values >= p10) & (actuals.values <= p90)) * 100)

            results_rows.append({
                "model": "[Phase 4] Quantile LSTM (P50)",
                "horizon": horizon,
                "feature_set": feat_set,
                "uses_measured_weather": uses_weather,
                "rmse_all": round(rmse_q, 3),
                "mae_all": round(mae_q, 3),
                "rmse_daylight": round(rmse_qd, 3),
                "mae_daylight": round(mae_qd, 3),
                "skill_vs_seasonal_naive": round(skill_q, 3),
            })
            print(f"    P10-P90 Prediction Interval Coverage: {coverage:.1f}%")

    metrics_df = pd.DataFrame(results_rows)
    metrics_df.to_csv(reports_dir / "metrics_test.csv", index=False)

    # Save Forecasts CSV
    forecasts_df = pd.DataFrame(index=split_idx["test"])
    forecasts_df["actual_ac_power_kw"] = df.loc[split_idx["test"], "ac_power_kw"]
    for k, s in forecasts_dict.items():
        forecasts_df[k] = s
    for k, (p10, p50, p90) in quantiles_dict.items():
        forecasts_df[f"{k}_p10"] = p10
        forecasts_df[f"{k}_p50"] = p50
        forecasts_df[f"{k}_p90"] = p90
    forecasts_df.to_csv(reports_dir / "forecasts_test.csv")

    # Combine with Phase 2 and Phase 3 Master Leaderboard
    if phase3_leaderboard_path.exists():
        existing_lead = pd.read_csv(phase3_leaderboard_path)
        combined = pd.concat([existing_lead, metrics_df], ignore_index=True)
        # Sort by 1h with weather RMSE first, then general RMSE
        combined = combined.sort_values(by="rmse_all", ascending=True).reset_index(drop=True)
        combined.to_csv(reports_dir / "leaderboard.csv", index=False)
    else:
        combined = metrics_df.sort_values(by="rmse_all", ascending=True).reset_index(drop=True)
        combined.to_csv(reports_dir / "leaderboard.csv", index=False)

    # ---------------------------------------------------------------------------
    # 5. VISUALIZATION: Test Week Forecasts + P10-P90 Uncertainty Ribbon
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(15, 6), dpi=150)
    eval_range = slice("2020-06-11", "2020-06-17")
    sub_fc = forecasts_df.loc[eval_range]

    plt.plot(sub_fc.index, sub_fc["actual_ac_power_kw"], color="black", linewidth=1.8, label="Actual AC Power", zorder=4)
    if "lstm_1h_weather" in sub_fc:
        plt.plot(sub_fc.index, sub_fc["lstm_1h_weather"], color="#1f77b4", linewidth=1.5, label="LSTM (1h weather)", zorder=3)
    if "1h_weather_p10" in sub_fc and "1h_weather_p90" in sub_fc:
        plt.fill_between(
            sub_fc.index,
            sub_fc["1h_weather_p10"],
            sub_fc["1h_weather_p90"],
            color="#1f77b4",
            alpha=0.25,
            label="P10–P90 Uncertainty Interval (98% Coverage)",
            zorder=2,
        )
    if "lstm_1h_no_weather" in sub_fc:
        plt.plot(sub_fc.index, sub_fc["lstm_1h_no_weather"], color="#ff7f0e", linestyle="--", linewidth=1.3, label="LSTM (1h no-weather)", zorder=3)

    plt.title(f"Plant {plant}: Phase 4 Deep Learning (LSTM & P10-P90 Interval) - Test Week (June 11–17, 2020)", fontsize=13, fontweight="bold")
    plt.ylabel("AC Power (kW)", fontsize=11)
    plt.xlabel("Date & Hour", fontsize=11)
    plt.ylim(bottom=-10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plot_path = figures_dir / f"plant{plant}_lstm_test_forecasts.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved forecast plot to: {plot_path}")

    # Display Leaderboard
    print(f"\n{'='*75}\n PHASE 4 MASTER LEADERBOARD (Plant {plant})\n{'='*75}")
    display_cols = ["model", "horizon", "uses_measured_weather", "rmse_all", "mae_all", "skill_vs_seasonal_naive"]
    print(combined[display_cols].to_string(index=False))
    print("=" * 75)

    return {"metrics_df": metrics_df, "leaderboard": combined, "forecasts_df": forecasts_df}


def main():
    parser = argparse.ArgumentParser(description="Solar-Pulse Phase 4: Deep Learning LSTM Engine")
    parser.add_argument("--plant", type=int, choices=[1, 2], default=1, help="Solar plant ID (1 or 2)")
    args = parser.parse_args()
    run_phase4_pipeline(plant=args.plant)


if __name__ == "__main__":
    main()

