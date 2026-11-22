"""Unit tests for Phase 4: LSTM & Quantile LSTM Deep Learning Engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm_model import PointLSTM, QuantileLSTM, prepare_sequences, multi_quantile_loss
from src.models.tree_features import FEATURE_SETS, add_extended_lags

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plant1_df():
    path = ROOT / "data/features/features_plant1.csv"
    raw = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return add_extended_lags(raw)


def test_lstm_sequence_shapes(plant1_df):
    cols = FEATURE_SETS["1h_weather"]
    seq_x, seq_y, split_idx, scaler, y_mean, y_std = prepare_sequences(plant1_df, cols, seq_len=12)

    assert "train" in seq_x and "val" in seq_x and "test" in seq_x
    assert seq_x["train"].ndim == 3
    assert seq_x["train"].shape[1] == 12
    assert seq_x["train"].shape[2] == len(cols)
    assert len(seq_x["test"]) == 168


def test_point_lstm_forward():
    model = PointLSTM(input_size=17, hidden_size=32, num_layers=1)
    x = torch.randn(4, 12, 17)
    out = model(x)
    assert out.shape == (4,)


def test_quantile_lstm_forward():
    model = QuantileLSTM(input_size=17, hidden_size=32, num_layers=1)
    x = torch.randn(4, 12, 17)
    p10, p50, p90 = model(x)
    assert p10.shape == (4,)
    assert p50.shape == (4,)
    assert p90.shape == (4,)


def test_phase4_output_files_exist():
    for plant in (1, 2):
        rep = ROOT / f"reports/phase4/plant{plant}"
        fig = ROOT / f"figures/phase4/plant{plant}"
        assert (rep / "metrics_test.csv").exists()
        assert (rep / "forecasts_test.csv").exists()
        assert (rep / "leaderboard.csv").exists()
        assert (fig / f"plant{plant}_lstm_test_forecasts.png").exists()

