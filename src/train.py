from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline import build_processed_dataset
from src.models.lstm_model import LSTMForecaster
from src.models.tcn_model import DilatedTCN
from src.utils.common import ensure_dirs, project_root, save_json, set_seed


@dataclass
class TrainConfig:
    seed: int = 42
    window_size: int = 24
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3


TIME_FEATURES = ["hour", "dayofweek", "month", "is_weekend", "sin_hour", "cos_hour"]
BASE_FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
] + TIME_FEATURES
TARGET = "Global_active_power"


def _temporal_splits(n: int) -> tuple[int, int]:
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return train_end, val_end


def _create_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str, window_size: int):
    x, y, timestamps = [], [], []
    values = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    times = pd.to_datetime(df["datetime"])
    for i in range(window_size, len(df)):
        x.append(values[i - window_size : i])
        y.append(targets[i])
        timestamps.append(times.iloc[i])
    return np.array(x), np.array(y), np.array(timestamps)


def _train_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig, model_path: Path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model, history


def _make_plot(history: dict[str, list[float]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_all_models() -> dict[str, str]:
    ensure_dirs()
    cfg = TrainConfig()
    set_seed(cfg.seed)
    root = project_root()
    processed_path = build_processed_dataset()
    results_dir = root / "results"
    logs_dir = root / "logs"
    models_dir = root / "src" / "models"

    df = pd.read_csv(processed_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Baseline data
    baseline_df = df.copy()
    baseline_df["lag1"] = baseline_df[TARGET].shift(1)
    baseline_df = baseline_df.dropna().reset_index(drop=True)
    n_base = len(baseline_df)
    train_end_b, val_end_b = _temporal_splits(n_base)

    # Baseline with time features
    feature_cols_with_time = ["lag1"] + TIME_FEATURES
    X_train = baseline_df.iloc[:train_end_b][feature_cols_with_time]
    X_val = baseline_df.iloc[train_end_b:val_end_b][feature_cols_with_time]
    X_test = baseline_df.iloc[val_end_b:][feature_cols_with_time]
    y_train = baseline_df.iloc[:train_end_b][TARGET]
    y_val = baseline_df.iloc[train_end_b:val_end_b][TARGET]
    y_test = baseline_df.iloc[val_end_b:][TARGET]

    linreg = LinearRegression()
    linreg.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    baseline_pred = linreg.predict(X_test)

    baseline_metrics = {
        "model": "linear_regression_with_time_features",
        "mae": float(mean_absolute_error(y_test, baseline_pred)),
        "mape": float(mean_absolute_percentage_error(y_test, baseline_pred)),
    }

    # Ablation: baseline without time features
    linreg_no_time = LinearRegression()
    linreg_no_time.fit(
        pd.concat([baseline_df.iloc[:train_end_b][["lag1"]], baseline_df.iloc[train_end_b:val_end_b][["lag1"]]]),
        pd.concat([y_train, y_val]),
    )
    baseline_pred_no_time = linreg_no_time.predict(baseline_df.iloc[val_end_b:][["lag1"]])
    baseline_no_time_metrics = {
        "model": "linear_regression_lag_only",
        "mae": float(mean_absolute_error(y_test, baseline_pred_no_time)),
        "mape": float(mean_absolute_percentage_error(y_test, baseline_pred_no_time)),
    }

    # Sequence models
    feature_cols = BASE_FEATURES
    x, y, timestamps = _create_sequences(df, feature_cols, TARGET, cfg.window_size)
    train_end, val_end = _temporal_splits(len(x))

    x_train_raw, x_val_raw, x_test_raw = x[:train_end], x[train_end:val_end], x[val_end:]
    y_train_seq, y_val_seq, y_test_seq = y[:train_end], y[train_end:val_end], y[val_end:]
    ts_test = timestamps[val_end:]

    feature_scaler = StandardScaler()
    train_2d = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    feature_scaler.fit(train_2d)

    def scale_seq(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, arr.shape[-1])
        scaled = feature_scaler.transform(flat)
        return scaled.reshape(arr.shape)

    x_train_scaled = scale_seq(x_train_raw)
    x_val_scaled = scale_seq(x_val_raw)
    x_test_scaled = scale_seq(x_test_raw)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_seq.reshape(-1, 1)).squeeze(1)
    y_val_scaled = y_scaler.transform(y_val_seq.reshape(-1, 1)).squeeze(1)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_scaled), torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val_scaled), torch.tensor(y_val_scaled, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    lstm_path = models_dir / "energy_lstm.pth"
    tcn_path = models_dir / "energy_tcn.pth"

    lstm = LSTMForecaster(input_size=len(feature_cols))
    lstm, lstm_hist = _train_torch_model(lstm, train_loader, val_loader, cfg, lstm_path)
    _make_plot(lstm_hist, "LSTM Training Curve", results_dir / "training_curve_lstm.png")

    tcn = DilatedTCN(input_size=len(feature_cols))
    tcn, tcn_hist = _train_torch_model(tcn, train_loader, val_loader, cfg, tcn_path)
    _make_plot(tcn_hist, "TCN Training Curve", results_dir / "training_curve_tcn.png")

    with torch.no_grad():
        lstm_pred_scaled = lstm(torch.tensor(x_test_scaled)).squeeze(-1).numpy()
        tcn_pred_scaled = tcn(torch.tensor(x_test_scaled)).squeeze(-1).numpy()

    lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).squeeze(1)
    tcn_pred = y_scaler.inverse_transform(tcn_pred_scaled.reshape(-1, 1)).squeeze(1)

    lstm_metrics = {
        "model": "lstm",
        "mae": float(mean_absolute_error(y_test_seq, lstm_pred)),
        "mape": float(mean_absolute_percentage_error(y_test_seq, lstm_pred)),
    }
    tcn_metrics = {
        "model": "tcn",
        "mae": float(mean_absolute_error(y_test_seq, tcn_pred)),
        "mape": float(mean_absolute_percentage_error(y_test_seq, tcn_pred)),
    }

    # Save prediction table
    pred_df = pd.DataFrame(
        {
            "datetime": ts_test,
            "actual": y_test_seq,
            "baseline_pred": baseline_pred[-len(y_test_seq):],
            "lstm_pred": lstm_pred,
            "tcn_pred": tcn_pred,
        }
    )
    pred_csv = results_dir / "forecast_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    # Actual vs predicted plot
    sample_n = min(240, len(pred_df))
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["datetime"].iloc[:sample_n], pred_df["actual"].iloc[:sample_n], label="Actual")
    plt.plot(pred_df["datetime"].iloc[:sample_n], pred_df["lstm_pred"].iloc[:sample_n], label="LSTM")
    plt.plot(pred_df["datetime"].iloc[:sample_n], pred_df["tcn_pred"].iloc[:sample_n], label="TCN")
    plt.xticks(rotation=45)
    plt.title("Forecast vs Actual (sample)")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "forecast_actual_vs_predicted.png")
    plt.close()

    metrics_df = pd.DataFrame([
        baseline_metrics,
        baseline_no_time_metrics,
        lstm_metrics,
        tcn_metrics,
    ])
    metrics_df.to_csv(results_dir / "forecast_metrics.csv", index=False)

    save_json(logs_dir / "training_config.json", cfg.__dict__)
    save_json(logs_dir / "forecast_metrics.json", metrics_df.set_index("model").to_dict(orient="index"))

    return {
        "processed_path": str(processed_path),
        "forecast_predictions_csv": str(pred_csv),
    }


if __name__ == "__main__":
    print(json.dumps(train_all_models(), indent=2))
