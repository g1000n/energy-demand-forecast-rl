from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
from src.utils.common import ensure_dirs, get_device, now_ts, minutes_elapsed, project_root, save_json, set_seed


@dataclass
class TrainConfig:
    seed: int = 42
    window_size: int = 24
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    num_workers: int = 0


TIME_FEATURES = ["hour", "dayofweek", "month", "is_weekend", "sin_hour", "cos_hour"]
BASE_POWER_FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
BASE_FEATURES = BASE_POWER_FEATURES + TIME_FEATURES
BASE_FEATURES_NO_TIME = BASE_POWER_FEATURES
TARGET = "Global_active_power"


def _temporal_splits(n: int) -> tuple[int, int]:
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return train_end, val_end


def _create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
):
    x, y, timestamps = [], [], []
    values = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    times = pd.to_datetime(df["datetime"])

    for i in range(window_size, len(df)):
        x.append(values[i - window_size : i])
        y.append(targets[i])
        timestamps.append(times.iloc[i])

    return np.array(x), np.array(y), np.array(timestamps)


def _scale_sequences(
    x_train_raw: np.ndarray,
    x_val_raw: np.ndarray,
    x_test_raw: np.ndarray,
):
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

    return x_train_scaled, x_val_scaled, x_test_scaled, feature_scaler


def _prepare_target_scaler(
    y_train: np.ndarray,
    y_val: np.ndarray,
):
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze(1)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).squeeze(1)
    return y_train_scaled, y_val_scaled, y_scaler


def _make_loaders(
    x_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    y_val_scaled: np.ndarray,
    cfg: TrainConfig,
):
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32),
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val_scaled, dtype=torch.float32),
            torch.tensor(y_val_scaled, dtype=torch.float32),
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader


def _train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    model_path: Path,
    device: torch.device,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    model = model.to(device)

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

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
                xb, yb = xb.to(device), yb.to(device)
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
                print(f"Early stopping triggered for {model.__class__.__name__} at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, history, best_val


def _predict_torch_model(model: nn.Module, x_test_scaled: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xb = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
        pred_scaled = model(xb).squeeze(-1).detach().cpu().numpy()
    return pred_scaled


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


def _metric_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def _save_error_analysis(
    pred_df: pd.DataFrame,
    results_dir: Path,
    primary_model_col: str = "lstm_pred",
) -> None:
    analysis_df = pred_df.copy()
    analysis_df["datetime"] = pd.to_datetime(analysis_df["datetime"])
    analysis_df["hour"] = analysis_df["datetime"].dt.hour
    analysis_df["dayofweek"] = analysis_df["datetime"].dt.dayofweek
    analysis_df["is_weekend"] = analysis_df["dayofweek"] >= 5

    analysis_df["abs_error"] = (analysis_df["actual"] - analysis_df[primary_model_col]).abs()
    analysis_df["ape"] = np.where(
        np.abs(analysis_df["actual"]) > 1e-8,
        analysis_df["abs_error"] / np.abs(analysis_df["actual"]),
        np.nan,
    )

    # Slice by hour
    hour_slice = (
        analysis_df.groupby("hour", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            mape=("ape", "mean"),
            count=("actual", "size"),
        )
        .sort_values("hour")
    )
    hour_slice.to_csv(results_dir / "error_slice_by_hour.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(hour_slice["hour"], hour_slice["mae"], marker="o")
    plt.title("LSTM MAE by Hour")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(results_dir / "error_slice_by_hour.png")
    plt.close()

    # Slice by weekend
    weekend_slice = (
        analysis_df.groupby("is_weekend", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            mape=("ape", "mean"),
            count=("actual", "size"),
        )
    )
    weekend_slice["is_weekend"] = weekend_slice["is_weekend"].map({False: "weekday", True: "weekend"})
    weekend_slice.to_csv(results_dir / "error_slice_weekend.csv", index=False)

    # Worst cases
    worst_cases = analysis_df.sort_values("abs_error", ascending=False).head(25)
    worst_cases.to_csv(results_dir / "forecast_worst_cases.csv", index=False)

    # Failure cases above threshold
    threshold = float(analysis_df["abs_error"].quantile(0.90))
    failure_cases = analysis_df[analysis_df["abs_error"] >= threshold].copy()
    failure_cases.to_csv(results_dir / "forecast_failure_cases.csv", index=False)

    # Combined slice file
    slice_summary = pd.concat(
        [
            hour_slice.assign(slice_type="hour", slice_value=lambda d: d["hour"].astype(str)),
            weekend_slice.assign(slice_type="weekend", slice_value=lambda d: d["is_weekend"].astype(str)),
        ],
        ignore_index=True,
        sort=False,
    )
    slice_summary.to_csv(results_dir / "forecast_slice_analysis.csv", index=False)


def train_all_models() -> dict[str, str | float]:
    pipeline_start = now_ts()

    results_dir, logs_dir = ensure_dirs()
    cfg = TrainConfig()
    set_seed(cfg.seed)

    root = project_root()
    device = get_device()
    processed_path = build_processed_dataset()
    models_dir = root / "src" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # -------------------------
    # Baseline models
    # -------------------------
    baseline_df = df.copy()
    baseline_df["lag1"] = baseline_df[TARGET].shift(1)
    baseline_df = baseline_df.dropna().reset_index(drop=True)

    n_base = len(baseline_df)
    train_end_b, val_end_b = _temporal_splits(n_base)

    # Baseline 1: Linear regression with lag + time features
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
    baseline_metrics = _metric_row("linear_regression_with_time_features", y_test.to_numpy(), baseline_pred)

    # Ablation 1: Linear regression with lag only
    linreg_no_time = LinearRegression()
    X_train_no_time = baseline_df.iloc[:train_end_b][["lag1"]]
    X_val_no_time = baseline_df.iloc[train_end_b:val_end_b][["lag1"]]
    X_test_no_time = baseline_df.iloc[val_end_b:][["lag1"]]

    linreg_no_time.fit(pd.concat([X_train_no_time, X_val_no_time]), pd.concat([y_train, y_val]))
    baseline_pred_no_time = linreg_no_time.predict(X_test_no_time)
    baseline_no_time_metrics = _metric_row("linear_regression_lag_only", y_test.to_numpy(), baseline_pred_no_time)

    # -------------------------
    # Sequence models with time features
    # -------------------------
    x, y, timestamps = _create_sequences(df, BASE_FEATURES, TARGET, cfg.window_size)
    train_end, val_end = _temporal_splits(len(x))

    x_train_raw, x_val_raw, x_test_raw = x[:train_end], x[train_end:val_end], x[val_end:]
    y_train_seq, y_val_seq, y_test_seq = y[:train_end], y[train_end:val_end], y[val_end:]
    ts_test = timestamps[val_end:]

    x_train_scaled, x_val_scaled, x_test_scaled, _ = _scale_sequences(x_train_raw, x_val_raw, x_test_raw)
    y_train_scaled, y_val_scaled, y_scaler = _prepare_target_scaler(y_train_seq, y_val_seq)
    train_loader, val_loader = _make_loaders(x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, cfg)

    lstm_path = models_dir / "energy_lstm.pth"
    tcn_path = models_dir / "energy_tcn.pth"

    lstm = LSTMForecaster(input_size=len(BASE_FEATURES))
    lstm, lstm_hist, lstm_best_val = _train_torch_model(lstm, train_loader, val_loader, cfg, lstm_path, device)
    _make_plot(lstm_hist, "LSTM Training Curve", results_dir / "training_curve_lstm.png")

    tcn = DilatedTCN(input_size=len(BASE_FEATURES))
    tcn, tcn_hist, tcn_best_val = _train_torch_model(tcn, train_loader, val_loader, cfg, tcn_path, device)
    _make_plot(tcn_hist, "TCN Training Curve", results_dir / "training_curve_tcn.png")

    lstm_pred_scaled = _predict_torch_model(lstm, x_test_scaled, device)
    tcn_pred_scaled = _predict_torch_model(tcn, x_test_scaled, device)

    lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).squeeze(1)
    tcn_pred = y_scaler.inverse_transform(tcn_pred_scaled.reshape(-1, 1)).squeeze(1)

    lstm_metrics = _metric_row("lstm", y_test_seq, lstm_pred)
    tcn_metrics = _metric_row("tcn", y_test_seq, tcn_pred)

    # -------------------------
    # Ablation 2: LSTM without time features
    # -------------------------
    x_nt, y_nt, _ = _create_sequences(df, BASE_FEATURES_NO_TIME, TARGET, cfg.window_size)
    train_end_nt, val_end_nt = _temporal_splits(len(x_nt))

    x_train_nt_raw = x_nt[:train_end_nt]
    x_val_nt_raw = x_nt[train_end_nt:val_end_nt]
    x_test_nt_raw = x_nt[val_end_nt:]

    y_train_nt = y_nt[:train_end_nt]
    y_val_nt = y_nt[train_end_nt:val_end_nt]
    y_test_nt = y_nt[val_end_nt:]

    x_train_nt_scaled, x_val_nt_scaled, x_test_nt_scaled, _ = _scale_sequences(
        x_train_nt_raw, x_val_nt_raw, x_test_nt_raw
    )
    y_train_nt_scaled, y_val_nt_scaled, y_scaler_nt = _prepare_target_scaler(y_train_nt, y_val_nt)
    train_loader_nt, val_loader_nt = _make_loaders(
        x_train_nt_scaled, y_train_nt_scaled, x_val_nt_scaled, y_val_nt_scaled, cfg
    )

    lstm_no_time_path = models_dir / "energy_lstm_no_time_ablation.pth"
    lstm_no_time = LSTMForecaster(input_size=len(BASE_FEATURES_NO_TIME))
    lstm_no_time, lstm_no_time_hist, lstm_no_time_best_val = _train_torch_model(
        lstm_no_time, train_loader_nt, val_loader_nt, cfg, lstm_no_time_path, device
    )
    _make_plot(
        lstm_no_time_hist,
        "LSTM No-Time Ablation Training Curve",
        results_dir / "training_curve_lstm_no_time.png",
    )

    lstm_no_time_pred_scaled = _predict_torch_model(lstm_no_time, x_test_nt_scaled, device)
    lstm_no_time_pred = y_scaler_nt.inverse_transform(lstm_no_time_pred_scaled.reshape(-1, 1)).squeeze(1)
    lstm_no_time_metrics = _metric_row("lstm_no_time_features", y_test_nt, lstm_no_time_pred)

    # -------------------------
    # Save main prediction table
    # -------------------------
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

    # Plot sample forecast
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

    # -------------------------
    # Save metrics
    # -------------------------
    metrics_df = pd.DataFrame(
        [
            baseline_metrics,
            baseline_no_time_metrics,
            lstm_metrics,
            tcn_metrics,
            lstm_no_time_metrics,
        ]
    )
    metrics_df.to_csv(results_dir / "forecast_metrics.csv", index=False)

    ablation_df = pd.DataFrame(
        [
            baseline_no_time_metrics,
            baseline_metrics,
            lstm_no_time_metrics,
            lstm_metrics,
            tcn_metrics,
        ]
    )
    ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)

    _save_error_analysis(pred_df, results_dir, primary_model_col="lstm_pred")

    runtime_minutes = minutes_elapsed(pipeline_start)

    training_summary = {
        "device": str(device),
        "seed": cfg.seed,
        "window_size": cfg.window_size,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "learning_rate": cfg.learning_rate,
        "runtime_minutes": runtime_minutes,
        "early_stopping_enabled": True,
        "lstm_best_val_loss": float(lstm_best_val),
        "tcn_best_val_loss": float(tcn_best_val),
        "lstm_no_time_best_val_loss": float(lstm_no_time_best_val),
    }

    save_json(logs_dir / "training_config.json", asdict(cfg))
    save_json(logs_dir / "training_summary.json", training_summary)
    save_json(logs_dir / "forecast_metrics.json", metrics_df.set_index("model").to_dict(orient="index"))

    return {
        "processed_path": str(processed_path),
        "forecast_predictions_csv": str(pred_csv),
        "forecast_metrics_csv": str(results_dir / "forecast_metrics.csv"),
        "ablation_results_csv": str(results_dir / "ablation_results.csv"),
        "runtime_minutes": runtime_minutes,
    }


if __name__ == "__main__":
    print(json.dumps(train_all_models(), indent=2))