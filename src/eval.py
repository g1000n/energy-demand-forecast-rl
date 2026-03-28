from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))))


def ensure_dirs(root: Path) -> tuple[Path, Path]:
    results_dir = root / "results"
    logs_dir = root / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, logs_dir


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_np = y_true.to_numpy(dtype=float)
    y_pred_np = y_pred.to_numpy(dtype=float)

    mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
    mape = mean_absolute_percentage_error(y_true_np, y_pred_np)

    return {
        "mae": mae,
        "mape": mape,
    }


def add_time_slices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    def hour_bucket(h: int) -> str:
        if 0 <= h <= 5:
            return "overnight"
        elif 6 <= h <= 11:
            return "morning"
        elif 12 <= h <= 17:
            return "afternoon"
        else:
            return "evening"

    out["time_of_day"] = out["hour"].apply(hour_bucket)
    out["day_type"] = np.where(out["is_weekend"] == 1, "weekend", "weekday")
    return out


def save_forecast_metrics(preds: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []

    model_cols = [
        ("baseline", "baseline_pred"),
        ("lstm", "lstm_pred"),
        ("tcn", "tcn_pred"),
    ]

    for model_name, col in model_cols:
        if col in preds.columns:
            m = compute_metrics(preds["actual"], preds[col])
            rows.append({
                "model": model_name,
                "mae": m["mae"],
                "mape": m["mape"],
            })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results_dir / "forecast_metrics.csv", index=False)
    return metrics_df


def save_ablation_results(preds: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []

    candidates = [
        ("Linear Regression Baseline", "baseline_pred", "Non-DL baseline with lag/time features"),
        ("LSTM", "lstm_pred", "Core deep learning forecaster"),
        ("TCN", "tcn_pred", "CNN-based time-series forecaster"),
    ]

    for model_name, col, note in candidates:
        if col in preds.columns:
            m = compute_metrics(preds["actual"], preds[col])
            rows.append({
                "experiment": model_name,
                "prediction_column": col,
                "mae": m["mae"],
                "mape": m["mape"],
                "notes": note,
            })

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)
    return ablation_df


def save_slice_analysis(preds: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    df = add_time_slices(preds)

    rows = []
    model_cols = [c for c in ["baseline_pred", "lstm_pred", "tcn_pred"] if c in df.columns]

    for model_col in model_cols:
        # By time of day
        for slice_name, slice_df in df.groupby("time_of_day"):
            m = compute_metrics(slice_df["actual"], slice_df[model_col])
            rows.append({
                "model": model_col,
                "slice_type": "time_of_day",
                "slice_value": slice_name,
                "count": len(slice_df),
                "mae": m["mae"],
                "mape": m["mape"],
            })

        # By weekday/weekend
        for slice_name, slice_df in df.groupby("day_type"):
            m = compute_metrics(slice_df["actual"], slice_df[model_col])
            rows.append({
                "model": model_col,
                "slice_type": "day_type",
                "slice_value": slice_name,
                "count": len(slice_df),
                "mae": m["mae"],
                "mape": m["mape"],
            })

    slice_df = pd.DataFrame(rows)
    slice_df.to_csv(results_dir / "forecast_slice_analysis.csv", index=False)
    return slice_df


def save_worst_cases(preds: pd.DataFrame, results_dir: Path, top_k: int = 20) -> pd.DataFrame:
    df = preds.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if "lstm_pred" in df.columns:
        df["abs_error_lstm"] = (df["actual"] - df["lstm_pred"]).abs()
    else:
        df["abs_error_lstm"] = np.nan

    if "tcn_pred" in df.columns:
        df["abs_error_tcn"] = (df["actual"] - df["tcn_pred"]).abs()
    else:
        df["abs_error_tcn"] = np.nan

    worst_df = df.sort_values("abs_error_lstm", ascending=False).head(top_k).copy()
    worst_df.to_csv(results_dir / "forecast_worst_cases.csv", index=False)
    return worst_df


def save_summary_json(
    metrics_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    logs_dir: Path,
) -> dict:
    summary = {
        "forecast_models": metrics_df.to_dict(orient="records"),
        "num_ablation_rows": int(len(ablation_df)),
        "num_slice_rows": int(len(slice_df)),
        "worst_case_count": int(len(worst_df)),
    }

    with open(logs_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    root = project_root()
    results_dir, logs_dir = ensure_dirs(root)

    preds_path = results_dir / "forecast_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"Missing forecast predictions file: {preds_path}. Run training first."
        )

    preds = pd.read_csv(preds_path)

    metrics_df = save_forecast_metrics(preds, results_dir)
    ablation_df = save_ablation_results(preds, results_dir)
    slice_df = save_slice_analysis(preds, results_dir)
    worst_df = save_worst_cases(preds, results_dir, top_k=20)
    summary = save_summary_json(metrics_df, ablation_df, slice_df, worst_df, logs_dir)

    # Keep a simple stdout summary for run.py
    lstm_row = metrics_df[metrics_df["model"] == "lstm"]
    tcn_row = metrics_df[metrics_df["model"] == "tcn"]

    output = {
        "lstm_mae": float(lstm_row["mae"].iloc[0]) if not lstm_row.empty else None,
        "lstm_mape": float(lstm_row["mape"].iloc[0]) if not lstm_row.empty else None,
        "tcn_mae": float(tcn_row["mae"].iloc[0]) if not tcn_row.empty else None,
        "tcn_mape": float(tcn_row["mape"].iloc[0]) if not tcn_row.empty else None,
        "worst_case_count": summary["worst_case_count"],
    }

    print(output)


if __name__ == "__main__":
    main()