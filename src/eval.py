from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.utils.common import ensure_dirs, project_root, save_json


def run_full_evaluation(processed_csv_path: str) -> dict[str, float]:
    ensure_dirs()
    root = project_root()
    results_dir = root / "results"
    logs_dir = root / "logs"

    pred_df = pd.read_csv(results_dir / "forecast_predictions.csv")
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
    processed = pd.read_csv(processed_csv_path)
    processed["datetime"] = pd.to_datetime(processed["datetime"])

    merged = pred_df.merge(processed[["datetime", "hour", "is_weekend"]], on="datetime", how="left")
    merged["abs_error_lstm"] = (merged["actual"] - merged["lstm_pred"]).abs()
    merged["abs_error_tcn"] = (merged["actual"] - merged["tcn_pred"]).abs()

    slice_summary = pd.DataFrame(
        {
            "hour": merged.groupby("hour")["abs_error_lstm"].mean().index,
            "lstm_hourly_mae": merged.groupby("hour")["abs_error_lstm"].mean().values,
            "tcn_hourly_mae": merged.groupby("hour")["abs_error_tcn"].mean().values,
        }
    )
    slice_summary.to_csv(results_dir / "error_slice_by_hour.csv", index=False)

    weekend_summary = pd.DataFrame(
        {
            "is_weekend": merged.groupby("is_weekend")["abs_error_lstm"].mean().index,
            "lstm_mae": merged.groupby("is_weekend")["abs_error_lstm"].mean().values,
            "tcn_mae": merged.groupby("is_weekend")["abs_error_tcn"].mean().values,
        }
    )
    weekend_summary.to_csv(results_dir / "error_slice_weekend.csv", index=False)

    worst_cases = merged.nlargest(20, "abs_error_lstm")[["datetime", "actual", "lstm_pred", "abs_error_lstm"]]
    worst_cases.to_csv(results_dir / "forecast_failure_cases.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(slice_summary["hour"], slice_summary["lstm_hourly_mae"])
    plt.title("LSTM Error Slice by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(results_dir / "error_slice_by_hour.png")
    plt.close()

    summary = {
        "lstm_mae": float(mean_absolute_error(merged["actual"], merged["lstm_pred"])),
        "lstm_mape": float(mean_absolute_percentage_error(merged["actual"], merged["lstm_pred"])),
        "tcn_mae": float(mean_absolute_error(merged["actual"], merged["tcn_pred"])),
        "tcn_mape": float(mean_absolute_percentage_error(merged["actual"], merged["tcn_pred"])),
        "worst_case_count": int(len(worst_cases)),
    }
    save_json(logs_dir / "evaluation_summary.json", summary)
    return summary


if __name__ == "__main__":
    root = project_root()
    print(run_full_evaluation(str(root / "data" / "processed_energy_hourly.csv")))
