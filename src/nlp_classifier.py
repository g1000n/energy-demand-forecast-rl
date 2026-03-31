from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> tuple[Path, Path]:
    experiments_dir = root / "experiments"
    results_dir = experiments_dir / "results"
    logs_dir = experiments_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, logs_dir

def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"]).reset_index(drop=True)

    data["hour"] = data["datetime"].dt.hour
    data["is_weekend"] = data["datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    def demand_band(x: float) -> str:
        if x < 1.0:
            return "low"
        elif x < 2.0:
            return "medium"
        return "high"

    def time_bucket(h: int) -> str:
        if 0 <= h <= 5:
            return "overnight"
        elif 6 <= h <= 11:
            return "morning"
        elif 12 <= h <= 17:
            return "afternoon"
        return "evening"

    data["next_power"] = data["Global_active_power"].shift(-1)
    data = data.dropna(subset=["next_power"]).copy()

    delta = data["next_power"] - data["Global_active_power"]

    def trend_label(d: float) -> str:
        if d > 0.10:
            return "increase"
        elif d < -0.10:
            return "decrease"
        return "stable"

    data["label"] = delta.apply(trend_label)

    data["demand_band"] = data["Global_active_power"].apply(demand_band)
    data["time_bucket"] = data["hour"].apply(time_bucket)
    data["week_type"] = np.where(data["is_weekend"] == 1, "weekend", "weekday")

    data["text"] = (
        data["demand_band"]
        + " "
        + data["time_bucket"]
        + " demand on "
        + data["week_type"]
    )

    return data[["datetime", "text", "label", "Global_active_power", "next_power"]].rename(
        columns={"Global_active_power": "current_power"}
    )


def run_nlp_experiment(processed_csv_path: str) -> dict[str, float]:
    root = project_root()
    results_dir, logs_dir = ensure_dirs(root)

    df = pd.read_csv(processed_csv_path)
    text_df = build_text_features(df)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        text_df["text"],
        text_df["label"],
        text_df[["datetime", "current_power", "next_power"]],
        test_size=0.2,
        random_state=42,
        stratify=text_df["label"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": len(text_df),
    }])
    metrics_df.to_csv(results_dir / "nlp_metrics.csv", index=False)

    labels = sorted(text_df["label"].unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("NLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "nlp_confusion_matrix.png")
    plt.close()

    pred_df = meta_test.copy()
    pred_df["text"] = X_test.values
    pred_df["true_label"] = y_test.values
    pred_df["predicted_label"] = y_pred
    pred_df = pred_df.sort_values("datetime")
    pred_df.to_csv(results_dir / "nlp_predictions.csv", index=False)

    summary = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": int(len(text_df)),
    }

    with open(logs_dir / "nlp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)
    return summary


if __name__ == "__main__":
    root = project_root()
    run_nlp_experiment(str(root / "data" / "processed_energy_hourly.csv"))