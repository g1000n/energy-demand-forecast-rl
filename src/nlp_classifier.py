from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.utils.common import ensure_dirs, project_root, save_json, set_seed


LABEL_ORDER = ["decrease", "stable", "increase"]


def _label_from_delta(delta: float) -> str:
    if delta < -0.08:
        return "decrease"
    if delta > 0.08:
        return "increase"
    return "stable"


def _band_from_value(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "medium"
    return "high"


def _build_text_dataset(df: pd.DataFrame) -> pd.DataFrame:
    values = df["Global_active_power"].values
    q1, q2 = df["Global_active_power"].quantile([0.33, 0.66]).tolist()
    rows = []
    for i in range(1, len(df)):
        current = values[i]
        prev = values[i - 1]
        delta = (current - prev) / max(abs(prev), 1e-6)
        label = _label_from_delta(delta)
        hour = int(df.iloc[i]["hour"])
        dow = int(df.iloc[i]["dayofweek"])
        demand_band = _band_from_value(current, q1, q2)
        prev_band = _band_from_value(prev, q1, q2)
        weekend = "weekend" if int(df.iloc[i]["is_weekend"]) == 1 else "weekday"
        text = f"{weekend} hour_{hour} day_{dow} current_{demand_band} previous_{prev_band}"
        rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def run_nlp_experiment(processed_csv_path: str) -> dict[str, float]:
    ensure_dirs()
    set_seed(42)
    root = project_root()
    results_dir = root / "results"
    logs_dir = root / "logs"

    df = pd.read_csv(processed_csv_path)
    text_df = _build_text_dataset(df)
    x_train, x_test, y_train, y_test = train_test_split(
        text_df["text"],
        text_df["label"],
        test_size=0.2,
        shuffle=False,
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_train_vec = vec.fit_transform(x_train)
    x_test_vec = vec.transform(x_test)

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(x_train_vec, y_train)
    preds = clf.predict(x_test_vec)

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    cm = confusion_matrix(y_test, preds, labels=LABEL_ORDER)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("NLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "nlp_confusion_matrix.png")
    plt.close()

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": int(len(text_df)),
    }
    pd.DataFrame([metrics]).to_csv(results_dir / "nlp_metrics.csv", index=False)
    save_json(logs_dir / "nlp_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    processed = str(project_root() / "data" / "processed_energy_hourly.csv")
    print(run_nlp_experiment(processed))
