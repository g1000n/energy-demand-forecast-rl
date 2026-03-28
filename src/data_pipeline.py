from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.common import ensure_dirs, project_root

RAW_FILENAME = "household_power_consumption.txt"
PROCESSED_FILENAME = "processed_energy_hourly.csv"


def build_processed_dataset() -> Path:
    ensure_dirs()
    root = project_root()
    raw_path = root / "data" / RAW_FILENAME
    processed_path = root / "data" / PROCESSED_FILENAME

    if processed_path.exists():
        return processed_path

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {raw_path}. Place {RAW_FILENAME} inside the data/ folder first."
        )

    df = pd.read_csv(raw_path, sep=";", low_memory=False, na_values=["?"])
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime")

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    hourly = df[numeric_cols].resample("h").mean().reset_index()

    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["dayofweek"] = hourly["datetime"].dt.dayofweek
    hourly["month"] = hourly["datetime"].dt.month
    hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
    hourly["sin_hour"] = hourly["hour"].apply(lambda x: __import__("math").sin(2 * __import__("math").pi * x / 24))
    hourly["cos_hour"] = hourly["hour"].apply(lambda x: __import__("math").cos(2 * __import__("math").pi * x / 24))

    hourly.to_csv(processed_path, index=False)
    return processed_path


if __name__ == "__main__":
    print(build_processed_dataset())
