from __future__ import annotations

from pathlib import Path
import pandas as pd

from ucimlrepo import fetch_ucirepo


RAW_FILENAME = "household_power_consumption.txt"


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    output_path = data_dir / RAW_FILENAME

    if output_path.exists():
        print(f"Dataset already exists at: {output_path}")
        return

    print("Fetching dataset from UCI via ucimlrepo...")

    dataset = fetch_ucirepo(id=235)

    # UCI repo returns features as a DataFrame.
    df = dataset.data.features.copy()

    # Expected column order for your existing pipeline / raw file format
    expected_columns = [
        "Date",
        "Time",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Fetched dataset is missing expected columns: "
            + ", ".join(missing)
        )

    df = df[expected_columns].copy()

    # Match raw-file missing value style as closely as possible
    df = df.fillna("?")

    # Save in the semicolon-separated format expected by your pipeline
    df.to_csv(output_path, sep=";", index=False)

    print(f"Saved dataset to: {output_path}")
    print("You can now run: python run.py")


if __name__ == "__main__":
    main()