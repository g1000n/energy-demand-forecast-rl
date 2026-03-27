import pandas as pd
import os

# Get the directory where this script is located (energy-demand-forecast-rl/src)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the root (up one level from src)
raw_data_path = os.path.join(base_dir, "..", "data", "household_power_consumption.txt")
processed_data_path = os.path.join(base_dir, "..", "data", "processed_energy_hourly.csv")

# Load raw dataset
try:
    df = pd.read_csv(
        raw_data_path,
        sep=";",
        low_memory=False,
        na_values=["?"]
    )
    print("Raw data loaded successfully.")
except FileNotFoundError:
    print(f"FAILED: Could not find {raw_data_path}")
    print("Make sure you have a folder named 'data' in your root directory containing the .txt file.")
    exit()

# --- Processing Logic ---
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
df = df.dropna(subset=["datetime"]).set_index("datetime")

numeric_columns = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_hourly = df[numeric_columns].ffill().resample("h").mean()

# Save to the root data folder
df_hourly.to_csv(processed_data_path)
print(f"Processed data saved to: {processed_data_path}")