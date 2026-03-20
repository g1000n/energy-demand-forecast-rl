import pandas as pd

# File paths
raw_data_path = "data/household_power_consumption.txt"
processed_data_path = "data/processed_energy_hourly.csv"

# Load raw dataset
df = pd.read_csv(
    raw_data_path,
    sep=";",
    low_memory=False,
    na_values=["?"]
)

print("Raw data loaded successfully.")
print("Shape before cleaning:", df.shape)
print(df.head())

# Combine Date and Time into one datetime column
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

# Drop rows with invalid datetime
df = df.dropna(subset=["datetime"])

# Convert relevant columns to numeric
numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numeric values
df = df.ffill()

# Set datetime as the index
df = df.set_index("datetime")

# Resample to hourly averages
df_hourly = df[numeric_columns].resample("h").mean()

print("\nHourly aggregated data:")
print(df_hourly.head())
print("Shape after hourly resampling:", df_hourly.shape)

# Save processed dataset
df_hourly.to_csv(processed_data_path)

print(f"\nProcessed hourly dataset saved to: {processed_data_path}")