import pandas as pd

# Load your processed dataset
df = pd.read_csv("data/processed_energy_hourly.csv")

# Check column names first
print("Columns:", df.columns)

# Change this if needed based on output above
col = "Global_active_power"

mean_val = df[col].mean()
median_val = df[col].median()
mode_val = df[col].mode()

print("\n=== DESCRIPTIVE STATS ===")
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val.head())