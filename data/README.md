# Data Instructions

## Primary Dataset
**UCI Individual Household Electric Power Consumption Dataset**

- Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Raw file required: `household_power_consumption.txt`
- License/usage: cite the dataset in the final report and README.

## Why this dataset
This dataset is appropriate for the project because it provides minute-level household electricity data over nearly 4 years, which supports:
- hourly demand forecasting
- temporal train/validation/test splitting
- simulation of flexible household loads using household consumption patterns and sub-metering variables

## Notes
- Do **not** commit the raw text file if your team wants a lightweight repo.
- The processed file `processed_energy_hourly.csv` will be generated automatically by the pipeline.
