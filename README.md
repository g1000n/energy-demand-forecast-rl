# Energy Demand Forecasting + RL Load Scheduling

## Project Overview

This project implements **6INTELSY Option 7: Energy Demand Forecasting + RL Load Scheduling**. The system uses historical household electricity data to forecast short-term energy demand and then applies a simple reinforcement learning scheduler to decide whether flexible appliances should run immediately or be delayed to reduce proxy energy cost.

The project includes all required components for the course specification:

- **Core deep learning model:** LSTM forecaster
- **CNN component:** TCN forecaster
- **NLP component:** auxiliary demand-text classifier
- **RL component:** Q-learning load scheduler in offline simulation
- **Full ML pipeline:** preprocessing, training, evaluation, ablations, error/slice analysis, and reproducible outputs

This project is implemented as an **offline simulation prototype**, not a real-time deployed smart-home controller.

---

## Course Requirement Coverage

This repository was designed to satisfy the major requirements of the 6INTELSY final project.

### Option 7 Coverage

- **Task:** Forecast demand and schedule loads to reduce proxy costs
- **MVP:** LSTM/TCN forecast + tabular baseline + simple RL
- **Metrics:** MAE/MAPE for forecasting; RL reward vs baseline policy
- **Ethics focus:** equity, user autonomy, limitations of automated scheduling

### Technical Requirement Coverage

- **Core deep learning model:** LSTM
- **CNN requirement:** TCN
- **NLP requirement:** auxiliary demand-text classification
- **RL requirement:** Q-learning load scheduling environment
- **Reproducibility:** `run.py`, `run.sh`, `requirements.txt`, organized outputs

---

## Problem Statement

The system aims to answer two connected questions:

1. **Forecasting**  
   Given historical household electricity usage, how much demand is expected in the next hour?

2. **Scheduling**  
   Given that forecast, should a flexible appliance run now or be delayed to a later hour to reduce proxy electricity cost?

---

## Dataset Choice

The primary dataset is the **UCI Individual Household Electric Power Consumption Dataset**.

It was chosen because:

- it is household-level data, which matches the project task
- it contains minute-level electricity measurements over almost 4 years
- it includes multiple electrical variables and sub-metering values
- it supports both forecasting and realistic household load-scheduling simulation

### Expected raw dataset file

Place this file in the `data/` folder:

- `household_power_consumption.txt`

---

## Data Preparation

The raw minute-level dataset is converted into an **hourly dataset** to make forecasting and simulation more manageable.

### Preprocessing steps

- combine `Date` and `Time` into one `datetime` column
- convert measurement fields to numeric
- handle missing values using forward-fill and backward-fill
- resample data into hourly averages
- generate time-based features:
  - `hour`
  - `dayofweek`
  - `month`
  - `is_weekend`
  - `sin_hour`
  - `cos_hour`

### Final split strategy

A **chronological temporal split** is used to avoid data leakage:

- **70% training**
- **15% validation**
- **15% testing**

---

## System Components

### 1. Forecasting Models

The forecasting stage predicts next-hour household electricity demand.

Models included:

- **Linear Regression baseline** using lag and time features
- **LSTM** as the main deep learning forecasting model
- **TCN** as the CNN-based time-series model

### 2. NLP Auxiliary Module

The NLP component is a lightweight text classifier that assigns simple demand-related labels to generated demand descriptions.

Its purpose is **auxiliary**, not central. It demonstrates the required NLP component and shows how textual demand context could support a broader intelligent energy system.

### 3. RL Scheduling Module

The reinforcement learning component is a **Q-learning scheduler** operating in an offline simulation environment.

It decides whether a flexible appliance should:

- **run now**
- **delay**

The RL environment uses:

- forecasted demand
- appliance properties
- proxy energy cost
- scheduling constraints

Example appliance templates in the simulation:

- dishwasher
- washing machine
- water heater

### 4. Dashboard / Interface

A Streamlit dashboard is included to visualize:

- forecast metrics
- actual vs predicted demand
- NLP results
- RL learning curves
- RL scheduling decisions
- baseline vs scheduled cost
- simulation walkthrough and what-if interaction

---

## Offline Simulation Explanation

This project does **not** wait in real time.

Instead, the system simulates time by stepping through the **test portion of the hourly dataset**.

For each simulated hour:

1. the system reads the hour and demand context
2. the forecaster predicts demand
3. the RL scheduler observes the state
4. the scheduler chooses whether to run or delay a flexible appliance
5. the system computes reward based on proxy cost reduction
6. results are saved for analysis

---

## Main Outputs

After the pipeline is executed, the project generates:

### Forecasting

- `forecast_predictions.csv`
- `forecast_metrics.csv`
- `forecast_actual_vs_predicted.png`
- `training_curve_lstm.png`
- `training_curve_tcn.png`

### Evaluation and Analysis

- `ablation_results.csv`
- `forecast_slice_analysis.csv`
- `forecast_worst_cases.csv`

### NLP

- `nlp_metrics.csv`
- `nlp_confusion_matrix.png`

### RL

- `rl_learning_curves.png`
- `rl_metrics_by_seed.csv`
- `rl_decisions_sample.csv`
- `rl_summary.json`

### Additional Outputs

- processed hourly dataset in `data/`
- logs and JSON summaries in `logs/`

---

## Models and Baselines

### Non-DL baseline

- Linear Regression with lag + time features

### DL models

- LSTM forecaster
- TCN forecaster

### Ablations

- LSTM vs TCN
- baseline comparison against deep learning models

### Error Analysis

- worst-case forecast examples
- time-of-day slices
- weekday vs weekend slices

## How to Run

Follow these steps to reproduce the full system.

### 1. Clone the repository

git clone https://github.com/g1000n/energy-demand-forecast-rl/tree/main

### cd energy-demand-forecast-rl

### 2. Create and activate a virtual environment

### Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1

If activation is blocked:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

### macOS / Linux / Git Bash

python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

### pip install -r requirements.txt

### 4. Prepare dataset

Download the dataset from:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Then place the file inside:
data/household_power_consumption.txt

### 5. Run the full pipeline

# Windows

python run.py

### macOS / Linux / Git Bash

bash run.sh

Outputs will be saved in:
results/
logs/

### 6. Run the dashboard

streamlit run app.py

Then open:
http://localhost:8501
