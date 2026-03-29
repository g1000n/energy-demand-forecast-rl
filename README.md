# Energy Demand Forecasting + RL Load Scheduling

## Project Overview

This project implements **6INTELSY Option 7: Energy Demand Forecasting + RL Load Scheduling**.

The system uses historical household electricity data to predict short-term energy demand and then applies a simple reinforcement learning scheduler to decide whether a flexible appliance should run immediately or be delayed to reduce proxy energy cost.

This project includes the required components for the course:

- **Core deep learning model:** LSTM forecaster
- **CNN component:** TCN forecaster
- **NLP component:** auxiliary demand-text classifier
- **RL component:** Q-learning load scheduler in offline simulation
- **ML pipeline:** preprocessing, training, evaluation, ablations, error/slice analysis, and reproducible outputs

This is an **offline simulation prototype** for academic use. It is not a real-time smart-home deployment system.

---

## Quick Start

### 1. Clone the repository

git clone https://github.com/g1000n/energy-demand-forecast-rl.git
cd energy-demand-forecast-rl

### 2. Create and activate a virtual environment

Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If activation is blocked:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux / Git Bash
python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Prepare the dataset

Download the UCI Individual Household Electric Power Consumption Dataset and place the raw file here:
data/household_power_consumption.txt

### 5. Run the full pipeline

##### Windows
python run.py
##### macOS / Linux / Git Bash
bash run.sh
This will run:
data preprocessing
forecasting training
evaluation
NLP experiment
RL simulation
Generated outputs will be saved in:
results/
logs/

### 6. Run the dashboard

streamlit run app.py
Then open:
http://localhost:8501

#### Results Summary

##### Forecasting

The forecasting module predicts next-hour household electricity demand using:
Linear Regression baseline

##### LSTM

##### TCN

Current evaluation results:
LSTM: MAE = 0.3148, MAPE = 0.4108
TCN: MAE = 0.3222, MAPE = 0.4519
The LSTM is the best-performing forecasting model in the current implementation.

##### NLP

The NLP component is a lightweight supporting classifier that labels demand-related text context.
Current results:
Accuracy: 0.5507
Macro-F1: 0.4977
This component is auxiliary and mainly supports interpretability and requirement coverage.

##### RL Scheduling

The RL module uses a Q-learning agent in an offline scheduling simulation.
It decides whether a simulated flexible appliance should:
run now
delay

##### Generated RL outputs include:

rl_learning_curves.png
rl_metrics_by_seed.csv
rl_decisions_sample.csv
logs/rl_summary.json
Ablations and Analysis
The project includes:
baseline vs deep learning comparison
LSTM vs TCN comparison
additional ablation outputs in ablation_results.csv
error and slice analysis outputs such as:
error_slice_by_hour.csv
error_slice_weekend.csv
forecast_worst_cases.csv
forecast_failure_cases.csv

#### Main Output Files

Forecasting
results/forecast_predictions.csv
results/forecast_metrics.csv
results/forecast_actual_vs_predicted.png
results/training_curve_lstm.png
results/training_curve_tcn.png

##### Analysis

results/ablation_results.csv
results/error_slice_by_hour.csv
results/error_slice_weekend.csv
results/forecast_slice_analysis.csv
results/forecast_worst_cases.csv
results/forecast_failure_cases.csv

##### NLP

results/nlp_metrics.csv
results/nlp_confusion_matrix.png

##### RL

results/rl_learning_curves.png
results/rl_metrics_by_seed.csv
results/rl_decisions_sample.csv
logs/rl_summary.json

#### Notes

The dataset is public and suitable for academic experimentation.
The system uses a chronological split to avoid data leakage.
The RL environment uses simulated flexible appliance requests because the original dataset does not include direct schedulable appliance events.
This project is intended for educational and experimental use only.
