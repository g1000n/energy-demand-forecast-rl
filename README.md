# Energy Demand Forecasting + RL Load Scheduling

## Project Overview

This project implements **6INTELSY Option 7: Energy Demand Forecasting + RL Load Scheduling**.

The system uses historical household electricity data to predict short-term energy demand and applies a reinforcement learning scheduler to decide whether a flexible appliance should run immediately or be delayed to reduce proxy energy cost.

## Highlights

- End-to-end reproducible pipeline using a single command (`python run.py`)
- Automatic dataset retrieval using a helper script (`data/get_data.py`)
- Forecasting models include Linear Regression, LSTM, and Temporal Convolutional Network (TCN)
- Reinforcement learning (Q-learning) used for load scheduling decisions
- Auxiliary NLP component for demand trend interpretation
- Interactive Streamlit dashboard for visualization and simulation
- Includes ablation studies, error analysis, and slice analysis
- Results stored in structured directories for easy inspection (`experiments/results`, `experiments/logs`)

### Components

- **Core Deep Learning Model:** LSTM forecaster
- **CNN Component:** TCN forecaster
- **NLP Component:** Auxiliary demand-text classifier
- **RL Component:** Q-learning load scheduler (offline simulation)
- **ML Pipeline:** Preprocessing, training, evaluation, ablations, error/slice analysis

This is an **offline simulation prototype** for academic use and is not intended for real-time deployment.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/g1000n/energy-demand-forecast-rl.git
cd energy-demand-forecast-rl
```

---

### 2. (Optional but Recommended) Create a Virtual Environment

#### Windows (PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. The Dataset

The dataset source is the **UCI Individual Household Electric Power Consumption Dataset**:

- Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The raw file will be automatically retrieved by the helper script during pipeline execution if it is not already present.

Expected local path:

```
data/household_power_consumption.txt
```

---

### 5. Run the Full Pipeline

#### Windows

```bash
python run.py
```

#### macOS / Linux / Git Bash

```bash
bash run.sh
```

This will execute:

- Data preprocessing
- Forecasting model training
- Evaluation
- NLP experiment
- Reinforcement learning simulation

Outputs will be saved in:

```
experiments/results/
experiments/logs/
```

---

### 6. Run the Dashboard

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## Results Summary

### Forecasting

Models evaluated:

- Linear Regression (baseline)
- LSTM
- TCN

Results:

- **LSTM:** MAE = 0.3148, MAPE = 0.4108
- **TCN:** MAE = 0.3222, MAPE = 0.4519

The LSTM model currently achieves the best performance.

---

### NLP Component

A lightweight classifier used for contextual demand labeling.

- **Accuracy:** 0.5507
- **Macro-F1:** 0.4977

This component is auxiliary and supports interpretability and requirement fulfillment.

---

### RL Scheduling

A Q-learning agent is used in an offline simulation environment.

The agent decides whether to:

- Run appliance now
- Delay appliance usage

#### Generated Outputs

- `results/rl_learning_curves.png`
- `results/rl_metrics_by_seed.csv`
- `results/rl_decisions_sample.csv`
- `logs/rl_summary.json`

---

### Ablations and Analysis

The project includes:

- Baseline vs deep learning comparison
- LSTM vs TCN comparison
- Additional ablations (`ablation_results.csv`)
- Error and slice analysis:
  - `error_slice_by_hour.csv`
  - `error_slice_weekend.csv`
  - `forecast_worst_cases.csv`
  - `forecast_failure_cases.csv`

---

## Output Files

### Forecasting

- `results/forecast_predictions.csv`
- `results/forecast_metrics.csv`
- `results/forecast_actual_vs_predicted.png`
- `results/training_curve_lstm.png`
- `results/training_curve_tcn.png`

---

### Analysis

- `results/ablation_results.csv`
- `results/error_slice_by_hour.csv`
- `results/error_slice_weekend.csv`
- `results/forecast_slice_analysis.csv`
- `results/forecast_worst_cases.csv`
- `results/forecast_failure_cases.csv`

---

### NLP

- `results/nlp_metrics.csv`
- `results/nlp_confusion_matrix.png`

---

### Reinforcement Learning

- `results/rl_learning_curves.png`
- `results/rl_metrics_by_seed.csv`
- `results/rl_decisions_sample.csv`
- `logs/rl_summary.json`

---

## Notes

- The dataset is publicly available and suitable for academic experimentation.
- A chronological split is used to prevent data leakage.
- The RL environment simulates flexible appliance scheduling due to dataset limitations.
- This system is designed for **educational and experimental purposes only**.

## Team

- **Ashley Dyriel V. Buenafe** — Data	&	Ethics	Lead
- **Rendel Gion B. Lobo** — Project	Lead	/	Integration and Modeling	Lead
- **Anoucshka Ysabeli A. Sison** —  Evaluation	&	MLOps	Lead

## Dataset License and Citation

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

**Citation:**
Hebrail, G., & Berard, A. (2006). _Individual Household Electric Power Consumption_ [Dataset].  
UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

**License:**
This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

We acknowledge and credit the original authors and the UCI Machine Learning Repository.
