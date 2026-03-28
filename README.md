# Energy Demand Forecasting + RL Load Scheduling

## Project Overview
This project implements the **6INTELSY Option 7** system: a time-series AI pipeline that forecasts short-term household electricity demand and uses a simple reinforcement learning scheduler to shift flexible loads away from higher-demand periods. The repository includes:

- **Core deep learning model:** LSTM forecaster
- **CNN component:** TCN forecaster
- **NLP component:** auxiliary demand-text classifier
- **RL component:** Q-learning load scheduler in offline simulation
- **Full ML pipeline:** preprocessing, training, evaluation, ablations, error/slice analysis, and reproducible outputs

## Dataset Choice
The primary dataset is the **UCI Individual Household Electric Power Consumption Dataset**. It is the best fit for this project because it contains household-level minute data over almost 4 years, includes multiple electrical measurements and sub-metering fields, and naturally supports both forecasting and household load-scheduling simulation.

Raw dataset file expected in `data/`:
- `household_power_consumption.txt`

## Quick Start
1. Put `household_power_consumption.txt` inside the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the full pipeline:
   ```bash
   bash run.sh
   ```

## Repository Structure
```text
project-root/
├── README.md
├── requirements.txt
├── run.py
├── run.sh
├── data/
├── src/
│   ├── data_pipeline.py
│   ├── train.py
│   ├── eval.py
│   ├── nlp_classifier.py
│   ├── energy_env.py
│   ├── rl_agent.py
│   ├── utils/
│   └── models/
├── configs/
├── logs/
├── results/
├── experiments/
├── notebooks/
└── docs/
```

## Main Outputs
After running the pipeline, the project generates:
- processed hourly dataset
- forecasting metrics (MAE, MAPE) for baseline, LSTM, and TCN
- ablation results
- error/slice analysis
- NLP metrics (Accuracy, Macro-F1, Confusion Matrix)
- RL metrics (reward curves, success rate, cost vs baseline, variance across seeds)
- plots and CSV/JSON summaries in `results/` and `logs/`

## Models and Baselines
- **Non-DL baseline:** Linear Regression with lag + time features
- **DL baseline:** TCN forecaster
- **Core model:** LSTM forecaster
- **Ablations:**
  1. LSTM vs TCN
  2. With time features vs without time features (Linear Regression)

## Reproducibility Notes
- Fixed seeds are used.
- Temporal train/validation/test split is used to avoid leakage.
- Early stopping is used for LSTM and TCN.
- Runtime is constrained to modest epochs so the full pipeline can finish well within the course limit on a normal laptop or mid-range GPU.

## Suggested Release Tags
- `v0.1` Proposal
- `v0.9` Release candidate
- `v1.0` Final submission
