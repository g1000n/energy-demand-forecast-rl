# Model Card

## Model Details
This repository contains three forecasting models and two support components:
- Linear Regression baseline
- LSTM forecaster (core deep learning model)
- TCN forecaster (CNN component)
- Auxiliary NLP demand-trend classifier
- Q-learning load scheduling agent

## Intended Use
Educational prototype for 6INTELSY final project. The system is intended to demonstrate forecasting and scheduling under offline simulation, not live utility control.

## Dataset
UCI Individual Household Electric Power Consumption Dataset. Household-level electricity measurements from a house in Sceaux, France, spanning nearly 4 years.

## Metrics
- Forecasting: MAE, MAPE
- NLP: Accuracy, Macro-F1, Confusion Matrix
- RL: cumulative reward, success rate, cost reduction vs baseline, seed variance

## Limitations
- Single-household data only
- Proxy cost rather than real tariff billing engine
- RL runs in offline simulation with simplified appliance constraints
- NLP module is auxiliary and synthetic-text-based

## Caveats
Not for operational deployment, smart-grid control, or consumer billing decisions.
