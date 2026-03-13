# Energy Demand Forecasting + RL Load Scheduling

## Overview
This project develops an artificial intelligence system that predicts electricity demand using time-series forecasting and optimizes flexible energy load scheduling using reinforcement learning.

The system forecasts short-term electricity demand based on historical consumption data and uses the predictions to guide a reinforcement learning agent that shifts energy usage away from peak demand periods.

## Components
- Baseline forecasting model (Linear Regression)
- LSTM deep learning forecasting model
- CNN-based Temporal Convolutional Network (TCN) comparison
- Reinforcement learning load scheduling agent
- Lightweight NLP text classifier

## Dataset
UCI Individual Household Electric Power Consumption Dataset  
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

The dataset contains electricity consumption measurements collected at one-minute intervals over approximately four years.

## Evaluation Metrics
The system will be evaluated using:

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Reinforcement learning reward and cost reduction compared to baseline scheduling.

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the training pipeline:

python src/train.py