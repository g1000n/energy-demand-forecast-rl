# Model Card

## 1. Model Overview

The proposed system is an integrated machine learning pipeline for residential energy management. It combines time-series forecasting, reinforcement learning, and an auxiliary NLP component to support demand-aware scheduling decisions.

The system follows a **forecast-then-schedule** paradigm:
- Forecast future electricity demand  
- Use predictions to guide scheduling decisions  
- Optimize energy cost through delayed appliance usage  

---

## 2. Model Components

### 2.1 Forecasting Models
- **Linear Regression (Baseline):** Provides a simple non-deep learning reference model  
- **LSTM (Primary Model):** Captures temporal dependencies in time-series data  
- **TCN (CNN Component):** Uses dilated convolutions for sequence modeling  

---

### 2.2 NLP Component
- **TF-IDF + Logistic Regression**
- Classifies demand trends into:
  - Increase  
  - Decrease  
  - Stable  

This component improves interpretability but does not directly influence scheduling decisions.

---

### 2.3 Reinforcement Learning Component
- **Q-learning agent**
- Action space:
  - Run  
  - Delay  
  - No_action  

The agent operates in a simulated environment and learns a scheduling policy that reduces proxy energy cost.

---

## 3. Dataset

- **Source:** UCI Individual Household Electric Power Consumption Dataset  
- **Type:** Time-series household electricity consumption  
- **Granularity:** Minute-level (aggregated to hourly)  
- **Scope:** Single household  

### Limitations:
- Not representative of diverse populations  
- No appliance-level scheduling data  
- Requires simulation for RL component  

---

## 4. Training Details

- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error (MSE)  
- **Max Epochs:** 20  
- **Early Stopping:** Enabled (patience = 4)  
- **Random Seed:** 42 (forecasting), multiple seeds for RL  

Training is performed on CPU in a local environment.

---

## 5. Evaluation Metrics

### Forecasting:
- Mean Absolute Error (MAE)  
- Mean Absolute Percentage Error (MAPE)  

### NLP:
- Accuracy  
- Macro-F1 Score  
- Confusion Matrix  

### Reinforcement Learning:
- Mean Final Reward  
- Cost Reduction  
- Success Rate  
- Learning Curves  

---

## 6. Performance Summary

### Forecasting Results
| Model | MAE | MAPE |
|------|-----|------|
| Linear Regression | 0.3790 | 0.5876 |
| LSTM | 0.3148 | 0.4108 |
| TCN | 0.3222 | 0.4519 |

- LSTM achieves the best performance  
- TCN performs competitively  
- Baseline model performs worst  

---

### NLP Results
- Accuracy: 0.5507  
- Macro-F1: 0.4977  

The NLP component provides moderate classification performance and serves primarily as an interpretability layer.

---

### Reinforcement Learning Results
- Mean Final Reward: 1372.67  
- Mean Cost Reduction: 1026.00  
- Mean Success Rate: 0.9996  

The agent successfully reduces cost compared to the baseline policy.

⚠️ Note:
The high success rate is influenced by the simulation structure and frequent `no_action` decisions.

---

## 7. Error Analysis

The model performs well under normal conditions but struggles in the following cases:
- Sudden spikes in energy demand  
- Irregular usage patterns  
- Peak demand periods (morning and evening)  

Slice analysis shows:
- Higher error during weekends  
- Increased MAE during peak hours  

---

## 8. Intended Use

The system is intended for:
- Academic research  
- Educational demonstrations  
- Simulated smart-home environments  

It is not intended for:
- Real-time deployment  
- Safety-critical applications  
- Direct operational use  

---

## 9. Limitations

- Single-household dataset limits generalizability  
- Simulated RL environment does not reflect real-world constraints  
- No real appliance-level data  
- Performance degrades during irregular demand  

---

## 10. Ethical Considerations

- No PII present in dataset  
- Potential privacy concerns in real-world deployment  
- Dataset bias affects fairness and generalization  
- Requires user control and transparency mechanisms  

For full details, see `ethical_statement.md`.

---

## 11. Deployment Considerations

To deploy this system in real-world environments, the following are required:
- IoT sensors or smart meters  
- Appliance-level monitoring systems  
- Real-time data processing pipeline  
- Dynamic pricing signals  
- User interfaces for control and preferences  

---

## 12. Reproducibility

The system is fully reproducible using:

```bash
pip install -r requirements.txt
python run.py
streamlit run app.py
````

Dataset must be placed in:

```
data/household_power_consumption.txt
```

All outputs are stored in:

* `results/`
* `logs/`

---

## 13. Summary

This model demonstrates the integration of forecasting and reinforcement learning for intelligent energy management.

While effective in a simulated environment, real-world deployment requires additional data, infrastructure, and validation.

````

