import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.tcn_model import DilatedTCN
from nlp_classifier import get_nlp_multiplier
from energy_env import EnergySchedulingEnv


# --- 1. SETUP PATHS ---
# This identifies '.../energy-demand-forecast-rl/src'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data is one level UP from src
data_path = os.path.join(BASE_DIR, "..", "data", "processed_energy_hourly.csv")
# Model is INSIDE src/models
model_path = os.path.join(BASE_DIR, "models", "energy_tcn.pth")

# --- 2. AGENT DEFINITION ---
# Hyperparams
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.2
STATES, ACTIONS = 10, 3 # Increased states to 10 for better resolution

class QLearningAgent:
    def __init__(self):
        # Q-Table: [State (Demand Level), Action (0=None, 1=Soft, 2=Hard)]
        self.q_table = np.zeros((STATES, ACTIONS))

    def get_bucket(self, val):
        """Maps forecast kW to a discrete state (0-9)"""
        # Assuming typical demand is 0-5kW; adjust multiplier if needed
        return min(max(int(val * 2), 0), STATES - 1)

    def act(self, state):
        if np.random.rand() < EPSILON: 
            return np.random.randint(ACTIONS)
        return np.argmax(self.q_table[state])

# --- 3. LOADING & INITIALIZATION ---
print(f"Loading data from: {data_path}")
if not os.path.exists(data_path):
    print("ERROR: Processed data not found! Run data_pipeline.py first.")
    exit()
# --- Execution ---
# Load TCN and Data
df = pd.read_csv(data_path)
series = df["Global_active_power"].values.astype(np.float32)
mean, std = series.mean(), series.std()

# Create Test Split (Last 15%)
test_start = int(len(series) * 0.85)
test_series = (series[test_start:] - mean) / std

X_test = []
for i in range(len(test_series) - 24):
    window = torch.tensor(test_series[i : i+24]).unsqueeze(0) # Shape: (1, 24)
    X_test.append(window)
X_test = torch.stack(X_test) # Final Shape: (N, 1, 24)

# Load TCN Brain
tcn = DilatedTCN()
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}! Run train.py first.")
    exit()

# Load weights (using weights_only=True is safer for modern PyTorch)
tcn.load_state_dict(torch.load(model_path, weights_only=True))
tcn.eval()

# Init Environment and Agent
env = EnergySchedulingEnv(tcn, X_test, mean, std)
agent = QLearningAgent()

# --- 4. EXECUTION ---
nlp_alert = "Energy demand increased significantly due to heatwave"
nlp_mod = get_nlp_multiplier(nlp_alert)
print(f"NLP Context applied: Multiplier = {nlp_mod}")

state_val = env.reset()
total_reward = 0
savings_history = []

print("Starting RL Simulation...")
for _ in range(len(X_test) - 1):
    state = agent.get_bucket(state_val[0])
    action = agent.act(state)
    
    next_state_val, reward, done = env.step(action, nlp_mod)
    
    # Q-Learning Update Rule (Bellman Equation)
    next_state = agent.get_bucket(next_state_val[0])
    best_future_q = np.max(agent.q_table[next_state])
    
    # Update the Q-value for the current state-action pair
    self_q = agent.q_table[state, action]
    agent.q_table[state, action] += ALPHA * (reward + GAMMA * best_future_q - self_q)
    
    state_val = next_state_val
    total_reward += reward
    savings_history.append(total_reward)
    
    if done: break

# --- 5. RESULTS ---
print("-" * 30)
print(f"Simulation Complete.")
print(f"Final Cumulative Savings: ${total_reward:.2f}")
print("-" * 30)

# Optional: Visualize the learning/savings progress
plt.figure(figsize=(10, 5))
plt.plot(savings_history)
plt.title("RL Agent: Cumulative Energy Cost Savings")
plt.xlabel("Hours Simulated")
plt.ylabel("Total Savings ($)")
plt.grid(True)
plt.show()

print("Final Q-Table (Decision Matrix):")
print("States (Demand) | Action 0 | Action 1 | Action 2")
print(agent.q_table)