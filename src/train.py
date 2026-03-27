import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from models.tcn_model import DilatedTCN

# 1. Load Data
# Get the folder where train.py is located (src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to find the 'data' folder
data_path = os.path.join(BASE_DIR, "..", "data", "processed_energy_hourly.csv")

# Load the file
df = pd.read_csv(data_path)
series = df["Global_active_power"].values.astype(np.float32)

# Normalization
mean, std = series.mean(), series.std()
series_norm = (series - mean) / std

# 2. Create Sequences (24h window)
def create_windows(data, window=24):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(x), np.array(y)

X, y = create_windows(series_norm)

# 3. Splits (70/15/15)
n = len(X)
train_idx, val_idx = int(n*0.7), int(n*0.85)
X_train, y_train = torch.tensor(X[:train_idx]).unsqueeze(1), torch.tensor(y[:train_idx])
X_val, y_val = torch.tensor(X[train_idx:val_idx]).unsqueeze(1), torch.tensor(y[train_idx:val_idx])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# 4. Train
model = DilatedTCN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_val = float('inf')

for epoch in range(15):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx).squeeze(), by)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_l = 0
    with torch.no_grad():
        for bx, by in val_loader:
            val_l += criterion(model(bx).squeeze(), by).item()
    
    avg_val = val_l / len(val_loader)
    if avg_val < best_val:
        best_val = avg_val
        # BASE_DIR is '.../src', so this points to '.../src/models/energy_tcn.pth'
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "energy_tcn.pth"))
        print(f"Epoch {epoch} - Model Saved (Val Loss: {avg_val:.4f})")