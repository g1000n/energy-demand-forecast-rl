import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from models.tcn_model import SimpleTCN

# -------------------------
# Config
# -------------------------
DATA_PATH = "data/processed_energy_hourly.csv"
TARGET_COL = "Global_active_power"
SEQ_LEN = 24
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)
series = df[TARGET_COL].values.astype(np.float32)

# normalize
mean = series.mean()
std = series.std()
series = (series - mean) / std

# -------------------------
# Create sequences
# -------------------------
X, y = [], []
for i in range(len(series) - SEQ_LEN):
    X.append(series[i:i+SEQ_LEN])
    y.append(series[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

# split
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# reshape for Conv1D: (batch, channels, seq_len)
X_train = torch.tensor(X_train).unsqueeze(1)
y_train = torch.tensor(y_train).unsqueeze(1)

X_val = torch.tensor(X_val).unsqueeze(1)
y_val = torch.tensor(y_val).unsqueeze(1)

X_test = torch.tensor(X_test).unsqueeze(1)
y_test = torch.tensor(y_test).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Model
# -------------------------
model = SimpleTCN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training
# -------------------------
print("Training TCN/CNN model...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# -------------------------
# Evaluation
# -------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().numpy()
    actual = y_test.squeeze().numpy()

# denormalize
preds = preds * std + mean
actual = actual * std + mean

mae = mean_absolute_error(actual, preds)
mape = mean_absolute_percentage_error(actual, preds)

print(f"Test MAE: {mae:.4f}")
print(f"Test MAPE: {mape:.4f}")