import torch
import torch.nn as nn

class SimpleTCN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, kernel_size=3):
        super(SimpleTCN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x