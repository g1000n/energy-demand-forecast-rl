import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class DilatedTCN(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2):
        super(DilatedTCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                     stride=1, padding=(kernel_size-1) * dilation_size // 2, 
                                     dilation=dilation_size)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (Batch, 1, 24)
        out = self.network(x)
        return self.fc(out[:, :, -1]) # Prediction based on last time step