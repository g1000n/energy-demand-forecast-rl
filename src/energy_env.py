import torch
import numpy as np

class EnergySchedulingEnv:
    def __init__(self, tcn_model, test_data, mean, std):
        """
        tcn_model: The trained DilatedTCN model (.pth loaded)
        test_data: The X_test tensor (normalized)
        mean: The mean from your training data (for denormalizing)
        std: The std from your training data (for denormalizing)
        """
        self.model = tcn_model
        self.data = test_data  # Shape: (N, 1, 24) // This should be the X_tensor
        self.mean, self.std = mean, std
        self.index = 0

    def get_state(self):
        """Returns the TCN forecast for the current window."""
        with torch.no_grad():
            window = self.data[self.index].unsqueeze(0)
            forecast = self.model(window).item()
        return np.array([forecast])
        
        # State is a 1D array containing the predicted kW
        return np.array([forecast])

    def step(self, action, nlp_modifier=1.0):
        """
        action: 0 (No change), 1 (Soft shift), 2 (Aggressive cut)
        nlp_modifier: multiplier from the NLP component (e.g., 1.5 for 'increase')
        """
        # 1. Get the actual demand from the dataset to calculate real reward
        # We look at the last value of the current window as a proxy or 
        # the target value if you have a separate y_test.
        actual_norm = self.data[self.index][0, -1].item()
        actual_kw = (actual_norm * self.std) + self.mean
        
        # Reductions: 0% reduction, 15% reduction, 30% reduction
        reduction = [1.0, 0.85, 0.70][action]
        savings = (actual_kw * 10 * nlp_modifier) * (1 - reduction)
        reward = savings - (0.2 if action == 2 else 0) # Discomfort penalty
        
        self.index += 1
        done = self.index >= len(self.data) - 1
        return self.get_state(), reward, done

    def reset(self):
        """Resets the simulation to the start of the test data."""
        self.index = 0
        return self.get_state()