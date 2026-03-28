from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ApplianceRequest:
    name: str
    power_kw: float
    duration_hours: int
    earliest_start_hour: int
    latest_finish_hour: int


class EnergySchedulingEnv:
    def __init__(self, forecast_df: pd.DataFrame, appliance_templates: list[ApplianceRequest]):
        self.forecast_df = forecast_df.reset_index(drop=True).copy()
        self.appliance_templates = appliance_templates
        self.idx = 0
        self.pending: ApplianceRequest | None = None
        self.pending_wait = 0
        self.current_episode_reward = 0.0
        self.successes = 0
        self.failures = 0

    def reset(self) -> tuple[int, int, int]:
        self.idx = 0
        self.pending = None
        self.pending_wait = 0
        self.current_episode_reward = 0.0
        self.successes = 0
        self.failures = 0
        return self._get_state()

    def _spawn_request(self, hour: int) -> ApplianceRequest | None:
        for template in self.appliance_templates:
            if hour == template.earliest_start_hour:
                return template
        return None

    def _demand_bucket(self, value: float) -> int:
        bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        return int(np.digitize([value], bins)[0])

    def _get_state(self) -> tuple[int, int, int]:
        row = self.forecast_df.iloc[self.idx]
        hour_bucket = int(row["hour"] // 6)
        demand_bucket = self._demand_bucket(float(row["predicted_demand_kw"]))
        pending_flag = 0 if self.pending is None else 1
        return hour_bucket, demand_bucket, pending_flag

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool, dict[str, Any]]:
        row = self.forecast_df.iloc[self.idx]
        hour = int(row["hour"])
        predicted = float(row["predicted_demand_kw"])
        if self.pending is None:
            self.pending = self._spawn_request(hour)
            self.pending_wait = 0

        price = 1.0 if predicted < 1.0 else 1.5 if predicted < 2.0 else 2.0
        baseline_cost = 0.0
        scheduled_cost = 0.0
        info: dict[str, Any] = {
            "hour": hour,
            "predicted": predicted,
            "pending": self.pending.name if self.pending else None,
            "action": action,
        }

        if self.pending is not None:
            baseline_cost = self.pending.power_kw * price
            must_run = hour >= self.pending.latest_finish_hour - self.pending.duration_hours
            if action == 0 or must_run:
                scheduled_cost = self.pending.power_kw * price
                reward = baseline_cost - scheduled_cost
                self.successes += 1
                self.pending = None
                self.pending_wait = 0
            else:
                scheduled_cost = 0.1  # inconvenience / standby penalty
                reward = baseline_cost - scheduled_cost
                self.pending_wait += 1
                if hour + 1 > self.pending.latest_finish_hour:
                    reward -= 3.0
                    self.failures += 1
                    self.pending = None
                    self.pending_wait = 0
        else:
            reward = 0.0

        self.current_episode_reward += reward
        info["baseline_cost"] = baseline_cost
        info["scheduled_cost"] = scheduled_cost
        info["reward"] = reward

        self.idx += 1
        done = self.idx >= len(self.forecast_df)
        next_state = self._get_state() if not done else (0, 0, 0)
        return next_state, reward, done, info
