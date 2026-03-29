from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ApplianceSpec:
    name: str
    power_kw: float
    max_delay_hours: int
    earliest_hour: int
    latest_hour: int
    probability: float


class EnergySchedulingEnv:
    """
    Offline RL environment for simple household load scheduling.

    Goal:
        Decide whether a flexible appliance should run now or be delayed
        to reduce proxy cost while respecting simple delay limits.

    Notes:
        - This is a simulation environment.
        - The dataset provides demand history, not true schedulable labels.
        - Appliance requests are simulated using time-window rules.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        predicted_demand_col: str = "lstm_pred",
        actual_demand_col: str = "actual",
        seed: int = 42,
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.predicted_demand_col = predicted_demand_col
        self.actual_demand_col = actual_demand_col
        self.rng = np.random.default_rng(seed)

        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors="coerce")
        else:
            raise ValueError("Environment requires a 'datetime' column.")

        required_cols = {"datetime", self.predicted_demand_col, self.actual_demand_col}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.appliances: Dict[str, ApplianceSpec] = {
            "dishwasher": ApplianceSpec(
                name="dishwasher",
                power_kw=1.0,
                max_delay_hours=3,
                earliest_hour=19,
                latest_hour=22,
                probability=0.30,
            ),
            "washing_machine": ApplianceSpec(
                name="washing_machine",
                power_kw=1.2,
                max_delay_hours=2,
                earliest_hour=8,
                latest_hour=18,
                probability=0.25,
            ),
            "water_heater": ApplianceSpec(
                name="water_heater",
                power_kw=0.8,
                max_delay_hours=1,
                earliest_hour=6,
                latest_hour=9,
                probability=0.40,
            ),
        }

        self.action_map = {
            0: "run_now",
            1: "delay",
        }

        self.reset()

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.pending_request: Optional[Dict] = None
        self.total_reward = 0.0
        self.total_baseline_cost = 0.0
        self.total_scheduled_cost = 0.0
        self.success_count = 0
        self.total_requests = 0
        self.decision_rows: List[Dict] = []

        return self._get_state()

    def _sample_request_for_current_hour(self) -> Optional[Dict]:
        row = self.df.iloc[self.idx]
        dt = row["datetime"]

        if pd.isna(dt):
            return None

        hour = int(dt.hour)

        candidates = []
        for spec in self.appliances.values():
            if spec.earliest_hour <= hour <= spec.latest_hour:
                if self.rng.random() < spec.probability:
                    candidates.append(spec)

        if not candidates:
            return None

        spec = candidates[0]
        return {
            "appliance_name": spec.name,
            "power_kw": spec.power_kw,
            "delay_count": 0,
            "max_delay_hours": spec.max_delay_hours,
            "created_idx": self.idx,
        }

    def _ensure_request_exists(self) -> None:
        if self.pending_request is None:
            request = self._sample_request_for_current_hour()
            if request is not None:
                self.pending_request = request
                self.total_requests += 1

    def _bin_demand(self, value: float) -> int:
        if value < 1.0:
            return 0
        if value < 2.0:
            return 1
        return 2

    def _get_state(self) -> np.ndarray:
        if self.idx >= len(self.df):
            return np.zeros(4, dtype=np.float32)

        self._ensure_request_exists()
        row = self.df.iloc[self.idx]

        pred = float(row[self.predicted_demand_col])
        hour = int(row["datetime"].hour)

        if self.pending_request is None:
            appliance_id = 0
            delay_ratio = 0.0
        else:
            appliance_name = self.pending_request["appliance_name"]
            appliance_id = {
                "dishwasher": 1,
                "washing_machine": 2,
                "water_heater": 3,
            }.get(appliance_name, 0)
            delay_count = int(self.pending_request["delay_count"])
            max_delay = max(int(self.pending_request["max_delay_hours"]), 1)
            delay_ratio = delay_count / max_delay

        state = np.array(
            [
                float(self._bin_demand(pred)),
                float(hour / 23.0),
                float(appliance_id),
                float(delay_ratio),
            ],
            dtype=np.float32,
        )
        return state

    def _current_costs(self) -> Tuple[float, float]:
        row = self.df.iloc[self.idx]
        pred = float(row[self.predicted_demand_col])

        if self.pending_request is None:
            baseline_cost = pred
            scheduled_cost = pred
        else:
            appliance_power = float(self.pending_request["power_kw"])
            baseline_cost = pred + appliance_power
            scheduled_cost = pred

        return baseline_cost, scheduled_cost

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.idx >= len(self.df):
            return np.zeros(4, dtype=np.float32), 0.0, True, {}

        self._ensure_request_exists()
        row = self.df.iloc[self.idx]
        dt = row["datetime"]
        pred = float(row[self.predicted_demand_col])
        actual = float(row[self.actual_demand_col])

        decision = self.action_map.get(action, "delay")
        info: Dict = {}
        baseline_cost, delay_cost = self._current_costs()

        scheduled_cost = baseline_cost
        reward = 0.0
        missed_deadline = False
        forced_run = False
        appliance_name = "none"
        appliance_power = 0.0
        delay_count = 0
        max_delay = 0

        if self.pending_request is None:
            decision = "no_action"
            scheduled_cost = pred
            reward = 0.0

        else:
            appliance_name = self.pending_request["appliance_name"]
            appliance_power = float(self.pending_request["power_kw"])
            delay_count = int(self.pending_request["delay_count"])
            max_delay = int(self.pending_request["max_delay_hours"])

            if delay_count >= max_delay:
                decision = "run_now"
                forced_run = True

            if decision == "run_now":
                scheduled_cost = baseline_cost

                # Reward timing quality for running
                if pred < 1.0:
                    reward = 0.8
                elif pred < 1.8:
                    reward = 0.3
                elif pred < 2.3:
                    reward = -0.1
                else:
                    reward = -0.6

                if forced_run:
                    reward -= 0.3

                self.success_count += 1
                self.pending_request = None

            elif decision == "delay":
                scheduled_cost = delay_cost
                cost_savings = baseline_cost - scheduled_cost
                delay_penalty = 0.15 * (delay_count + 1)

                # Delaying is bad when demand is already low,
                # and more acceptable when current demand is high.
                if pred < 1.0:
                    timing_penalty = 0.8
                elif pred < 1.8:
                    timing_penalty = 0.3
                elif pred < 2.3:
                    timing_penalty = 0.0
                else:
                    timing_penalty = -0.2

                reward = cost_savings - delay_penalty - timing_penalty

                self.pending_request["delay_count"] += 1

                if self.pending_request["delay_count"] > max_delay:
                    missed_deadline = True
                    reward -= 2.0
                    scheduled_cost = baseline_cost
                    self.success_count += 1
                    self.pending_request = None

            else:
                scheduled_cost = pred
                reward = 0.0

        self.total_reward += reward
        self.total_baseline_cost += baseline_cost
        self.total_scheduled_cost += scheduled_cost

        self.decision_rows.append(
            {
                "datetime": dt,
                "predicted_demand_kw": pred,
                "actual_demand_kw": actual,
                "appliance_name": appliance_name,
                "appliance_power_kw": appliance_power,
                "delay_count": delay_count,
                "max_delay": max_delay,
                "decision": decision,
                "forced_run": forced_run,
                "missed_deadline": missed_deadline,
                "baseline_cost": baseline_cost,
                "scheduled_cost": scheduled_cost,
                "reward": reward,
            }
        )

        self.idx += 1
        done = self.idx >= len(self.df)
        next_state = self._get_state() if not done else np.zeros(4, dtype=np.float32)

        info["total_reward"] = self.total_reward
        info["total_baseline_cost"] = self.total_baseline_cost
        info["total_scheduled_cost"] = self.total_scheduled_cost
        info["success_rate"] = (
            self.success_count / self.total_requests if self.total_requests > 0 else 1.0
        )

        return next_state, reward, done, info