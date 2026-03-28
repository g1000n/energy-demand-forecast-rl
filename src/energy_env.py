from __future__ import annotations

from dataclasses import dataclass
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
        self.df = forecast_df.copy().reset_index(drop=True)
        self.appliance_templates = appliance_templates
        self.index = 0
        self.successes = 0
        self.failures = 0

    def _get_active_appliance(self, hour: int):
        for appliance in self.appliance_templates:
            if appliance.earliest_start_hour <= hour <= appliance.latest_finish_hour:
                return appliance
        return None

    def _bucket_demand(self, demand: float) -> int:
        if demand < 1.0:
            return 0
        elif demand < 2.0:
            return 1
        elif demand < 3.0:
            return 2
        return 3

    def _bucket_hour(self, hour: int) -> int:
        return min(hour // 3, 7)

    def _get_state(self):
        row = self.df.iloc[self.index]
        predicted = float(row["predicted_demand_kw"])
        hour = int(row["hour"]) if "hour" in row else pd.to_datetime(row["datetime"]).hour
        appliance = self._get_active_appliance(hour)
        has_appliance = 1 if appliance is not None else 0
        return (self._bucket_demand(predicted), self._bucket_hour(hour), has_appliance)

    def reset(self):
        self.index = 0
        self.successes = 0
        self.failures = 0
        return self._get_state()

    def step(self, action: int):
        row = self.df.iloc[self.index]

        dt = pd.to_datetime(row["datetime"]) if "datetime" in row else None
        predicted = float(row["predicted_demand_kw"])
        actual = float(row["actual_demand_kw"])
        hour = int(row["hour"]) if "hour" in row else (dt.hour if dt is not None else 0)

        appliance = self._get_active_appliance(hour)

        reward = 0.0
        baseline_cost = 0.0
        scheduled_cost = 0.0
        appliance_name = "none"

        if appliance is not None:
            appliance_name = appliance.name

            # baseline = always run now
            baseline_cost = predicted + appliance.power_kw

            if action == 0:
                # run now
                scheduled_cost = predicted + appliance.power_kw
                reward = baseline_cost - scheduled_cost
                self.successes += 1
                decision = "run_now"
            else:
                # delay
                scheduled_cost = predicted
                reward = baseline_cost - scheduled_cost
                self.successes += 1
                decision = "delay"
        else:
            baseline_cost = predicted
            scheduled_cost = predicted
            reward = 0.0
            decision = "no_appliance"

        info = {
            "datetime": str(dt) if dt is not None else None,
            "hour": hour,
            "predicted_demand_kw": predicted,
            "actual_demand_kw": actual,
            "appliance_name": appliance_name,
            "decision": decision,
            "baseline_cost": float(baseline_cost),
            "scheduled_cost": float(scheduled_cost),
            "reward": float(reward),
        }

        self.index += 1
        done = self.index >= len(self.df)

        if done:
            next_state = (0, 0, 0)
        else:
            next_state = self._get_state()

        return next_state, reward, done, info