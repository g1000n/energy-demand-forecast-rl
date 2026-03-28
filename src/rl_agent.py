from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.energy_env import ApplianceRequest, EnergySchedulingEnv
from src.utils.common import ensure_dirs, project_root, save_json, set_seed


@dataclass
class RLConfig:
    episodes: int = 40
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    seeds: tuple[int, int, int] = (7, 21, 42)


class QLearningAgent:
    def __init__(self, num_actions: int = 2):
        self.num_actions = num_actions
        self.q_table = np.zeros((4, 8, 2, num_actions), dtype=np.float32)

    def act(self, state: tuple[int, int, int], epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: tuple[int, int, int], action: int, reward: float, next_state: tuple[int, int, int]) -> None:
        alpha, gamma = 0.1, 0.95
        old_q = self.q_table[state][action]
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] = old_q + alpha * (reward + gamma * best_next - old_q)


def _prepare_forecast_df(processed_csv_path, forecast_csv_path):
    import pandas as pd

    actual = pd.read_csv(processed_csv_path)
    preds = pd.read_csv(forecast_csv_path)

    actual["datetime"] = pd.to_datetime(actual["datetime"], errors="coerce")
    preds["datetime"] = pd.to_datetime(preds["datetime"], errors="coerce")

    actual = actual.dropna(subset=["datetime"])
    preds = preds.dropna(subset=["datetime"])

    merged = preds.merge(
        actual[["datetime", "hour", "dayofweek", "is_weekend"]],
        on="datetime",
        how="left"
    )

    merged["predicted_demand_kw"] = merged["lstm_pred"]
    merged["actual_demand_kw"] = merged["actual"]

    return merged[
        ["datetime", "hour", "dayofweek", "is_weekend", "predicted_demand_kw", "actual_demand_kw"]
    ]


def _run_single_seed(forecast_df: pd.DataFrame, seed: int, cfg: RLConfig):
    set_seed(seed)
    appliance_templates = [
        ApplianceRequest("dishwasher", power_kw=1.0, duration_hours=1, earliest_start_hour=19, latest_finish_hour=23),
        ApplianceRequest("washing_machine", power_kw=1.2, duration_hours=1, earliest_start_hour=18, latest_finish_hour=22),
        ApplianceRequest("water_heater", power_kw=0.8, duration_hours=1, earliest_start_hour=6, latest_finish_hour=9),
    ]
    env = EnergySchedulingEnv(forecast_df, appliance_templates)
    agent = QLearningAgent()

    episode_rewards = []
    success_rates = []

    for _ in range(cfg.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state, cfg.epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        denom = max(env.successes + env.failures, 1)
        success_rates.append(env.successes / denom)

    # Deterministic evaluation
    env_eval = EnergySchedulingEnv(forecast_df, appliance_templates)
    state = env_eval.reset()
    done = False
    total_reward = 0.0
    decisions = []
    while not done:
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done, info = env_eval.step(action)
        total_reward += reward
        info["decision"] = "run_now" if action == 0 else "delay"
        decisions.append(info)
        state = next_state

    decisions_df = pd.DataFrame(decisions)
    baseline_cost = float(decisions_df["baseline_cost"].sum())
    scheduled_cost = float(decisions_df["scheduled_cost"].sum())
    rl_success_rate = float(max(env_eval.successes / max(env_eval.successes + env_eval.failures, 1), 0.0))

    return {
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_success_rates": success_rates,
        "final_reward": float(total_reward),
        "baseline_cost": baseline_cost,
        "scheduled_cost": scheduled_cost,
        "success_rate": rl_success_rate,
        "decisions_df": decisions_df,
    }


def run_rl_experiment(processed_csv_path: str, forecast_csv_path: str) -> dict[str, float]:
    ensure_dirs()
    root = project_root()
    results_dir = root / "results"
    logs_dir = root / "logs"
    cfg = RLConfig()
    forecast_df = _prepare_forecast_df(processed_csv_path, forecast_csv_path)

    runs = [_run_single_seed(forecast_df, seed, cfg) for seed in cfg.seeds]

    plt.figure(figsize=(8, 4))
    for run in runs:
        plt.plot(run["episode_rewards"], label=f"seed {run['seed']}")
    plt.title("RL Learning Curves")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rl_learning_curves.png")
    plt.close()

    summary = {
        "mean_final_reward": float(np.mean([r["final_reward"] for r in runs])),
        "reward_std": float(np.std([r["final_reward"] for r in runs])),
        "mean_success_rate": float(np.mean([r["success_rate"] for r in runs])),
        "mean_baseline_cost": float(np.mean([r["baseline_cost"] for r in runs])),
        "mean_scheduled_cost": float(np.mean([r["scheduled_cost"] for r in runs])),
        "mean_cost_reduction": float(np.mean([r["baseline_cost"] - r["scheduled_cost"] for r in runs])),
    }

    pd.DataFrame([
        {
            "seed": run["seed"],
            "final_reward": run["final_reward"],
            "success_rate": run["success_rate"],
            "baseline_cost": run["baseline_cost"],
            "scheduled_cost": run["scheduled_cost"],
            "cost_reduction": run["baseline_cost"] - run["scheduled_cost"],
        }
        for run in runs
    ]).to_csv(results_dir / "rl_metrics_by_seed.csv", index=False)

    best_run = max(runs, key=lambda x: x["final_reward"])
    best_run["decisions_df"].to_csv(results_dir / "rl_decisions_sample.csv", index=False)
    save_json(logs_dir / "rl_summary.json", summary)
    return summary


if __name__ == "__main__":
    root = project_root()
    print(
        run_rl_experiment(
            str(root / "data" / "processed_energy_hourly.csv"),
            str(root / "results" / "forecast_predictions.csv"),
        )
    )
