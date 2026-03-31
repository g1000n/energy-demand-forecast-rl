from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.energy_env import EnergySchedulingEnv


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_forecast_predictions() -> pd.DataFrame:
    path = RESULTS_DIR / "forecast_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            "results/forecast_predictions.csv not found. Run forecasting first."
        )

    df = pd.read_csv(path)

    expected_candidates = [
        ("datetime", "actual", "lstm_pred"),
        ("datetime", "actual", "tcn_pred"),
        ("datetime", "actual", "baseline_pred"),
    ]

    valid = False
    for cols in expected_candidates:
        if all(c in df.columns for c in cols):
            valid = True
            break

    if not valid:
        raise ValueError(
            "forecast_predictions.csv must contain at least datetime, actual, and one prediction column."
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    return df


def discretize_state(state: np.ndarray) -> Tuple[int, int, int, int]:
    demand_bin = int(state[0])
    hour_bin = min(int(state[1] * 4), 3)
    appliance_id = int(state[2])
    delay_bin = min(int(state[3] * 3), 2)
    return demand_bin, hour_bin, appliance_id, delay_bin


def choose_action(
    q_table: np.ndarray,
    state_key: Tuple[int, int, int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, 2))
    return int(np.argmax(q_table[state_key]))


def train_q_learning(
    env_df: pd.DataFrame,
    prediction_col: str = "lstm_pred",
    episodes: int = 30,
    alpha: float = 0.10,
    gamma: float = 0.95,
    epsilon_start: float = 1.00,
    epsilon_end: float = 0.10,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)

    q_table = np.zeros((3, 4, 4, 3, 2), dtype=np.float32)
    rewards_per_episode: List[float] = []

    for episode in range(episodes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / max(episodes - 1, 1))
        env = EnergySchedulingEnv(
            df=env_df,
            predicted_demand_col=prediction_col,
            actual_demand_col="actual",
            seed=seed + episode,
        )

        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_key = discretize_state(state)
            action = choose_action(q_table, state_key, epsilon, rng)

            next_state, reward, done, _ = env.step(action)
            next_key = discretize_state(next_state)

            best_next = np.max(q_table[next_key])
            old_value = q_table[state_key][action]
            new_value = old_value + alpha * (reward + gamma * best_next - old_value)
            q_table[state_key][action] = new_value

            state = next_state
            episode_reward += reward

        rewards_per_episode.append(episode_reward)

    eval_env = EnergySchedulingEnv(
        df=env_df,
        predicted_demand_col=prediction_col,
        actual_demand_col="actual",
        seed=seed + 999,
    )

    state = eval_env.reset()
    done = False

    while not done:
        state_key = discretize_state(state)
        action = int(np.argmax(q_table[state_key]))
        next_state, _, done, info = eval_env.step(action)
        state = next_state

    decisions_df = pd.DataFrame(eval_env.decision_rows)

    result = {
        "seed": seed,
        "prediction_col": prediction_col,
        "episode_rewards": rewards_per_episode,
        "final_reward": float(rewards_per_episode[-1] if rewards_per_episode else 0.0),
        "mean_reward": float(np.mean(rewards_per_episode) if rewards_per_episode else 0.0),
        "baseline_cost": float(info.get("total_baseline_cost", 0.0)),
        "scheduled_cost": float(info.get("total_scheduled_cost", 0.0)),
        "cost_reduction": float(
            info.get("total_baseline_cost", 0.0) - info.get("total_scheduled_cost", 0.0)
        ),
        "success_rate": float(info.get("success_rate", 1.0)),
        "decisions_df": decisions_df,
        "q_table": q_table,
    }
    return result


def save_learning_curve(all_runs: List[Dict]) -> None:
    plt.figure(figsize=(10, 5))
    for run in all_runs:
        rewards = run["episode_rewards"]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"seed={run['seed']}")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("RL Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rl_learning_curves.png")
    plt.close()


def save_metrics(all_runs: List[Dict]) -> None:
    rows = []
    for run in all_runs:
        rows.append(
            {
                "seed": run["seed"],
                "prediction_col": run["prediction_col"],
                "final_reward": run["final_reward"],
                "mean_reward": run["mean_reward"],
                "baseline_cost": run["baseline_cost"],
                "scheduled_cost": run["scheduled_cost"],
                "cost_reduction": run["cost_reduction"],
                "success_rate": run["success_rate"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "rl_metrics_by_seed.csv", index=False)

    summary = {
        "mean_final_reward": float(metrics_df["final_reward"].mean()),
        "reward_std": float(metrics_df["final_reward"].std(ddof=0)),
        "mean_success_rate": float(metrics_df["success_rate"].mean()),
        "mean_baseline_cost": float(metrics_df["baseline_cost"].mean()),
        "mean_scheduled_cost": float(metrics_df["scheduled_cost"].mean()),
        "mean_cost_reduction": float(metrics_df["cost_reduction"].mean()),
    }

    with open(LOGS_DIR / "rl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ensure_dirs()
    df = load_forecast_predictions()

    seeds = [42, 52, 62]
    all_runs: List[Dict] = []

    for seed in seeds:
        run = train_q_learning(
            env_df=df,
            prediction_col="lstm_pred" if "lstm_pred" in df.columns else "actual",
            episodes=30,
            alpha=0.10,
            gamma=0.95,
            epsilon_start=1.00,
            epsilon_end=0.10,
            seed=seed,
        )
        all_runs.append(run)

    save_learning_curve(all_runs)
    save_metrics(all_runs)

    best_run = max(all_runs, key=lambda x: x["cost_reduction"])
    best_run["decisions_df"].to_csv(RESULTS_DIR / "rl_decisions_sample.csv", index=False)

    print("RL scheduling complete.")
    print(f"Saved: {RESULTS_DIR / 'rl_learning_curves.png'}")
    print(f"Saved: {RESULTS_DIR / 'rl_metrics_by_seed.csv'}")
    print(f"Saved: {RESULTS_DIR / 'rl_decisions_sample.csv'}")
    print(f"Saved: {LOGS_DIR / 'rl_summary.json'}")


if __name__ == "__main__":
    main()