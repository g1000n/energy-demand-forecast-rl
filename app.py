from pathlib import Path
import json

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

st.set_page_config(page_title="Energy Demand Forecasting + RL Dashboard", layout="wide")
st.title("Energy Demand Forecasting + RL Load Scheduling")
st.caption("6INTELSY Final Project Dashboard")

# -------------------------------
# Helpers
# -------------------------------
def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None

def safe_read_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

forecast_metrics = safe_read_csv(RESULTS / "forecast_metrics.csv")
forecast_predictions = safe_read_csv(RESULTS / "forecast_predictions.csv")
nlp_metrics = safe_read_csv(RESULTS / "nlp_metrics.csv")
rl_metrics = safe_read_csv(RESULTS / "rl_metrics_by_seed.csv")
rl_decisions = safe_read_csv(RESULTS / "rl_decisions_sample.csv")
rl_summary = safe_read_json(LOGS / "rl_summary.json")

tabs = st.tabs([
    "Forecasting",
    "NLP",
    "RL Scheduling",
    "Simulation Walkthrough",
    "System Summary"
])

# -------------------------------
# Forecasting Tab
# -------------------------------
with tabs[0]:
    st.header("Forecasting Results")
    st.write(
        "This tab shows how well the forecasting models predicted next-hour electricity demand "
        "on the test portion of the dataset."
    )

    if forecast_metrics is not None:
        st.subheader("Forecast Metrics")
        st.dataframe(forecast_metrics, use_container_width=True)
    else:
        st.warning("forecast_metrics.csv not found.")

    if forecast_predictions is not None:
        df = forecast_predictions.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        st.subheader("Actual vs Predicted Demand")

        available_pred_cols = [c for c in ["baseline_pred", "lstm_pred", "tcn_pred"] if c in df.columns]
        if available_pred_cols:
            model_choice = st.selectbox(
                "Select prediction series",
                available_pred_cols,
                index=min(1, len(available_pred_cols) - 1)
            )

            max_points = st.slider("Number of points to display", 100, 1000, 300, 50)
            plot_df = df.head(max_points)

            fig, ax = plt.subplots(figsize=(12, 5))
            if "datetime" in plot_df.columns:
                ax.plot(plot_df["datetime"], plot_df["actual"], label="Actual")
                ax.plot(plot_df["datetime"], plot_df[model_choice], label=model_choice)
                ax.set_xlabel("Datetime")
            else:
                ax.plot(plot_df["actual"].values, label="Actual")
                ax.plot(plot_df[model_choice].values, label=model_choice)
                ax.set_xlabel("Step")

            ax.set_title("Actual vs Predicted Demand")
            ax.set_ylabel("Demand (kW)")
            ax.legend()
            st.pyplot(fig)

            st.info(
                "Interpretation: for each test hour, the model predicts the next-hour demand. "
                "The closer the prediction line is to the actual line, the better the model performed."
            )

        st.subheader("Forecast Predictions Sample")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("forecast_predictions.csv not found.")

# -------------------------------
# NLP Tab
# -------------------------------
with tabs[1]:
    st.header("NLP Supporting Module")
    st.write(
        "This module performs text classification on short demand-context descriptions generated from "
        "the household energy data. Its role is to provide an interpretable text-based demand trend layer "
        "that supports the forecasting and scheduling pipeline."
    )

    if nlp_metrics is not None:
        st.subheader("NLP Metrics")
        st.dataframe(nlp_metrics, use_container_width=True)
    else:
        st.warning("nlp_metrics.csv not found.")

    cm_path = RESULTS / "nlp_confusion_matrix.png"
    if cm_path.exists():
        st.subheader("Confusion Matrix")
        st.image(str(cm_path), use_container_width=True)

    nlp_preds = safe_read_csv(RESULTS / "nlp_predictions.csv")
    if nlp_preds is not None:
        st.subheader("Example NLP Predictions")
        show_cols = [c for c in ["datetime", "text", "true_label", "predicted_label"] if c in nlp_preds.columns]
        st.dataframe(nlp_preds[show_cols].head(20), use_container_width=True)

    st.info(
        "Interpretation: the NLP component classifies demand-context text such as "
        "'high evening demand on weekday' into trend categories like increase, stable, or decrease. "
        "This supports the pipeline by making forecast conditions easier to interpret in human-readable form."
    )

# -------------------------------
# RL Scheduling Tab
# -------------------------------
with tabs[2]:
    st.header("RL Scheduling Results")
    st.write(
        "This tab shows the results of the Q-learning scheduler after training in an offline simulation. "
        "It compares RL scheduling against a baseline policy that runs appliances immediately."
    )

    if rl_summary is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Final Reward", f"{rl_summary.get('mean_final_reward', 0):.2f}")
        c2.metric("Mean Success Rate", f"{rl_summary.get('mean_success_rate', 0):.2f}")
        c3.metric("Mean Cost Reduction", f"{rl_summary.get('mean_cost_reduction', 0):.2f}")

        c4, c5 = st.columns(2)
        c4.metric("Mean Baseline Cost", f"{rl_summary.get('mean_baseline_cost', 0):.2f}")
        c5.metric("Mean Scheduled Cost", f"{rl_summary.get('mean_scheduled_cost', 0):.2f}")
    else:
        st.warning("rl_summary.json not found.")

    if rl_metrics is not None:
        st.subheader("RL Metrics by Seed")
        st.dataframe(rl_metrics, use_container_width=True)
    else:
        st.warning("rl_metrics_by_seed.csv not found.")

    curve_path = RESULTS / "rl_learning_curves.png"
    if curve_path.exists():
        st.subheader("RL Learning Curves")
        st.image(str(curve_path), use_container_width=True)

    if rl_decisions is not None:
        st.subheader("Sample Scheduling Decisions")
        st.dataframe(rl_decisions.head(50), use_container_width=True)

    st.info(
        "Interpretation: the RL agent learned a policy for deciding whether a flexible appliance should "
        "run now or be delayed. The reported reward and cost reduction summarize how that policy performed."
    )

# -------------------------------
# Simulation Walkthrough Tab
# -------------------------------
with tabs[3]:
    st.header("Simulation Walkthrough")
    st.write(
        "This tab is the easiest way to understand the system as a mini simulation. "
        "It shows a chronological view of what happened during RL scheduling."
    )

    if rl_decisions is not None:
        df = rl_decisions.copy()

        st.subheader("Saved RL Decisions File")
        st.write("Available columns in `rl_decisions_sample.csv`:")
        st.code(", ".join(df.columns.astype(str).tolist()))

        has_datetime = "datetime" in df.columns
        if has_datetime:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["date"] = df["datetime"].dt.date.astype(str)

            available_dates = sorted(df["date"].dropna().unique().tolist())
            if available_dates:
                selected_date = st.selectbox("Choose a simulated day", available_dates)
                day_df = df[df["date"] == selected_date].copy().sort_values("datetime")

                st.subheader(f"Hourly Simulation for {selected_date}")

                show_cols = [c for c in [
                    "datetime",
                    "predicted_demand_kw",
                    "actual_demand_kw",
                    "appliance_name",
                    "decision",
                    "baseline_cost",
                    "scheduled_cost",
                    "reward"
                ] if c in day_df.columns]

                if show_cols:
                    st.dataframe(day_df[show_cols], use_container_width=True)
                else:
                    st.dataframe(day_df, use_container_width=True)

                if {"datetime", "predicted_demand_kw", "actual_demand_kw"}.issubset(day_df.columns):
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(day_df["datetime"], day_df["predicted_demand_kw"], label="Predicted Demand")
                    ax.plot(day_df["datetime"], day_df["actual_demand_kw"], label="Actual Demand")
                    ax.set_title("Predicted vs Actual Demand During Simulation")
                    ax.set_xlabel("Hour")
                    ax.set_ylabel("Demand (kW)")
                    ax.legend()
                    st.pyplot(fig)

                st.subheader("Step-by-Step Explanation")
                step_idx = st.slider("Choose a simulation step", 0, max(len(day_df) - 1, 0), 0, 1)
                row = day_df.iloc[step_idx]

                st.markdown("### Current Step")
                if "datetime" in row.index:
                    st.write(f"**Hour:** {row['datetime']}")
                if "predicted_demand_kw" in row.index:
                    st.write(f"**Predicted demand:** {row['predicted_demand_kw']:.3f} kW")
                if "actual_demand_kw" in row.index:
                    st.write(f"**Actual demand:** {row['actual_demand_kw']:.3f} kW")
                if "appliance_name" in row.index:
                    st.write(f"**Appliance request:** {row['appliance_name']}")
                if "decision" in row.index:
                    st.write(f"**RL action:** {row['decision']}")
                if "baseline_cost" in row.index:
                    st.write(f"**Baseline cost (run immediately):** {row['baseline_cost']:.3f}")
                if "scheduled_cost" in row.index:
                    st.write(f"**Scheduled cost (RL policy):** {row['scheduled_cost']:.3f}")
                if "reward" in row.index:
                    st.write(f"**Reward at this step:** {row['reward']:.3f}")

                st.success(
                    "Read this tab like a timeline: at each hour, the system forecasts demand, "
                    "checks whether a flexible appliance is waiting, chooses an action, and computes cost/reward."
                )
            else:
                st.warning("The datetime column exists, but no valid dates were found.")
        else:
            st.warning(
                "No datetime column found in rl_decisions_sample.csv. "
                "The app can still show the saved decisions table below, but the timeline view needs "
                "the RL file to be regenerated with datetime included."
            )
            st.dataframe(df.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive What-If Simulator")
        st.write(
            "This is a manual mini-simulation to help explain the project. "
            "It does not retrain the RL agent live; it just demonstrates the effect of choosing run_now or delay."
        )

        appliance_choice = st.selectbox(
            "Choose appliance",
            ["dishwasher", "washing_machine", "water_heater"]
        )

        appliance_power_map = {
            "dishwasher": 1.0,
            "washing_machine": 1.2,
            "water_heater": 0.8,
        }

        selected_hour = st.slider("Select simulated hour of day", 0, 23, 18, 1)
        predicted_kw = st.number_input("Predicted demand (kW)", min_value=0.0, value=2.5, step=0.1)
        actual_kw = st.number_input("Actual demand (kW)", min_value=0.0, value=2.7, step=0.1)
        action_choice = st.radio("Choose RL action", ["run_now", "delay"])

        appliance_power = appliance_power_map[appliance_choice]

        baseline_cost = predicted_kw + appliance_power
        if action_choice == "run_now":
            scheduled_cost = predicted_kw + appliance_power
        else:
            scheduled_cost = predicted_kw

        reward = baseline_cost - scheduled_cost

        st.markdown("### What-If Result")
        st.write(f"**Hour:** {selected_hour}:00")
        st.write(f"**Appliance:** {appliance_choice}")
        st.write(f"**Predicted demand:** {predicted_kw:.2f} kW")
        st.write(f"**Actual demand:** {actual_kw:.2f} kW")
        st.write(f"**Baseline cost (run immediately):** {baseline_cost:.2f}")
        st.write(f"**Scheduled cost:** {scheduled_cost:.2f}")
        st.write(f"**Reward:** {reward:.2f}")

        if action_choice == "delay":
            st.success("The appliance was delayed, so the scheduler avoided adding its load to the current hour.")
        else:
            st.info("The appliance was run immediately, so its power was added to the current hour.")

    else:
        st.warning("RL decisions file not found. Run the pipeline first.")

# -------------------------------
# Summary Tab
# -------------------------------
with tabs[4]:
    st.header("System Summary")
    st.markdown("""
### What the system does
1. Historical household electricity data is preprocessed into hourly observations.
2. Forecasting models predict next-hour electricity demand.
3. The NLP auxiliary module classifies simple demand-related text.
4. A Q-learning scheduler simulates appliance scheduling decisions.
5. The system compares RL scheduling against a baseline immediate-run policy.

### Expected input
- Raw household electricity file: `household_power_consumption.txt`

### Expected outputs
- Processed hourly dataset
- Forecast predictions and metrics
- NLP metrics and confusion matrix
- RL learning curves, decisions, and cost reduction metrics
""")

    st.info(
        "Important: this project is an offline simulation prototype. "
        "It does not wait in real time. Instead, it steps through the test dataset hour by hour."
    )