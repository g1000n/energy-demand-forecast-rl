from pathlib import Path
import json

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
LOGS = EXPERIMENTS / "logs"


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


def format_bool(value):
    if pd.isna(value):
        return "No"
    return "Yes" if bool(value) else "No"


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
        "This tab shows how well the forecasting models predicted next-hour household "
        "electricity demand on the test portion of the dataset."
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
        if available_pred_cols and "actual" in df.columns:
            model_choice = st.selectbox(
                "Select prediction series",
                available_pred_cols,
                index=min(1, len(available_pred_cols) - 1),
                key="forecast_model_choice"
            )

            sort_latest = st.checkbox("Show most recent rows first", value=False, key="forecast_recent_toggle")
            plot_df = df.sort_values("datetime", ascending=not sort_latest) if "datetime" in df.columns else df.copy()

            max_points = st.slider(
                "Number of points to display",
                100, 1000, 300, 50,
                key="forecast_points_slider"
            )
            plot_df = plot_df.head(max_points)

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
        display_df = df.copy()
        if "datetime" in display_df.columns:
            display_df = display_df.sort_values("datetime", ascending=False)
        st.dataframe(display_df.head(20), use_container_width=True)
    else:
        st.warning("forecast_predictions.csv not found.")


# -------------------------------
# NLP Tab
# -------------------------------
with tabs[1]:
    st.header("NLP Supporting Module")
    st.write(
        "This module performs text classification on short demand-context descriptions "
        "generated from the household energy data. Its role is to provide an interpretable "
        "text-based demand trend layer that supports the forecasting and scheduling pipeline."
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

        nlp_display = nlp_preds.copy()
        if "datetime" in nlp_display.columns:
            nlp_display["datetime"] = pd.to_datetime(nlp_display["datetime"], errors="coerce")
            nlp_display = nlp_display.sort_values("datetime", ascending=False)

        label_options = ["All"]
        if "predicted_label" in nlp_display.columns:
            label_options += sorted(nlp_display["predicted_label"].dropna().unique().tolist())

        selected_label = st.selectbox(
            "Filter by predicted label",
            label_options,
            key="nlp_label_filter"
        )

        if selected_label != "All" and "predicted_label" in nlp_display.columns:
            nlp_display = nlp_display[nlp_display["predicted_label"] == selected_label]

        search_text = st.text_input("Search NLP text", key="nlp_search_text").strip()
        if search_text and "text" in nlp_display.columns:
            nlp_display = nlp_display[
                nlp_display["text"].astype(str).str.contains(search_text, case=False, na=False)
            ]

        nlp_rows = st.slider("NLP rows to show", 10, 200, 20, 10, key="nlp_rows_limit")

        show_cols = [c for c in ["datetime", "text", "true_label", "predicted_label"] if c in nlp_display.columns]
        if show_cols:
            st.dataframe(nlp_display[show_cols].head(nlp_rows), use_container_width=True)

    st.info(
        "Interpretation: the NLP component classifies demand-context text such as "
        "'high evening demand on weekday' into trend categories like increase, stable, or decrease. "
        "This supports the system by making energy conditions easier to interpret in human-readable form."
    )


# -------------------------------
# RL Scheduling Tab
# -------------------------------
with tabs[2]:
    st.header("RL Scheduling Results")
    st.write(
        "This tab shows the results of the Q-learning scheduler in an offline simulation. "
        "The agent decides whether a simulated flexible appliance should run now or be delayed "
        "within a limited allowable window to reduce proxy cost."
    )

    st.info(
        "Important: the RL environment uses simulated flexible appliance requests "
        "(such as dishwasher, washing machine, and water heater) placed on top of real household demand data. "
        "This is because the dataset provides household energy usage patterns, but not direct schedulable appliance events."
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

        preview_cols = [
            c for c in [
                "datetime",
                "predicted_demand_kw",
                "actual_demand_kw",
                "appliance_name",
                "appliance_power_kw",
                "delay_count",
                "max_delay",
                "decision",
                "forced_run",
                "missed_deadline",
                "baseline_cost",
                "scheduled_cost",
                "reward"
            ]
            if c in rl_decisions.columns
        ]

        rl_display = rl_decisions.copy()
        if "datetime" in rl_display.columns:
            rl_display["datetime"] = pd.to_datetime(rl_display["datetime"], errors="coerce")
            rl_display = rl_display.sort_values("datetime", ascending=False)

        appliance_filter = ["All"]
        if "appliance_name" in rl_display.columns:
            appliance_filter += sorted(
                [x for x in rl_display["appliance_name"].dropna().unique().tolist() if x != "none"]
            )

        selected_appliance = st.selectbox(
            "Filter by appliance",
            appliance_filter,
            key="rl_appliance_filter"
        )

        if selected_appliance != "All" and "appliance_name" in rl_display.columns:
            rl_display = rl_display[rl_display["appliance_name"] == selected_appliance]

        decision_filter = ["All"]
        if "decision" in rl_display.columns:
            decision_filter += sorted(rl_display["decision"].dropna().unique().tolist())

        selected_decision = st.selectbox(
            "Filter by decision",
            decision_filter,
            key="rl_decision_filter"
        )

        if selected_decision != "All" and "decision" in rl_display.columns:
            rl_display = rl_display[rl_display["decision"] == selected_decision]

        row_limit = st.slider("Rows to show", 10, 200, 50, 10, key="rl_rows_limit")

        if preview_cols:
            pretty_rl = rl_display[preview_cols].copy()
            if "forced_run" in pretty_rl.columns:
                pretty_rl["forced_run"] = pretty_rl["forced_run"].apply(format_bool)
            if "missed_deadline" in pretty_rl.columns:
                pretty_rl["missed_deadline"] = pretty_rl["missed_deadline"].apply(format_bool)
            st.dataframe(pretty_rl.head(row_limit), use_container_width=True)
        else:
            st.dataframe(rl_display.head(row_limit), use_container_width=True)

        if "decision" in rl_decisions.columns:
            st.subheader("Decision Distribution")
            decision_counts = rl_decisions["decision"].fillna("unknown").value_counts()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(decision_counts.index.astype(str), decision_counts.values)
            ax.set_title("Distribution of RL Decisions")
            ax.set_xlabel("Decision")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    st.success(
        "How to interpret this tab: the baseline policy runs a waiting appliance immediately. "
        "The RL policy may delay a flexible appliance if that reduces current proxy cost, but it must still respect simple delay limits."
    )


# -------------------------------
# Simulation Walkthrough Tab
# -------------------------------
with tabs[3]:
    st.header("Simulation Walkthrough")
    st.write(
        "This tab shows the RL component as a chronological simulation. "
        "At each hour, the system observes predicted demand, checks whether a flexible appliance request exists, "
        "chooses an action, and computes the resulting proxy cost and reward."
    )

    if rl_decisions is not None:
        df = rl_decisions.copy()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).copy()
            df["date"] = df["datetime"].dt.date.astype(str)

        st.subheader("Saved RL Decisions File")
        st.write("Available columns in `rl_decisions_sample.csv`:")
        st.code(", ".join(df.columns.astype(str).tolist()))

        has_datetime = "datetime" in df.columns and not df.empty
        if has_datetime:
            available_dates = sorted(df["date"].dropna().unique().tolist())
            if available_dates:
                selected_date = st.selectbox(
                    "Choose a simulated day",
                    available_dates,
                    key="sim_selected_date"
                )

                day_df = df[df["date"] == selected_date].copy().sort_values("datetime")

                st.write(f"Number of decision rows for this day: {len(day_df)}")

                if len(day_df) < 6:
                    st.warning(
                        "This selected day has only a few appliance decision rows, so the line charts may look sparse or jumpy. "
                        "That does not necessarily mean the model is bad; it means few scheduling events occurred on this date."
                    )

                st.subheader(f"Hourly Simulation for {selected_date}")

                show_cols = [
                    c for c in [
                        "datetime",
                        "predicted_demand_kw",
                        "actual_demand_kw",
                        "appliance_name",
                        "appliance_power_kw",
                        "delay_count",
                        "max_delay",
                        "decision",
                        "forced_run",
                        "missed_deadline",
                        "baseline_cost",
                        "scheduled_cost",
                        "reward"
                    ]
                    if c in day_df.columns
                ]

                if show_cols:
                    pretty_df = day_df[show_cols].copy()
                    if "forced_run" in pretty_df.columns:
                        pretty_df["forced_run"] = pretty_df["forced_run"].apply(format_bool)
                    if "missed_deadline" in pretty_df.columns:
                        pretty_df["missed_deadline"] = pretty_df["missed_deadline"].apply(format_bool)
                    st.dataframe(pretty_df, use_container_width=True)
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

                if {"datetime", "baseline_cost", "scheduled_cost"}.issubset(day_df.columns):
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(day_df["datetime"], day_df["baseline_cost"], label="Baseline Cost")
                    ax.plot(day_df["datetime"], day_df["scheduled_cost"], label="Scheduled Cost")
                    ax.set_title("Baseline Cost vs RL Scheduled Cost")
                    ax.set_xlabel("Hour")
                    ax.set_ylabel("Proxy Cost")
                    ax.legend()
                    st.pyplot(fig)

                st.subheader("Step-by-Step Explanation")
                step_idx = st.slider(
                    "Choose a simulation step",
                    0,
                    max(len(day_df) - 1, 0),
                    0,
                    1,
                    key="sim_step_slider"
                )
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
                if "appliance_power_kw" in row.index:
                    st.write(f"**Appliance power:** {row['appliance_power_kw']:.3f} kW")
                if "delay_count" in row.index and "max_delay" in row.index:
                    st.write(f"**Delay progress:** {int(row['delay_count'])} / {int(row['max_delay'])}")
                if "decision" in row.index:
                    st.write(f"**RL action:** {row['decision']}")
                if "forced_run" in row.index:
                    st.write(f"**Forced run due to delay limit:** {format_bool(row['forced_run'])}")
                if "missed_deadline" in row.index:
                    st.write(f"**Missed deadline:** {format_bool(row['missed_deadline'])}")
                if "baseline_cost" in row.index:
                    st.write(f"**Baseline cost (immediate run):** {row['baseline_cost']:.3f}")
                if "scheduled_cost" in row.index:
                    st.write(f"**RL scheduled cost:** {row['scheduled_cost']:.3f}")
                if "reward" in row.index:
                    st.write(f"**Reward at this step:** {row['reward']:.3f}")

                explanation = []
                if "decision" in row.index:
                    if row["decision"] == "run_now":
                        explanation.append(
                            "The agent chose to run the appliance in the current hour, so the appliance load was added immediately."
                        )
                    elif row["decision"] == "delay":
                        explanation.append(
                            "The agent chose to delay the appliance, which avoided adding its load to the current hour."
                        )
                    elif row["decision"] == "no_action":
                        explanation.append(
                            "No appliance request was waiting during this hour, so no scheduling action was needed."
                        )

                if "forced_run" in row.index and bool(row["forced_run"]):
                    explanation.append(
                        "This was a forced run because the appliance had already reached its maximum allowed delay."
                    )

                if "missed_deadline" in row.index and bool(row["missed_deadline"]):
                    explanation.append(
                        "A missed deadline penalty was applied because delaying exceeded the configured limit."
                    )

                if explanation:
                    for item in explanation:
                        st.write(f"- {item}")

                st.success(
                    "Read this tab like a timeline: forecasting provides the demand context, "
                    "the RL scheduler makes a decision for a simulated flexible appliance, and the system compares the RL outcome against an immediate-run baseline."
                )
            else:
                st.warning("The datetime column exists, but no valid dates were found.")
        else:
            st.warning(
                "No valid datetime column found in rl_decisions_sample.csv. "
                "The app can still show the decisions table, but the timeline view needs the RL file to be regenerated with datetime included."
            )
            st.dataframe(df.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive What-If Simulator")
        st.write(
            "This mini-simulation is for explanation only. "
            "It does not retrain the RL agent live. Instead, it demonstrates how proxy cost changes depending on "
            "predicted demand, appliance choice, and whether the appliance is run immediately or delayed."
        )

        appliance_choice = st.selectbox(
            "Choose appliance",
            ["dishwasher", "washing_machine", "water_heater"],
            key="what_if_appliance"
        )

        appliance_specs = {
            "dishwasher": {"power_kw": 1.0, "max_delay": 3},
            "washing_machine": {"power_kw": 1.2, "max_delay": 2},
            "water_heater": {"power_kw": 0.8, "max_delay": 1},
        }

        selected_hour = st.slider(
            "Select simulated hour of day",
            0, 23, 18, 1,
            key="what_if_hour"
        )
        predicted_kw = st.number_input(
            "Predicted demand (kW)",
            min_value=0.0, value=2.5, step=0.1,
            key="what_if_pred"
        )
        actual_kw = st.number_input(
            "Actual demand (kW)",
            min_value=0.0, value=2.7, step=0.1,
            key="what_if_actual"
        )
        current_delay = st.slider(
            "Current delay count",
            0,
            appliance_specs[appliance_choice]["max_delay"],
            0,
            1,
            key="what_if_delay_count"
        )

        action_choice = st.radio(
            "Choose RL action",
            ["run_now", "delay"],
            key="what_if_action"
        )

        appliance_power = appliance_specs[appliance_choice]["power_kw"]
        max_delay = appliance_specs[appliance_choice]["max_delay"]

        baseline_cost = predicted_kw + appliance_power

        forced_run = False
        missed_deadline = False

        if action_choice == "run_now":
            scheduled_cost = predicted_kw + appliance_power
            if predicted_kw < 1.0:
                reward = 0.8
            elif predicted_kw < 1.8:
                reward = 0.3
            elif predicted_kw < 2.3:
                reward = -0.1
            else:
                reward = -0.6
            final_action = "run_now"
        else:
            if current_delay >= max_delay:
                forced_run = True
                final_action = "run_now"
                scheduled_cost = predicted_kw + appliance_power
                if predicted_kw < 1.0:
                    reward = 0.5
                elif predicted_kw < 1.8:
                    reward = 0.0
                elif predicted_kw < 2.3:
                    reward = -0.4
                else:
                    reward = -0.9
            else:
                final_action = "delay"
                scheduled_cost = predicted_kw
                cost_savings = baseline_cost - scheduled_cost
                delay_penalty = 0.15 * (current_delay + 1)

                if predicted_kw < 1.0:
                    timing_penalty = 0.8
                elif predicted_kw < 1.8:
                    timing_penalty = 0.3
                elif predicted_kw < 2.3:
                    timing_penalty = 0.0
                else:
                    timing_penalty = -0.2

                reward = cost_savings - delay_penalty - timing_penalty

                if current_delay + 1 > max_delay:
                    missed_deadline = True
                    reward -= 2.0

        st.markdown("### What-If Result")
        st.write(f"**Hour:** {selected_hour}:00")
        st.write(f"**Appliance:** {appliance_choice}")
        st.write(f"**Appliance power:** {appliance_power:.2f} kW")
        st.write(f"**Predicted demand:** {predicted_kw:.2f} kW")
        st.write(f"**Actual demand:** {actual_kw:.2f} kW")
        st.write(f"**Current delay count:** {current_delay}")
        st.write(f"**Maximum delay allowed:** {max_delay}")
        st.write(f"**Baseline cost (immediate run):** {baseline_cost:.2f}")
        st.write(f"**Final scheduled action:** {final_action}")
        st.write(f"**Scheduled cost:** {scheduled_cost:.2f}")
        st.write(f"**Reward:** {reward:.2f}")
        st.write(f"**Forced run:** {'Yes' if forced_run else 'No'}")
        st.write(f"**Missed deadline:** {'Yes' if missed_deadline else 'No'}")

        if final_action == "delay":
            st.success(
                "The appliance was delayed, so its load was not added to the current hour. "
                "This can be helpful when current demand is high, but delaying during a low-demand window can be a worse choice."
            )
        else:
            if forced_run:
                st.warning(
                    "The appliance was forced to run because it reached its maximum allowed delay."
                )
            else:
                st.info(
                    "The appliance was run immediately, so its power was added to the current hour."
                )

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
4. A Q-learning scheduler simulates flexible appliance scheduling decisions.
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
        "The dataset provides real household demand history, while the RL environment adds simulated flexible appliance requests "
        "to evaluate load shifting and proxy cost reduction."
    )

    st.success(
        "Final interpretation: the project does not try to reduce total energy demand directly. "
        "Instead, it forecasts demand and schedules flexible appliance usage at better times to reduce proxy cost."
    )