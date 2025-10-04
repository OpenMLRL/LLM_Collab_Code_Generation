import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import wandb


def plot_mt_humaneval_reward():
    """Plot Multi-turn HumanEval rewards from WandB runs with 2 curves: Turn 1 and Turn 2."""

    # ------------------------------------------------------------------
    # W&B setup and project selection
    # ------------------------------------------------------------------

    # Initialize wandb API
    api = wandb.Api()

    # Project settings
    entity = "nu-llpr"
    project = "mlrl-archiv"

    # Get runs and filter for multi-turn HumanEval
    runs = api.runs(f"{entity}/{project}")
    mt_humaneval_runs = [
        run for run in runs if "mt_expert_humaneval_level_reward_all" in run.name
    ]

    print(f"Found {len(mt_humaneval_runs)} Multi-turn HumanEval runs")

    # ------------------------------------------------------------------
    # Collect and normalize run histories
    # ------------------------------------------------------------------
    all_data = []
    for run in mt_humaneval_runs:
        try:
            history = run.history()

            # Use _step column and filter to reasonable range
            if "_step" in history.columns:
                history = history[history["_step"] <= 2200]

            # Add run metadata
            history["run_name"] = run.name
            history["run_id"] = run.id

            all_data.append(history)

        except Exception as e:
            print(f"Error loading {run.name}: {e}")

    if not all_data:
        print("No data found")
        return

    # Combine and clean data
    df = pd.concat(all_data, ignore_index=True)

    # Define the columns we need for multi-turn HumanEval
    required_columns = [
        "_step",
        "eval/turn_1/avg_total_reward",
        "eval/turn_2/avg_total_reward",
        "run_name",
        "run_id",
    ]

    # Check which columns exist and filter accordingly
    existing_columns = [col for col in required_columns if col in df.columns]
    df = df[existing_columns]

    # Drop rows with NaN values in essential columns
    essential_columns = [
        col for col in existing_columns if col not in ["run_name", "run_id"]
    ]
    df = df.dropna(subset=essential_columns)

    print(f"Plotting {len(df)} data points from {df['run_id'].nunique()} runs")

    # Find minimum run length to ensure all runs contribute to every point
    run_lengths = df.groupby("run_id").size()
    min_length = run_lengths.min()
    print(f"Run lengths: min={min_length}, max={run_lengths.max()}")

    # ------------------------------------------------------------------
    # Create uniform x-axis per run (truncate to min length for alignment)
    # ------------------------------------------------------------------
    uniform_data = []
    for run_id in df["run_id"].unique():
        run_data = df[df["run_id"] == run_id].copy()
        run_data = run_data.sort_values("_step")

        # Truncate to minimum length and create uniform x-axis from 0-2200
        run_data = run_data.head(min_length)
        run_data["uniform_step"] = np.linspace(0, 2200, len(run_data))

        uniform_data.append(run_data)

    # Combine uniform data
    df_uniform = pd.concat(uniform_data, ignore_index=True)

    # Aggregate data first
    aggregated = (
        df_uniform.groupby("uniform_step")
        .agg(
            {
                "eval/turn_1/avg_total_reward": ["mean", "std"],
                "eval/turn_2/avg_total_reward": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    aggregated.columns = [
        "uniform_step",
        "turn1_mean",
        "turn1_std",
        "turn2_mean",
        "turn2_std",
    ]

    # ------------------------------------------------------------------
    # Scale aggregated means and stds (total reward assumed in [0,4])
    # ------------------------------------------------------------------
    total_min, total_max = 0, 4.0

    # Scale Turn 1
    aggregated["turn1_mean_scaled"] = (aggregated["turn1_mean"] - total_min) / (
        total_max - total_min
    )
    aggregated["turn1_std_scaled"] = aggregated["turn1_std"] / (total_max - total_min)

    # Scale Turn 2
    aggregated["turn2_mean_scaled"] = (aggregated["turn2_mean"] - total_min) / (
        total_max - total_min
    )
    aggregated["turn2_std_scaled"] = aggregated["turn2_std"] / (total_max - total_min)

    # ------------------------------------------------------------------
    # Plot with bounded error bands
    # ------------------------------------------------------------------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot with error bars manually to ensure they stay in bounds
    def plot_with_bounded_error(x, y_mean, y_std, label, **kwargs):
        y_lower = np.maximum(y_mean - y_std, 0)  # Bound lower at 0
        y_upper = np.minimum(y_mean + y_std, 1)  # Bound upper at 1

        ax.plot(x, y_mean, label=label, linewidth=4, **kwargs)
        ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=kwargs.get("color", None))

    # Plot Turn 1
    plot_with_bounded_error(
        aggregated["uniform_step"],
        aggregated["turn1_mean_scaled"],
        aggregated["turn1_std_scaled"],
        "Turn 1",
        linestyle="-",
        alpha=1,
        color="steelblue",
    )

    # Plot Turn 2
    plot_with_bounded_error(
        aggregated["uniform_step"],
        aggregated["turn2_mean_scaled"],
        aggregated["turn2_std_scaled"],
        "Turn 2",
        linestyle="-",
        alpha=1,
        color="indianred",
    )

    # Style the plot
    ax.set_xlabel("(K) Steps", fontsize=18)
    ax.set_ylabel("Normalized Reward (%)", fontsize=18)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    ax.grid(True, alpha=0.3)
    ax.legend(handlelength=3, fontsize=16, loc="lower right")

    plt.tight_layout()
    plt.ylim(-0.1, 1.05)
    # plt.show()
    plt.savefig(f"./mt_humaneval_{time.time()}.pdf", dpi=300)

    return aggregated


def plot_mt_coophumaneval_reward():
    """Plot Multi-turn HumanEval rewards from WandB runs with 2 curves: Turn 1 and Turn 2."""

    # Initialize wandb API
    api = wandb.Api()

    # Project settings
    entity = "nu-llpr"
    project = "mlrl-archiv"

    # Get runs and filter for multi-turn HumanEval
    runs = api.runs(f"{entity}/{project}")
    mt_humaneval_runs = [
        run for run in runs if "mt_expert_coophumaneval_level_reward_all" in run.name
    ]

    print(f"Found {len(mt_humaneval_runs)} Multi-turn HumanEval runs")

    # Collect data from all runs
    all_data = []
    for run in mt_humaneval_runs:
        try:
            history = run.history()

            # Use _step column and filter to reasonable range
            if "_step" in history.columns:
                history = history[history["_step"] <= 2200]

            # Add run metadata
            history["run_name"] = run.name
            history["run_id"] = run.id

            all_data.append(history)

        except Exception as e:
            print(f"Error loading {run.name}: {e}")

    if not all_data:
        print("No data found")
        return

    # Combine and clean data
    df = pd.concat(all_data, ignore_index=True)

    # Define the columns we need for multi-turn HumanEval
    required_columns = [
        "_step",
        "eval/turn_1/avg_total_reward",
        "eval/turn_2/avg_total_reward",
        "run_name",
        "run_id",
    ]

    # Check which columns exist and filter accordingly
    existing_columns = [col for col in required_columns if col in df.columns]
    df = df[existing_columns]

    # Drop rows with NaN values in essential columns
    essential_columns = [
        col for col in existing_columns if col not in ["run_name", "run_id"]
    ]
    df = df.dropna(subset=essential_columns)

    print(f"Plotting {len(df)} data points from {df['run_id'].nunique()} runs")

    # Find minimum run length to ensure all runs contribute to every point
    run_lengths = df.groupby("run_id").size()
    min_length = run_lengths.min()
    print(f"Run lengths: min={min_length}, max={run_lengths.max()}")

    # Create uniform x-axis mapping for each run, truncated to minimum length
    uniform_data = []
    for run_id in df["run_id"].unique():
        run_data = df[df["run_id"] == run_id].copy()
        run_data = run_data.sort_values("_step")

        # Truncate to minimum length and create uniform x-axis from 0-2200
        run_data = run_data.head(min_length)
        run_data["uniform_step"] = np.linspace(0, 2200, len(run_data))

        uniform_data.append(run_data)

    # Combine uniform data
    df_uniform = pd.concat(uniform_data, ignore_index=True)

    # Aggregate data first
    aggregated = (
        df_uniform.groupby("uniform_step")
        .agg(
            {
                "eval/turn_1/avg_total_reward": ["mean", "std"],
                "eval/turn_2/avg_total_reward": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    aggregated.columns = [
        "uniform_step",
        "turn1_mean",
        "turn1_std",
        "turn2_mean",
        "turn2_std",
    ]

    # Scale the aggregated means and stds
    # Assuming total reward range is 0-4 as in the original code
    total_min, total_max = 0, 4.0

    # Scale Turn 1
    aggregated["turn1_mean_scaled"] = (aggregated["turn1_mean"] - total_min) / (
        total_max - total_min
    )
    aggregated["turn1_std_scaled"] = aggregated["turn1_std"] / (total_max - total_min)

    # Scale Turn 2
    aggregated["turn2_mean_scaled"] = (aggregated["turn2_mean"] - total_min) / (
        total_max - total_min
    )
    aggregated["turn2_std_scaled"] = aggregated["turn2_std"] / (total_max - total_min)

    # Create plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot with error bars manually to ensure they stay in bounds
    def plot_with_bounded_error(x, y_mean, y_std, label, **kwargs):
        y_lower = np.maximum(y_mean - y_std, 0)  # Bound lower at 0
        y_upper = np.minimum(y_mean + y_std, 1)  # Bound upper at 1

        ax.plot(x, y_mean, label=label, linewidth=4, **kwargs)
        ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=kwargs.get("color", None))

    # Plot Turn 1
    plot_with_bounded_error(
        aggregated["uniform_step"],
        aggregated["turn1_mean_scaled"],
        aggregated["turn1_std_scaled"],
        "Turn 1",
        linestyle="-",
        alpha=1,
        color="steelblue",
    )

    # Plot Turn 2
    plot_with_bounded_error(
        aggregated["uniform_step"],
        aggregated["turn2_mean_scaled"],
        aggregated["turn2_std_scaled"],
        "Turn 2",
        linestyle="-",
        alpha=1,
        color="indianred",
    )

    # Style the plot
    ax.set_xlabel("(K) Steps", fontsize=18)
    ax.set_ylabel("Normalized Reward (%)", fontsize=18)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    ax.grid(True, alpha=0.3)
    ax.legend(handlelength=3, fontsize=16, loc="lower right")

    plt.tight_layout()
    plt.ylim(-0.1, 1.05)
    # plt.show()
    plt.savefig(f"./mt_coophumaneval_{time.time()}.pdf", dpi=300)

    return aggregated


# Run the plotting function
if __name__ == "__main__":
    plot_mt_humaneval_reward()
    plot_mt_coophumaneval_reward()
