import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import seaborn as sns
import wandb
from PyPDF2.generic import RectangleObject


def plot_combined_two_panels():
    """Plot two panels: Single Turn and Multi-Turn Average (Turn1+Turn2)/2 with larger fonts."""

    # Initialize wandb API
    api = wandb.Api()

    # Project settings
    entity = "nu-llpr"
    project = "mlrl-archiv"

    # Get runs
    runs = api.runs(f"{entity}/{project}")

    # Separate runs for single-turn and multi-turn
    humaneval_runs = [run for run in runs if "coophumaneval" == run.name]
    mt_expert_runs = [
        run for run in runs if "mt_expert_coophumaneval_level_reward_all" in run.name
    ]

    print(f"Found {len(humaneval_runs)} CoopHumanEval runs")
    print(f"Found {len(mt_expert_runs)} Multi-Turn Expert HumanEval runs")

    # Process single-turn data
    single_turn_data = []
    for run in humaneval_runs:
        try:
            history = run.history()
            if "_step" in history.columns:
                history = history[history["_step"] <= 1500]
            history["run_name"] = run.name
            history["run_id"] = run.id
            single_turn_data.append(history)
        except Exception as e:
            print(f"Error loading {run.name}: {e}")

    # Process multi-turn data
    mt_data = []
    for run in mt_expert_runs:
        try:
            history = run.history()
            if "_step" in history.columns:
                history = history[history["_step"] <= 2200]
            history["run_name"] = run.name
            history["run_id"] = run.id
            mt_data.append(history)
        except Exception as e:
            print(f"Error loading {run.name}: {e}")

    # Process single-turn dataframe
    df_single = pd.concat(single_turn_data, ignore_index=True)

    # Define columns for single-turn
    single_columns = [
        "_step",
        "eval/turn_1/avg_total_reward",
        "eval/turn_1/avg_level_1_reward",
        "eval/turn_1/avg_level_2_reward",
        "eval/turn_1/avg_level_3_reward",
        "eval/turn_1/avg_bonus_reward",
        "run_name",
        "run_id",
    ]

    existing_single_columns = [
        col for col in single_columns if col in df_single.columns
    ]
    df_single = df_single[existing_single_columns]
    essential_single = [
        col for col in existing_single_columns if col not in ["run_name", "run_id"]
    ]
    df_single = df_single.dropna(subset=essential_single)

    # Calculate single-turn component rewards
    df_single["structure_reward"] = df_single["eval/turn_1/avg_level_1_reward"]
    df_single["syntax_reward"] = df_single["eval/turn_1/avg_level_2_reward"]
    df_single["tests_reward"] = (
        df_single["eval/turn_1/avg_level_3_reward"]
        - df_single["eval/turn_1/avg_bonus_reward"]
    )
    df_single["cooperation_reward"] = df_single["eval/turn_1/avg_bonus_reward"]

    # Process multi-turn dataframe
    df_mt = pd.concat(mt_data, ignore_index=True)

    # Define columns for multi-turn
    mt_columns = [
        "_step",
        # Turn 1 columns
        "eval/turn_1/avg_total_reward",
        "eval/turn_1/avg_level_1_reward",
        "eval/turn_1/avg_level_2_reward",
        "eval/turn_1/avg_level_3_reward",
        "eval/turn_1/avg_bonus_reward",
        # Turn 2 columns
        "eval/turn_2/avg_total_reward",
        "eval/turn_2/avg_level_1_reward",
        "eval/turn_2/avg_level_2_reward",
        "eval/turn_2/avg_level_3_reward",
        "eval/turn_2/avg_bonus_reward",
        "run_name",
        "run_id",
    ]

    existing_mt_columns = [col for col in mt_columns if col in df_mt.columns]
    df_mt = df_mt[existing_mt_columns]
    essential_mt = [
        col for col in existing_mt_columns if col not in ["run_name", "run_id"]
    ]
    df_mt = df_mt.dropna(subset=essential_mt)

    # Calculate averaged multi-turn component rewards (Turn1 + Turn2) / 2
    df_mt["avg_structure_reward"] = (
        df_mt["eval/turn_1/avg_level_1_reward"]
        + df_mt["eval/turn_2/avg_level_1_reward"]
    ) / 2
    df_mt["avg_syntax_reward"] = (
        df_mt["eval/turn_1/avg_level_2_reward"]
        + df_mt["eval/turn_2/avg_level_2_reward"]
    ) / 2
    df_mt["avg_tests_reward"] = (
        (
            df_mt["eval/turn_1/avg_level_3_reward"]
            - df_mt["eval/turn_1/avg_bonus_reward"]
        )
        + (
            df_mt["eval/turn_2/avg_level_3_reward"]
            - df_mt["eval/turn_2/avg_bonus_reward"]
        )
    ) / 2
    df_mt["avg_cooperation_reward"] = (
        df_mt["eval/turn_1/avg_bonus_reward"] + df_mt["eval/turn_2/avg_bonus_reward"]
    ) / 2
    df_mt["avg_total_reward"] = (
        df_mt["eval/turn_1/avg_total_reward"] + df_mt["eval/turn_2/avg_total_reward"]
    ) / 2

    # Create uniform x-axis for single-turn
    single_run_lengths = df_single.groupby("run_id").size()
    single_min_length = single_run_lengths.min()

    uniform_single_data = []
    for run_id in df_single["run_id"].unique():
        run_data = df_single[df_single["run_id"] == run_id].copy()
        run_data = run_data.sort_values("_step")
        run_data = run_data.head(single_min_length)
        run_data["uniform_step"] = np.linspace(0, 1500, len(run_data))
        uniform_single_data.append(run_data)

    df_single_uniform = pd.concat(uniform_single_data, ignore_index=True)

    # Create uniform x-axis for multi-turn
    mt_run_lengths = df_mt.groupby("run_id").size()
    mt_min_length = mt_run_lengths.min()

    uniform_mt_data = []
    for run_id in df_mt["run_id"].unique():
        run_data = df_mt[df_mt["run_id"] == run_id].copy()
        run_data = run_data.sort_values("_step")
        run_data = run_data.head(mt_min_length)
        run_data["uniform_step"] = np.linspace(0, 2200, len(run_data))
        uniform_mt_data.append(run_data)

    df_mt_uniform = pd.concat(uniform_mt_data, ignore_index=True)

    # Aggregate single-turn data
    aggregated_single = (
        df_single_uniform.groupby("uniform_step")
        .agg(
            {
                "eval/turn_1/avg_total_reward": ["mean", "std"],
                "structure_reward": ["mean", "std"],
                "syntax_reward": ["mean", "std"],
                "tests_reward": ["mean", "std"],
                "cooperation_reward": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names for single-turn
    aggregated_single.columns = [
        "uniform_step",
        "total_mean",
        "total_std",
        "structure_mean",
        "structure_std",
        "syntax_mean",
        "syntax_std",
        "tests_mean",
        "tests_std",
        "cooperation_mean",
        "cooperation_std",
    ]

    # Aggregate multi-turn AVERAGED data
    aggregated_mt = (
        df_mt_uniform.groupby("uniform_step")
        .agg(
            {
                "avg_total_reward": ["mean", "std"],
                "avg_structure_reward": ["mean", "std"],
                "avg_syntax_reward": ["mean", "std"],
                "avg_tests_reward": ["mean", "std"],
                "avg_cooperation_reward": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names for multi-turn
    aggregated_mt.columns = [
        "uniform_step",
        "total_mean",
        "total_std",
        "structure_mean",
        "structure_std",
        "syntax_mean",
        "syntax_std",
        "tests_mean",
        "tests_std",
        "cooperation_mean",
        "cooperation_std",
    ]

    # Scale single-turn data to 0-100 range
    total_min, total_max = 0, 4.0
    aggregated_single["total_mean_scaled"] = (
        (aggregated_single["total_mean"] - total_min) / (total_max - total_min) * 100
    )
    aggregated_single["total_std_scaled"] = (
        aggregated_single["total_std"] / (total_max - total_min) * 100
    )

    # Scale components for single-turn to 0-100 range
    components = {
        "structure": (0, 1.0),
        "syntax": (0, 0.5),
        "tests": (0, 1.0),
        "cooperation": (0, 1.5),
    }

    for component, (min_val, max_val) in components.items():
        aggregated_single[f"{component}_mean_scaled"] = (
            (aggregated_single[f"{component}_mean"] - min_val)
            / (max_val - min_val)
            * 100
        )
        aggregated_single[f"{component}_std_scaled"] = (
            aggregated_single[f"{component}_std"] / (max_val - min_val) * 100
        )

    # Scale multi-turn AVERAGED data to 0-100 range
    aggregated_mt["total_mean_scaled"] = (
        (aggregated_mt["total_mean"] - total_min) / (total_max - total_min) * 100
    )
    aggregated_mt["total_std_scaled"] = (
        aggregated_mt["total_std"] / (total_max - total_min) * 100
    )

    for component, (min_val, max_val) in components.items():
        aggregated_mt[f"{component}_mean_scaled"] = (
            (aggregated_mt[f"{component}_mean"] - min_val) / (max_val - min_val) * 100
        )
        aggregated_mt[f"{component}_std_scaled"] = (
            aggregated_mt[f"{component}_std"] / (max_val - min_val) * 100
        )

    # Apply smoothing to single-turn (window_size = 2)
    single_window_size = 2
    for col in [
        "total_mean_scaled",
        "total_std_scaled",
        "structure_mean_scaled",
        "structure_std_scaled",
        "syntax_mean_scaled",
        "syntax_std_scaled",
        "tests_mean_scaled",
        "tests_std_scaled",
        "cooperation_mean_scaled",
        "cooperation_std_scaled",
    ]:
        aggregated_single[col] = (
            aggregated_single[col]
            .rolling(window=single_window_size, center=True, min_periods=1)
            .mean()
        )

    # Apply smoothing to multi-turn averaged data (window_size = 1)
    mt_window_size = 1
    for col in [
        "total_mean_scaled",
        "total_std_scaled",
        "structure_mean_scaled",
        "structure_std_scaled",
        "syntax_mean_scaled",
        "syntax_std_scaled",
        "tests_mean_scaled",
        "tests_std_scaled",
        "cooperation_mean_scaled",
        "cooperation_std_scaled",
    ]:
        aggregated_mt[col] = (
            aggregated_mt[col]
            .rolling(window=mt_window_size, center=True, min_periods=1)
            .mean()
        )

    # Create plot with TWO subplots - match writing plot dimensions
    plt.style.use("default")

    # Use same figure size as writing plot for consistency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # Match the spacing used in writing plot
    plt.subplots_adjust(left=0.07, right=0.83, wspace=0.35, bottom=0.20, top=0.90)

    # Define colors
    colors = {
        "Structure": "grey",
        "Syntax": "forestgreen",
        "Tests": "indianred",
        "Cooperation": "orange",
        "Total": "steelblue",
    }

    # Plot function with bounded error for -10 to 110 range
    def plot_with_bounded_error(
        ax, x, y_mean, y_std, label, color, linestyle, alpha=1.0
    ):
        # Ensure error bands stay within 0-100 range even though axis is -10 to 110
        y_lower = np.maximum(y_mean - y_std, 0)
        y_upper = np.minimum(y_mean + y_std, 100)

        ax.plot(
            x,
            y_mean,
            label=label,
            linewidth=5,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

    # INCREASED FONT SIZES
    title_size = 24  # Increased from 20
    label_size = 24  # Increased from 20
    tick_size = 24  # Increased from 20

    # Plot Single Turn (ax1)
    for component in ["Structure", "Syntax", "Tests", "Cooperation"]:
        plot_with_bounded_error(
            ax1,
            aggregated_single["uniform_step"],
            aggregated_single[f"{component.lower()}_mean_scaled"],
            aggregated_single[f"{component.lower()}_std_scaled"],
            component,
            colors[component],
            "--",
            alpha=0.7,
        )

    plot_with_bounded_error(
        ax1,
        aggregated_single["uniform_step"],
        aggregated_single["total_mean_scaled"],
        aggregated_single["total_std_scaled"],
        "Total",
        colors["Total"],
        "-",
        alpha=1.0,
    )

    ax1.set_title("CHE | Single Turn", fontsize=title_size, pad=10)
    ax1.set_xlabel("(K) Steps", fontsize=label_size)
    ax1.set_ylabel("Normalized Return (%)", fontsize=label_size)
    ax1.set_xticks([0, 300, 600, 900, 1200, 1500])
    ax1.set_xticklabels(["0", "0.3", "0.6", "0.9", "1.2", "1.5"], fontsize=tick_size)
    ax1.yaxis.set_tick_params(labelsize=tick_size)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 110)

    # Plot Multi-Turn Average (ax2)
    for component in ["Structure", "Syntax", "Tests", "Cooperation"]:
        plot_with_bounded_error(
            ax2,
            aggregated_mt["uniform_step"],
            aggregated_mt[f"{component.lower()}_mean_scaled"],
            aggregated_mt[f"{component.lower()}_std_scaled"],
            component,
            colors[component],
            "--",
            alpha=0.7,
        )

    plot_with_bounded_error(
        ax2,
        aggregated_mt["uniform_step"],
        aggregated_mt["total_mean_scaled"],
        aggregated_mt["total_std_scaled"],
        "Total",
        colors["Total"],
        "-",
        alpha=1.0,
    )

    ax2.set_title("CHE | Multi-Turn", fontsize=title_size, pad=10)
    ax2.set_xlabel("(K) Steps", fontsize=label_size)
    ax2.set_ylabel("Normalized Return (%)", fontsize=label_size)
    ax2.set_xticks([0, 500, 1000, 1500, 2000])
    ax2.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=tick_size)
    ax2.yaxis.set_tick_params(labelsize=tick_size)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-10, 110)

    # Add legend to match writing plot style
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    legend_elements = []

    # Create custom handles with line and surrounding error band
    for label, color in colors.items():
        # Determine line style based on label
        linestyle = "-" if label == "Total" else "--"
        alpha = 1.0 if label == "Total" else 0.7

        # Create a line for the legend
        line = Line2D(
            [0], [0], color=color, linewidth=5, linestyle=linestyle, alpha=alpha
        )

        # Create a rectangle to represent the error band
        rect = Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2, edgecolor="none")

        # Combine them into a single artist
        legend_elements.append((line, rect, label))

    # Custom legend handler to display both line and error band
    class HandlerLineWithErrorBand(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            line, rect, label = orig_handle

            # Create the error band rectangle
            rect_artist = Rectangle(
                (xdescent, ydescent),
                width,
                height,
                facecolor=rect.get_facecolor(),
                alpha=rect.get_alpha(),
                edgecolor="none",
                transform=trans,
            )

            # Create the line in the middle
            line_artist = Line2D(
                [xdescent, xdescent + width],
                [ydescent + height / 2, ydescent + height / 2],
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                alpha=line.get_alpha(),
                transform=trans,
            )

            return [rect_artist, line_artist]

    # Create the legend with custom handles - match writing plot position
    legend_handles = [(h[0], h[1], h[2]) for h in legend_elements]
    legend_labels = [h[2] for h in legend_elements]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=20,
        frameon=True,
        handlelength=2.2,
        title="Metrics",
        title_fontsize=24,
        handler_map={tuple: HandlerLineWithErrorBand()},
    )

    # Save the main plot WITH legend integrated
    timestamp = time.time()
    plt.savefig(f"./che_two_panel_{timestamp}.pdf", dpi=300, bbox_inches="tight")

    # Close the figure to free memory
    plt.close(fig)

    # Split the PDF into two separate files
    with open(f"./che_two_panel_{timestamp}.pdf", "rb") as file:
        reader = PyPDF2.PdfReader(file)
        left_writer = PyPDF2.PdfWriter()
        right_writer = PyPDF2.PdfWriter()

        for page in reader.pages:
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)
            split_point = (
                page_width * 2 / 5
            )  # Split at 2/5 position to give more space for multi-turn with legend

            # Create left half (Single Turn)
            left_page = PyPDF2.PageObject.create_blank_page(
                width=split_point, height=page_height
            )
            left_page.merge_page(page)
            left_page.mediabox = RectangleObject([0, 0, split_point, page_height])

            # Create right half (Multi-Turn with legend)
            right_page = PyPDF2.PageObject.create_blank_page(
                width=page_width - split_point, height=page_height
            )
            page_copy = PyPDF2.PageObject.create_blank_page(
                width=page_width, height=page_height
            )
            page_copy.merge_page(page)
            page_copy.add_transformation(
                PyPDF2.Transformation().translate(-split_point, 0)
            )
            right_page.merge_page(page_copy)
            right_page.mediabox = RectangleObject(
                [0, 0, page_width - split_point, page_height]
            )

            left_writer.add_page(left_page)
            right_writer.add_page(right_page)

        # Save left half (Single Turn)
        with open(f"./che_single_turn_{timestamp}.pdf", "wb") as left_file:
            left_writer.write(left_file)

        # Save right half (Multi-Turn with legend)
        with open(f"./che_multi_turn_{timestamp}.pdf", "wb") as right_file:
            right_writer.write(right_file)

    print(f"Saved complete two-panel plot as: che_two_panel_{timestamp}.pdf")
    print(f"Saved single turn plot as: che_single_turn_{timestamp}.pdf")
    print(f"Saved multi-turn plot (with legend) as: che_multi_turn_{timestamp}.pdf")


# Run the plotting function
if __name__ == "__main__":
    plot_combined_two_panels()
