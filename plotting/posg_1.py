 

import matplotlib.pyplot as plt
import numpy as np

# Set Chinese fonts
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# Figure setup
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Chicken Game payoff matrix with modified values
# Payoff matrix: [[agent1_payoff, agent2_payoff], ...]
# Layout: (Cooperate, Cooperate), (Cooperate, Defect), (Defect, Cooperate), (Defect, Defect)
payoff_matrix = np.array(
    [
        [[5, 5], [4, 3]],  # Agent 1 chooses Cooperate
        [[3, 4], [0, 0]],  # Agent 1 chooses Defect
    ]
)

# Generate all possible joint returns from mixed strategy combinations
joint_returns_x = []
joint_returns_y = []

# Generate mixed strategy probabilities for both agents
n_points = 30  # Points per dimension
p1_range = np.linspace(0, 1, n_points)
p2_range = np.linspace(0, 1, n_points)

for p1 in p1_range:
    for p2 in p2_range:
        # Calculate expected payoffs
        # Agent 1's expected payoff
        expected_1 = (
            p1 * p2 * payoff_matrix[0, 0, 0]
            + p1 * (1 - p2) * payoff_matrix[0, 1, 0]
            + (1 - p1) * p2 * payoff_matrix[1, 0, 0]
            + (1 - p1) * (1 - p2) * payoff_matrix[1, 1, 0]
        )

        # Agent 2's expected payoff
        expected_2 = (
            p1 * p2 * payoff_matrix[0, 0, 1]
            + p1 * (1 - p2) * payoff_matrix[0, 1, 1]
            + (1 - p1) * p2 * payoff_matrix[1, 0, 1]
            + (1 - p1) * (1 - p2) * payoff_matrix[1, 1, 1]
        )

        joint_returns_x.append(expected_2)
        joint_returns_y.append(expected_1)

# Plot in specific order for legend
grey_scatter = ax.scatter(
    joint_returns_x, joint_returns_y, c="grey", s=1, alpha=0.6, label="Joint Return"
)

# Add Joint Optimal solution (Cooperate, Cooperate) = (5,5)
joint_optimal_x = [5]
joint_optimal_y = [5]

# Deterministic Nash Equilibria (pure strategies)
# For this game, (Cooperate, Cooperate) is the only pure strategy Nash equilibrium
# (C,C): Agent 1 gets 5, Agent 2 gets 5
deterministic_ne_x = [5]  # Agent 2's payoff
deterministic_ne_y = [5]  # Agent 1's payoff

# Create custom legend handles with different marker scales
from matplotlib.lines import Line2D

# Create custom legend elements
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="grey",
        markersize=1 * 5.7,
        alpha=0.6,
        label="Joint Return",
        linestyle="None",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="steelblue",
        markersize=np.sqrt(250) * 0.6,
        label="Deterministic NE",
        linestyle="None",
        markeredgecolor="black",
        markeredgewidth=1.5,
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="indianred",
        markersize=np.sqrt(240) * 0.6,
        label="Joint Optima",
        linestyle="None",
        markeredgecolor="black",
        markeredgewidth=1.5,
    ),
]

# Now plot the actual data (Joint Optimal first, then NE on top)
ax.scatter(
    joint_optimal_x,
    joint_optimal_y,
    c="indianred",
    s=240,
    marker="s",
    edgecolors="black",
    linewidth=1.5,
)

# Mixed Strategy Nash Equilibrium
# For the chicken game with payoffs (5,5), (4,3), (3,4), (0,0)
# Let p = probability that Agent 1 cooperates
# Let q = probability that Agent 2 cooperates

# Agent 1's indifference condition:
# EU₁(C) = EU₁(D)
# 5q + 4(1-q) = 3q + 0(1-q)
# 5q + 4 - 4q = 3q
# q + 4 = 3q
# 4 = 2q
# q = 2 (impossible since q must be ≤ 1)

# Agent 2's indifference condition:
# EU₂(C) = EU₂(D)
# 5p + 4(1-p) = 3p + 0(1-p)
# 5p + 4 - 4p = 3p
# p + 4 = 3p
# 4 = 2p
# p = 2 (impossible since p must be ≤ 1)

# This game has NO mixed strategy Nash equilibrium because both
# indifference conditions lead to probabilities > 1

# Since there's no mixed strategy NE, we won't plot the orange point

# Plot Deterministic NE last so it appears on top
ax.scatter(
    deterministic_ne_x,
    deterministic_ne_y,
    c="steelblue",
    s=250,
    marker="^",
    edgecolors="black",
    linewidth=1.5,
)

# Set figure properties
ax.set_xlabel("Agent 2 Return", fontsize=18)
ax.set_ylabel("Agent 1 Return", fontsize=18)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.3)

# Set custom ticks for both x and y axes
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 2, 4, 6, 8])

# Remove the duplicate 0 at the origin
# Get current tick labels
xticks = ax.get_xticks()
yticks = ax.get_yticks()

# Set x-axis labels (keep all)
ax.set_xticklabels([str(int(x)) for x in xticks])

# Set y-axis labels (skip the first one which is 0)
yticklabels = ["" if i == 0 else str(int(y)) for i, y in enumerate(yticks)]
ax.set_yticklabels(yticklabels)

# Set axis tick font sizes
ax.tick_params(axis="both", which="major", labelsize=16)

# Add legend with custom handles
ax.legend(handles=legend_elements, loc="upper right", fontsize=14, labelspacing=0.5)

plt.tight_layout()

# Save as PDF file
plt.savefig("game1.pdf", format="pdf", bbox_inches="tight", dpi=300)

plt.show()
