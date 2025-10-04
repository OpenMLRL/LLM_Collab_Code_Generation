 

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Set Chinese fonts
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Chicken Game payoff matrix with your specified values
# Payoff matrix: [[agent1_payoff, agent2_payoff], ...]
# Layout: (Cooperate, Cooperate), (Cooperate, Defect), (Defect, Cooperate), (Defect, Defect)
payoff_matrix = np.array(
    [
        [[5, 5], [1, 6]],  # Agent 1 chooses Cooperate
        [[6, 1], [0, 0]],  # Agent 1 chooses Defect
    ]
)

# Generate all possible joint returns from mixed strategy combinations
joint_returns_x = []
joint_returns_y = []

# Generate mixed strategy probabilities for both agents
n_points = 40  # Points per dimension
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

# Plot joint return points
ax.scatter(
    joint_returns_x, joint_returns_y, c="grey", s=1, alpha=0.6, label="Joint Return"
)

# Deterministic Nash Equilibria (pure strategies)
deterministic_ne_x = []
deterministic_ne_y = []

# Extract pure strategy Nash equilibria - (Defect, Cooperate) and (Cooperate, Defect)
pure_strategies = [(0, 1), (1, 0)]  # (Cooperate, Defect) and (Defect, Cooperate)
for s1, s2 in pure_strategies:
    payoff_1 = payoff_matrix[s1, s2, 0]
    payoff_2 = payoff_matrix[s1, s2, 1]
    deterministic_ne_x.append(payoff_2)
    deterministic_ne_y.append(payoff_1)

# Add Joint Optimal solution (Cooperate, Cooperate) = (5,5)
joint_optimal_x = [5]
joint_optimal_y = [5]

ax.scatter(
    deterministic_ne_x,
    deterministic_ne_y,
    c="steelblue",
    s=250,
    marker="^",
    label="Deterministic NE",
    edgecolors="black",
    linewidth=1.5,
)

# Probabilistic Nash Equilibrium (mixed strategies)
# CORRECTED CALCULATION:
# For mixed strategy equilibrium, each player must be indifferent between their pure strategies
#
# Agent 1 plays Cooperate with probability p, Agent 2 plays Cooperate with probability q
#
# For Agent 1 to be indifferent (Agent 2's strategy makes Agent 1 indifferent):
# EU₁(Cooperate) = EU₁(Defect)
# 5q + 1(1-q) = 6q + 0(1-q)
# 5q + 1 - q = 6q
# 4q + 1 = 6q
# 1 = 2q
# q = 1/2
#
# For Agent 2 to be indifferent (Agent 1's strategy makes Agent 2 indifferent):
# EU₂(Cooperate) = EU₂(Defect)
# 5p + 1(1-p) = 6p + 0(1-p)
# 5p + 1 - p = 6p
# 4p + 1 = 6p
# 1 = 2p
# p = 1/2

# Mixed strategy Nash Equilibrium: p = q = 1/2
p_ne = 1 / 2
q_ne = 1 / 2

expected_1_mixed = (
    p_ne * q_ne * 5
    + p_ne * (1 - q_ne) * 1
    + (1 - p_ne) * q_ne * 6
    + (1 - p_ne) * (1 - q_ne) * 0
)

expected_2_mixed = (
    p_ne * q_ne * 5
    + p_ne * (1 - q_ne) * 6
    + (1 - p_ne) * q_ne * 1
    + (1 - p_ne) * (1 - q_ne) * 0
)

probabilistic_ne_x = [expected_2_mixed]
probabilistic_ne_y = [expected_1_mixed]
print(
    f"Mixed Strategy NE payoffs: Agent 1 = {expected_1_mixed}, Agent 2 = {expected_2_mixed}"
)

ax.scatter(
    probabilistic_ne_x,
    probabilistic_ne_y,
    c="orange",
    s=240,
    marker="o",
    label="Probabilistic NE",
    edgecolors="black",
    linewidth=1.5,
)

ax.scatter(
    joint_optimal_x,
    joint_optimal_y,
    c="indianred",
    s=240,
    marker="s",
    label="Joint Optimal",
    edgecolors="black",
    linewidth=1.5,
)

# Create custom legend elements with different marker scales
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
        marker="o",
        color="w",
        markerfacecolor="orange",
        markersize=np.sqrt(240) * 0.6,
        label="Probabilistic NE",
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

# Set figure properties
ax.set_xlabel("Agent 2 Return", fontsize=18)
ax.set_ylabel("Agent 1 Return", fontsize=18)
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
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
plt.savefig("game2.pdf", format="pdf", bbox_inches="tight", dpi=300)

plt.show()
