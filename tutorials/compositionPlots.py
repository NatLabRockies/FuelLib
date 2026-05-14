import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

import fuellib as fl

fuel_name = "posf10325"

fuel = fl.fuel(fuel_name)

# Create a DataFrame with the compounds and their families
# Use the hydrocarbon type classification from group decompositions
family_names = ["n-alkane", "iso-alkane", "cyclo-alkane", "aromatic"]
df = pd.DataFrame({
    "Compound": fuel.compounds,
    "Weight %": fuel.Y_0 * 100,
    "Family": fuel.hc_type,
})

# Determine carbon number from compound name
def determine_carbon_number(compound):
    """Extract carbon number from compound name."""
    if "Toluene" in compound:
        return 7
    elif "benzene" in compound.lower():
        match = re.search(r"C(\d+)", compound)
        if match:
            try:
                return int(match.group(1)) + 6
            except ValueError:
                return np.nan
        return np.nan
    else:
        match = re.search(r"C(\d+)", compound)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return np.nan
        return np.nan

df["nC"] = df["Compound"].apply(determine_carbon_number)

# Remove rows with weight % <= 0.01
df = df[df["Weight %"] > 0.01]

# Calculate family weights
family_weights = df.groupby("Family")["Weight %"].sum()

# Print composition table
print(f"\n{'=' * 50}")
print("Relative Weight % of Each Compound Family")
print(f"Fuel: {fuel_name}")
print("=" * 50)
for family, weight in family_weights.items():
    print(f"  {family:<20} {weight:>8.2f}%")
print("-" * 50)
print(f"  {'Total':<20} {family_weights.sum():>8.2f}%")
print("=" * 50 + "\n")

# Color scheme
colors = {
    "n-alkane": "#063C61",
    "iso-alkane": "#2980B9",
    "cyclo-alkane": "#91BCD8",
    "aromatic": "#7f7f7f",
}

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Plot 1: Bar chart grouped by carbon number
spacing = [-0.2985, -0.099, 0.099, 0.2985]
N = df.nC.unique()
families_in_data = df["Family"].unique()

for k, family in enumerate(family_names):
    if family not in families_in_data:
        continue
    df_family = df[df["Family"] == family]
    nC = df_family.nC
    weight = df_family["Weight %"]

    # Check for duplicate carbon numbers and sum weights
    if len(nC) != len(set(nC)):
        df_grouped = df_family.groupby("nC")["Weight %"].sum().reset_index()
        nC = df_grouped["nC"]
        weight = df_grouped["Weight %"]

    ax1.bar(
        nC + spacing[k],
        weight,
        label=family,
        alpha=1,
        color=colors.get(family, "#7f7f7f"),
        width=0.2,
    )

ax1.set_xlabel("Carbon Number", fontsize=16)
ax1.set_xticks(sorted(N))
ax1.set_xticklabels(sorted(N), fontsize=14)
ax1.set_xlim(min(N) - 0.5, max(N) + 0.5)
ax1.set_ylabel("Weight %", fontsize=16)
ax1.tick_params(axis="y", labelsize=14)
ax1.grid(axis="y", alpha=0.3)
ax1.legend(fontsize=12, loc="upper left")

# Plot 2: Pie chart of family composition
# Only include families that have weight > 0
families_present = [f for f in family_names if f in family_weights.index and family_weights[f] > 0]
family_weights_sorted = family_weights[families_present]

# Create pie chart without labels/percentages (we'll add them outside)
wedges, texts = ax2.pie(
    family_weights_sorted,
    labels=None,
    autopct=None,
    startangle=140,
    colors=[colors.get(family, "#7f7f7f") for family in family_weights_sorted.index],
)

# Add percentages outside the pie with arrows
for wedge, value, family in zip(wedges, family_weights_sorted.values, family_weights_sorted.index):
    angle = (wedge.theta2 + wedge.theta1) / 2
    radius = 1.3
    x = radius * np.cos(np.radians(angle))
    y = radius * np.sin(np.radians(angle))
    
    # Determine horizontal alignment based on position
    ha = "left" if x > 0 else "right"
    
    # Add annotation with arrow
    ax2.annotate(
        f"{value:.1f}%",
        xy=(np.cos(np.radians(angle)), np.sin(np.radians(angle))),
        xytext=(x, y),
        ha=ha,
        va="center",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
    )

ax2.axis("equal")

# Add a single figure-level legend for all families
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc=colors.get(family, "#7f7f7f"))
    for family in family_weights_sorted.index
]
fig.legend(
    legend_handles,
    family_weights_sorted.index,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    fontsize=13,
    frameon=True,
)

fig.suptitle("Fuel Composition", fontsize=16, fontweight="bold")

plt.show()

