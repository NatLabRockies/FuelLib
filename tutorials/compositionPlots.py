import os
import sys
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
import paths
import FuelLib as fl

fuel_name = "posf10325"

fuel = fl.fuel(fuel_name)

# Classify compounds into families
aromatic = [
    True if re.search(r"Toluene|Benzene|Aromatic", comp, re.IGNORECASE) else False
    for comp in fuel.compounds
]
n_alkane = [
    True if re.search(r"n-C", comp, re.IGNORECASE) else False for comp in fuel.compounds
]
isoalkane = [
    True if re.search(r"Isoparaffin", comp, re.IGNORECASE) else False
    for comp in fuel.compounds
]
cycloalkane = [
    True if re.search(r"Cycloparaffin", comp, re.IGNORECASE) else False
    for comp in fuel.compounds
]

# Create a DataFrame with the compounds and their families
colNames = ["Compound", "Weight %"]
df = pd.DataFrame({"Compounds": fuel.compounds, "Weight %": fuel.Y_0 * 100})
# Append classification as a new column
family_names = ["n-alkane", "iso-alkane", "cyclo-alkane", "aromatic"]
df["Family"] = np.select(
    [n_alkane, isoalkane, cycloalkane, aromatic], family_names, default="unknown"
)


# Determine carbon number by row:
def determine_carbon_number(compound):
    if "Toluene" in compound:
        return 7
    elif "benzene" in compound.lower():
        # Extract the number after "C" and add 7
        match = re.search(r"C(\d+)", compound)
        if match:
            try:
                carbon_number = int(match.group(1))
                return carbon_number + 6
            except ValueError:
                return np.nan  # Handle cases where extraction fails
        else:
            return np.nan  # No match found
    else:
        # Extract the number after "C" in either format
        match = re.search(r"C(\d+)", compound)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return np.nan  # Handle cases where extraction fails
        else:
            return np.nan  # No match found


# Apply the function to the column and append as a new column
df["nC"] = df.Compounds.apply(determine_carbon_number)

# Determine relative weight % of each family
family_weights = df.groupby("Family")["Weight %"].sum()

# Print the sum of weight % for each family
print("\n" + "=" * 45)
print("Relative Weight % of Each Compound Family")
print(f"Fuel: {fuel_name}")
print("=" * 45)
for family, weight in family_weights.items():
    print(f"  {family:<20} {weight:>8.2f}%")
print("-" * 45)
print(f"  {'Total':<20} {family_weights.sum():>8.2f}%")
print("=" * 45 + "\n")

# Remove rows <= 0.01 in weight % column at max(nC)
df = df[df["Weight %"] > 0.01]

# Plotting parameters
spacing = [-0.2985, -0.099, 0.099, 0.2985]
colors = {
    "n-alkane": "#063C61",
    "iso-alkane": "#2980B9",
    "cyclo-alkane": "#91BCD8",
    "aromatic": "#7f7f7f",
}

# Bar plot of the number of compounds in each family
plt.figure(figsize=(7, 5))
N = df.nC.unique()
for k, family in enumerate(family_names):
    nC = df[df["Family"] == family].nC
    weight = df[df["Family"] == family]["Weight %"]

    # check duplicate nC values
    if len(nC) != len(set(nC)):
        # If there are duplicates, sum the weights for each nC
        df_grouped = (
            df[df["Family"] == family].groupby("nC")["Weight %"].sum().reset_index()
        )
        nC = df_grouped["nC"]
        weight = df_grouped["Weight %"]
    plt.bar(
        nC + spacing[k], weight, label=family, alpha=1, color=colors[family], width=0.2
    )
    plt.xticks(df.nC.unique(), fontsize=14)
    plt.xlim(min(N) - 0.5, max(N) + 0.5)

plt.xlabel("Carbon Number", fontsize=16)
plt.xticks(df.nC.unique(), fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Weight %", fontsize=16)
plt.title("Fuel Composition", fontsize=16, fontweight="bold")
plt.legend(fontsize=14)
plt.tight_layout()

# Plot pie chart of family weights
plt.figure(figsize=(7, 5))
plt.pie(
    family_weights,
    labels=None,
    autopct="%1.1f%%",
    startangle=140,
    colors=[colors[family] for family in family_names],
    textprops={"fontsize": 20, "weight": "bold", "color": "white"},
)
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc=colors[family]) for family in family_names
]
plt.legend(
    legend_handles,
    family_names,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=14,
)
plt.axis("equal")
plt.tight_layout()
plt.show()
