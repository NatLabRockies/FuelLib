import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fuellib as fl

# -----------------------------------------------------------------------------
# Calculate mixture properties from the group contribution properties
# -----------------------------------------------------------------------------

# HEFA fuels from various feedstocks (see fuelData/propertiesData for fuels)
fuel_names = ["hefa-mfat", "hefa-came", "hefa-tall"]
conv_fuel_name = "jet-a"
blends = fl.Units.Quantity(
    np.linspace(0, 100, 100), "%"
)  # Weight percentages of HEFA in blend

# Properties to plot
prop_names = ["Density", "Viscosity"]

# Plotting parameters
fsize = 18
ticksize = 18
line_thickness = 4
marker_size = 75


# Line specifications for plotting
def linespecs(name):
    if "came" in name:
        return "#7f7f7f", "o"  # 50% Gray
    elif "mfat" in name:
        return "#333333", "o"  # Dark Gray
    elif "tall" in name:
        return "#2980B9", "o"  # Primary Blue
    else:
        return "#2980B9", "o"  # Fallback: Primary Blue


# Legend labels for plotting
def leglab(name):
    if "hefa" in name:
        return name.upper()
    else:
        return "Jet-A"


# y-axis label
def ylab(prop_name, temp, prop_units):
    temp_c = temp.to("celsius").magnitude
    return rf"{prop_name} at {temp_c:g} °C [$\mathrm{{{prop_units}}}$]"


def getPredAndData(fuel_name, prop_name, blend):
    blend = blend.to_base_units()  # Convert percent to mass fraction

    # Get the fuel properties based on the GCM
    fuel = fl.Fuel(fuel_name, "hefa")
    jetA = fl.Fuel(conv_fuel_name)

    data_file = "hefa-jet-a-blends.csv"
    data_path = os.path.join(fl.get_fueldata_props_dir(), data_file)
    data_units = pd.read_csv(data_path, nrows=1)
    data = pd.read_csv(data_path, skiprows=[1])
    col = f"{prop_name}_{fuel_name[5:].upper()}"
    prop_units = data_units.at[0, col]
    prop_data = fl.Units.Quantity(data[col].to_numpy(), prop_units)
    blend_data = data["HEFA_concentration"]

    # Separate properties and associated temperatures from data
    if prop_name == "Density":
        T = fl.Units.Quantity(15, "celsius").to("K")
        prop_pred = fl.Units.Quantity(np.zeros_like(blend.magnitude), prop_units)
    elif prop_name == "Viscosity":
        T = fl.Units.Quantity(-20, "celsius").to("K")
        prop_pred = fl.Units.Quantity(np.zeros_like(blend), prop_units)

    for i in range(0, len(prop_pred)):
        # Initial liquid mass fractions
        Y_li = blend[i] * fuel.Y_0 + (1 - blend[i]) * jetA.Y_0

        if prop_name == "Density":
            prop_pred[i] = fuel.mixture_density(Y_li, T).to(prop_units)

        if prop_name == "Viscosity":
            # initial liquid mass fractions
            Y_li = blend[i] * fuel.Y_0 + (1 - blend[i]) * jetA.Y_0

            prop_pred[i] = fuel.mixture_kinematic_viscosity(Y_li, T).to(prop_units)

    return T, prop_units, prop_data.magnitude, blend_data, prop_pred.magnitude


figW = 5.25 * len(prop_names)
fig, ax = plt.subplots(1, len(prop_names), figsize=(figW, 5.5), constrained_layout=True)

for i in range(len(prop_names)):
    for fuel_name in fuel_names:
        T, prop_units, prop_data, blend_data, pred = getPredAndData(
            fuel_name, prop_names[i], blends
        )
        line_color, marker_style = linespecs(fuel_name)

        # Plot GCM predictions and data
        ax[i].plot(
            blends.magnitude,
            pred,
            "-",
            color=line_color,
            label=leglab(fuel_name),
            linewidth=line_thickness,
        )

        ax[i].scatter(
            blend_data[1:],
            prop_data[1:],
            marker=marker_style,
            label=None,
            facecolors=line_color,
            s=marker_size,
        )

    # Data for pure Jet-A
    ax[i].scatter(
        blend_data[0],
        prop_data[0],
        marker="o",
        label="Jet-A",
        facecolors="darkorange",
        s=marker_size + 2,
    )

    # Add labels and adjust ticks
    ax[i].set_xlabel("HEFA Concentration [wt %]", fontsize=fsize)
    ax[i].set_xticks([0, 20, 40, 60, 80, 100])
    ax[i].set_ylabel(ylab(prop_names[i], T, prop_units), fontsize=fsize)
    ax[i].tick_params(labelsize=ticksize)

handles, labels = ax[0].get_legend_handles_labels()
ax[i].legend(handles, labels, ncol=1, fontsize=fsize - 2)

plt.show()
