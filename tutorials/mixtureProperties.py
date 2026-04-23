import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
from paths import *
import FuelLib as fl

# -----------------------------------------------------------------------------
# Calculate mixture properties from the group contribution properties
# -----------------------------------------------------------------------------

# Fuel for GCM and data for validation (see fuelData/propertiesData for fuels)
# Options: 'decane','dodecane', 'heptane', 'posf10264', 'posf10325', 'posf10289'
# fuel_names = ["heptane","decane","dodecane"]
fuel_names = ["posf10264", "posf10325", "posf10289"]

# Properties to plot
prop_names = [
    "Density",
    "Viscosity",
    "VaporPressure",
    "SurfaceTension",
    "ThermalConductivity",
]

# Plotting parameters
fsize = 18
ticksize = 18
line_thickness = 4
marker_size = 75


# Line specifications for plotting
def linespecs(name):
    if (name == "decane") or (name == "posf10325"):
        return "#7f7f7f", "o"  # 50% Gray
    elif (name == "dodecane") or (name == "posf10289"):
        return "#333333", "s"  # Dark Gray
    elif (name == "heptane") or (name == "posf10264"):
        return "#2980B9", "D"  # Primary Blue
    else:
        return "#2980B9", "o"  # Fallback: Primary Blue


# Legend labels for plotting
def leglab(name):
    if "posf" in name:
        return name[4:]
    else:
        return name.capitalize()


# Ticks for x-axis of posf fuels only
xticks_posf = {
    "Density": [-40, -20, 0, 20, 40],
    "Viscosity": [-40, 0, 40, 100],
    "VaporPressure": [0, 25, 75, 125],
    "SurfaceTension": [-10, 20, 40],
    "ThermalConductivity": [0, 40, 80, 125],
}

# Ticks for x-axis of simple fuels only
xticks_simp = {
    "Density": [-50, 0, 50, 100],
    "Viscosity": [-50, 0, 50, 100],
    "VaporPressure": [-20, 30, 80, 130],
    "SurfaceTension": [-20, 30, 80, 130],
    "ThermalConductivity": [-10, 40, 90, 140],
}

# y-axis label
ylab = {
    "Density": r"Density [g/cm$^3$]",
    "Viscosity": r"Viscosity [mm$^2$/s]",
    "VaporPressure": r"Vapor Pressure [kPa]",
    "SurfaceTension": r"Surface Tension [N/m]",
    "ThermalConductivity": r"Thermal Conductivity [W/m/K]",
}


def getPredAndData(fuel_name, prop_name):
    # Get the fuel properties based on the GCM
    fuel = fl.fuel(fuel_name)

    data_file = f"{fuel_name}.csv"
    try:
        data = pd.read_csv(os.path.join(FUELDATA_PROPS_DIR, data_file), skiprows=[1])
        # Separate properties and associated temperatures from data
        T_data = data.Temperature[data[prop_name].notna()]
        prop_data = data[prop_name].dropna()
        # Vectors for temperature (convert from C to K)
        T_pred = fl.C2K(np.linspace(min(T_data), max(T_data), 100))
    except (FileNotFoundError, KeyError):
        # If propertyData is not found, set temp min/max from xticks_posf
        T_data = pd.Series(dtype=float)
        prop_data = pd.Series(dtype=float)
        temp_min = min(xticks_posf[prop_name])
        temp_max = max(xticks_posf[prop_name])
        T_pred = fl.C2K(np.linspace(temp_min, temp_max, 100))

    # Vectors for density, viscosity and vapor pressure
    pred = np.zeros_like(T_pred)

    for i in range(0, len(T_pred)):
        Y_li = fuel.Y_0

        if prop_name == "Density":
            # Mixture density (returns rho in kg/m^3)
            pred[i] = fuel.mixture_density(Y_li, T_pred[i])
            # Convert density to CGS (g/cm^3)
            pred[i] *= 1.0e-03

        if prop_name == "VaporPressure":
            # Mixture vapor pressure (returns pv in Pa)
            pred[i] = fuel.mixture_vapor_pressure(Y_li, T_pred[i])
            # Convert vapor pressure to kPa
            pred[i] *= 1.0e-03

        if prop_name == "Viscosity":
            pred[i] = fuel.mixture_kinematic_viscosity(Y_li, T_pred[i])
            # Convert viscosity to mm^2/s
            pred[i] *= 1.0e06

        if prop_name == "SurfaceTension":
            pred[i] = fuel.mixture_surface_tension(Y_li, T_pred[i])

        if prop_name == "ThermalConductivity":
            pred[i] = fuel.mixture_thermal_conductivity(Y_li, T_pred[i])

    return T_data, prop_data, T_pred, pred


figW = 4.25 * len(prop_names)
fig, ax = plt.subplots(1, len(prop_names), figsize=(figW, 5.5), constrained_layout=True)

for i in range(len(prop_names)):
    for fuel_name in fuel_names:
        T_data, prop_data, T, pred = getPredAndData(fuel_name, prop_names[i])
        line_color, marker_style = linespecs(fuel_name)

        # Plot GCM predictions and data
        ax[i].plot(
            fl.K2C(T),
            pred,
            "-",
            color=line_color,
            label=f"FuelLib: {leglab(fuel_name)}",
            linewidth=line_thickness,
        )

        if "posf" in fuel_name:
            data_label = "AFRL: "
        else:
            data_label = "NIST: "
        ax[i].scatter(
            T_data,
            prop_data,
            marker=marker_style,
            label=f"{data_label}{leglab(fuel_name)}",
            facecolors=line_color,
            s=marker_size,
        )

    # Add labels and adjust ticks
    ax[i].set_xlabel("T [°C]", fontsize=fsize)
    if "posf" in fuel_name:
        ax[i].set_xticks(xticks_posf[prop_names[i]])
    else:
        ax[i].set_xticks(xticks_simp[prop_names[i]])
    ax[i].set_ylabel(ylab[prop_names[i]], fontsize=fsize)
    ax[i].tick_params(labelsize=ticksize)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="outside lower center", ncol=len(fuel_names), fontsize=fsize
)

plt.show()
