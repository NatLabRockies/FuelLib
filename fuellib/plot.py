"""
Utilities for plotting fuel properties and composition.

This module provides functions for visualizing:
- Fuel composition by compound and chemical family
- Mixture properties over a temperature range
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._data_locator import get_fueldata_dir, get_metadata_props_data
from .fuel import Fuel
from .types import FloatArray, PintArray
from .units import PintUnits as Units

# ANSI codes
BLUE = "\033[94m"  # ANSI code for blue text
RESET = "\033[0m"  # ANSI code to reset text color


def plot_composition(
    fuel_name: str,
    fuel_data_dir: str | None = None,
    output_dir: str | None = None,
    title: str | None = None,
    decomp_name: str | None = None,
    save: bool = True,
    display: bool = False,
) -> None:
    """
    Plot the composition of a given fuel.

    :param fuel_name: Name of the fuel to plot.
    :type fuel_name: str
    :param fuel_data_dir: Directory where fuel data files are located (optional).
    :type fuel_data_dir: str, optional
    :param output_dir: Directory to save the plot (optional, default: current directory).
    :type output_dir: str, optional
    :param title: Title for the plots (optional, default: fuel_name, or "none"/"None" to disable).
    :type title: str, optional
    :param decomp_name: Name of the decomposition file to use (optional, default: fuel_name).
    :type decomp_name: str, optional
    :param save: Whether to save the plot to a file (optional, default: True).
    :type save: bool, optional
    :param display: Whether to display the plot with plt.show() (optional, default: False).
    :type display: bool, optional
    """
    fuel_data_dir = fuel_data_dir or get_fueldata_dir()
    output_dir = output_dir or os.getcwd()
    title = (
        (title or "Fuel Composition") if title != "none" and title != "None" else None
    )

    if save:
        os.makedirs(output_dir, exist_ok=True)

    fuel = Fuel(fuel_name, decompName=decomp_name, fuelDataDir=fuel_data_dir)

    df = pd.DataFrame(
        {
            "Compound": fuel.compounds,
            "Weight %": fuel.Y_0 * 100.0,
            "Family": fuel.hc_type,
            "nC": fuel.nC,
        }
    )
    # Remove rows with weight percentage less than 0.01%
    df = df[df["Weight %"] > 0.01]

    # Get unique families from the fuel data in canonical order
    canonical_order = ["n-alkane", "iso-alkane", "cyclo-alkane", "aromatic", "alkene"]
    unique_families = list(np.unique(fuel.hc_type))
    family_names = [f for f in canonical_order if f in unique_families]

    family_weights = df.groupby("Family")["Weight %"].sum()

    # Print composition table
    lines = [
        f"{BLUE}Relative Weight % of Each Compound Family ({fuel.name}){RESET}",
        f"{'=' * 25}",
        f"{'Family':<15}{'Weight %':<10}",
        f"{'=' * 25}",
    ]
    for family in family_names:
        if family in family_weights.index:
            lines.append(f"{family:<15}{family_weights[family]:<10.2f}")
    lines.append(f"{'-' * 25}")
    lines.append(f"{'Total':<15}{family_weights.sum():<10.2f}")
    print("\n".join(lines))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ## Color scheme
    colors = {
        "n-alkane": "#063C61",
        "iso-alkane": "#2980B9",
        "cyclo-alkane": "#91BCD8",
        "alkene": "#663399",
        "aromatic": "#7f7f7f",
    }

    # Plot 1: Bar chart grouped by carbon number, colored by hydrocarbon type
    spacing = [-0.2985, -0.099, 0.099, 0.2985]
    nC_values = sorted(df["nC"].unique())

    # Get unique families that are in the filtered data
    families_in_data = [f for f in family_names if f in df["Family"].values]

    # Create bars for each family at each carbon number
    for k, family in enumerate(families_in_data):
        df_family = df[df["Family"] == family]

        # Group by carbon number and sum weights
        family_by_nC = df_family.groupby("nC")["Weight %"].sum()

        ax1.bar(
            family_by_nC.index + spacing[k],
            family_by_nC.values,
            label=family,
            alpha=1,
            color=colors.get(family, "#7f7f7f"),
            width=0.2,
        )

    ax1.set_xlabel("Carbon Number", fontsize=16)
    ax1.set_ylabel("Weight %", fontsize=16)
    ax1.set_xticks(nC_values)
    ax1.set_xticklabels(
        [int(n) if n == int(n) else f"{n:.1f}" for n in nC_values], fontsize=14
    )
    ax1.set_xlim(min(nC_values) - 0.5, max(nC_values) + 0.5)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Pie chart of family composition
    # Only include families that have weight > 0, in canonical order
    families_present = [
        f for f in family_names if f in family_weights.index and family_weights[f] > 0
    ]
    family_weights_sorted = family_weights[families_present]

    # Create pie chart without labels/percentages (we'll add them outside)
    wedges, _texts = ax2.pie(
        family_weights_sorted,
        labels=None,
        autopct=None,
        startangle=140,
        colors=[
            colors.get(family, "#7f7f7f") for family in family_weights_sorted.index
        ],
    )

    # Add percentages outside the pie with arrows
    for wedge, value, family in zip(
        wedges, family_weights_sorted.values, family_weights_sorted.index
    ):
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
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.5},
        )

    ax2.axis("equal")

    # Adjust layout to make room for legend BEFORE adding it
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    # Add a single figure-level legend for all families
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors.get(family, "#7f7f7f"))
        for family in family_weights_sorted.index
    ]
    fig.legend(
        legend_handles,
        family_weights_sorted.index,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=13,
        frameon=True,
    )

    # Add title if specified
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    # Save the plot if requested
    if save:
        plot_file = os.path.join(output_dir, f"composition_{fuel_name}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"Composition plot saved to {plot_file}")

    # Display the plot if requested
    plt.show() if display else plt.close()


def _get_pred_and_data(
    fuel_name: str,
    prop_name: str,
    *,
    decomp_name: str | None = None,
    fuel_data_dir: str | None = None,
) -> tuple[PintArray | None, PintArray | None, PintArray, PintArray]:
    """Get predicted and experimental data for a given fuel and property."""
    default_ranges_by_property = {
        "Density": Units.Quantity([-40, 40], "celsius"),
        "Viscosity": Units.Quantity([-40, 100], "celsius"),
        "VaporPressure": Units.Quantity([0, 125], "celsius"),
        "SurfaceTension": Units.Quantity([-10, 40], "celsius"),
        "ThermalConductivity": Units.Quantity([0, 60], "celsius"),
    }

    # Deprecated: fuel-specific ranges (kept for reference, now using property-based)
    _default_ranges = {
        "posf10264": [-40, 125],
        "posf10325": [-40, 125],
        "posf10289": [-40, 125],
        "posf11498": [-40, 125],
        "jet-a": [-40, 125],
        "hefa": [-40, 125],
        "decane": [-50, 100],
        "dodecane": [-50, 100],
        "heptane": [-50, 100],
    }

    # Get the fuel properties based on the GCM
    fuel = Fuel(fuel_name, decompName=decomp_name, fuelDataDir=fuel_data_dir)
    props_dir = fuel.fuelDataPropsDir

    if props_dir and os.path.exists(props_dir):
        # Check if metadata specifies a different props_data filename
        props_data_name = get_metadata_props_data(fuel_name, fuel_data_dir)
        data_filename = props_data_name if props_data_name else fuel_name

        data_temps = None
        data_props = None
        data_units = ""
        data_file = os.path.join(props_dir, f"{data_filename}.csv")
        if os.path.exists(data_file):
            try:
                data = pd.read_csv(data_file)
                if prop_name in data.columns:
                    t_vals: FloatArray = data.Temperature.iloc[1:].to_numpy(dtype=float)
                    t_units: str = data.Temperature.iloc[0]
                    data_temps: PintArray = Units.Quantity(t_vals, t_units).to("K")

                    data_vals: FloatArray = (
                        data[prop_name].iloc[1:].to_numpy(dtype=float)
                    )
                    data_units: str = data[prop_name].iloc[0]
                    data_props: PintArray = Units.Quantity(data_vals, data_units)

                    valid_idxs = ~np.isnan(data_props)
                    data_temps = data_temps[valid_idxs]
                    data_props = data_props[valid_idxs]

            except (OSError, KeyError, ValueError):
                pass

    if data_temps is not None and np.size(data_temps) > 0:
        pred_temps = Units.Quantity(
            np.linspace(np.min(data_temps), np.max(data_temps), 100), "K"
        )
    else:
        min_temp, max_temp = default_ranges_by_property.get(
            prop_name, Units.Quantity([0, 100], "celsius")
        ).to("K")
        pred_temps = Units.Quantity(np.linspace(min_temp, max_temp, 100), "K")

    default_units = {
        "Density": "g/cm^3",
        "VaporPressure": "kPa",
        "Viscosity": "mm^2/s",
        "SurfaceTension": "N/m",
        "ThermalConductivity": "W/m/K",
    }

    data_units = default_units.get(prop_name, "") if data_units == "" else data_units
    if data_units == "":
        raise NotImplementedError(f"Property '{prop_name}' is not implemented.")

    pred_props = Units.Quantity(np.zeros_like(pred_temps), data_units)
    for i, t in enumerate(pred_temps):
        if prop_name == "Density":
            pred_props[i] = fuel.mixture_density(fuel.Y_0, t).to("g/cm^3")
        elif prop_name == "VaporPressure":
            pred_props[i] = fuel.mixture_vapor_pressure(fuel.Y_0, t).to("kPa")
        elif prop_name == "Viscosity":
            pred_props[i] = fuel.mixture_kinematic_viscosity(fuel.Y_0, t).to("mm^2/s")
        elif prop_name == "SurfaceTension":
            pred_props[i] = fuel.mixture_surface_tension(fuel.Y_0, t).to("N/m")
        elif prop_name == "ThermalConductivity":
            pred_props[i] = fuel.mixture_thermal_conductivity(fuel.Y_0, t).to("W/m/K")
        else:
            raise NotImplementedError(f"Property '{prop_name}' is not implemented.")

    return data_temps, data_props, pred_temps, pred_props


def plot_mixture_properties(
    fuel_names: str | list[str],
    property_names: list[str] | None = None,
    fuel_data_dir: str | None = None,
    output_dir: str | None = None,
    title: str | None = None,
    decomp_name: str | None = None,
    save: bool = True,
    display: bool = False,
):
    """
    Plot mixture properties for fuel(s) over a temperature range.

    :param fuel_names: Name or list of fuel names to plot.
    :type fuel_names: str or list[str]
    :param property_names: Properties to plot (optional, defaults to standard set).
    :type property_names: list[str], optional
    :param fuel_data_dir: Directory where fuel data files are located (optional).
    :type fuel_data_dir: str, optional
    :param output_dir: Directory to save the plot (optional, default: current directory).
    :type output_dir: str, optional
    :param title: Title for the plot (optional, default: None).
    :type title: str, optional
    :param decomp_name: Name of the decomposition file to use (optional, default: fuel_name).
    :type decomp_name: str, optional
    :param save: Whether to save the plot to a file (optional, default: True).
    :type save: bool, optional
    :param display: Whether to display the plot with plt.show() (optional, default: False).
    :type display: bool, optional
    """
    fuel_data_dir = fuel_data_dir or get_fueldata_dir()
    output_dir = output_dir or os.getcwd()
    title = (
        (title or "Fuel Composition") if title != "none" and title != "None" else None
    )
    fuel_names = [fuel_names] if isinstance(fuel_names, str) else fuel_names

    if save:
        os.makedirs(output_dir, exist_ok=True)

    property_names = property_names or [
        "Density",
        "Viscosity",
        "VaporPressure",
        "SurfaceTension",
        "ThermalConductivity",
    ]

    # Y-axis labels
    ylab = {
        "Density": r"Density [g/cm$^3$]",
        "Viscosity": r"Viscosity [mm$^2$/s]",
        "VaporPressure": r"Vapor Pressure [kPa]",
        "SurfaceTension": r"Surface Tension [N/m]",
        "ThermalConductivity": r"Thermal Conductivity [W/m/K]",
    }
    # Line specs for different fuels (marker styles)
    line_specs_map = {
        "decane": "o",
        "posf10325": "o",
        "dodecane": "s",
        "posf10289": "s",
        "heptane": "D",
        "posf10264": "D",
        "posf11498": "^",
        "jet-a": "v",
        "hefa": "p",
    }
    # Fuel-specific colors
    fuel_color_map = {
        "posf10264": "#2980B9",  # Primary Blue
        "posf10325": "#7f7f7f",  # 50% Gray
        "posf10289": "#333333",  # Dark Gray
        "heptane": "#2980B9",  # Primary Blue
        "decane": "#7f7f7f",  # 50% Gray
        "dodecane": "#333333",  # Dark Gray
    }
    # Color palette for cycling through distinct colors
    color_palette = [
        "#2980B9",  # Medium blue
        "#e74c3c",  # Red
        "#27ae60",  # Green
        "#8e44ad",  # Purple
        "#f39c12",  # Orange
        "#1abc9c",  # Turquoise
        "#c0392b",  # Dark red
        "#16a085",  # Dark turquoise
        "#d35400",  # Dark orange
    ]

    n_props = len(property_names)
    figW = 4.25 * n_props
    fig, ax = plt.subplots(1, n_props, figsize=(figW, 5.5), constrained_layout=True)

    # Handle single subplot case
    ax = [ax] if n_props == 1 else ax

    for i, prop_name in enumerate(property_names):
        for fuel_idx, fuel_name in enumerate(fuel_names):
            data_temps, data_props, pred_temps, pred_props = _get_pred_and_data(
                fuel_name,
                prop_name,
                decomp_name=decomp_name,
                fuel_data_dir=fuel_data_dir,
            )

            line_color = fuel_color_map.get(
                fuel_name.lower(), color_palette[fuel_idx % len(color_palette)]
            )
            marker_style = line_specs_map.get(fuel_name.lower(), "o")

            legend_label = lambda name: (
                name.upper()
                if "hefa" in name.lower()
                else name[4:].upper()
                if "posf" in name.lower()
                else name.capitalize()
            )

            ax[i].plot(
                pred_temps.to("celsius"),
                pred_props,
                "-",
                color=line_color,
                label=f"FuelLib: {legend_label(fuel_name)}",
            )

            if (
                data_props is not None
                and data_temps is not None
                and len(data_props) > 0
                and len(data_temps) > 0
            ):
                props_data_name = get_metadata_props_data(fuel_name, fuel_data_dir)
                data_label = props_data_name if props_data_name else fuel_name
                ax[i].scatter(
                    data_temps.to("celsius"),
                    data_props,
                    marker=marker_style,
                    label=f"Data: {legend_label(data_label)}",
                    facecolors=line_color,
                    s=75,
                    zorder=5,
                )

        ax[i].set_xlabel("Temperature (°C)")
        ax[i].set_ylabel(ylab.get(prop_name, prop_name))
        ax[i].tick_params(labelsize=18)
        ax[i].grid(alpha=0.3)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=len(fuel_names), fontsize=18
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    if save:
        fuel_str = "_".join(fuel_names)
        plot_file = os.path.join(output_dir, f"mixture_properties_{fuel_str}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Properties plot saved to {plot_file}")

    plt.show() if display else plt.close(fig)
