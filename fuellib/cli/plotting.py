"""
Utilities for plotting fuel properties and composition.

This module provides functions for visualizing:
- Fuel composition by compound and chemical family
- Mixture properties over a temperature range
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import fuellib as fl


def plot_composition(
    fuel_name,
    fuel_data_dir=None,
    output_dir=None,
    title=None,
    decomp_name=None,
    save=True,
    display=False,
):
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
    if output_dir is None:
        output_dir = os.getcwd()

    # Handle title
    plot_title = None
    if title is None:
        plot_title = "Fuel Composition"
    elif title.lower() != "none":
        plot_title = title

    if fuel_data_dir is None:
        fuel_data_dir = fl.get_fueldata_dir()

    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the fuel
    fuel = fl.fuel(fuel_name, decompName=decomp_name, fuelDataDir=fuel_data_dir)

    # Create DataFrame with compound data and carbon numbers from fuel object
    df = pd.DataFrame(
        {
            "Compound": fuel.compounds,
            "Weight %": fuel.Y_0 * 100,
            "Family": fuel.hc_type,
            "nC": fuel.nC,
        }
    )

    # Get unique families from the fuel data in canonical order
    canonical_order = ["n-alkane", "iso-alkane", "cyclo-alkane", "aromatic", "alkene"]
    unique_families = list(np.unique(fuel.hc_type))
    family_names = [f for f in canonical_order if f in unique_families]

    # Remove rows with weight % <= 0.01
    df = df[df["Weight %"] > 0.01]

    # Calculate family weights
    family_weights = df.groupby("Family")["Weight %"].sum()

    # Print composition table
    print(f"\n{'=' * 50}")
    print("Relative Weight % of Each Compound Family")
    print(f"Fuel: {fuel_name}")
    print("=" * 50)
    for family in family_names:
        if family in family_weights.index:
            weight = family_weights[family]
            print(f"  {family:<20} {weight:>8.2f}%")
    print("-" * 50)
    print(f"  {'Total':<20} {family_weights.sum():>8.2f}%")
    print("=" * 50 + "\n")

    # Color scheme
    colors = {
        "n-alkane": "#063C61",
        "iso-alkane": "#2980B9",
        "cyclo-alkane": "#91BCD8",
        "alkene": "#663399",
        "aromatic": "#7f7f7f",
    }

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

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
    wedges, texts = ax2.pie(
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

    # Add title if specified
    if plot_title:
        fig.suptitle(plot_title, fontsize=16, fontweight="bold")

    # Save the plot if requested
    if save:
        plot_file = os.path.join(output_dir, f"composition_{fuel_name}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Composition plot saved to {plot_file}")

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_mixture_properties(
    fuel_names,
    property_names=None,
    fuel_data_dir=None,
    output_dir=None,
    title=None,
    decomp_name=None,
    save=True,
    display=False,
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
    if output_dir is None:
        output_dir = os.getcwd()

    if fuel_data_dir is None:
        fuel_data_dir = fl.get_fueldata_dir()

    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle single fuel_name or list
    if isinstance(fuel_names, str):
        fuel_names = [fuel_names]

    # Default properties if not specified
    if property_names is None:
        property_names = [
            "Density",
            "Viscosity",
            "VaporPressure",
            "SurfaceTension",
            "ThermalConductivity",
        ]

    # Default temperature ranges (in °C) for different properties
    default_ranges_by_property = {
        "Density": [-40, 40],
        "Viscosity": [-40, 100],
        "VaporPressure": [0, 125],
        "SurfaceTension": [-10, 40],
        "ThermalConductivity": [0, 60],
    }

    # Deprecated: fuel-specific ranges (kept for reference, now using property-based)
    default_ranges = {
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

    def get_line_spec(fuel_name, fuel_index=0):
        """Get line color and marker style for fuel.

        :param fuel_name: Name of the fuel
        :param fuel_index: Index of fuel in the plot (for cycling through colors)
        :return: Tuple of (color, marker_style)
        """
        # Get marker style from map
        marker_style = "o"  # Default
        for key, spec in line_specs_map.items():
            if key in fuel_name.lower():
                marker_style = spec
                break

        # Get color: use fuel-specific mapping if available, otherwise use palette
        if fuel_name in fuel_color_map:
            color = fuel_color_map[fuel_name]
        else:
            color = color_palette[fuel_index % len(color_palette)]
        return (color, marker_style)

    def get_legend_label(fuel_name):
        """Create legend label for fuel."""
        if "hefa" in fuel_name.lower():
            return fuel_name.upper()
        elif "posf" in fuel_name.lower():
            return fuel_name[4:].upper()
        else:
            return fuel_name.capitalize()

    def get_temp_range(prop_name):
        """Get default temperature range for a property."""
        return default_ranges_by_property.get(
            prop_name, [0, 100]
        )  # Fallback to [0, 100]

    def get_predictions_and_data(fuel_name, prop_name):
        """Get predicted and experimental data for a property."""
        fuel = fl.fuel(fuel_name, decompName=decomp_name, fuelDataDir=fuel_data_dir)

        # Try to load experimental data
        props_dir = fuel.fuelDataPropsDir

        T_data = pd.Series(dtype=float)
        prop_data = pd.Series(dtype=float)

        if props_dir and os.path.exists(props_dir):
            # Check if metadata specifies a different props_data filename
            props_data_name = fl.get_metadata_props_data(fuel_name, fuel_data_dir)
            data_filename = props_data_name if props_data_name else fuel_name

            data_file = os.path.join(props_dir, f"{data_filename}.csv")
            if os.path.exists(data_file):
                try:
                    data = pd.read_csv(data_file, skiprows=[1])
                    if prop_name in data.columns:
                        mask = data[prop_name].notna()
                        T_data = data.loc[mask, "Temperature"]
                        prop_data = data.loc[mask, prop_name]
                except Exception:
                    pass

        # Generate predictions over temperature range
        # First check if experimental data exists - use its range if available
        if len(T_data) > 0:
            # Use data range if available
            T_pred = fl.convert.C2K(np.linspace(T_data.min(), T_data.max(), 100))
        else:
            # Use property-specific default range
            temp_min, temp_max = get_temp_range(prop_name)
            T_pred = fl.convert.C2K(np.linspace(temp_min, temp_max, 100))

        pred = np.zeros_like(T_pred)
        Y_li = fuel.Y_0

        for i, T in enumerate(T_pred):
            try:
                if prop_name == "Density":
                    pred[i] = (
                        fuel.mixture_density(Y_li, T) * 1.0e-03
                    )  # Convert to g/cm^3
                elif prop_name == "VaporPressure":
                    pred[i] = (
                        fuel.mixture_vapor_pressure(Y_li, T) * 1.0e-03
                    )  # Convert to kPa
                elif prop_name == "Viscosity":
                    pred[i] = (
                        fuel.mixture_kinematic_viscosity(Y_li, T) * 1.0e6
                    )  # Convert to mm^2/s
                elif prop_name == "SurfaceTension":
                    pred[i] = fuel.mixture_surface_tension(Y_li, T)
                elif prop_name == "ThermalConductivity":
                    pred[i] = fuel.mixture_thermal_conductivity(Y_li, T)
            except Exception:
                pred[i] = np.nan

        return T_data, prop_data, T_pred, pred

    # Create figure with subplots
    n_props = len(property_names)
    figW = 4.25 * n_props
    fig, ax = plt.subplots(1, n_props, figsize=(figW, 5.5), constrained_layout=True)

    # Handle single subplot case
    if n_props == 1:
        ax = [ax]

    # Plot properties for each fuel
    for i, prop_name in enumerate(property_names):
        for fuel_idx, fuel_name in enumerate(fuel_names):
            T_data, prop_data, T_pred, pred = get_predictions_and_data(
                fuel_name, prop_name
            )
            line_color, marker_style = get_line_spec(fuel_name, fuel_index=fuel_idx)

            # Plot predictions
            ax[i].plot(
                fl.convert.K2C(T_pred),
                pred,
                "-",
                color=line_color,
                label=f"FuelLib: {get_legend_label(fuel_name)}",
                linewidth=4,
            )

            # Plot experimental data if available
            if len(prop_data) > 0:
                # Get props_data name for the legend
                props_data_name = fl.get_metadata_props_data(fuel_name, fuel_data_dir)
                data_label = props_data_name if props_data_name else fuel_name
                ax[i].scatter(
                    T_data,
                    prop_data,
                    marker=marker_style,
                    label=f"Data: {get_legend_label(data_label)}",
                    facecolors=line_color,
                    s=75,
                    zorder=5,
                )

        # Format subplot
        ax[i].set_xlabel("T [°C]", fontsize=18)
        ax[i].set_ylabel(ylab.get(prop_name, prop_name), fontsize=18)
        ax[i].tick_params(labelsize=18)
        ax[i].grid(alpha=0.3)

    # Add legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=len(fuel_names), fontsize=18
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # Save the plot if requested
    if save:
        fuel_str = "_".join(fuel_names)
        plot_file = os.path.join(output_dir, f"mixture_properties_{fuel_str}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Mixture properties plot saved to {plot_file}")

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig)


def comp_main():
    """
    Entry point for fl-plt-comp command - Plot fuel composition.
    """
    parser = argparse.ArgumentParser(
        description="Plot fuel composition by compound and chemical family."
    )

    # Fuel name (required)
    parser.add_argument(
        "-f",
        "--fuel_name",
        required=True,
        metavar="NAME",
        help="Name of the fuel to plot (required).",
    )
    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        metavar="PATH",
        help="Directory to save the plot (optional, default: current directory).",
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        metavar="TITLE",
        help="Title for the plots (optional, default: fuel_name, or 'none' to disable).",
    )
    parser.add_argument(
        "-decomp",
        "--decomp_name",
        default=None,
        metavar="NAME",
        help="Name of the decomposition file to use (optional, default: fuel_name).",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display the plot with plt.show() (optional, default: False).",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=lambda x: str(x).lower() not in ["false", "0"],
        default=True,
        metavar="{true,false}",
        help="Save the plot to a file (optional, default: True).",
    )

    args = parser.parse_args()

    try:
        plot_composition(
            args.fuel_name,
            fuel_data_dir=args.fuel_data_dir,
            output_dir=args.output_dir,
            title=args.title,
            decomp_name=args.decomp_name,
            save=args.save,
            display=args.display,
        )
    except Exception as e:
        print(f"Error plotting composition: {e}")
        exit(1)


def props_main():
    """
    Entry point for fl-plt-props command - Plot mixture properties.
    """
    parser = argparse.ArgumentParser(
        description="Plot mixture properties over temperature range for fuel(s)."
    )

    parser.add_argument(
        "-f",
        "--fuel_names",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Name(s) of fuel(s) to plot (required, space-separated for multiple).",
    )
    parser.add_argument(
        "-p",
        "--property_names",
        nargs="+",
        default=None,
        metavar="PROP",
        help="Properties to plot (optional). Options: Density, Viscosity, VaporPressure, SurfaceTension, ThermalConductivity",
    )
    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        metavar="PATH",
        help="Directory to save the plot (optional, default: current directory).",
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        metavar="TITLE",
        help="Title for the plot (optional).",
    )
    parser.add_argument(
        "-decomp",
        "--decomp_name",
        default=None,
        metavar="NAME",
        help="Name of the decomposition file to use (optional, default: fuel_name).",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display the plot with plt.show() (optional, default: False).",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=lambda x: str(x).lower() not in ["false", "0"],
        default=True,
        metavar="{true,false}",
        help="Save the plot to a file (optional, default: True).",
    )

    args = parser.parse_args()

    try:
        plot_mixture_properties(
            args.fuel_names,
            property_names=args.property_names,
            fuel_data_dir=args.fuel_data_dir,
            output_dir=args.output_dir,
            title=args.title,
            decomp_name=args.decomp_name,
            save=args.save,
            display=args.display,
        )
    except Exception as e:
        print(f"Error plotting mixture properties: {e}")
        exit(1)


def main():
    """
    Main entry point for CLI usage.

    This function handles routing between composition and mixture properties plotting.
    """
    parser = argparse.ArgumentParser(
        description="Plot fuel composition or mixture properties."
    )

    # Subparsers for different plot types
    subparsers = parser.add_subparsers(
        dest="plot_type", help="Type of plot to generate"
    )

    # Composition plotter subcommand
    comp_parser = subparsers.add_parser("comp", help="Plot fuel composition")
    comp_parser.add_argument(
        "-f",
        "--fuel_name",
        required=True,
        metavar="NAME",
        help="Name of the fuel to plot (required).",
    )
    comp_parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    comp_parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        metavar="PATH",
        help="Directory to save the plot (optional, default: current directory).",
    )
    comp_parser.add_argument(
        "-t",
        "--title",
        default=None,
        metavar="TITLE",
        help="Title for the plots (optional, default: fuel_name, or 'none' to disable).",
    )

    # Mixture properties plotter subcommand
    props_parser = subparsers.add_parser(
        "props", help="Plot mixture properties over temperature range"
    )
    props_parser.add_argument(
        "-f",
        "--fuel_names",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Name(s) of fuel(s) to plot (required, space-separated for multiple).",
    )
    props_parser.add_argument(
        "-p",
        "--property_names",
        nargs="+",
        default=None,
        metavar="PROP",
        help="Properties to plot (optional). Options: Density, Viscosity, VaporPressure, SurfaceTension, ThermalConductivity",
    )
    props_parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    props_parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        metavar="PATH",
        help="Directory to save the plot (optional, default: current directory).",
    )
    props_parser.add_argument(
        "-t",
        "--title",
        default=None,
        metavar="TITLE",
        help="Title for the plot (optional).",
    )

    args = parser.parse_args()

    if args.plot_type == "comp":
        try:
            plot_composition(
                args.fuel_name,
                fuel_data_dir=args.fuel_data_dir,
                output_dir=args.output_dir,
                title=args.title,
            )
        except Exception as e:
            print(f"Error plotting composition: {e}")
            exit(1)

    elif args.plot_type == "props":
        try:
            plot_mixture_properties(
                args.fuel_names,
                property_names=args.property_names,
                fuel_data_dir=args.fuel_data_dir,
                output_dir=args.output_dir,
                title=args.title,
            )
        except Exception as e:
            print(f"Error plotting mixture properties: {e}")
            exit(1)

    else:
        # If no subcommand specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()
