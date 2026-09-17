"""Command-line interface for plotting fuel compositions and properties."""

import argparse
import sys

from fuellib.plot import plot_composition, plot_mixture_properties


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
        type=lambda x: str(x).lower() not in ["false", "0"],
        default=True,
        metavar="{true,false}",
        help="Display the plot with plt.show() (optional, default: True).",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the plot to a file (optional, default: False).",
    )

    args = parser.parse_args()

    plot_composition(
        args.fuel_name,
        fuel_data_dir=args.fuel_data_dir,
        output_dir=args.output_dir,
        title=args.title,
        decomp_name=args.decomp_name,
        save=args.save,
        display=args.display,
    )


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
        type=lambda x: str(x).lower() not in ["false", "0"],
        default=True,
        metavar="{true,false}",
        help="Display the plot with plt.show() (optional, default: True).",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the plot to a file (optional, default: False).",
    )

    args = parser.parse_args()

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
