"""Command-line tools for plotting fuel composition and properties.

The plotting functions are in :mod:`fuellib.utils.plotting`.
"""

import argparse
import sys

from ..utils.plotting import plot_composition, plot_mixture_properties


def comp_main():
    """Entry point for fl-plt-comp command - Plot fuel composition."""
    parser = argparse.ArgumentParser(
        description="Plot fuel composition by compound and chemical family."
    )

    # Fuel name (required)
    parser.add_argument(
        "-f",
        "--fuel-name",
        required=True,
        metavar="NAME",
        help="Name of the fuel to plot (required).",
    )
    parser.add_argument(
        "-dir",
        "--fuel-data-dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
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

    try:
        plot_composition(
            args.fuel_name,
            fuel_data_dir=args.fuel_data_dir,
            output_dir=args.output_dir,
            title=args.title,
            save=args.save,
            display=args.display,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error plotting composition: {e}")
        sys.exit(1)


def props_main():
    """Entry point for fl-plt-props command - Plot mixture properties."""
    parser = argparse.ArgumentParser(
        description="Plot mixture properties over temperature range for fuel(s)."
    )

    parser.add_argument(
        "-f",
        "--fuel-names",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Name(s) of fuel(s) to plot (required, space-separated for multiple).",
    )
    parser.add_argument(
        "-p",
        "--property-names",
        nargs="+",
        default=None,
        metavar="PROP",
        help="Properties to plot (optional). Options: Density, Viscosity, VaporPressure, SurfaceTension, ThermalConductivity",
    )
    parser.add_argument(
        "-dir",
        "--fuel-data-dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
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
        "-leg",
        "--leg-labels",
        nargs="+",
        metavar="LABEL",
        help="Literal legend label for each fuel, in the same order as --fuel-names.",
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

    try:
        plot_mixture_properties(
            args.fuel_names,
            property_names=args.property_names,
            fuel_data_dir=args.fuel_data_dir,
            output_dir=args.output_dir,
            title=args.title,
            leg=args.leg_labels,
            save=args.save,
            display=args.display,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error plotting mixture properties: {e}")
        sys.exit(1)


def main():
    """Main entry point for CLI usage.

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
        "--fuel-name",
        required=True,
        metavar="NAME",
        help="Name of the fuel to plot (required).",
    )
    comp_parser.add_argument(
        "-dir",
        "--fuel-data-dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    comp_parser.add_argument(
        "-o",
        "--output-dir",
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
        "--fuel-names",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Name(s) of fuel(s) to plot (required, space-separated for multiple).",
    )
    props_parser.add_argument(
        "-p",
        "--property-names",
        nargs="+",
        default=None,
        metavar="PROP",
        help="Properties to plot (optional). Options: Density, Viscosity, VaporPressure, SurfaceTension, ThermalConductivity",
    )
    props_parser.add_argument(
        "-dir",
        "--fuel-data-dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (optional).",
    )
    props_parser.add_argument(
        "-o",
        "--output-dir",
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
    props_parser.add_argument(
        "-leg",
        "--leg-labels",
        nargs="+",
        metavar="LABEL",
        help="Literal legend label for each fuel, in the same order as --fuel-names.",
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
        except Exception as e:  # noqa: BLE001
            print(f"Error plotting composition: {e}")
            sys.exit(1)

    elif args.plot_type == "props":
        try:
            plot_mixture_properties(
                args.fuel_names,
                property_names=args.property_names,
                fuel_data_dir=args.fuel_data_dir,
                output_dir=args.output_dir,
                title=args.title,
                leg=args.leg_labels,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Error plotting mixture properties: {e}")
            sys.exit(1)

    else:
        # If no subcommand specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()
