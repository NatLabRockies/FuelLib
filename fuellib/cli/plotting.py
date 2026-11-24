"""Command-line tools for plotting fuel composition and properties.

The plotting functions are in :mod:`fuellib.utils.plotting`.
"""

import argparse

from ..utils.plotting import plot_composition, plot_mixture_properties


def _add_comp_parser(plots: "argparse._SubParsersAction") -> None:
    """Add the ``plot comp`` command."""
    description = "Plot fuel composition by compound and chemical family."
    parser = plots.add_parser("comp", help=description, description=description)

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

    parser.set_defaults(func=_comp)


def _comp(args: argparse.Namespace) -> None:
    """Plot the composition of a fuel."""
    plot_composition(
        args.fuel_name,
        fuel_data_dir=args.fuel_data_dir,
        output_dir=args.output_dir,
        title=args.title,
        save=args.save,
        display=args.display,
    )


def _add_props_parser(plots: "argparse._SubParsersAction") -> None:
    """Add the ``plot props`` command."""
    description = "Plot mixture properties over temperature range for fuel(s)."
    parser = plots.add_parser("props", help=description, description=description)

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

    parser.set_defaults(func=_props)


def _props(args: argparse.Namespace) -> None:
    """Plot mixture properties of one or more fuels."""
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


def add_parser(commands: "argparse._SubParsersAction") -> None:
    """Add the ``plot`` command and its subcommands.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = (
        "Plot fuel composition or mixture properties. To use other correlations, "
        "use the Python API."
    )
    plot = commands.add_parser("plot", help=description, description=description)
    plots = plot.add_subparsers(dest="plot_type", metavar="PLOT", required=True)
    _add_comp_parser(plots)
    _add_props_parser(plots)
