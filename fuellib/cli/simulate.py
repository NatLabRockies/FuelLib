"""Command-line simulations of fuel tests."""

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fuellib.fuel import Fuel
from fuellib.utils import plotting


def _d86_greenfield(fuels: list[Fuel], air_cstr: bool) -> Figure:
    """Simulate the ASTM D86 distillation curve with the Greenfield et al. model.

    Args:
        fuels: Fuel objects, overlaid on the same axes with the colors, symbols
            and legend labels of ``fl plot props``, with their measured D86 curves
            if available.
        air_cstr: Whether to include the air-filled CSTR in the simulation.

    Returns:
        Figure with the D86 curves.
    """
    return plotting.plot_distillation(fuels, air_cstr=air_cstr)


DISTILLATION_MODELS: dict[str, Callable[[list[Fuel], bool], Figure]] = {
    "greenfield": _d86_greenfield,
}
"""Distillation curve models available from the command line, by name."""


def add_parser(commands: "argparse._SubParsersAction") -> None:
    """Add the ``simulate`` command and its subcommands.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = "Simulate a fuel test and plot the results."
    simulate = commands.add_parser(
        "simulate", help=description, description=description
    )
    tests = simulate.add_subparsers(dest="test", metavar="TEST", required=True)

    description = "Simulate the ASTM D86 distillation curve of fuels."
    parser = tests.add_parser("distillation", help=description, description=description)
    parser.add_argument(
        "--model",
        default="greenfield",
        type=str.casefold,
        choices=DISTILLATION_MODELS,
        help="Distillation curve model (default: greenfield).",
    )
    parser.add_argument(
        "-f",
        "--fuel_names",
        required=True,
        nargs="+",
        metavar="NAME",
        help="Name(s) of the fuel(s) (required, space-separated for multiple).",
    )
    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=None,
        type=Path,
        metavar="PATH",
        help="Directory where fuel data files are located (default: fuellib/data).",
    )
    parser.add_argument(
        "-r",
        "--use-ref-props",
        dest="use_ref_props",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference compound properties (default: True).",
    )
    parser.add_argument(
        "--air-cstr",
        dest="air_cstr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the air-filled CSTR in the distillation model (default: True).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        metavar="FILE",
        help="File to export the plot to (default: show the plot).",
    )
    parser.set_defaults(func=_distillation)


def _distillation(args: argparse.Namespace) -> None:
    """Simulate the distillation curves of fuels and export or show the plot."""
    fuels = [
        Fuel(
            name,
            data_dir=args.fuel_data_dir,
            use_ref_props=args.use_ref_props,
        )
        for name in args.fuel_names
    ]
    fig = DISTILLATION_MODELS[args.model](fuels, args.air_cstr)

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {args.output}")
    plt.close(fig)
