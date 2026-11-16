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
            and legend labels of ``fl-plt-props``, with their measured D86 curves
            if available.
        air_cstr: Whether to include the air-filled CSTR in the simulation.

    Returns:
        Figure with the D86 curves.
    """
    return plotting.plot_distillation(fuels, air_cstr=air_cstr)


SIMULATIONS: dict[str, Callable[[list[Fuel], bool], Figure]] = {
    "d86_greenfield": _d86_greenfield,
}
"""Simulations available from the command line, by name."""


def main() -> None:
    """Run a simulation for fuels and export or show the plot of the results."""
    parser = argparse.ArgumentParser(
        description="Simulate a fuel test and plot the results."
    )
    parser.add_argument(
        "simulation",
        type=str.casefold,
        choices=SIMULATIONS,
        help="Simulation to perform.",
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
    args = parser.parse_args()

    fuels = [
        Fuel(
            name,
            data_dir=args.fuel_data_dir,
            use_ref_props=args.use_ref_props,
        )
        for name in args.fuel_names
    ]
    fig = SIMULATIONS[args.simulation](fuels, args.air_cstr)

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
