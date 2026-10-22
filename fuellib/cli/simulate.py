"""Command-line simulations of fuel tests."""

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fuellib.fuel import Fuel
from fuellib.simulate.distillation import greenfield
from fuellib.utils import Units, types


def _d86_data(fuel: Fuel) -> tuple[types.Quantity1D, types.Quantity1D] | None:
    """Measured D86 curve of a fuel from its properties data.

    Distillation rows store the D86 temperature in ``Temp`` and the recovered volume
    in ``Property_Value``.

    Args:
        fuel: Fuel object.

    Returns:
        Recovered volume fractions and D86 temperatures, or None if the fuel has no
        distillation data.
    """
    if fuel.propData is None:
        return None
    df = fuel.propData[fuel.propData["Property"] == "Distillation"]
    if df.empty:
        return None
    # pint does not define "percentage"
    vol_units = df["Property_Units"].iloc[0].replace("percentage", "percent")
    vol = Units.Quantity(df["Property_Value"].to_numpy(float), vol_units)
    T = Units.Quantity(df["Temp"].to_numpy(float), df["Temp_Units"].iloc[0])
    return vol, T


def _d86_greenfield(fuels: list[Fuel]) -> Figure:
    """Simulate the ASTM D86 distillation curve with the Greenfield et al. model.

    Args:
        fuels: Fuel objects, overlaid on the same axes and labeled by name, with
            their measured D86 curves if available.

    Returns:
        Figure with the D86 curves.
    """
    ax = None
    for fuel in fuels:
        ax = greenfield.distill(fuel).plot(ax=ax, label=fuel.name, data=_d86_data(fuel))
    return plt.gcf()


SIMULATIONS: dict[str, Callable[[list[Fuel]], Figure]] = {
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
            fuelDataDir=args.fuel_data_dir,
            useRefProperties=args.use_ref_props,
        )
        for name in args.fuel_names
    ]
    fig = SIMULATIONS[args.simulation](fuels)

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
