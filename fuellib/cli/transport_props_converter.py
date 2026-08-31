"""Command-line tool to convert transport properties for combustion simulations."""

import argparse

from .. import convert
from ..units import Q_


def eps2K_main():
    """Convert Lennard-Jones epsilon from J/mol to K via command line."""
    parser = argparse.ArgumentParser(
        description="Convert Lennard-Jones well depth epsilon from J/mol to Kelvin"
    )
    parser.add_argument(
        "epsilon",
        type=float,
        metavar="EPSILON",
        help="Lennard-Jones well depth in J/mol",
    )

    args = parser.parse_args()
    result = convert.epsilon_to_characteristic_temperature(Q_(args.epsilon, "J/mol"))
    print(f"Characteristic temperature: {result.magnitude:.3f} K")
