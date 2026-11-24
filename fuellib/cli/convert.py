"""Command-line tools to convert temperatures (non-pint quantities)."""

import argparse

from ..utils import utility
from ..utils.types import Units

_TEMPERATURE_UNITS = {
    "C": ("celsius", "Celsius"),
    "K": ("kelvin", "Kelvin"),
    "F": ("fahrenheit", "Fahrenheit"),
}
"""Pint units and display names of the temperature scales, by abbreviation."""

_TEMPERATURE_CONVERSIONS = ["c2k", "k2c", "c2f", "f2c", "f2k", "k2f"]
"""Temperature conversions available from the command line, as ``<from>2<to>``."""


def _convert_temperature(args: argparse.Namespace) -> None:
    """Convert a temperature between scales and print the result."""
    temp = Units.Quantity(args.temperature, args.source_units)
    result = temp.to(args.target_units)
    print(f"{temp:~P.2f} = {result:~P.2f}")


def _convert_epsilon(args: argparse.Namespace) -> None:
    """Convert Lennard-Jones epsilon from J/mol to K and print the result."""
    result = utility.epsilon_to_characteristic_temperature(args.epsilon)
    print(f"Characteristic temperature: {result:~P.3f}")


def add_parser(commands: "argparse._SubParsersAction") -> None:
    """Add the ``convert`` command and its subcommands.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = "Convert temperatures and Lennard-Jones parameters."
    convert = commands.add_parser("convert", help=description, description=description)
    conversions = convert.add_subparsers(
        dest="conversion", metavar="CONVERSION", required=True
    )

    for name in _TEMPERATURE_CONVERSIONS:
        source, target = name.upper().split("2")
        source_units, source_name = _TEMPERATURE_UNITS[source]
        target_units, target_name = _TEMPERATURE_UNITS[target]
        description = f"Convert temperature from {source_name} to {target_name}"
        parser = conversions.add_parser(name, help=description, description=description)
        parser.add_argument(
            "temperature",
            type=float,
            metavar="TEMP",
            help=f"Temperature in {source_name}",
        )
        parser.set_defaults(
            func=_convert_temperature,
            source_units=source_units,
            target_units=target_units,
        )

    description = "Convert Lennard-Jones well depth epsilon from J/mol to Kelvin"
    parser = conversions.add_parser("eps2k", help=description, description=description)
    parser.add_argument(
        "epsilon",
        type=float,
        metavar="EPSILON",
        help="Lennard-Jones well depth in J/mol",
    )
    parser.set_defaults(func=_convert_epsilon)
