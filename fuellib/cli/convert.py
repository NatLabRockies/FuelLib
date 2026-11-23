"""Command-line tools to convert temperatures (non-pint quantities)."""

import argparse

from ..utils import utility
from ..utils.types import Units


def c2k_main():
    """Convert temperature from Celsius to Kelvin via command line."""
    parser = argparse.ArgumentParser(
        description="Convert temperature from Celsius to Kelvin"
    )
    parser.add_argument(
        "temperature",
        type=float,
        metavar="TEMP",
        help="Temperature in Celsius",
    )

    args = parser.parse_args()
    temp = Units.Quantity(args.temperature, "celsius")
    result = temp.to("kelvin")
    print(f"{temp:.2f~P} = {result:.2f~P}")


def k2c_main():
    """Convert temperature from Kelvin to Celsius via command line."""
    parser = argparse.ArgumentParser(
        description="Convert temperature from Kelvin to Celsius"
    )
    parser.add_argument(
        "temperature",
        type=float,
        metavar="TEMP",
        help="Temperature in Kelvin",
    )

    args = parser.parse_args()
    temp = Units.Quantity(args.temperature, "kelvin")
    result = temp.to("celsius")
    print(f"{temp:.2f~P} = {result:.2f~P}")


def c2f_main():
    """Convert temperature from Celsius to Fahrenheit via command line."""
    parser = argparse.ArgumentParser(
        description="Convert temperature from Celsius to Fahrenheit"
    )
    parser.add_argument(
        "temperature",
        type=float,
        metavar="TEMP",
        help="Temperature in Celsius",
    )

    args = parser.parse_args()
    temp = Units.Quantity(args.temperature, "celsius")
    result = temp.to("fahrenheit")
    print(f"{temp:.2f~P} = {result:.2f~P}")


def f2c_main():
    """Convert temperature from Fahrenheit to Celsius via command line."""
    parser = argparse.ArgumentParser(
        description="Convert temperature from Fahrenheit to Celsius"
    )
    parser.add_argument(
        "temperature",
        type=float,
        metavar="TEMP",
        help="Temperature in Fahrenheit",
    )

    args = parser.parse_args()
    temp = Units.Quantity(args.temperature, "fahrenheit")
    result = temp.to("celsius")
    print(f"{temp:.2f~P} = {result:.2f~P}")


def f2k_main():
    """Convert temperature from Fahrenheit to Kelvin via command line."""
    parser = argparse.ArgumentParser(
        description="Convert temperature from Fahrenheit to Kelvin"
    )
    parser.add_argument(
        "temperature",
        type=float,
        metavar="TEMP",
        help="Temperature in Fahrenheit",
    )

    args = parser.parse_args()
    temp = Units.Quantity(args.temperature, "fahrenheit")
    result = temp.to("kelvin")
    print(f"{temp:.2f~P} = {result:.2f~P}")


def k2f_main():
    """Convert temperature from Kelvin to Fahrenheit via command line."""
    parser = argparse.ArgumentParser(
        description="Convert temperature from Kelvin to Fahrenheit"
    )
    parser.add_argument(
        "temperature",
        type=float,
        metavar="TEMP",
        help="Temperature in Kelvin",
    )

    args = parser.parse_args()
    temp = Units.Quantity(args.temperature, "kelvin")
    result = temp.to("fahrenheit")
    print(f"{temp:.2f~P} = {result:.2f~P}")


def eps2k_main():
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
    result = utility.epsilon_to_characteristic_temperature(args.epsilon)
    print(f"Characteristic temperature: {result:.3f~P}")
