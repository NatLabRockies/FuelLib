"""Command-line tools to convert temperatures."""

import argparse
from .. import convert


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
    result = convert.C2K(args.temperature)
    print(f"{args.temperature} °C = {result:.2f} K")


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
    result = convert.K2C(args.temperature)
    print(f"{args.temperature} K = {result:.2f} °C")


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
    result = convert.C2F(args.temperature)
    print(f"{args.temperature} °C = {result:.2f} °F")


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
    result = convert.F2C(args.temperature)
    print(f"{args.temperature} °F = {result:.2f} °C")


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
    result = convert.F2K(args.temperature)
    print(f"{args.temperature} °F = {result:.2f} K")


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
    result = convert.K2F(args.temperature)
    print(f"{args.temperature} K = {result:.2f} °F")
