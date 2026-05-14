"""Command-line tools to convert temperature between Celsius and Kelvin."""

import sys
from .. import C2K, K2C


def c2k_main():
    """Convert temperature from Celsius to Kelvin via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-C2K <temperature_in_celsius>")
        print("\nConvert temperature from Celsius to Kelvin")
        sys.exit(1)

    try:
        temp_c = float(sys.argv[1])
        result = C2K(temp_c)
        print(f"{temp_c} °C = {result:.2f} K")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)


def k2c_main():
    """Convert temperature from Kelvin to Celsius via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-K2C <temperature_in_kelvin>")
        print("\nConvert temperature from Kelvin to Celsius")
        sys.exit(1)

    try:
        temp_k = float(sys.argv[1])
        result = K2C(temp_k)
        print(f"{temp_k} K = {result:.2f} °C")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)
