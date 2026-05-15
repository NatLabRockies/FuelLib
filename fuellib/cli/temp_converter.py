"""Command-line tools to convert temperatures."""

import sys
from .. import C2K, K2C, C2F, F2C, F2K, K2F


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


def c2f_main():
    """Convert temperature from Celsius to Fahrenheit via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-C2F <temperature_in_celsius>")
        print("\nConvert temperature from Celsius to Fahrenheit")
        sys.exit(1)

    try:
        temp_c = float(sys.argv[1])
        result = C2F(temp_c)
        print(f"{temp_c} °C = {result:.2f} °F")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)


def f2c_main():
    """Convert temperature from Fahrenheit to Celsius via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-F2C <temperature_in_fahrenheit>")
        print("\nConvert temperature from Fahrenheit to Celsius")
        sys.exit(1)

    try:
        temp_f = float(sys.argv[1])
        result = F2C(temp_f)
        print(f"{temp_f} °F = {result:.2f} °C")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)


def f2k_main():
    """Convert temperature from Fahrenheit to Kelvin via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-F2K <temperature_in_fahrenheit>")
        print("\nConvert temperature from Fahrenheit to Kelvin")
        sys.exit(1)

    try:
        temp_f = float(sys.argv[1])
        result = F2K(temp_f)
        print(f"{temp_f} °F = {result:.2f} K")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)


def k2f_main():
    """Convert temperature from Kelvin to Fahrenheit via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-K2F <temperature_in_kelvin>")
        print("\nConvert temperature from Kelvin to Fahrenheit")
        sys.exit(1)

    try:
        temp_k = float(sys.argv[1])
        result = K2F(temp_k)
        print(f"{temp_k} K = {result:.2f} °F")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)
