"""Command-line tool to convert transport properties for combustion simulations."""

import sys
from .. import epsilon_to_characteristic_temperature


def eps2K_main():
    """Convert Lennard-Jones epsilon from J/mol to K via command line."""
    if len(sys.argv) != 2:
        print("Usage: fl-eps2K <epsilon_in_J_per_mol>")
        print("\nConvert Lennard-Jones well depth epsilon from J/mol to Kelvin")
        sys.exit(1)

    try:
        epsilon = float(sys.argv[1])
        result = epsilon_to_characteristic_temperature(epsilon)
        print(f"Characteristic temperature: {result:.3f} K")
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)
