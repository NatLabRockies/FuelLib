"""
Utilities for managing and querying the fuel library.

This module provides tools for listing and discovering available fuels
in the FuelLib database, including source information and metadata.
"""

import os
import argparse
import fuellib as fl

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_fuel_metadata(fuel_data_dir=None):
    """
    Load fuel metadata from YAML file if available.

    :param fuel_data_dir: Optional directory containing fuel data (e.g., customFuels/fuelData).
                         If None, loads from embedded FuelLib data.
    :type fuel_data_dir: str, optional
    :return: Dictionary of fuel metadata or empty dict if not available
    :rtype: dict
    """
    if not HAS_YAML:
        return {}

    # Determine which metadata file to load
    if fuel_data_dir is None:
        # Load from embedded data
        metadata_file = os.path.join(fl.get_fueldata_dir(), "fuel_metadata.yaml")
    else:
        # Load from custom directory
        metadata_file = os.path.join(fuel_data_dir, "fuel_metadata.yaml")

    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                data = yaml.safe_load(f)
                return data.get("fuels", {}) if data else {}
    except Exception as e:
        pass

    return {}


def list_fuels_main():
    """
    Entry point for fl-fuels command - List all available fuels in the library.
    """
    parser = argparse.ArgumentParser(
        description="List all available fuels in the FuelLib library."
    )

    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=None,
        metavar="PATH",
        help="Directory where fuel data files are located (e.g., customFuels/fuelData). If not specified, uses embedded FuelLib data.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information including source and category for each fuel.",
    )

    args = parser.parse_args()

    if args.fuel_data_dir is None:
        fuel_data_dir = fl.get_fueldata_gc_dir()
        metadata_dir = None  # Use embedded metadata
    else:
        fuel_data_dir = os.path.join(args.fuel_data_dir, "gcData")
        metadata_dir = args.fuel_data_dir  # Load metadata from same custom directory

    try:
        # List all fuel files in the gcData directory
        if not os.path.exists(fuel_data_dir):
            print(f"Error: Fuel data directory not found: {fuel_data_dir}")
            exit(1)

        # Extract fuel names from *_init.csv files
        fuel_files = [f for f in os.listdir(fuel_data_dir) if f.endswith("_init.csv")]
        fuel_names = sorted([f.replace("_init.csv", "") for f in fuel_files])

        if not fuel_names:
            print("No fuels found in the specified directory.")
            exit(0)

        # Load metadata from the appropriate location
        metadata = load_fuel_metadata(metadata_dir)

        print("\n" + "=" * 80)
        if args.fuel_data_dir:
            print(f"Available Fuels in {args.fuel_data_dir}")
        else:
            print("Available Fuels in FuelLib")
        print("=" * 80)

        if args.verbose and metadata:
            # Verbose output with metadata
            for i, fuel_name in enumerate(fuel_names, 1):
                meta = metadata.get(fuel_name, {})
                category = meta.get("category", "Unknown")
                source = meta.get("source")
                description = meta.get("description", "")

                print(f"{i:2d}. {fuel_name}")
                print(f"    Category:      {category}")
                if source:
                    print(f"    Source:        {source}")
                if description:
                    print(f"    Note:          {description}")
                print()
        else:
            # Simple list output
            for i, fuel_name in enumerate(fuel_names, 1):
                if metadata and fuel_name in metadata:
                    source = metadata[fuel_name].get("source", "")
                    if source:
                        print(f"{i:2d}. {fuel_name:<20} [{source}]")
                    else:
                        print(f"{i:2d}. {fuel_name}")
                else:
                    print(f"{i:2d}. {fuel_name}")

        print("=" * 80)
        print(f"Total: {len(fuel_names)} fuel(s)")
        if not args.verbose and metadata:
            print(
                "Use -v/--verbose for detailed information including source and category"
            )
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"Error listing fuels: {e}")
        exit(1)
