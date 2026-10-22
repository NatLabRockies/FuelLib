"""Data locator module for FuelLib.

This module provides functions to locate data directories and files embedded
within the fuellib package using importlib.resources.
"""

import os
from importlib.resources import files

__all__ = [
    "get_data_dir",
    "get_fueldata_dir",
    "get_fueldata_gc_dir",
    "get_fueldata_props_dir",
]


def _get_props_dir_for_fueldata(fuel_data_dir: str) -> str | None:
    """Get the properties directory for a fuel data directory, or None if not found.

    Args:
        fuel_data_dir: Path to the fuel data directory.

    Returns:
        Path to the properties directory, or None if not found.
    """
    props_dir = os.path.join(fuel_data_dir, "propertiesData")
    return props_dir if os.path.isdir(props_dir) else None


def get_data_dir() -> str:
    """Get the path to FuelLib's data directory.

    Returns:
        Absolute path to the data directory.
    """
    data_ref = files("fuellib").joinpath("data")
    # Convert to a concrete path
    return str(data_ref)


# Alias for get_data_dir (backward compatibility)
get_fueldata_dir = get_data_dir


def get_fueldata_gc_dir() -> str:
    """Get the path to FuelLib's GC data subdirectory.

    Returns:
        Absolute path to embedded data/gcData directory.
    """
    return os.path.join(get_fueldata_dir(), "gcData")


def get_fueldata_props_dir() -> str | None:
    """Get the path to FuelLib's properties data subdirectory, or None if not found.

    This directory is optional.

    Returns:
        Absolute path to embedded data/propertiesData directory, or None if not found.
    """
    return _get_props_dir_for_fueldata(get_fueldata_dir())
