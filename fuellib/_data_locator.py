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


def _get_props_dir_for_fueldata(fuel_data_dir):
    """Get the properties directory for a fuel data directory, or None if it doesn't exist.

    :param fuel_data_dir: Path to fuel data directory.
    :type fuel_data_dir: str
    :return: Path to properties directory, or None if not found.
    :rtype: str or None
    """
    props_dir = os.path.join(fuel_data_dir, "propertiesData")
    return props_dir if os.path.isdir(props_dir) else None


def get_data_dir():
    """Get the path to FuelLib's data directory.

    :return: Absolute path to the data directory.
    :rtype: str
    """
    data_ref = files("fuellib").joinpath("data")
    # Convert to a concrete path
    return str(data_ref)


def get_fueldata_dir():
    """Get the path to FuelLib's fuel data directory.

    :return: Absolute path to the embedded data directory.
    :rtype: str
    """
    return get_data_dir()


def get_fueldata_gc_dir():
    """Get the path to FuelLib's GC data subdirectory.

    :return: Absolute path to embedded data/gcData directory.
    :rtype: str
    """
    return os.path.join(get_fueldata_dir(), "gcData")


def get_fueldata_props_dir():
    """Get the path to FuelLib's properties data subdirectory, or None if not found.

    This directory is optional.

    :return: Absolute path to embedded data/propertiesData directory, or None if not found.
    :rtype: str or None
    """
    return _get_props_dir_for_fueldata(get_fueldata_dir())
