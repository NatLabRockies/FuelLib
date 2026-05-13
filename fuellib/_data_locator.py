"""
Data locator module for FuelLib.

This module provides functions to locate data directories and files embedded
within the fuellib package using importlib.resources.
"""

import os
import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files


def get_data_dir():
    """
    Get the path to the embedded data directory.

    :return: Absolute path to the data directory.
    :rtype: str
    """
    data_ref = files("fuellib").joinpath("data")
    # Convert to a concrete path
    return str(data_ref)


def get_gcmtable_dir():
    """
    Get the path to the GCM table data directory.

    :return: Absolute path to gcmTableData directory.
    :rtype: str
    """
    return os.path.join(get_data_dir(), "gcmTableData")


def get_fueldata_dir():
    """
    Get the path to the fuel data directory.

    :return: Absolute path to fuelData directory.
    :rtype: str
    """
    return os.path.join(get_data_dir(), "fuelData")


def get_fueldata_gc_dir():
    """
    Get the path to the GC data subdirectory.

    :return: Absolute path to fuelData/gcData directory.
    :rtype: str
    """
    return os.path.join(get_fueldata_dir(), "gcData")


def get_fueldata_decomp_dir():
    """
    Get the path to the group decomposition data subdirectory.

    :return: Absolute path to fuelData/groupDecompositionData directory.
    :rtype: str
    """
    return os.path.join(get_fueldata_dir(), "groupDecompositionData")


def get_fueldata_props_dir():
    """
    Get the path to the properties data subdirectory.

    :return: Absolute path to fuelData/propertiesData directory.
    :rtype: str
    """
    return os.path.join(get_fueldata_dir(), "propertiesData")
