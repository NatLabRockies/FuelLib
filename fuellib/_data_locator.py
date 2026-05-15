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

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _validate_fuel_data_dir(fuel_data_dir):
    """
    Validate that a custom fuel data directory has required subdirectories.
    
    :param fuel_data_dir: Path to fuel data directory.
    :type fuel_data_dir: str
    :raises ValueError: If required subdirectories are missing.
    """
    if fuel_data_dir is None:
        return
    
    gc_dir = os.path.join(fuel_data_dir, "gcData")
    decomp_dir = os.path.join(fuel_data_dir, "groupDecompositionData")
    
    if not os.path.isdir(gc_dir):
        raise ValueError(
            f"Custom fuel data directory is missing 'gcData' subdirectory:\n"
            f"  Expected: {gc_dir}"
        )
    
    if not os.path.isdir(decomp_dir):
        raise ValueError(
            f"Custom fuel data directory is missing 'groupDecompositionData' subdirectory:\n"
            f"  Expected: {decomp_dir}"
        )


def _get_props_dir_for_fueldata(fuel_data_dir):
    """
    Get the properties directory for a fuel data directory, or None if it doesn't exist.
    
    :param fuel_data_dir: Path to fuel data directory.
    :type fuel_data_dir: str
    :return: Path to properties directory, or None if not found.
    :rtype: str or None
    """
    props_dir = os.path.join(fuel_data_dir, "propertiesData")
    return props_dir if os.path.isdir(props_dir) else None


def get_data_dir():
    """
    Get the path to FuelLib's data directory.

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
    Get the path to FuelLib's fuel data directory.

    :return: Absolute path to embedded fuelData directory.
    :rtype: str
    """
    return os.path.join(get_data_dir(), "fuelData")


def get_fueldata_gc_dir():
    """
    Get the path to FuelLib's GC data subdirectory.

    :return: Absolute path to embedded fuelData/gcData directory.
    :rtype: str
    """
    return os.path.join(get_fueldata_dir(), "gcData")


def get_fueldata_decomp_dir():
    """
    Get the path to FuelLib's group decomposition data subdirectory.

    :return: Absolute path to embedded fuelData/groupDecompositionData directory.
    :rtype: str
    """
    return os.path.join(get_fueldata_dir(), "groupDecompositionData")


def get_fueldata_props_dir():
    """
    Get the path to FuelLib's properties data subdirectory, or None if not found.
    
    This directory is optional.

    :return: Absolute path to embedded fuelData/propertiesData directory, or None if not found.
    :rtype: str or None
    """
    return _get_props_dir_for_fueldata(get_fueldata_dir())


def get_decomp_name_from_metadata(fuel_name, fuel_data_dir=None):
    """
    Load decomposition name mapping from fuel_metadata.yaml.

    :param fuel_name: Name of the fuel to look up.
    :type fuel_name: str
    :param fuel_data_dir: Directory containing fuel data. If None, uses embedded data.
    :type fuel_data_dir: str, optional
    :return: Decomposition name from metadata.
    :rtype: str
    :raises FileNotFoundError: If fuel_metadata.yaml is missing or fuel not found in metadata
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to use custom fuels. Install it with: pip install pyyaml"
        )

    if fuel_data_dir is None:
        # Use embedded data
        metadata_file = os.path.join(get_fueldata_dir(), "fuel_metadata.yaml")
        data_dir_display = "FuelLib embedded data"
    else:
        # Use custom data directory
        metadata_file = os.path.join(fuel_data_dir, "fuel_metadata.yaml")
        data_dir_display = fuel_data_dir

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"fuel_metadata.yaml not found in {data_dir_display}.\n\n"
            f"This file is required for all fuels. Please create:\n"
            f"  {metadata_file}\n\n"
            f"Minimal example:\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}  # or name of your .csv file in groupDecompositionData/\n\n"
            f"See the 'Adding Custom Fuels' documentation for more details."
        )

    try:
        with open(metadata_file, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(
            f"Error parsing {metadata_file}:\n{e}\n\n"
            f"Make sure the file is valid YAML with proper indentation."
        )

    if not data or "fuels" not in data:
        raise ValueError(
            f"Invalid metadata file {metadata_file}.\n"
            f"File must contain a 'fuels' section.\n\n"
            f"Example:\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}"
        )

    if fuel_name not in data["fuels"]:
        available = list(data["fuels"].keys())
        raise KeyError(
            f"Fuel '{fuel_name}' not found in {metadata_file}.\n\n"
            f"Available fuels: {', '.join(available) if available else 'none'}\n\n"
            f"Add an entry for '{fuel_name}':\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}"
        )

    fuel_meta = data["fuels"][fuel_name]

    if "decomp_name" not in fuel_meta:
        raise ValueError(
            f"Incomplete metadata for fuel '{fuel_name}' in {metadata_file}.\n\n"
            f"Required fields:\n"
            f"  - decomp_name: Name of the .csv file in groupDecompositionData/ (without .csv extension)\n\n"
            f"Current entry:\n"
            f"  {fuel_name}: {fuel_meta}"
        )

    return fuel_meta["decomp_name"]


def get_props_data_from_metadata(fuel_name, fuel_data_dir=None):
    """
    Load properties data name mapping from fuel_metadata.yaml.
    
    Returns None if props_data is not specified in metadata (it's optional).

    :param fuel_name: Name of the fuel to look up.
    :type fuel_name: str
    :param fuel_data_dir: Directory containing fuel data. If None, uses embedded data.
    :type fuel_data_dir: str, optional
    :return: Properties data name from metadata, or None if not specified.
    :rtype: str or None
    :raises FileNotFoundError: If fuel_metadata.yaml is missing or fuel not found in metadata
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to use custom fuels. Install it with: pip install pyyaml"
        )

    if fuel_data_dir is None:
        # Use embedded data
        metadata_file = os.path.join(get_fueldata_dir(), "fuel_metadata.yaml")
    else:
        # Use custom data directory
        metadata_file = os.path.join(fuel_data_dir, "fuel_metadata.yaml")

    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not data or "fuels" not in data or fuel_name not in data["fuels"]:
        return None

    fuel_meta = data["fuels"][fuel_name]
    
    # Return props_data if present, otherwise None
    return fuel_meta.get("props_data", None)
