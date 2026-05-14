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
        metadata_file = os.path.join(
            get_fueldata_dir(),
            "fuel_metadata.yaml"
        )
        data_dir_display = "FuelLib embedded data"
    else:
        # Use custom data directory
        metadata_file = os.path.join(
            fuel_data_dir,
            "fuel_metadata.yaml"
        )
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
        with open(metadata_file, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(
            f"Error parsing {metadata_file}:\n{e}\n\n"
            f"Make sure the file is valid YAML with proper indentation."
        )
    
    if not data or 'fuels' not in data:
        raise ValueError(
            f"Invalid metadata file {metadata_file}.\n"
            f"File must contain a 'fuels' section.\n\n"
            f"Example:\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}"
        )
    
    if fuel_name not in data['fuels']:
        available = list(data['fuels'].keys())
        raise KeyError(
            f"Fuel '{fuel_name}' not found in {metadata_file}.\n\n"
            f"Available fuels: {', '.join(available) if available else 'none'}\n\n"
            f"Add an entry for '{fuel_name}':\n"
            f"  fuels:\n"
            f"    {fuel_name}:\n"
            f"      decomp_name: {fuel_name}"
        )
    
    fuel_meta = data['fuels'][fuel_name]
    
    if 'decomp_name' not in fuel_meta:
        raise ValueError(
            f"Incomplete metadata for fuel '{fuel_name}' in {metadata_file}.\n\n"
            f"Required fields:\n"
            f"  - decomp_name: Name of the .csv file in groupDecompositionData/ (without .csv extension)\n\n"
            f"Current entry:\n"
            f"  {fuel_name}: {fuel_meta}"
        )
    
    return fuel_meta['decomp_name']
