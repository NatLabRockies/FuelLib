"""
FuelLib: Fuel Library for Group Contribution Method calculations.

FuelLib utilizes the Group Contribution Method (GCM) as proposed by Constantinou
and Gani (1994, 1995) to calculate thermodynamic and mixture properties of fuels.

See :class:`fuel` for the main class and complete API documentation.
"""

try:
    from importlib.metadata import version

    __version__ = version("fuellib")
except ImportError:
    __version__ = "unknown"

# Import fuel class
# Import submodules for namespacing
from . import constants, convert, units, utility

# Import data locator functions
from ._data_locator import *
from .fuel import fuel

__all__ = [
    "constants",
    "convert",
    "fuel",
    "get_data_dir",
    "get_fueldata_decomp_dir",
    "get_fueldata_dir",
    "get_fueldata_gc_dir",
    "get_fueldata_props_dir",
    "get_gcmtable_dir",
    "get_metadata_decomp_name",
    "get_metadata_props_data",
    "units",
    "utility",
]
