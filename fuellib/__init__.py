"""
FuelLib: Fuel Library for Group Contribution Method calculations.

FuelLib utilizes the Group Contribution Method (GCM) as proposed by Constantinou
and Gani (1994, 1995) to calculate thermodynamic and mixture properties of fuels.

See :class:`fuel` for the main class and complete API documentation.
"""

try:
    from importlib.metadata import version

    __version__ = version("fuellib")
except Exception:
    __version__ = "unknown"

# Import constants
from .constants import k_B, N_A

# Import fuel class
from .fuel import fuel

# Import data locator functions for convenience
from ._data_locator import (
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_decomp_dir,
    get_fueldata_props_dir,
    get_gcmtable_dir,
)

# Import submodules for namespacing
from . import constants
from . import convert
from . import utility

__all__ = [
    "fuel",
    "k_B",
    "N_A",
    "get_fueldata_dir",
    "get_fueldata_gc_dir",
    "get_fueldata_decomp_dir",
    "get_fueldata_props_dir",
    "get_gcmtable_dir",
    "constants",
    "convert",
    "utility",
]
