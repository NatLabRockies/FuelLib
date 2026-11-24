"""FuelLib: Fuel property prediction based on group contribution methods.

FuelLib predicts the thermophysical properties of fuels and their individual
compounds from group contribution methods (Constantinou and Gani, Nannoolal,
UNIFAC, Boehm) and, when available, measured reference data.

See :class:`Fuel` for the main class and complete API documentation.
"""

try:
    from importlib.metadata import version

    __version__ = version("fuellib")
except ImportError:
    __version__ = "unknown"

# Import fuel class
# Import submodules for namespacing
from . import correlate

# Import data locator functions
from .data._data_locator import (
    get_data_dir,
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_props_dir,
)
from .data._reference_data import DEFAULT_DATA_DIR
from .fuel import Fuel
from .utils import Units, color_maps, constants, plotting, utility

__all__ = [
    "DEFAULT_DATA_DIR",
    "Fuel",
    "Units",
    "color_maps",
    "constants",
    "correlate",
    "get_data_dir",
    "get_fueldata_dir",
    "get_fueldata_gc_dir",
    "get_fueldata_props_dir",
    "plotting",
    "utility",
]
