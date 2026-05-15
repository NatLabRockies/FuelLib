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

# Import temperature conversions and characteristic temperature
from .conversions import (
    C2K,
    K2C,
    C2F,
    F2C,
    F2K,
    K2F,
    epsilon_to_characteristic_temperature,
)

# Import utility functions
from .utilities import mixing_rule, droplet_volume, droplet_mass

# Import fuel class
from .fuel import fuel

# Import data locator functions
from ._data_locator import (
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_decomp_dir,
    get_fueldata_props_dir,
    get_decomp_name_from_metadata,
    get_props_data_from_metadata,
)

__all__ = [
    "fuel",
    "k_B",
    "N_A",
    "C2K",
    "K2C",
    "C2F",
    "F2C",
    "F2K",
    "K2F",
    "epsilon_to_characteristic_temperature",
    "mixing_rule",
    "droplet_volume",
    "droplet_mass",
    "get_fueldata_dir",
    "get_fueldata_gc_dir",
    "get_fueldata_decomp_dir",
    "get_fueldata_props_dir",
    "get_decomp_name_from_metadata",
    "get_props_data_from_metadata",
]
