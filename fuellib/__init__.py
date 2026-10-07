"""FuelLib: Fuel Library for Group Contribution Method calculations.

FuelLib utilizes the Group Contribution Method (GCM) as proposed by Constantinou
and Gani (1994, 1995) to calculate thermodynamic and mixture properties of fuels.

See :class:`Fuel` for the main class and complete API documentation.
"""

try:
    from importlib.metadata import version

    __version__ = version("fuellib")
except ImportError:
    __version__ = "unknown"

# Import fuel class
# Import submodules for namespacing
from . import correlate, data
from .data import Property
from .fuel import Fuel
from .utils import Units, constants, convert, set_log_level, utility

__all__ = [
    "FLLogger",
    "Fuel",
    "Property",
    "Units",
    "constants",
    "convert",
    "correlate",
    "data",
    "set_log_level",
    "utility",
]
