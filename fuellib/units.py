"""Units handling and conversions."""

from typing import Literal

import astropy.units as u
import numpy as np

# astropy does not have some common unit strings; we need to define and register them
## Pressure (defined in terms of Pa)
atm = u.def_unit("atm", 101325 * u.Pa, doc="Standard atmosphere")
dyne_cm2 = u.def_unit("dyne/cm^2", 0.1 * u.Pa, doc="Dyne per square centimeter")
cgs = u.def_unit("cgs", 0.1 * u.Pa, doc="CGS unit of pressure")
mks = u.def_unit("mks", 1 * u.Pa, doc="MKS unit of pressure")
## Temperature
fahrenheit = u.def_unit(
    "Fahrenheit",
    1 * u.imperial.deg_F,
    doc="Fahrenheit temperature unit",
)
## Undefined
dimensionless = u.def_unit(
    "dimensionless", 1 * u.dimensionless_unscaled, doc="Dimensionless unit"
)

## Register the new units with astropy so they can be used in unxt.Quantity objects.
u.add_enabled_units([atm, mks, dyne_cm2, cgs, fahrenheit, dimensionless])


def convert_temperature(
    temp: u.Quantity, target_unit: Literal["K", "Kelvin", "Celsius", "Fahrenheit"]
) -> u.Quantity:
    """
    Convert a temperature quantity to a different unit.

    *NOTE*: Astropy does not automatically handle conversions between temperature units
    without enabling the temperature equivalencies.

    :param temp: Temperature quantity to convert.
    :type temp: ut.QuantityLike
    :param target_unit: Unit to convert to.
    :type target_unit: str
    :return: Converted temperature quantity.
    :rtype: u.Quantity
    """
    with u.add_enabled_equivalencies(u.temperature()):
        return temp.to(target_unit)


def ustrip(quant: u.Quantity) -> np.ndarray:
    """
    Strip the unit from an astropy Quantity, returning the raw value as a numpy array.

    :param quant: Quantity to strip the unit from.
    :type quant: u.Quantity
    :return: Raw value without units.
    :rtype: np.ndarray
    """
    return np.array(quant.value)


__all__ = ["convert_temperature", "ustrip"]
