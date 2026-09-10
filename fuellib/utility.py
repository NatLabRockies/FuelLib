"""Utility functions for mixture calculations and droplet properties."""

from collections.abc import Sequence
from typing import Literal

import astropy.units as u
import numpy as np
import numpy.typing as npt

type FloatArrayLike = npt.NDArray[np.float64] | Sequence[float]


def mixing_rule(
    var_n: u.Quantity,
    X: FloatArrayLike,
    *,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> u.Quantity:
    """
    Mixing rules for computing mixture properties.

    :param var_n: Individual compound properties.
    :type var_n: u.Quantity
    :param X: Mole fractions of the compounds.
    :type X: FloatArrayLike
    :param pseudo_prop: Type of mean ("arithmetic" or "geometric").
    :type pseudo_prop: str, optional
    :return: Mixture property value.
    :rtype: u.Quantity
    """
    num_comps = len(var_n)
    var_mix = u.Quantity(0.0, var_n.unit)
    for i in range(num_comps):
        for j in range(num_comps):
            if pseudo_prop.casefold() == "geometric":
                # Use geometric mean definition for the pseudo property
                var_ij = (var_n[i] * var_n[j]) ** (0.5)
            else:
                # Use arithmetic definition for the pseudo property
                var_ij = (var_n[i] + var_n[j]) / 2
            var_mix += X[i] * X[j] * var_ij
    return var_mix


def droplet_volume(r):
    """
    Calculate spherical volume of a droplet given the radius.

    :param r: Radius of the droplet in meters.
    :type r: float
    :return: Spherical volume of droplet in cubic meters.
    :rtype: float
    """
    return 4.0 / 3.0 * np.pi * r**3


def droplet_mass(fuel, r, Yi, T):
    """
    Calculate the mass of each compound in the fuel provided the radius of the droplet.

    :param fuel: An instance of the fuel class.
    :type fuel: fuel object
    :param r: Radius of the droplet in meters.
    :type r: float
    :param Yi: Mass fractions of each compound.
    :type Yi: np.ndarray
    :param T: Droplet temperature in Kelvin.
    :type T: float
    :return: Mass of each compound in droplet in kg.
    :rtype: np.ndarray
    """
    volume = droplet_volume(r)  # m^3
    if volume > 0:
        return volume / (fuel.molar_liquid_vol(T) @ Yi) * Yi * fuel.MW
    else:
        return np.zeros_like(fuel.MW)


__all__ = ["droplet_mass", "droplet_volume", "mixing_rule"]
