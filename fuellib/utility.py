"""Utility functions for mixture calculations and droplet properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pint

from .utils import types

if TYPE_CHECKING:
    from .fuel import Fuel


@overload
def mixing_rule(
    var_n: types.Array1D,
    X: types.Array1D,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> float: ...
@overload
def mixing_rule(
    var_n: types.Quantity1D,
    X: types.Array1D,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> types.Quantity0D: ...
def mixing_rule(
    var_n: types.Array1D | types.Quantity1D,
    X: types.Array1D,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> float | types.Quantity0D:
    """Mixing rules for computing mixture properties.

    Args:
        var_n: Individual compound properties.
        X: Mole fractions of the compounds.
        pseudo_prop: Type of mean ("arithmetic" or "geometric").

    Returns:
        Mixture property value.
    """
    if isinstance(var_n, pint.Quantity):
        units = var_n.units
        values = var_n.magnitude
    else:
        units = None
        values = np.asarray(var_n)
    mole_fractions = np.asarray(X)

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            if pseudo_prop.casefold() == "geometric":
                # Use geometric mean definition for the pseudo property
                var_ij = (values[i] * values[j]) ** 0.5
            else:
                # Use arithmetic definition for the pseudo property
                var_ij = (values[i] + values[j]) / 2
            var_mix += mole_fractions[i] * mole_fractions[j] * var_ij

    return cast("types.Quantity0D", var_mix * units) if units is not None else var_mix


def droplet_volume(r: float) -> float:
    """Calculate spherical volume of a droplet given the radius.

    Args:
        r: Radius of the droplet in meters.

    Returns:
        Spherical volume of droplet in cubic meters.
    """
    return 4.0 / 3.0 * np.pi * r**3


def droplet_mass(
    fuel: Fuel, r: float, Yi: types.Array1D, T: types.Quantity0D
) -> types.Array1D:
    """Calculate the mass of each compound in the fuel provided the radius of the droplet.

    Args:
        fuel: An instance of the fuel class.
        r: Radius of the droplet in meters.
        Yi: Mass fractions of each compound.
        T: Droplet temperature in Kelvin.

    Returns:
        Mass of each compound in droplet in kg.
    """
    volume = droplet_volume(r)  # m^3
    if volume > 0:
        return (volume / (fuel.molar_liquid_vol(T) @ Yi) * Yi * fuel.MW).magnitude
    else:
        return np.zeros_like(fuel.MW)


__all__ = ["droplet_mass", "droplet_volume", "mixing_rule"]
