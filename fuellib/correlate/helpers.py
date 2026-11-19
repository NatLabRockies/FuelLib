"""Helper functions for correlations."""

from typing import TYPE_CHECKING

import numpy as np

from ..utils import Units, types

if TYPE_CHECKING:
    from ..fuel import Fuel


def mass_to_mass_fractions(fuel: "Fuel", mass: types.Quantity1D) -> types.Quantity1D:
    """Convert mass of each component to mass fractions (yi).

    Args:
        fuel: Fuel object.
        mass: Mass of each compound.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    # Normalize to get group mole fractions
    mass = mass.to("kg")
    total_mass = mass.magnitude.sum()
    if total_mass != 0:
        yi = (mass / total_mass).magnitude
    else:
        yi = np.zeros_like(fuel.comps.mw.magnitude)

    return Units.Quantity(yi, "dimensionless")


def mass_to_mole_fractions(fuel: "Fuel", mass: types.Quantity1D) -> types.Quantity1D:
    """Convert mass of each component to mole fractions (xi).

    Args:
        fuel: Fuel object.
        mass: Mass of each compound in the mixture.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    mass = mass.to("kg")

    # Calculate the number of moles for each compound
    num_mole = mass / fuel.comps.mw

    # Normalize to get group mole fractions
    total_moles = np.sum(num_mole)
    if total_moles != 0:
        xi = (num_mole / total_moles).magnitude
    else:
        xi = np.zeros_like(fuel.comps.mw.magnitude)

    return Units.Quantity(xi, "dimensionless")


def mole_fractions_to_mass_fractions(
    fuel: "Fuel", xi: types.Quantity1D
) -> types.Quantity1D:
    """Convert mole fractions (xi) to mass fractions (yi).

    Args:
        fuel: Fuel object.
        xi: Mole fractions of each compound in the mixture.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    # Calculate the mass for each compound
    mass = fuel.comps.mw * xi
    # Normalize to get group mass fractions
    total_mass = np.sum(mass)
    yi: types.Array1D = (
        (mass / total_mass).magnitude
        if total_mass != 0
        else np.zeros_like(fuel.comps.mw.magnitude)
    )

    return Units.Quantity(yi, "dimensionless")


def mass_fractions_to_mole_fractions(
    fuel: "Fuel", yi: types.Quantity1D
) -> types.Quantity1D:
    """Convert mass fractions (yi) to mole fractions (xi).

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    mw_bar = fuel.mix.mw(yi)
    if np.sum(yi) != 0:
        xi = (mw_bar * yi / fuel.comps.mw).magnitude
    else:
        xi = np.zeros_like(fuel.comps.mw.magnitude)

    return Units.Quantity(xi, "dimensionless")


def mass_fractions_to_volume_fractions(
    fuel: "Fuel", yi: types.Quantity1D, temp: types.Quantity0D
) -> types.Quantity1D:
    """Convert mass fractions (yi) to volume fractions (vi).

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
        temp: Temperature at which to evaluate the mixture density.

    Returns:
        Volume fractions of the compounds (shape: num_compounds,).
    """
    density = fuel.comps.rho(temp)
    if np.sum(yi) != 0:
        v = yi / density
        vi = (v / np.sum(v)).magnitude
    else:
        vi = np.zeros_like(fuel.comps.mw.magnitude)

    return Units.Quantity(vi, "dimensionless")


__all__ = [
    "mass_fractions_to_mole_fractions",
    "mass_fractions_to_volume_fractions",
    "mass_to_mass_fractions",
    "mass_to_mole_fractions",
    "mole_fractions_to_mass_fractions",
]
