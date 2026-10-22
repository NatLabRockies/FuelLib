"""Utility functions for mixture calculations and droplet properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .. import correlate
from . import types
from .constants import N_A, k_B
from .types import Units

if TYPE_CHECKING:
    from ..fuel import Fuel


def mixing_rule(
    var_n: types.Quantity1D,
    X: types.Quantity1D,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> types.Quantity0D:
    """Mixing rules for computing mixture properties.

    Args:
        var_n: Individual compound properties.
        X: Mole fractions of the compounds.
        pseudo_prop: Type of mean ("arithmetic" or "geometric").

    Returns:
        Mixture property value.
    """
    if pseudo_prop.casefold() == "geometric":
        return correlate.mixture.geometric(X, var_n)
    return correlate.mixture.arithmetic(X, var_n)


def droplet_volume(r: types.Quantity0D) -> types.Quantity0D:
    """Calculate the spherical volume of a droplet.

    Args:
        r: Radius of the droplet in meters.

    Returns:
        Spherical volume of the droplet in cubic meters.
    """
    radius = r.to("m")
    return 4.0 / 3.0 * np.pi * radius**3


def droplet_mass(
    fuel: Fuel, r: types.Quantity0D, Yi: types.Quantity1D, T: types.Quantity0D
) -> types.Quantity1D:
    """Calculate the mass of each compound in the fuel provided droplet radius.

    Args:
        fuel: An instance of the fuel class.
        r: Radius of the droplet.
        Yi: Mass fractions of each compound.
        T: Droplet temperature in Kelvin.

    Returns:
        Mass of each compound in the droplet in kg.
    """
    Yi_magnitude = Yi.to("dimensionless").magnitude
    volume = droplet_volume(r)
    if volume > 0:
        return (
            volume / (fuel.molar_liquid_vol(T) @ Yi_magnitude) * Yi_magnitude * fuel.MW
        ).to("kg")
    else:
        return Units.Quantity(np.zeros_like(fuel.MW.magnitude), "kg")


def epsilon_to_characteristic_temperature(epsilon_j_per_mol: float) -> types.Quantity0D:
    """Convert Lennard-Jones epsilon from J/mol to characteristic temperature in Kelvin.

    The characteristic temperature (epsilon/k_B) is used in transport property
    correlations and is required by combustion codes like CHEMKIN.

    Uses the relation: T* = (epsilon_J/mol) / (N_A * k_B)

    Args:
        epsilon_j_per_mol: Lennard-Jones well depth epsilon in J/mol.

    Returns:
        Characteristic temperature (epsilon/k_B) in Kelvin.
    """
    eps = Units.Quantity(epsilon_j_per_mol, "J/mol")
    return (eps / (N_A * k_B)).to("K")


__all__ = [
    "droplet_mass",
    "droplet_volume",
    "epsilon_to_characteristic_temperature",
    "mixing_rule",
]
