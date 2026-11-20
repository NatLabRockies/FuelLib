"""Utility functions for mixture calculations and droplet properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .. import correlate
from . import types
from .constants import K_B, N_A
from .types import Units

if TYPE_CHECKING:
    from ..fuel import Fuel


def mixing_rule(
    var_i: types.Quantity1D,
    x: types.Quantity1D,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> types.Quantity0D:
    """Mixing rules for computing mixture properties.

    Args:
        var_i: Individual compound properties.
        x: Mole fractions of the compounds.
        pseudo_prop: Type of mean ("arithmetic" or "geometric").

    Returns:
        Mixture property value.
    """
    if pseudo_prop.casefold() == "geometric":
        return correlate.mixture.geometric(x, var_i)
    return correlate.mixture.arithmetic(x, var_i)


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
    fuel: Fuel, r: types.Quantity0D, yi: types.Quantity1D, temp: types.Quantity0D
) -> types.Quantity1D:
    """Calculate the mass of each compound in the fuel provided droplet radius.

    Args:
        fuel: An instance of the fuel class.
        r: Radius of the droplet.
        yi: Mass fractions of each compound.
        temp: Droplet temperature.

    Returns:
        Mass of each compound in the droplet in kg.
    """
    yi_mag = yi.to("dimensionless").magnitude
    volume = droplet_volume(r)
    if volume > 0:
        return (volume / (fuel.comps.vm(temp) @ yi_mag) * yi_mag * fuel.comps.mw).to(
            "kg"
        )
    else:
        return Units.Quantity(np.zeros_like(fuel.comps.mw.magnitude), "kg")


def epsilon_to_characteristic_temperature(epsilon: float) -> types.Quantity0D:
    """Convert Lennard-Jones epsilon from J/mol to characteristic temperature in Kelvin.

    The characteristic temperature (epsilon/K_B) is used in transport property
    correlations and is required by combustion codes like CHEMKIN.

    Uses the relation: T* = epsilon / (N_A * K_B)

    Args:
        epsilon: Lennard-Jones well depth epsilon in J/mol.

    Returns:
        Characteristic temperature (epsilon/K_B) in Kelvin.
    """
    eps = Units.Quantity(epsilon, "J/mol")
    return (eps / (N_A * K_B)).to("K")


__all__ = [
    "droplet_mass",
    "droplet_volume",
    "epsilon_to_characteristic_temperature",
    "mixing_rule",
]
