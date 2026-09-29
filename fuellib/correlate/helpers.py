"""Helper functions for correlations."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils import Units, types

if TYPE_CHECKING:
    from ..fuel import Fuel


def arithmetic_mixing_rule(
    X: types.Quantity1D,
    var_n: types.Quantity1D,
) -> types.Quantity0D:
    """Arithmetic mixing rule for computing mixture properties.

    Args:
        X: Mole fractions of the compounds.
        var_n: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_n.magnitude
    mag_X = X.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use arithmetic definition for the pseudo property
            var_ij = (values[i] + values[j]) / 2
            var_mix += mag_X[i] * mag_X[j] * var_ij

    return Units.Quantity(var_mix, var_n.units)  # Attach original units from var_n


def geometric_mixing_rule(
    X: types.Quantity1D,
    var_n: types.Quantity1D,
) -> types.Quantity0D:
    """Geometric mixing rule for computing mixture properties.

    Args:
        X: Mole fractions of the compounds.
        var_n: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_n.magnitude
    mag_X = X.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use geometric definition for the pseudo property
            var_ij = (values[i] * values[j]) ** 0.5
            var_mix += mag_X[i] * mag_X[j] * var_ij

    return Units.Quantity(var_mix, var_n.units)  # Attach original units from var_n


def mass_to_mass_fractions(fuel: "Fuel", mass: types.Quantity1D) -> types.Quantity1D:
    """Convert mass of each component to mass fractions (Yi).

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
        Yi = (mass / total_mass).magnitude
    else:
        Yi = np.zeros_like(fuel.MW.magnitude)

    return Units.Quantity(Yi, "dimensionless")


def mass_to_mole_fractions(fuel: "Fuel", mass: types.Quantity1D) -> types.Quantity1D:
    """Convert mass of each component to mole fractions (Xi).

    Args:
        fuel: Fuel object.
        mass: Mass of each compound in the mixture.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    mass = mass.to("kg")

    # Calculate the number of moles for each compound
    num_mole = mass / fuel.MW

    # Normalize to get group mole fractions
    total_moles = np.sum(num_mole)
    if total_moles != 0:
        Xi = (num_mole / total_moles).magnitude
    else:
        Xi = np.zeros_like(fuel.MW.magnitude)

    return Units.Quantity(Xi, "dimensionless")


def mole_fractions_to_mass_fractions(
    fuel: "Fuel", Xi: types.Quantity1D
) -> types.Quantity1D:
    """Convert mole fractions (Xi) to mass fractions (Yi).

    Args:
        fuel: Fuel object.
        Xi: Mole fractions of each compound in the mixture.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    # Calculate the mass for each compound
    mass = fuel.MW * Xi
    # Normalize to get group mass fractions
    total_mass = np.sum(mass)
    Yi: types.Array1D = (
        (mass / total_mass).magnitude
        if total_mass != 0
        else np.zeros_like(fuel.MW.magnitude)
    )

    return Units.Quantity(Yi, "dimensionless")


def mass_fractions_to_mole_fractions(
    fuel: "Fuel", Yi: types.Quantity1D
) -> types.Quantity1D:
    """Convert mass fractions (Yi) to mole fractions (Xi).

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the mixture.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    Mbar = fuel.mean_molecular_weight(Yi)
    if np.sum(Yi) != 0:
        Xi = (Mbar * Yi / fuel.MW).magnitude
    else:
        Xi = np.zeros_like(fuel.MW.magnitude)

    return Units.Quantity(Xi, "dimensionless")


__all__ = [
    "arithmetic_mixing_rule",
    "geometric_mixing_rule",
    "mass_fractions_to_mole_fractions",
    "mass_to_mass_fractions",
    "mass_to_mole_fractions",
    "mole_fractions_to_mass_fractions",
]
