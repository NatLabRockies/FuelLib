"""Command Line Unit conversion functions."""

from typing import overload

from .constants import N_A, k_B
from .utils import types


@overload
def C2K(T: float) -> float: ...
@overload
def C2K(T: types.Array1D) -> types.Array1D: ...
def C2K(T: float | types.Array1D) -> float | types.Array1D:
    """Convert temperature from Celsius to Kelvin.

    Args:
        T: Temperature in Celsius.

    Returns:
        Temperature in Kelvin.
    """
    return T + 273.15


@overload
def K2C(T: float) -> float: ...
@overload
def K2C(T: types.Array1D) -> types.Array1D: ...
def K2C(T: float | types.Array1D) -> float | types.Array1D:
    """Convert temperature from Kelvin to Celsius.

    Args:
        T: Temperature in Kelvin.

    Returns:
        Temperature in Celsius.
    """
    return T - 273.15


@overload
def C2F(T: float) -> float: ...
@overload
def C2F(T: types.Array1D) -> types.Array1D: ...
def C2F(T: float | types.Array1D) -> float | types.Array1D:
    """Convert temperature from Celsius to Fahrenheit.

    Args:
        T: Temperature in Celsius.

    Returns:
        Temperature in Fahrenheit.
    """
    return T * 9 / 5 + 32


@overload
def F2C(T: float) -> float: ...
@overload
def F2C(T: types.Array1D) -> types.Array1D: ...
def F2C(T: float | types.Array1D) -> float | types.Array1D:
    """Convert temperature from Fahrenheit to Celsius.

    Args:
        T: Temperature in Fahrenheit.

    Returns:
        Temperature in Celsius.
    """
    return (T - 32) * 5 / 9


@overload
def F2K(T: float) -> float: ...
@overload
def F2K(T: types.Array1D) -> types.Array1D: ...
def F2K(T: float | types.Array1D) -> float | types.Array1D:
    """Convert temperature from Fahrenheit to Kelvin.

    Args:
        T: Temperature in Fahrenheit.

    Returns:
        Temperature in Kelvin.
    """
    return C2K(F2C(T))


@overload
def K2F(T: float) -> float: ...
@overload
def K2F(T: types.Array1D) -> types.Array1D: ...
def K2F(T: float | types.Array1D) -> float | types.Array1D:
    """Convert temperature from Kelvin to Fahrenheit.

    Args:
        T: Temperature in Kelvin.

    Returns:
        Temperature in Fahrenheit.
    """
    return C2F(K2C(T))


def epsilon_to_characteristic_temperature(epsilon_j_per_mol: float) -> float:
    """Convert Lennard-Jones epsilon from J/mol to characteristic temperature in Kelvin.

    The characteristic temperature (epsilon/k_B) is used in transport property
    correlations and is required by combustion codes like CHEMKIN.

    Uses the relation: T* = (epsilon_J/mol) / (N_A * k_B)

    Args:
        epsilon_j_per_mol: Lennard-Jones well depth epsilon in J/mol.

    Returns:
        Characteristic temperature (epsilon/k_B) in Kelvin.
    """
    epsilon_per_molecule = epsilon_j_per_mol / N_A.magnitude
    lj_welldepth_K = epsilon_per_molecule / k_B.magnitude
    return lj_welldepth_K


__all__ = [
    "C2F",
    "C2K",
    "F2C",
    "F2K",
    "K2C",
    "K2F",
    "epsilon_to_characteristic_temperature",
]
