"""Unit conversion functions."""

import numpy as np
from .constants import k_B, N_A


def C2K(T):
    """
    Convert temperature from Celsius to Kelvin.

    :param T: Temperature in Celsius.
    :type T: float or np.ndarray
    :return: Temperature in Kelvin.
    :rtype: float or np.ndarray
    """
    return T + 273.15


def K2C(T):
    """
    Convert temperature from Kelvin to Celsius.

    :param T: Temperature in Kelvin.
    :type T: float or np.ndarray
    :return: Temperature in Celsius.
    :rtype: float or np.ndarray
    """
    return T - 273.15


def C2F(T):
    """
    Convert temperature from Celsius to Fahrenheit.

    :param T: Temperature in Celsius.
    :type T: float or np.ndarray
    :return: Temperature in Fahrenheit.
    :rtype: float or np.ndarray
    """
    return T * 9 / 5 + 32


def F2C(T):
    """
    Convert temperature from Fahrenheit to Celsius.

    :param T: Temperature in Fahrenheit.
    :type T: float or np.ndarray
    :return: Temperature in Celsius.
    :rtype: float or np.ndarray
    """
    return (T - 32) * 5 / 9


def F2K(T):
    """
    Convert temperature from Fahrenheit to Kelvin.

    :param T: Temperature in Fahrenheit.
    :type T: float or np.ndarray
    :return: Temperature in Kelvin.
    :rtype: float or np.ndarray
    """
    return C2K(F2C(T))


def K2F(T):
    """
    Convert temperature from Kelvin to Fahrenheit.

    :param T: Temperature in Kelvin.
    :type T: float or np.ndarray
    :return: Temperature in Fahrenheit.
    :rtype: float or np.ndarray
    """
    return C2F(K2C(T))


def epsilon_to_characteristic_temperature(epsilon_j_per_mol):
    """
    Convert Lennard-Jones epsilon from J/mol to characteristic temperature in Kelvin.

    The characteristic temperature (epsilon/k_B) is used in transport property
    correlations and is required by combustion codes like CHEMKIN.

    Uses the relation: T* = (epsilon_J/mol) / (N_A * k_B)

    :param epsilon_j_per_mol: Lennard-Jones well depth epsilon in J/mol.
    :type epsilon_j_per_mol: float
    :return: Characteristic temperature (epsilon/k_B) in Kelvin.
    :rtype: float
    """
    epsilon_per_molecule = epsilon_j_per_mol / N_A
    lj_welldepth_K = epsilon_per_molecule / k_B
    return lj_welldepth_K


__all__ = [
    "C2K",
    "K2C",
    "C2F",
    "F2C",
    "F2K",
    "K2F",
    "epsilon_to_characteristic_temperature",
]
