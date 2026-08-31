"""Unit conversion functions."""

from .constants import N_A, k_B


def C2K(T):
    """
    Convert temperature from Celsius to Kelvin.

    :param T: Temperature (pint Quantity with temperature units).
    :type T: pint.Quantity
    :return: Temperature in Kelvin.
    :rtype: pint.Quantity
    """
    return T.to("kelvin")


def K2C(T):
    """
    Convert temperature from Kelvin to Celsius.

    :param T: Temperature (pint Quantity with temperature units).
    :type T: pint.Quantity
    :return: Temperature in Celsius.
    :rtype: pint.Quantity
    """
    return T.to("degC")


def C2F(T):
    """
    Convert temperature from Celsius to Fahrenheit.

    :param T: Temperature (pint Quantity with temperature units).
    :type T: pint.Quantity
    :return: Temperature in Fahrenheit.
    :rtype: pint.Quantity
    """
    return T.to("degF")


def F2C(T):
    """
    Convert temperature from Fahrenheit to Celsius.

    :param T: Temperature (pint Quantity with temperature units).
    :type T: pint.Quantity
    :return: Temperature in Celsius.
    :rtype: pint.Quantity
    """
    return T.to("degC")


def F2K(T):
    """
    Convert temperature from Fahrenheit to Kelvin.

    :param T: Temperature (pint Quantity with temperature units).
    :type T: pint.Quantity
    :return: Temperature in Kelvin.
    :rtype: pint.Quantity
    """
    return T.to("kelvin")


def K2F(T):
    """
    Convert temperature from Kelvin to Fahrenheit.

    :param T: Temperature (pint Quantity with temperature units).
    :type T: pint.Quantity
    :return: Temperature in Fahrenheit.
    :rtype: pint.Quantity
    """
    return T.to("degF")


def epsilon_to_characteristic_temperature(epsilon_j_per_mol):
    """
    Convert Lennard-Jones epsilon to characteristic temperature.

    The characteristic temperature (epsilon/k_B) is used in transport property
    correlations and is required by combustion codes like CHEMKIN.

    Uses the relation: T* = (epsilon_J/mol) / (N_A * k_B)

    :param epsilon_j_per_mol: Lennard-Jones well depth epsilon (energy/substance).
    :type epsilon_j_per_mol: pint.Quantity
    :return: Characteristic temperature (epsilon/k_B) in Kelvin.
    :rtype: pint.Quantity
    """
    epsilon_per_molecule = epsilon_j_per_mol / N_A
    lj_welldepth_K = (epsilon_per_molecule / k_B).to("kelvin")
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
