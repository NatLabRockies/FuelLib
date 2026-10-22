"""Type definitions for FuelLib."""

from __future__ import annotations

from dataclasses import field
from typing import TypeAlias

import numpy as np
import pint

# NumPy
Array1D: TypeAlias = np.ndarray[tuple[int,]]
"""One-dimensional NumPy array."""
Array2D: TypeAlias = np.ndarray[tuple[int, int]]
"""Two-dimensional NumPy array."""

# Pint
Units = pint.UnitRegistry()
"""Pint unit registry and quantity type definitions."""
Quantity0D: TypeAlias = pint.Quantity[float]
"""Unit-aware scalar quantity with a float/zero-dimensional array magnitude."""
Quantity1D: TypeAlias = pint.Quantity[Array1D]
"""Unit-aware quantity with a one-dimensional array magnitude."""
Quantity2D: TypeAlias = pint.Quantity[Array2D]
"""Unit-aware quantity with a two-dimensional array magnitude."""


# Dataclasses
def quantity_field(value: float, units: str) -> Quantity0D:
    """Dataclass field with a default quantity.

    Args:
        value: Default magnitude.
        units: Units of the default value.

    Returns:
        Dataclass field whose default is a new quantity.
    """
    return field(default_factory=lambda: Units.Quantity(value, units))


__all__ = [
    "Array1D",
    "Array2D",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
    "quantity_field",
]
