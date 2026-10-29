"""Type definitions for FuelLib."""

from __future__ import annotations

from dataclasses import dataclass
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


# Result classes
@dataclass(frozen=True)
class DistillationResult:
    """Result of a distillation calculation."""

    T0: Quantity0D
    T5: Quantity0D
    T10: Quantity0D
    T30: Quantity0D
    T50: Quantity0D
    T70: Quantity0D
    T90: Quantity0D
    Tf: Quantity0D


__all__ = [
    "Array1D",
    "Array2D",
    "DistillationResult",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
]
