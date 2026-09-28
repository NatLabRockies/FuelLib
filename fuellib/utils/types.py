"""Type definitions for FuelLib."""

from __future__ import annotations
from typing import ClassVar, TypeVar, TypeAlias

import numpy as np
import pint

# NOTE: This can be expanded out to provide the same type alias for different types of
# objects based on the availability of different numerical libraries (i.e., JAX, NumPy,
# Pint, Unxt, ...). Moving Units here allows for the Units registry to be dynamically
# set based on the available numerical library.

# NumPy
_ = np.array([1])  # Check that NumPy is available
Array1D: TypeAlias = np.ndarray[tuple[int,]]
"""One-dimensional NumPy array."""
Array2D: TypeAlias = np.ndarray[tuple[int, int]]
"""Two-dimensional NumPy array."""

# Pint
ureg = pint.UnitRegistry()
"""Pint unit registry and quantity type definitions."""

## NOTE: TypeVar provides a way to define generic types that can be used for type
## hinting. This will be important when we begin to implement optional dependencies, as
## it allows us to define types that can adapt to the available numerical library.
PintQuantityT = TypeVar("PintQuantityT", bound=pint.Quantity)
"""Pint quantity type variable."""
Quantity0D: TypeAlias = pint.Quantity[float]
"""Unit-aware scalar quantity with a float/zero-dimensional array magnitude."""
Quantity1D: TypeAlias = pint.Quantity[Array1D]
"""Unit-aware quantity with a one-dimensional array magnitude."""
Quantity2D: TypeAlias = pint.Quantity[Array2D]
"""Unit-aware quantity with a two-dimensional array magnitude."""

UnxtQuantityT = TypeVar("UnxtQuantityT", bound=object)
"""Placeholder for Unxt quantity type variable."""


QuantityT = PintQuantityT if "pint" in globals() else UnxtQuantityT
"""Resolved quantity type variable based on the available numerical library."""


class Units:
    """Wrapper class for the resolved Units registry.

    This class provides a unified interface for accessing the quantity type
    and unit registry based on the available numerical library.

    Currently a placeholder but will be expanded to include the Unxt registry.
    """

    Quantity: ClassVar[type[QuantityT]] = ureg.Quantity
    """Resolved quantity type based on the available numerical library."""
    Q: ClassVar[type[QuantityT]] = ureg.Quantity  # Alias for Quantity
    """Alias for the resolved quantity type based on the available numerical library."""


__all__ = [
    "Array1D",
    "Array2D",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
]
