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
#: One-dimensional NumPy array.
Array1D: TypeAlias = np.ndarray[tuple[int,]]
#: Two-dimensional NumPy array.
Array2D: TypeAlias = np.ndarray[tuple[int, int]]

# Pint
ureg = pint.UnitRegistry()
## NOTE: TypeVar provides a way to define generic types that can be used for type
## hinting. This will be important when we begin to implement optional dependencies, as
## it allows us to define types that can adapt to the available numerical library.
PintQuantityT = TypeVar("PintQuantityT", bound=pint.Quantity)
#: Unit-aware scalar quantity.
Quantity0D: TypeAlias = pint.Quantity[float]
#: Unit-aware quantity with a one-dimensional array magnitude.
Quantity1D: TypeAlias = pint.Quantity[Array1D]
#: Unit-aware quantity with a two-dimensional array magnitude.
Quantity2D: TypeAlias = pint.Quantity[Array2D]

UnxtQuantityT = TypeVar(
    "UnxtQuantityT", bound=object
)  # Placeholder for Unxt quantity type


QuantityT = (
    PintQuantityT if "pint" in globals() else UnxtQuantityT
)  # Placeholder for type selection


class Units:
    """Wrapper class for the resolved Units registry.

    Currently a placeholder but will be expanded to include the Unxt registry
    """

    Quantity: ClassVar[type[QuantityT]] = ureg.Quantity
    Q: ClassVar[type[QuantityT]] = ureg.Quantity  # Alias for Quantity


__all__ = [
    "Array1D",
    "Array2D",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
]
