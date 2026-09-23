"""Type definitions for FuelLib."""

import numpy as np
import pint

# Aliases for readability
NumCompounds = int

# NumPy
FloatArray = np.ndarray[tuple[NumCompounds,], np.dtype[np.float64]]

# Pint
PintScalar = pint.Quantity[float]
PintArray = pint.Quantity[FloatArray]
