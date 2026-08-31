"""Physical constants used in FuelLib calculations."""

from .units import Q_

# Physical constants
#: Boltzmann's constant.
k_B = Q_(1.380649e-23, "J/K")

#: Avogadro's number.
N_A = Q_(6.02214076e23, "1/mol")

__all__ = ["N_A", "k_B"]
