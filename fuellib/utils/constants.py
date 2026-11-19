"""Physical constants used in FuelLib calculations."""

from .types import Units

# Physical constants
#: Boltzmann's constant in J/K.
K_B = Units.Quantity(1.380649e-23, "J/K")

#: Avogadro's number in 1/mol.
N_A = Units.Quantity(6.02214076e23, "1/mol")

#: Standard temperature
T_STP = Units.Quantity(298.15, "K")

#: Lennard-Jones default parameters for ambient gas.
SIGMA_GAS = Units.Quantity(3.62, "angstrom")
LJ_EPS_BY_KB_GAS = Units.Quantity(97.0, "K")
MW_GAS = Units.Quantity(28.97e-3, "kg/mol")

# R gas-constant
R = Units.Quantity(8.31446, "J/(mol*K)")

__all__ = [
    "K_B",
    "LJ_EPS_BY_KB_GAS",
    "MW_GAS",
    "N_A",
    "SIGMA_GAS",
    "T_STP",
    "R",
]
