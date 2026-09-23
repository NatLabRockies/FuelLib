"""Physical constants used in FuelLib calculations."""

from .units import Units

# Physical constants
#: Boltzmann's constant in J/K.
k_B = Units.Quantity(1.380649e-23, "J/K")

#: Avogadro's number in 1/mol.
N_A = Units.Quantity(6.02214076e23, "1/mol")

#: Standard temperature
T_stp = Units.Quantity(298.15, "K")

#: Lennard-Jones default parameters for ambient gas.
Sigma_gas = Units.Quantity(3.62, "angstrom")
EpsilonByKB_gas = Units.Quantity(97.0, "K")
MW_gas = Units.Quantity(28.97e-3, "kg/mol")

__all__ = ["N_A", "EpsilonByKB_gas", "MW_gas", "Sigma_gas", "T_stp", "k_B"]
