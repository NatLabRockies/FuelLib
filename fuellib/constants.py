"""Physical constants used in FuelLib calculations."""

from .units import PintUnits

# Physical constants
#: Boltzmann's constant in J/K.
k_B = PintUnits.Quantity(1.380649e-23, "J/K")

#: Avogadro's number in 1/mol.
N_A = PintUnits.Quantity(6.02214076e23, "1/mol")

#: Standard temperature 
T_stp = PintUnits.Quantity(298.15, "K")

#: Lennard-Jones default parameters for ambient gas.
Sigma_gas = PintUnits.Quantity(3.62, "angstrom")
EpsilonByKB_gas = PintUnits.Quantity(97.0, "K")
MW_gas = PintUnits.Quantity(28.97e-3, "kg/mol")

__all__ = ["N_A", "k_B", "T_stp", "Sigma_gas", "EpsilonByKB_gas", "MW_gas"]
