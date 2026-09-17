"""Physical constants used in FuelLib calculations."""

from .units import PintUnits

# Physical constants
#: Boltzmann's constant in J/K.
k_B = 1.380649e-23

#: Avogadro's number in 1/mol.
N_A = 6.02214076e23

#: Standard Temperature and Pressure
T_STP = PintUnits.Quantity(298.15, "K")

#: Lennard-Jones default parameters for ambient gas.
Sigma_g = PintUnits.Quantity(3.62, "angstrom")
EpsilonByKB_g = PintUnits.Quantity(97.0, "K")
MW_g = PintUnits.Quantity(28.97e-3, "kg/mol")

__all__ = ["N_A", "T_STP", "EpsilonByKB_g", "MW_g", "Sigma_g", "k_B"]
