"""Mixture correlation functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import curve_fit

from .. import constants
from ..utils import Units, types
from . import components, helpers

if TYPE_CHECKING:
    from ..fuel import Fuel


def mean_molecular_weight(
    fuel: "Fuel", Yi: types.Quantity1D | None = None
) -> types.Quantity0D:
    """Calculate the mean molecular weight of the mixture.

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound.
            Defaults to `fuel.Y_0` (initial mass fractions).

    Returns:
        Mean molecular weight of the mixture in kg/mol.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    MW = fuel.MW.to("kg/mol")  # Can't be None after `rdkit` implementation
    return 1 / np.sum(Yi / MW)


def density(
    fuel: "Fuel",
    T: types.Quantity0D,
    Yi: types.Quantity1D | None = None,
) -> types.Quantity1D:
    """Calculate mixture density.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        Yi: Mass fractions of each compound.
            Defaults to `fuel.Y_0` (initial mass fractions).

    Returns:
        Mixture density in kg/m^3.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    rho_i = components.density(fuel, T).to("kg/m^3")
    return Yi @ rho_i


def kinematic_viscosity_dutt(
    fuel: "Fuel",
    T: types.Quantity0D,
    Yi: types.Quantity1D | None = None,
    *,
    correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> types.Quantity0D:
    """Calculate kinematic viscosity of the mixture.

    Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        Yi: Mass fractions of each compound.
            Defaults to `fuel.Y_0` (initial mass fractions).
        correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
            Defaults to "Kendall-Monroe".

    Returns:
        Mixture kinematic viscosity in m^2/s.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    nu_i = components.kinematic_viscosity_dutt(fuel, T).to("m^2/s").magnitude
    # Calculate mole fractions for each species
    Xi = helpers.mass_fractions_to_mole_fractions(fuel, Yi).magnitude
    if correlation.casefold() == "Arrhenius".casefold():
        # Arrhenius mixing correlation
        nu = np.exp(np.sum(Xi * np.log(nu_i)))
    else:
        # Default: Kendall-Monroe mixing correlation
        nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** 3.0

    return Units.Quantity(nu, "m^2/s")


def dynamic_viscosity_dutt(
    fuel: "Fuel",
    T: types.Quantity0D,
    Yi: types.Quantity1D | None = None,
    *,
    correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> types.Quantity0D:
    """Calculate dynamic viscosity of the mixture.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        Yi: Mass fractions of each compound.
            Defaults to `fuel.Y_0` (initial mass fractions).
        correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
            Defaults to "Kendall-Monroe".

    Returns:
        Mixture dynamic viscosity in Pa*s.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    nu = kinematic_viscosity_dutt(fuel, T, Yi, correlation=correlation)
    rho = density(fuel, T, Yi)
    return (rho * nu).to("Pa*s")


def saturated_vapor_pressure(
    fuel: "Fuel",
    T: types.Quantity0D,
    Yi: types.Quantity1D | None = None,
    *,
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> types.Quantity0D:
    """Calculate saturated vapor pressure of the mixture.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
            Defaults to "Lee-Kesler".

    Returns:
        Mixture saturated vapor pressure in Pa.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    # Mole fraction for each compound
    Xi = helpers.mass_fractions_to_mole_fractions(fuel, Yi)
    # Saturated vapor pressure for each compound (Pa)
    p_sati = components.saturated_vapor_pressure(fuel, T, correlation=correlation).to(
        "Pa"
    )
    # Mixture vapor pressure via Raoult's law
    return p_sati @ Xi


def saturated_vapor_pressure_antoine_coeffs(
    fuel: "Fuel",
    Tvals: types.Quantity1D | None = None,
    Yi: types.Quantity1D | None = None,
    *,
    units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> tuple[float, float, float, float]:
    """Estimate Antoine coefficients for saturated vapor pressure of the mixture.

    Args:
        fuel: Fuel object.
        Tvals: Temperature range or nodes for Antoine fit in Kelvin.
            Defaults to [273.15, min(Tb_mix)].
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).
        units: Units for pressure in fit.
            Defaults to "mks" (Pa).
        correlation: Correlation method.
            Defaults to "Lee-Kesler".

    Returns:
        Coefficients A, B, C, D.

    Raises:
        ValueError: If units or Tvals are invalid.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    if units == "cgs":
        units = "dyne/cm^2"
    elif units == "mks":
        units = "Pa"
    elif units not in ["dyne/cm^2", "Pa"]:
        raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

    if Tvals is not None:
        Tvals = Tvals.to("K")

    # Define or get temperature nodes for fit
    if Tvals is None:
        print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
        Xi = helpers.mass_fractions_to_mole_fractions(fuel, Yi)
        Tb = helpers.arithmetic_mixing_rule(Xi, fuel.Tb)
        T = Units.Quantity(np.linspace(273.15, np.min(Tb.to("K").magnitude), 20), "K")
    elif len(Tvals) == 2:
        T = Units.Quantity(np.linspace(Tvals[0].magnitude, Tvals[1].magnitude, 20), "K")
    elif len(Tvals) > 2:
        T = Tvals
    else:
        raise ValueError("Tvals must be None, length 2, or length > 2.")

    # Antoine equation log10(p) = A - B/(C + T)
    def antoine_eq(
        T: float | types.Array1D, A: float, B: float, C: float
    ) -> float | types.Array1D:
        """Antoine equation for vapor pressure.

        Args:
            T: Temperature.
            A: Antoine coefficient A.
            B: Antoine coefficient B.
            C: Antoine coefficient C.

        Returns:
            log10(pressure).
        """
        return A - B / (T + C)

    # Fit A, B, C against pressure in Pa (mks base) so the coefficients are
    # unit independent. "mks" (meter-kilogram-second) and "cgs"
    # D is the Pa-to-target-unit conversion factor, applied only when evaluating
    # psat(T) = D * 10**(A - B/(T + C)) in Pele.
    D = Units.Quantity(1, "Pa").to(units)

    T_magnitude = T.to("K").magnitude
    Pvals = np.zeros_like(T_magnitude)
    for k in range(len(T)):
        Pvals[k] = (
            saturated_vapor_pressure(fuel, T[k], Yi, correlation=correlation)
            .to(units)
            .magnitude
        )

    logP = np.log10(Pvals)
    popt, _ = curve_fit(antoine_eq, T_magnitude, logP, p0=[1, 1e3, -1])
    A, B, C = popt

    return A, B, C, D.magnitude


def surface_tension(
    fuel: "Fuel",
    T: types.Quantity0D,
    Yi: types.Quantity1D | None = None,
    *,
    correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
) -> types.Quantity0D:
    """Calculate surface tension of the mixture.

    Uses arithmetic pseudo-property method recommended by Hugill and van
    Welsenes (1986).

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).
        correlation: Correlation method ("Pitzer" or "Brock-Bird").
            Defaults to "Brock-Bird".

    Returns:
        Mixture surface tension in N/m.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    Xi = helpers.mass_fractions_to_mole_fractions(fuel, Yi)
    # Surface tension for each compound (N/m)
    sti = components.surface_tension(fuel, T, correlation=correlation)
    # Mixture surface tension via arithmetic mean, Poling (12-5.2)
    st = helpers.arithmetic_mixing_rule(Xi, sti)
    return st.to("N/m")


def thermal_conductivity_latini(
    fuel: "Fuel",
    T: types.Quantity0D,
    Yi: types.Quantity1D | None = None,
) -> types.Quantity0D:
    """Calculate thermal conductivity of the mixture.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).

    Returns:
        Thermal conductivity in W/m/K.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    tci = components.thermal_conductivity_latini(fuel, T).to("W/(m*K)").magnitude
    tc = np.sum(Yi.magnitude * tci ** (-2)) ** (-0.5)
    return Units.Quantity(tc, "W/(m*K)")
