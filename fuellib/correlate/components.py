"""Component correlation functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import curve_fit

from ..gcm.gani import gani_gcm
from ..gcm.nannoolal import nannoolal_gcm
from ..rdk import mol
from ..utils import Units, constants, types

if TYPE_CHECKING:
    from ..fuel import Fuel


def molar_liquid_volume(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Compute molar liquid volume with temperature correction.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Molar liquid volume in m^3/mol.
    """
    Tstp = Units.Quantity(298, "K")
    T = T.to("K")
    Tc = fuel.comps.tc.to("K")
    phi = np.zeros_like(Tc.magnitude)
    for i in range(len(Tc)):
        if T > Tc[i]:
            phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
        else:
            phi[i] = (1 - (T / Tc[i])) ** (2.0 / 7.0) - (1 - (Tstp / Tc[i])) ** (
                2.0 / 7.0
            )
    z = 0.29056 - 0.08775 * fuel.comps.omega
    return fuel.comps.vm_stp * z**phi


def density(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate the density of each component at temperature T.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Density of each compound in kg/m^3.
    """
    return (fuel.comps.mw / molar_liquid_volume(fuel, T)).to("kg/m^3")


def viscosity_kinematic_dutt(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate the kinematic viscosity using Dutt's equation.

    Uses Dutt's equation (4.23) from "Viscosity of Liquids". The equation
    predicts viscosity in mm^2/s and is converted to SI units.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Viscosity of each component in m^2/s.
    """
    # Convert temperature to Celsius
    T: float = T.to("celsius").magnitude
    Tb = fuel.comps.tb.to("celsius").magnitude
    # RHS of Dutt's equation (4.23) in Viscosity of Liquids
    rhs = -3.0171 + (442.78 + 1.6452 * Tb) / (T + 239 - 0.19 * Tb)
    return Units.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")


def viscosity_dynamic_dutt(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate the dynamic viscosity using Dutt's equation.

    Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Dynamic viscosity in Pa*s.
    """
    nu_i = viscosity_kinematic_dutt(fuel, T)
    rho_i = density(fuel, T)
    return (nu_i * rho_i).to("Pa*s")


def viscosity_dynamic_nannoolal(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate the dynamic viscosity using Nannoolal (2009) GCM.

    Uses equation (6) from Nannoolal (2009).

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Component dynamic viscosities in Pa*s.
    """
    T_mag = T.to("kelvin").magnitude
    Tb_mag = fuel.comps.tb.to("kelvin").magnitude
    dBvi = nannoolal_gcm.predict("dBv", fuel).magnitude
    Tvi = nannoolal_gcm.predict("Tv", fuel).magnitude
    ni = np.array([
        sum(n for atom, n in mol.atom_counts(m).items() if atom != "H")
        for m in fuel.comps.rdkit_mols
    ])
    dBvi = dBvi / (ni**-2.5635 + 0.0685) + 3.7777
    Tvi = 21.8444 * np.sqrt(Tb_mag) + Tvi**0.9315 / (ni**0.6577 + 4.9259) - 231.1361
    denom = T_mag - Tvi / 16.0
    eta = 1.3e-3 * np.exp(-dBvi * (T_mag - Tvi) / denom)
    return Units.Quantity(eta, "Pa*s")


def viscosity_kinematic_nannoolal(
    fuel: "Fuel", T: types.Quantity0D
) -> types.Quantity1D:
    """Calculate the kinematic viscosity using Nannoolal (2009) GCM.

        Divides equation (6) from Nannoolal (2009) by density.

    Args:
        fuel: Fuel object.
        T: Temperature to calculate the kinematic viscosities.

    Returns:
        Component kinematic viscosities in m^2/s.
    """
    eta = viscosity_dynamic_nannoolal(fuel, T)
    rho = density(fuel, T)
    return (eta / rho).to("m^2/s")


def specific_heat_capacity_molar(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Compute molar specific heat capacity at a given temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Molar specific heat capacity in J/mol/K.
    """
    T = T.to("K")
    theta = (T - Units.Quantity(298, "K")) / Units.Quantity(700, "K")
    cp_stp = gani_gcm.predict("Cp_stp", fuel)
    cp_b = gani_gcm.predict("Cp_B", fuel)
    cp_c = gani_gcm.predict("Cp_C", fuel)
    cp = cp_stp + cp_b * theta + cp_c * theta**2
    return cp.to("J/(mol*K)")


def specific_heat_capacity_molar_ruzicka(
    fuel: "Fuel", T: types.Quantity0D
) -> types.Quantity1D:
    """Compute molar specific heat capacity in J/mol/K at a given temperature.

    Uses the Ruzicka-Domalski correlation.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Molar specific heat capacity in J/mol/K.
    """
    Tr = T.to("K") / 100.0
    rd_A = gani_gcm.predict("rd_A", fuel)
    rd_B = gani_gcm.predict("rd_B", fuel)
    rd_D = gani_gcm.predict("rd_D", fuel)
    cp_molar = constants.R * (rd_A + rd_B * Tr + rd_D * Tr**2)
    return cp_molar.to("J/(mol*K)")


def specific_heat_capacity_mass(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Mass specific heat capacity in J/kg/K.
    """
    T = T.to("K")
    MW = fuel.comps.mw
    cp = specific_heat_capacity_molar(fuel, T)
    return (cp / MW).to("J/(kg*K)")


def specific_heat_capacity_mass_ruzicka(
    fuel: "Fuel", T: types.Quantity0D
) -> types.Quantity1D:
    """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

    Uses the Ruzicka-Domalski correlation.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Mass specific heat capacity in J/kg/K.
    """
    cp_molar = specific_heat_capacity_molar_ruzicka(fuel, T)
    return (cp_molar / fuel.comps.mw).to("J/(kg*K)")


def saturated_vapor_pressure(
    fuel: "Fuel",
    T: types.Quantity0D,
    *,
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> types.Quantity1D:
    """Compute saturated vapor pressure (P_sat).

    Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

    Returns:
        Saturated vapor pressure in Pa.
    """
    T = T.to("K")
    Tr = T / fuel.comps.tc
    Pc = fuel.comps.pc
    omega = fuel.comps.omega

    if correlation.casefold() == "Ambrose-Walton".casefold():
        # May cause trouble at high temperatures
        tau = 1 - Tr
        f0 = (
            -5.97616 * tau
            + 1.29874 * tau**1.5
            - 0.60394 * tau**2.5
            - 1.06841 * tau**5.0
        )
        f0 /= Tr
        f1 = (
            -5.03365 * tau
            + 1.11505 * tau**1.5
            - 5.41217 * tau**2.5
            - 7.46628 * tau**5.0
        )
        f1 /= Tr
        f2 = (
            -0.64771 * tau
            + 2.41539 * tau**1.5
            - 4.26979 * tau**2.5
            - 3.25259 * tau**5.0
        )
        f2 /= Tr
        rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

    else:  # Default correlation is Lee-Kesler
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
        rhs = np.exp(f0 + omega * f1)

    return (Pc * rhs).to("Pa")


def saturated_vapor_pressure_antoine_coeffs(
    fuel: "Fuel",
    Tvals: types.Quantity1D | None = None,
    *,
    units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
    """Estimate Antoine coefficients for vapor pressure of an individual compound.

    Args:
        fuel: Fuel object.
        Tvals: Temperature range or nodes for Antoine fit in Kelvin.
            Defaults to [273.15, Tb_i].
        units: Units for pressure in fit ("mks", "cgs", "dyne/cm^2", "Pa").
            Defaults to "mks".
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
            Defaults to "Lee-Kesler".

    Returns:
        Coefficients A, B, C, D for each compound.

    Raises:
        ValueError: If units or Tvals are invalid.
    """
    if units == "cgs" or units == "dyne/cm^2":
        units = "dyne/cm^2"
    elif units == "mks" or units == "Pa":
        units = "Pa"
    else:
        raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

    if Tvals is not None:
        Tvals = Tvals.to("K")

    # Define or get temperature nodes for fit
    if Tvals is None:
        print("Tvals not specified, using [273.15, Tb_i] for each compound.")
        # Initialize as zeros for now, calculated for each compound later
        T = Units.Quantity(np.zeros(20), "K")
    elif len(Tvals) == 2:
        T_low = Tvals[0].magnitude
        T_high = Tvals[1].magnitude
        T = Units.Quantity(
            np.linspace(T_low, T_high, 20),
            "K",
        )
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

    # Fit Antoine coefficients for each compound
    A = np.zeros(fuel.num_compounds)
    B = np.zeros(fuel.num_compounds)
    C = np.zeros(fuel.num_compounds)
    for i in range(fuel.num_compounds):
        # Update T if not specified
        if Tvals is None:
            T = Units.Quantity(np.linspace(273.15, fuel.comps.tb[i].magnitude, 20), "K")
        T_magnitude = T.to("K").magnitude
        Pvals = np.zeros_like(T_magnitude)
        for k in range(len(T)):
            Pvals[k] = (
                saturated_vapor_pressure(fuel, T[k], correlation=correlation)[i]
                .to("Pa")
                .magnitude
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T_magnitude, logP, p0=[1, 1e3, -1])
        A[i], B[i], C[i] = popt
    D = D.magnitude + np.zeros(fuel.num_compounds)  # make D an array
    return A, B, C, D


def enthalpy_vaporization_molar(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate molar enthalpy of vaporization adjusted for temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Molar enthalpy of vaporization in J/mol.
    """
    T = T.to("K")
    Tc = fuel.comps.tc.to("K")
    Tb = fuel.comps.tb.to("K")
    hv_stp = fuel.comps.hv_stp.to("J/mol")

    # Reduced temperatures
    Tr = T / Tc
    Trb = Tb / Tc

    hvi = Units.Quantity(np.zeros_like(Tc.magnitude), "J/mol")
    for i in range(len(Tc)):
        if T > Tc[i]:
            hvi.magnitude[i] = 0.0
        else:
            hvi[i] = hv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

    return hvi


def enthalpy_vaporization_mass(fuel: "Fuel", T: types.Quantity0D) -> types.Quantity1D:
    """Calculate mass-based enthalpy of vaporization adjusted for temperature.

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Mass-bass enthalpy of vaporization in J/kg.
    """
    return (enthalpy_vaporization_molar(fuel, T) / fuel.comps.mw).to("J/kg")


def diffusion_coeffs_wilke(
    fuel: "Fuel",
    p: types.Quantity0D,
    T: types.Quantity0D,
    *,
    sigma_gas: types.Quantity0D = constants.Sigma_gas,
    epsilonByKB_gas: types.Quantity0D = constants.EpsilonByKB_gas,
    MW_gas: types.Quantity0D = constants.MW_gas,
    correlation: Literal["Tee", "Wilke"] = "Tee",
) -> types.Quantity1D:
    """Compute diffusion coefficients using Lennard-Jones parameters.

    Uses Wilke and Lee method (Poling, equation 11-4.1). Ambient gas
    defaults to air parameters.

    Args:
        fuel: Fuel object.
        p: Pressure to compute property.
        T: Temperature to compute property.
        sigma_gas: Collision diameter.
            Default is 3.62 Angstroms.
        epsilonByKB_gas: Well depth over Boltzmann constant.
            Default is 97.0 K.
        MW_gas: Mean molecular weight of ambient gas.
            Default is 28.97 g/mol.
        correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
            Default is "Tee".

    Returns:
        Diffusion coefficient in m^2/s.
    """
    p = p.to("bar")
    T = T.to("K")
    sigma_gas = sigma_gas.to("angstrom")
    epsilonByKB_gas = epsilonByKB_gas.to("K")
    MW_gas = MW_gas.to("g/mol")

    # Method of Tee for calculating liquid sigma and epsilon
    if correlation.casefold() == "Tee".casefold():
        sigma_i = fuel.comps.lj_sigma.to("angstrom").magnitude
        epsilonByKB_i = fuel.comps.lj_eps_by_kb.to("K").magnitude
    else:
        # Method of Wilke & Lee calculating liquid sigma and epsilon
        Vmb_i = np.zeros_like(fuel.comps.tb.magnitude)
        for n in range(fuel.num_compounds):
            Vmb_i[n] = (
                molar_liquid_volume(fuel, fuel.comps.tb[n])[n].to("cm^3/mol").magnitude
            )
        sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
        epsilonByKB_i = 1.15 * fuel.comps.tb.to("K").magnitude  # K, Poling (11-4.3)

    # Compute binary sigma and epsilon
    sigma_gas: float = sigma_gas.magnitude
    sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
    epsilonAB_byKB_i = (
        epsilonByKB_gas.magnitude * epsilonByKB_i
    ) ** 0.5  # K, Poling (11-3.4)

    # Dimensionless collision integral for diffusion: Poling (11-3.6)
    T: float = T.magnitude
    Tstar_i = T / epsilonAB_byKB_i  # [1]
    A = 1.06036
    B = 0.15610
    C = 0.193
    D = 0.47635
    E = 1.03587
    F = 1.52996
    G = 1.76474
    H = 3.89411
    omegaD_i = (
        A / (Tstar_i**B)
        + C / np.exp(D * Tstar_i)
        + E / np.exp(F * Tstar_i)
        + G / np.exp(H * Tstar_i)
    )

    # Convert molecular weights from kg/mol to g/mol then calculate M_AB
    MW_gas: float = MW_gas.magnitude
    MW_i = fuel.comps.mw.to("g/mol").magnitude
    M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

    # Pressure is already in bar.
    p: float = p.magnitude

    # Binary diffusion coefficients, Poling (11-4.1)
    D_AB_i = (
        1e-3
        * (3.03 - 0.98 / (M_AB_i**0.5))
        * (T**1.5)
        / (p * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
    )  # cm^2/s
    return Units.Quantity(D_AB_i, "cm^2/s").to("m^2/s")


def surface_tension(
    fuel: "Fuel",
    T: types.Quantity0D,
    correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
) -> types.Quantity1D:
    """Calculate surface tension of each compound at a given temperature.

    Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.
        correlation: Correlation method ("Brock-Bird" or "Pitzer").
            Default is "Brock-Bird".

    Returns:
        Surface tension in N/m.
    """
    T = T.to("K")
    Tc = fuel.comps.tc.to("K").magnitude
    Pc = fuel.comps.pc.to("Pa").magnitude
    Tb = fuel.comps.tb.to("K").magnitude
    omega = fuel.comps.omega.magnitude

    Tr = T.magnitude / Tc
    Pc = Pc * 1e-5  # convert from Pa to bar

    if correlation.casefold() == "Brock-Bird".casefold():
        Tbr = Tb / Tc
        Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279
    else:
        w = omega
        Q = (
            (1.86 + 1.18 * w)
            / 19.05
            * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
        )

    st = Pc ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)

    return Units.Quantity(st, "dyn/cm").to("N/m")


def thermal_conductivity_latini(
    fuel: "Fuel",
    T: types.Quantity0D,
) -> types.Quantity1D:
    """Calculate thermal conductivity at a given temperature.

    Uses Latini et al. method (Poling equation 10-9.1).

    Args:
        fuel: Fuel object.
        T: Temperature to compute property.

    Returns:
        Thermal conductivity in W/m/K.
    """
    T = T.to("K")
    MW = fuel.comps.mw.to("kg/mol").magnitude
    Tc = fuel.comps.tc.to("K").magnitude
    Tb = fuel.comps.tb.to("K").magnitude
    fam = fuel.comps.fam

    Astar = 0.00350 + np.zeros_like(Tc)
    alpha = 1.2
    beta = 0.5 + np.zeros_like(Tc)
    gamma = 0.167
    MW_beta = MW * 1e3  # convert from kg/mol to g/mol
    Tr = T.magnitude / Tc

    for i in range(len(Tc)):
        if fam[i] == 1:
            # Aromatics
            Astar[i] = 0.0346
            beta[i] = 1.0
        elif fam[i] == 2:
            # Cycloparaffins
            Astar[i] = 0.0310
            beta[i] = 1.0
        elif fam[i] == 3:
            # Olefins
            Astar[i] = 0.0361
            beta[i] = 1.0
        MW_beta[i] = MW_beta[i] ** beta[i]

    A = Astar * Tb**alpha / (MW_beta * Tc**gamma)
    tc = A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6))
    return Units.Quantity(tc, "W/(m*K)")


def flash_point_alqaheem(fuel: "Fuel") -> types.Quantity1D:
    """Calculate component flash points with Alqaheem-Riazi (2017) correlation eq. (18).

    Args:
        fuel: Fuel object.

    Returns:
        Component flash points in K.
    """
    return 0.70 * fuel.comps.tb


def flash_point_alibakhshi(fuel: "Fuel") -> types.Quantity1D:
    """Calculate component flash points with Alibakhshi (2017) correlation eq. (5).

    Args:
        fuel: Fuel object.

    Returns:
        Component flash points in K.
    """
    phi = gani_gcm.predict("alibakhshi_phi", fuel).to("K")
    return Units.Quantity(12.14, "K") + 0.73 * fuel.comps.tb + phi


def lower_heating_value(fuel: "Fuel") -> types.Quantity1D:
    """Calculate component lower heating values from a Hess cycle.

    This estimate corresponds to hydrocarbons only; it does not account for
    heteroatom-containing groups. The result is an engineering estimate related to ASTM
    D4809/D3338 heating-value characterization, not a simulated bomb-calorimeter test.

    Args:
        fuel: Fuel object.

    Returns:
        Lower heating values of the components in J/kg.
    """
    dH_form_co2 = Units.Quantity(-393.51, "kJ/mol")  # At 298.15 K
    dH_form_h2o = Units.Quantity(-241.83, "kJ/mol")  # At 298.15 K
    Hf_liq = (fuel.comps.hf_stp - fuel.comps.hv_stp).to("J/mol")
    return (
        -(
            fuel.comps.num_c * dH_form_co2
            + 0.5 * fuel.comps.num_h * dH_form_h2o
            - Hf_liq
        )
        / fuel.comps.mw
    )
