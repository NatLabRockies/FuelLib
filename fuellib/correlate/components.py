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


def molar_liquid_volume(fuel: "Fuel", temp: types.Quantity0D) -> types.Quantity1D:
    """Compute molar liquid volume with temperature correction.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Molar liquid volume in m^3/mol.
    """
    temp_stp = Units.Quantity(298, "K")
    temp = temp.to("K")
    tc = fuel.comps.tc.to("K")
    phi = np.zeros_like(tc.magnitude)
    for i in range(len(tc)):
        if temp > tc[i]:
            phi[i] = -((1 - (temp_stp / tc[i])) ** (2.0 / 7.0))
        else:
            phi[i] = (1 - (temp / tc[i])) ** (2.0 / 7.0) - (1 - (temp_stp / tc[i])) ** (
                2.0 / 7.0
            )
    z = 0.29056 - 0.08775 * fuel.comps.omega
    return fuel.comps.vm_stp * z**phi


def density(fuel: "Fuel", temp: types.Quantity0D) -> types.Quantity1D:
    """Calculate the density of each component at a given temperature.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Density of each compound in kg/m^3.
    """
    return (fuel.comps.mw / molar_liquid_volume(fuel, temp)).to("kg/m^3")


def viscosity_kinematic_dutt(fuel: "Fuel", temp: types.Quantity0D) -> types.Quantity1D:
    """Calculate the kinematic viscosity using Dutt's equation.

    Uses Dutt's equation (4.23) from "Viscosity of Liquids". The equation
    predicts viscosity in mm^2/s and is converted to SI units.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Viscosity of each component in m^2/s.
    """
    # Dutt's equation (4.23) uses temperatures in Celsius
    temp_mag: float = temp.to("celsius").magnitude
    tb_mag = fuel.comps.tb.to("celsius").magnitude
    # RHS of Dutt's equation (4.23) in Viscosity of Liquids
    rhs = -3.0171 + (442.78 + 1.6452 * tb_mag) / (temp_mag + 239 - 0.19 * tb_mag)
    return Units.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")


def viscosity_dynamic_dutt(fuel: "Fuel", temp: types.Quantity0D) -> types.Quantity1D:
    """Calculate the dynamic viscosity using Dutt's equation.

    Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Dynamic viscosity in Pa*s.
    """
    nu_i = viscosity_kinematic_dutt(fuel, temp)
    rho_i = density(fuel, temp)
    return (nu_i * rho_i).to("Pa*s")


def viscosity_dynamic_nannoolal(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Calculate the dynamic viscosity using Nannoolal (2009) GCM.

    Uses equation (6) from Nannoolal (2009).

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Component dynamic viscosities in Pa*s.
    """
    temp_mag = temp.to("kelvin").magnitude
    tb_mag = fuel.comps.tb.to("kelvin").magnitude
    dbv_i = nannoolal_gcm.predict("dbv", fuel).magnitude
    tv_i = nannoolal_gcm.predict("tv", fuel).magnitude
    n_i = np.array([
        sum(n for atom, n in mol.atom_counts(m).items() if atom != "H")
        for m in fuel.comps.rdkit_mols
    ])
    dbv_i = dbv_i / (n_i**-2.5635 + 0.0685) + 3.7777
    tv_i = 21.8444 * np.sqrt(tb_mag) + tv_i**0.9315 / (n_i**0.6577 + 4.9259) - 231.1361
    denom = temp_mag - tv_i / 16.0
    eta = 1.3e-3 * np.exp(-dbv_i * (temp_mag - tv_i) / denom)
    return Units.Quantity(eta, "Pa*s")


def viscosity_kinematic_nannoolal(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Calculate the kinematic viscosity using Nannoolal (2009) GCM.

        Divides equation (6) from Nannoolal (2009) by density.

    Args:
        fuel: Fuel object.
        temp: Temperature to calculate the kinematic viscosities.

    Returns:
        Component kinematic viscosities in m^2/s.
    """
    eta = viscosity_dynamic_nannoolal(fuel, temp)
    rho = density(fuel, temp)
    return (eta / rho).to("m^2/s")


def specific_heat_capacity_molar(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Compute molar specific heat capacity at a given temperature.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Molar specific heat capacity in J/mol/K.
    """
    temp = temp.to("K")
    theta = (temp - Units.Quantity(298, "K")) / Units.Quantity(700, "K")
    cp_stp = gani_gcm.predict("cp_stp", fuel)
    cp_b = gani_gcm.predict("cp_b", fuel)
    cp_c = gani_gcm.predict("cp_c", fuel)
    cp = cp_stp + cp_b * theta + cp_c * theta**2
    return cp.to("J/(mol*K)")


def specific_heat_capacity_molar_ruzicka(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Compute molar specific heat capacity in J/mol/K at a given temperature.

    Uses the Ruzicka-Domalski correlation.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Molar specific heat capacity in J/mol/K.
    """
    temp_100 = temp.to("K") / 100.0
    rd_a = gani_gcm.predict("rd_a", fuel)
    rd_b = gani_gcm.predict("rd_b", fuel)
    rd_d = gani_gcm.predict("rd_d", fuel)
    cp_molar = constants.R * (rd_a + rd_b * temp_100 + rd_d * temp_100**2)
    return cp_molar.to("J/(mol*K)")


def specific_heat_capacity_mass(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Mass specific heat capacity in J/kg/K.
    """
    temp = temp.to("K")
    mw = fuel.comps.mw
    cp = specific_heat_capacity_molar(fuel, temp)
    return (cp / mw).to("J/(kg*K)")


def specific_heat_capacity_mass_ruzicka(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

    Uses the Ruzicka-Domalski correlation.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Mass specific heat capacity in J/kg/K.
    """
    cp_molar = specific_heat_capacity_molar_ruzicka(fuel, temp)
    return (cp_molar / fuel.comps.mw).to("J/(kg*K)")


def vapor_pressure(
    fuel: "Fuel",
    temp: types.Quantity0D,
    *,
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> types.Quantity1D:
    """Compute vapor pressure.

    Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

    Returns:
        Vapor pressure in Pa.
    """
    temp = temp.to("K")
    tr = temp / fuel.comps.tc
    pc = fuel.comps.pc
    omega = fuel.comps.omega

    if correlation.casefold() == "Ambrose-Walton".casefold():
        # May cause trouble at high temperatures
        tau = 1 - tr
        f0 = (
            -5.97616 * tau
            + 1.29874 * tau**1.5
            - 0.60394 * tau**2.5
            - 1.06841 * tau**5.0
        )
        f0 /= tr
        f1 = (
            -5.03365 * tau
            + 1.11505 * tau**1.5
            - 5.41217 * tau**2.5
            - 7.46628 * tau**5.0
        )
        f1 /= tr
        f2 = (
            -0.64771 * tau
            + 2.41539 * tau**1.5
            - 4.26979 * tau**2.5
            - 3.25259 * tau**5.0
        )
        f2 /= tr
        rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

    else:  # Default correlation is Lee-Kesler
        f0 = 5.92714 - (6.09648 / tr) - 1.28862 * np.log(tr) + 0.169347 * (tr**6)
        f1 = 15.2518 - (15.6875 / tr) - 13.4721 * np.log(tr) + 0.43577 * (tr**6)
        rhs = np.exp(f0 + omega * f1)

    return (pc * rhs).to("Pa")


def vapor_pressure_antoine_coeffs(
    fuel: "Fuel",
    temp_vals: types.Quantity1D | None = None,
    *,
    units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
    """Estimate Antoine coefficients for vapor pressure of an individual compound.

    Args:
        fuel: Fuel object.
        temp_vals: Temperature range or nodes for Antoine fit in Kelvin.
            Defaults to [273.15, Tb_i].
        units: Units for pressure in fit ("mks", "cgs", "dyne/cm^2", "Pa").
            Defaults to "mks".
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
            Defaults to "Lee-Kesler".

    Returns:
        Coefficients A, B, C, D for each compound.

    Raises:
        ValueError: If units or temp_vals are invalid.
    """
    if units == "cgs" or units == "dyne/cm^2":
        units = "dyne/cm^2"
    elif units == "mks" or units == "Pa":
        units = "Pa"
    else:
        raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

    if temp_vals is not None:
        temp_vals = temp_vals.to("K")

    # Define or get temperature nodes for fit
    use_tb_range = temp_vals is None
    if use_tb_range:
        # Nodes are calculated for each compound later
        print("temp_vals not specified, using [273.15, Tb_i] for each compound.")
    elif len(temp_vals) == 2:
        temp_vals = Units.Quantity(
            np.linspace(temp_vals[0].magnitude, temp_vals[1].magnitude, 20), "K"
        )
    elif len(temp_vals) < 2:
        raise ValueError("temp_vals must be None, length 2, or length > 2.")

    # Antoine equation log10(p) = A - B/(C + T)
    def antoine_eq(
        temp: float | types.Array1D, a: float, b: float, c: float
    ) -> float | types.Array1D:
        """Antoine equation for vapor pressure.

        Args:
            temp: Temperature.
            a: Antoine coefficient A.
            b: Antoine coefficient B.
            c: Antoine coefficient C.

        Returns:
            log10(pressure).
        """
        return a - b / (temp + c)

    # Fit A, B, C against pressure in Pa (mks base) so the coefficients are
    # unit independent. "mks" (meter-kilogram-second) and "cgs"
    # D is the Pa-to-target-unit conversion factor, applied only when evaluating
    # pvap(T) = D * 10**(A - B/(T + C)) in Pele.
    d = Units.Quantity(1, "Pa").to(units)

    # Fit Antoine coefficients for each compound
    a = np.zeros(fuel.num_compounds)
    b = np.zeros(fuel.num_compounds)
    c = np.zeros(fuel.num_compounds)
    for i in range(fuel.num_compounds):
        # Update temp_vals if not specified
        if use_tb_range:
            temp_vals = Units.Quantity(
                np.linspace(273.15, fuel.comps.tb[i].magnitude, 20), "K"
            )
        temp_vals_mag = temp_vals.to("K").magnitude
        pvals = np.zeros_like(temp_vals_mag)
        for k in range(len(temp_vals)):
            pvals[k] = (
                vapor_pressure(fuel, temp_vals[k], correlation=correlation)[i]
                .to("Pa")
                .magnitude
            )

        log_p = np.log10(pvals)
        popt, _ = curve_fit(antoine_eq, temp_vals_mag, log_p, p0=[1, 1e3, -1])
        a[i], b[i], c[i] = popt
    d = d.magnitude + np.zeros(fuel.num_compounds)  # make d an array
    return a, b, c, d


def enthalpy_vaporization_molar(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Calculate molar enthalpy of vaporization adjusted for temperature.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Molar enthalpy of vaporization in J/mol.
    """
    temp = temp.to("K")
    tc = fuel.comps.tc.to("K")
    tb = fuel.comps.tb.to("K")
    hv_stp = fuel.comps.hv_stp.to("J/mol")

    # Reduced temperatures
    tr = temp / tc
    tbr = tb / tc

    hvi = Units.Quantity(np.zeros_like(tc.magnitude), "J/mol")
    for i in range(len(tc)):
        if temp > tc[i]:
            hvi.magnitude[i] = 0.0
        else:
            hvi[i] = hv_stp[i] * (((1.0 - tr[i]) / (1.0 - tbr[i])) ** 0.38)

    return hvi


def enthalpy_vaporization_mass(
    fuel: "Fuel", temp: types.Quantity0D
) -> types.Quantity1D:
    """Calculate mass-based enthalpy of vaporization adjusted for temperature.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Mass-bass enthalpy of vaporization in J/kg.
    """
    return (enthalpy_vaporization_molar(fuel, temp) / fuel.comps.mw).to("J/kg")


def diffusion_coeffs_wilke(
    fuel: "Fuel",
    p: types.Quantity0D,
    temp: types.Quantity0D,
    *,
    sigma_gas: types.Quantity0D = constants.SIGMA_GAS,
    lj_eps_by_kb_gas: types.Quantity0D = constants.LJ_EPS_BY_KB_GAS,
    mw_gas: types.Quantity0D = constants.MW_GAS,
    correlation: Literal["Tee", "Wilke"] = "Tee",
) -> types.Quantity1D:
    """Compute diffusion coefficients using Lennard-Jones parameters.

    Uses Wilke and Lee method (Poling, equation 11-4.1). Ambient gas
    defaults to air parameters.

    Args:
        fuel: Fuel object.
        p: Pressure to compute property.
        temp: Temperature to compute property.
        sigma_gas: Collision diameter.
            Default is 3.62 Angstroms.
        lj_eps_by_kb_gas: Well depth over Boltzmann constant.
            Default is 97.0 K.
        mw_gas: Mean molecular weight of ambient gas.
            Default is 28.97 g/mol.
        correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
            Default is "Tee".

    Returns:
        Diffusion coefficient in m^2/s.
    """
    p = p.to("bar")
    temp = temp.to("K")
    sigma_gas = sigma_gas.to("angstrom")
    lj_eps_by_kb_gas = lj_eps_by_kb_gas.to("K")
    mw_gas = mw_gas.to("g/mol")

    # Method of Tee for calculating liquid sigma and epsilon
    if correlation.casefold() == "Tee".casefold():
        sigma_i = fuel.comps.lj_sigma.to("angstrom").magnitude
        eps_by_kb_i = fuel.comps.lj_eps_by_kb.to("K").magnitude
    else:
        # Method of Wilke & Lee calculating liquid sigma and epsilon
        vmb_i = np.zeros_like(fuel.comps.tb.magnitude)
        for n in range(fuel.num_compounds):
            vmb_i[n] = (
                molar_liquid_volume(fuel, fuel.comps.tb[n])[n].to("cm^3/mol").magnitude
            )
        sigma_i = 1.18 * vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
        eps_by_kb_i = 1.15 * fuel.comps.tb.to("K").magnitude  # K, Poling (11-4.3)

    # Compute binary sigma and epsilon
    sigma_gas: float = sigma_gas.magnitude
    sigma_ab_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
    eps_ab_by_kb_i = (
        lj_eps_by_kb_gas.magnitude * eps_by_kb_i
    ) ** 0.5  # K, Poling (11-3.4)

    # Dimensionless collision integral for diffusion: Poling (11-3.6)
    temp: float = temp.magnitude
    t_star_i = temp / eps_ab_by_kb_i  # [1]
    a = 1.06036
    b = 0.15610
    c = 0.193
    d = 0.47635
    e = 1.03587
    f = 1.52996
    g = 1.76474
    h = 3.89411
    omega_d_i = (
        a / (t_star_i**b)
        + c / np.exp(d * t_star_i)
        + e / np.exp(f * t_star_i)
        + g / np.exp(h * t_star_i)
    )

    # Convert molecular weights from kg/mol to g/mol then calculate M_AB
    mw_gas: float = mw_gas.magnitude
    mw_i = fuel.comps.mw.to("g/mol").magnitude
    m_ab_i = 2 * (mw_i * mw_gas) / (mw_i + mw_gas)  # g/mol, see Poling (11-3.1)

    # Pressure is already in bar.
    p: float = p.magnitude

    # Binary diffusion coefficients, Poling (11-4.1)
    d_ab_i = (
        1e-3
        * (3.03 - 0.98 / (m_ab_i**0.5))
        * (temp**1.5)
        / (p * m_ab_i**0.5 * sigma_ab_i**2 * omega_d_i)
    )  # cm^2/s
    return Units.Quantity(d_ab_i, "cm^2/s").to("m^2/s")


def surface_tension(
    fuel: "Fuel",
    temp: types.Quantity0D,
    correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
) -> types.Quantity1D:
    """Calculate surface tension of each compound at a given temperature.

    Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        correlation: Correlation method ("Brock-Bird" or "Pitzer").
            Default is "Brock-Bird".

    Returns:
        Surface tension in N/m.
    """
    temp = temp.to("K")
    tc = fuel.comps.tc.to("K").magnitude
    pc = fuel.comps.pc.to("bar").magnitude  # bar
    tb = fuel.comps.tb.to("K").magnitude
    omega = fuel.comps.omega.magnitude

    tr = temp.magnitude / tc

    if correlation.casefold() == "Brock-Bird".casefold():
        tbr = tb / tc
        q = 0.1196 * (1.0 + (tbr * np.log(pc / 1.01325)) / (1.0 - tbr)) - 0.279
    else:
        w = omega
        q = (
            (1.86 + 1.18 * w)
            / 19.05
            * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
        )

    st = pc ** (2.0 / 3.0) * tc ** (1.0 / 3.0) * q * (1 - tr) ** (11.0 / 9.0)

    return Units.Quantity(st, "dyn/cm").to("N/m")


def thermal_conductivity_latini(
    fuel: "Fuel",
    temp: types.Quantity0D,
) -> types.Quantity1D:
    """Calculate thermal conductivity at a given temperature.

    Uses Latini et al. method (Poling equation 10-9.1).

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.

    Returns:
        Thermal conductivity in W/m/K.
    """
    temp = temp.to("K")
    mw = fuel.comps.mw.to("kg/mol").magnitude
    tc = fuel.comps.tc.to("K").magnitude
    tb = fuel.comps.tb.to("K").magnitude
    fam = fuel.comps.fam

    a_star = 0.00350 + np.zeros_like(tc)
    alpha = 1.2
    beta = 0.5 + np.zeros_like(tc)
    gamma = 0.167
    mw_beta = mw * 1e3  # convert from kg/mol to g/mol
    tr = temp.magnitude / tc

    for i in range(len(tc)):
        if fam[i] == 1:
            # Aromatics
            a_star[i] = 0.0346
            beta[i] = 1.0
        elif fam[i] == 2:
            # Cycloparaffins
            a_star[i] = 0.0310
            beta[i] = 1.0
        elif fam[i] == 3:
            # Olefins
            a_star[i] = 0.0361
            beta[i] = 1.0
        mw_beta[i] = mw_beta[i] ** beta[i]

    a = a_star * tb**alpha / (mw_beta * tc**gamma)
    k_i = a * (1 - tr) ** (0.38) / (tr ** (1 / 6))
    return Units.Quantity(k_i, "W/(m*K)")


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
    dh_form_co2 = Units.Quantity(-393.51, "kJ/mol")  # At 298.15 K
    dh_form_h2o = Units.Quantity(-241.83, "kJ/mol")  # At 298.15 K
    hf_liq = (fuel.comps.hf_stp - fuel.comps.hv_stp).to("J/mol")
    return (
        -(
            fuel.comps.num_c * dh_form_co2
            + 0.5 * fuel.comps.num_h * dh_form_h2o
            - hf_liq
        )
        / fuel.comps.mw
    )
