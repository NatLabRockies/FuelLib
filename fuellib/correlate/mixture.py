"""Mixture correlation functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import curve_fit, root

from fuellib import constants

from ..utils import Units, types
from . import components, helpers

if TYPE_CHECKING:
    from ..fuel import Fuel


def arithmetic(
    X: types.Quantity1D,
    var_n: types.Quantity1D,
) -> types.Quantity0D:
    """Arithmetic mixing rule for computing mixture properties.

    Args:
        X: Mole fractions of the compounds.
        var_n: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_n.magnitude
    mag_X = X.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use arithmetic definition for the pseudo property
            var_ij = (values[i] + values[j]) / 2
            var_mix += mag_X[i] * mag_X[j] * var_ij

    return Units.Quantity(var_mix, var_n.units)  # Attach original units from var_n


def geometric(
    X: types.Quantity1D,
    var_n: types.Quantity1D,
) -> types.Quantity0D:
    """Geometric mixing rule for computing mixture properties.

    Args:
        X: Mole fractions of the compounds.
        var_n: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_n.magnitude
    mag_X = X.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use geometric definition for the pseudo property
            var_ij = (values[i] * values[j]) ** 0.5
            var_mix += mag_X[i] * mag_X[j] * var_ij

    return Units.Quantity(var_mix, var_n.units)  # Attach original units from var_n


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
        Tb = arithmetic(Xi, fuel.Tb)
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
    st = arithmetic(Xi, sti)
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


def freeze_point_boehm(
    fuel: "Fuel",
    Yi: types.Quantity1D | None = None,
    *,
    alpha: float = 1.0,
) -> types.Quantity0D:
    """Calculate the freeze point of the mixture using the Boehm method.

    Solves the solid-liquid equilibrium model of Boehm et al. (2022),
    equation 21, for every compound using its mole fraction in the mixture,
    and returns the highest candidate temperature, which marks the first
    crystal to form on cooling. A scipy root finder is used in place of the
    fixed-point iteration used in the original reference implementation.

    For each hydrocarbon family, `fusion_families.csv` supplies the entropy
    of fusion `dS_fus_i = max(A_f + B_f * (nC_i - C_ref_f), 20)` in
    J/(mol*K); unclassified compounds fall back to the Walden-rule estimate
    of 56.5 J/mol/K. The enthalpy of fusion is `dH_fus_i = Tm_i * dS_fus_i`,
    and the solid-minus-liquid heat-capacity approximation is
    `dCp_i = -0.35 * Cp_L_i(298.15 K)` on a molar basis. Boehm's equation 21
    is solved per compound `j` against the ideal binary mixing entropy of
    that compound relative to the rest of the mixture,

        dS_mix_j = -(R / x_j) * [(1 - x_j) * ln(1 - x_j) + x_j * ln(x_j)],

        T_j = (dH_fus_j + x_j * dCp_j * (Tm_j - T_j))
              / (dS_fus_j + x_j * dCp_j * ln(T_j / Tm_j) + alpha * dS_mix_j).

    Components with mole fraction at or below 1e-6 are excluded from the
    result. This is an equilibrium screening model; it does not represent
    cooling rate, supercooling, crystal kinetics, or detailed solid-phase
    nonideality.

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).
        alpha: Scaling applied to the ideal mixing-entropy term.
            Defaults to 1.0 (classical ideal-solution entropy term).

    Returns:
        Mixture freeze point in K.

    Raises:
        RuntimeError: If the root finder fails to converge.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    Xi = (
        helpers.mass_fractions_to_mole_fractions(fuel, Yi).to("dimensionless").magnitude
    )

    R = constants.gas_constant.to("J/(mol*K)")
    # NOTE: Tm should be from ASTM, not Gani.
    Tm = fuel.get_property("gani", "Tm").to("K")
    dS_fus = fuel.get_property("boehm", "dS_fus").to("J/(mol*K)")
    dH_fus = Tm * dS_fus
    dCp = (
        -0.35
        * components.liquid_mass_specific_heat_capacity_ruzicka(
            fuel, Units.Quantity(298.15, "K")
        )
        * fuel.MW
    ).to("J/(mol*K)")

    # Ideal mixing entropy for each compound at its mole fraction in the
    # mixture. This is the only composition-dependent term in the model.
    Xi_safe = np.clip(Xi, 1e-6, 1.0 - 1e-6)
    dS_mix = (
        -R
        / Xi_safe
        * ((1.0 - Xi_safe) * np.log(1.0 - Xi_safe) + Xi_safe * np.log(Xi_safe))
    )

    Tm_mag = Tm.magnitude
    dH_fus_mag = dH_fus.magnitude
    dS_fus_mag = dS_fus.magnitude
    dCp_mag = dCp.magnitude
    dS_mix_mag = dS_mix.magnitude

    def residual(T: types.Array1D) -> types.Array1D:
        """Residual of Boehm et al. (2022), equation 21, for each compound.

        Args:
            T: Candidate freeze temperature of each compound in K.

        Returns:
            Residual of equation 21 for each compound.
        """
        T_safe = np.maximum(T, 1.0)
        lhs = T_safe * (
            dS_fus_mag
            + Xi_safe * dCp_mag * np.log(T_safe / Tm_mag)
            + alpha * dS_mix_mag
        )
        rhs = dH_fus_mag + Xi_safe * dCp_mag * (Tm_mag - T_safe)
        return lhs - rhs

    sol = root(residual, Tm_mag)
    if not sol.success:
        raise RuntimeError(
            f"Freeze point root-finding failed to converge: {sol.message}"
        )

    # Mark non-physical (non-positive or non-finite) solutions as -inf, and
    # only let compounds actually present in the mixture (Xi > 1e-6) set the
    # freeze point, matching the reference implementation's convention.
    T_candidates = np.where(np.isfinite(sol.x) & (sol.x > 0), sol.x, -np.inf)
    T_freeze = np.max(np.where(Xi > 1e-6, T_candidates, -np.inf))

    return Units.Quantity(T_freeze, "K")


def _liaw_chiu_flash_point(
    Xi: types.Array1D,
    Tf_i: types.Array1D,
    Tc: types.Array1D,
    Pc: types.Array1D,
    omega: types.Array1D,
) -> float:
    """Solve the ideal Liaw-Chiu mixture flash-point criterion.

    Solves for the mixture flash point T such that
    sum(Xi * psat(T) / psat(Tf_i)) = 1, where psat is evaluated with the
    Lee-Kesler correlation for each compound at its own critical properties.
    A scipy root finder is used in place of the fixed-point iteration used
    in the original reference implementation.

    Args:
        Xi: Mole fractions of each compound in the mixture.
        Tf_i: Flash point of each compound in K.
        Tc: Critical temperature of each compound in K.
        Pc: Critical pressure of each compound.
        omega: Acentric factor of each compound.

    Returns:
        Mixture flash point in K.

    Raises:
        RuntimeError: If the root finder fails to converge.
    """

    def _psat_lee_kesler(
        T: types.Array1D | float,
        Tc: types.Array1D,
        Pc: types.Array1D,
        omega: types.Array1D,
    ) -> types.Array1D:
        """Lee-Kesler saturated vapor pressure for each compound.

        Copies `components.saturated_vapor_pressure(..., correlation="lee_kesler")`
        in a scipy-based root-finding context.

        Args:
            T: Temperature in K.
            Tc: Critical temperature of each compound in K.
            Pc: Critical pressure of each compound.
            omega: Acentric factor of each compound.

        Returns:
            Saturated vapor pressure (units follow whatever `Pc` is given in).
        """
        Tr = T / Tc
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
        return Pc * np.exp(f0 + omega * f1)

    psat_ref = _psat_lee_kesler(Tf_i, Tc, Pc, omega)

    def residual(T: types.Array1D) -> types.Array1D:
        """Residual of the ideal Liaw-Chiu mixture flash-point criterion.

        Args:
            T: Candidate mixture flash point in K.

        Returns:
            Residual of the flash-point criterion.
        """
        psat_T = _psat_lee_kesler(T[0], Tc, Pc, omega)
        return np.array([np.sum(Xi * psat_T / psat_ref) - 1.0])

    T0 = np.sum(Xi * Tf_i)
    sol = root(residual, [T0])
    if not sol.success:
        raise RuntimeError(
            f"Flash point root-finding failed to converge: {sol.message}"
        )

    return sol.x[0]


def flash_point_alqaheem(
    fuel: "Fuel",
    Yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["linear", "Liaw"] = "Liaw",
) -> types.Quantity0D:
    """Calculate the flash point of the mixture using the Alqaheem method.

    Uses the Alqaheem-Riazi pure-component correlation
    (`components.flash_point_alqaheem`) combined with either the Liaw-Chiu
    (default) or linear mixing rule; see `flash_point_alibashki` for the
    Liaw-Chiu mixture-rule formulation shared by both methods.

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).
        mixing_rule: Mixing rule to use.
            Defaults to "Liaw".

    Returns:
        Mixture flash point in K.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    T_fpi = components.flash_point_alqaheem(fuel)
    if mixing_rule.casefold() == "linear".casefold():
        return np.sum(Yi * T_fpi)

    # Liaw-Chiu mixing rule: solve for the mixture flash point T such that
    # sum(Xi * psat(T) / psat(Tf_i)) = 1, where psat is evaluated with the
    # Lee-Kesler correlation for each compound at its own critical properties.
    Xi = (
        helpers.mass_fractions_to_mole_fractions(fuel, Yi).to("dimensionless").magnitude
    )
    Tc = fuel.Tc.to("K").magnitude
    Pc = fuel.Pc.magnitude
    omega = fuel.omega.magnitude
    Tf_i = T_fpi.to("K").magnitude

    T_flash = _liaw_chiu_flash_point(Xi, Tf_i, Tc, Pc, omega)
    return Units.Quantity(T_flash, "K")


def flash_point_alibashki(
    fuel: "Fuel",
    Yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["linear", "Liaw"] = "Liaw",
) -> types.Quantity0D:
    """Calculate the flash point of the mixture using the Alibashki method.

    Uses the Alibakhshi et al. pure-component correlation
    (`components.flash_point_alibashki`) combined with either the Liaw-Chiu
    (default) or linear mixing rule.

    For `mixing_rule="linear"`, the mixture flash point is the mass-fraction
    weighted average, `Tfp_mix = sum(Yi * Tfp_i)`.

    For `mixing_rule="Liaw"` (default), mole fractions `Xi` are used with the
    ideal-activity Liaw-Chiu (2006) relation, which solves for the mixture
    flash point `Tfp_mix` satisfying

        sum(Xi * Psat_i(Tfp_mix) / Psat_i(Tfp_i)) = 1,

    where `Psat_i` is evaluated with the Lee-Kesler correlation using each
    compound's own critical properties (see
    `components.saturated_vapor_pressure`). The residual is solved with a
    bounded Newton iteration (`helpers.liaw_chiu_flash_point`) initialized
    from the mole-fraction-weighted pure-component flash point. This ideal
    mixing rule does not model nonideal liquid activity coefficients.

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).
        mixing_rule: Mixing rule to use.
            Defaults to "Liaw".

    Returns:
        Mixture flash point in K.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    T_fpi = components.flash_point_alibashki(fuel)
    if mixing_rule.casefold() == "linear".casefold():
        return np.sum(Yi * T_fpi)

    # Liaw-Chiu mixing rule: solve for the mixture flash point T such that
    # sum(Xi * psat(T) / psat(Tf_i)) = 1, where psat is evaluated with the
    # Lee-Kesler correlation for each compound at its own critical properties.
    Xi = (
        helpers.mass_fractions_to_mole_fractions(fuel, Yi).to("dimensionless").magnitude
    )
    Tc = fuel.Tc.to("K").magnitude
    Pc = fuel.Pc.magnitude
    omega = fuel.omega.magnitude
    Tf_i = T_fpi.to("K").magnitude

    T_flash = _liaw_chiu_flash_point(Xi, Tf_i, Tc, Pc, omega)
    return Units.Quantity(T_flash, "K")


def heat_of_combustion(
    fuel: "Fuel", Yi: types.Quantity1D | None = None
) -> types.Quantity1D:
    """Calculate the heat of combustion of the fuel using a Hess cycle.

    Combines each component's lower heating value
    (`components.lower_heating_value`) with a mass-fraction weighted mixing
    rule, `dHc_mix = sum(Yi * LHV_i)`, reported as the positive heat released
    per unit mass. This is a net (lower) basis,
    consistent with gaseous-water combustion products; it is an engineering
    estimate related to ASTM D4809/D3338 heating-value characterization, not
    a simulated bomb-calorimeter test.

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.Y_0` (initial mass fractions).

    Returns:
        Heat of combustion of the mixture in J/kg.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    lhv_i = components.lower_heating_value(fuel)
    return np.sum(Yi * lhv_i)
