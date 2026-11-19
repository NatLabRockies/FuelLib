"""Mixture correlation functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import curve_fit, root

from ..gcm.boehm import boehm_gcm
from ..rdk import mol
from ..utils import Units, constants, types
from . import components, helpers

if TYPE_CHECKING:
    from ..fuel import Fuel


def arithmetic(
    x: types.Quantity1D,
    var_i: types.Quantity1D,
) -> types.Quantity0D:
    """Arithmetic mixing rule for computing mixture properties.

    Args:
        x: Mole fractions of the compounds.
        var_i: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_i.magnitude
    mag_x = x.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use arithmetic definition for the pseudo property
            var_ij = (values[i] + values[j]) / 2
            var_mix += mag_x[i] * mag_x[j] * var_ij

    return Units.Quantity(var_mix, var_i.units)  # Attach original units from var_i


def geometric(
    x: types.Quantity1D,
    var_i: types.Quantity1D,
) -> types.Quantity0D:
    """Geometric mixing rule for computing mixture properties.

    Args:
        x: Mole fractions of the compounds.
        var_i: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_i.magnitude
    mag_x = x.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use geometric definition for the pseudo property
            var_ij = (values[i] * values[j]) ** 0.5
            var_mix += mag_x[i] * mag_x[j] * var_ij

    return Units.Quantity(var_mix, var_i.units)  # Attach original units from var_i


def arrhenius(x: types.Quantity1D, var_i: types.Quantity1D) -> types.Quantity0D:
    """Apply the Arrhenius mixing rule to a positive component property.

    Args:
        x: Mole fractions of the compounds.
        var_i: Positive component properties, typically kinematic viscosities.

    Returns:
        The mixture property in the units of `var_i`.
    """
    values = var_i.magnitude
    mole_fractions = x.to("dimensionless").magnitude
    return Units.Quantity(np.exp(np.sum(mole_fractions * np.log(values))), var_i.units)


def kendall_monroe(x: types.Quantity1D, var_i: types.Quantity1D) -> types.Quantity0D:
    """Apply the Kendall-Monroe mixing rule to a positive component property.

    Args:
        x: Mole fractions of the compounds.
        var_i: Positive component properties, typically kinematic viscosities.

    Returns:
        The mixture property in the units of `var_i`.
    """
    values = var_i.magnitude
    mole_fractions = x.to("dimensionless").magnitude
    value = np.sum(mole_fractions * values ** (1.0 / 3.0)) ** 3.0
    return Units.Quantity(value, var_i.units)


def mean_molecular_weight(
    fuel: "Fuel", yi: types.Quantity1D | None = None
) -> types.Quantity0D:
    """Calculate the mean molecular weight of the mixture.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Mean molecular weight of the mixture in kg/mol.
    """
    yi = yi if yi is not None else fuel.y0
    mw = fuel.comps.mw.to("kg/mol")  # Can't be None after `rdkit` implementation
    return 1 / np.sum(yi / mw)


def density(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
) -> types.Quantity0D:
    """Calculate mixture density.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Mixture density in kg/m^3.
    """
    yi = yi if yi is not None else fuel.y0
    rho_i = components.density(fuel, temp).to("kg/m^3")
    return yi @ rho_i


def viscosity_kinematic_dutt(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> types.Quantity0D:
    """Calculate kinematic viscosity of the mixture.

    Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound.
            Defaults to `fuel.y0` (initial mass fractions).
        mixing_rule: Mixture rule ("Kendall-Monroe" or "Arrhenius").
            Defaults to "Kendall-Monroe".

    Returns:
        Mixture kinematic viscosity in m^2/s.
    """
    yi = yi if yi is not None else fuel.y0
    nu_i = components.viscosity_kinematic_dutt(fuel, temp).to("m^2/s")
    # Calculate mole fractions for each species
    xi = helpers.mass_fractions_to_mole_fractions(fuel, yi)
    if mixing_rule.casefold() == "Arrhenius".casefold():
        # Arrhenius mixing correlation
        return arrhenius(xi, nu_i)
    else:
        # Default: Kendall-Monroe mixing correlation
        return kendall_monroe(xi, nu_i)


def viscosity_dynamic_dutt(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> types.Quantity0D:
    """Calculate dynamic viscosity of the mixture.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound.
            Defaults to `fuel.y0` (initial mass fractions).
        mixing_rule: Mixture rule ("Kendall-Monroe" or "Arrhenius").
            Defaults to "Kendall-Monroe".

    Returns:
        Mixture dynamic viscosity in Pa*s.
    """
    yi = yi if yi is not None else fuel.y0
    nu = viscosity_kinematic_dutt(fuel, temp, yi, mixing_rule=mixing_rule)
    rho = density(fuel, temp, yi)
    return (rho * nu).to("Pa*s")


def viscosity_kinematic_nannoolal(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> types.Quantity0D:
    """Calculate kinematic viscosity of the mixture from Nannoolal GCM.

    Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound.
            Defaults to `fuel.y0` (initial mass fractions).
        mixing_rule: Mixture rule ("Kendall-Monroe" or "Arrhenius").
            Defaults to "Kendall-Monroe".

    Returns:
        Mixture kinematic viscosity in m^2/s.
    """
    yi = yi if yi is not None else fuel.y0
    nu_i = components.viscosity_kinematic_nannoolal(fuel, temp).to("m^2/s")
    # Calculate mole fractions for each species
    xi = helpers.mass_fractions_to_mole_fractions(fuel, yi)
    if mixing_rule.casefold() == "Arrhenius".casefold():
        # Arrhenius mixing correlation
        return arrhenius(xi, nu_i)
    else:
        # Default: Kendall-Monroe mixing correlation
        return kendall_monroe(xi, nu_i)


def viscosity_dynamic_nannoolal(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> types.Quantity0D:
    """Calculate dynamic viscosity of the mixture from Nannoolal GCM.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound.
            Defaults to `fuel.y0` (initial mass fractions).
        mixing_rule: Mixture rule ("Kendall-Monroe" or "Arrhenius").
            Defaults to "Kendall-Monroe".

    Returns:
        Mixture dynamic viscosity in Pa*s.
    """
    yi = yi if yi is not None else fuel.y0
    nu = viscosity_kinematic_nannoolal(fuel, temp, yi, mixing_rule=mixing_rule)
    rho = density(fuel, temp, yi)
    return (rho * nu).to("Pa*s")


def vapor_pressure(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
    *,
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> types.Quantity0D:
    """Calculate vapor pressure of the mixture.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
            Defaults to "Lee-Kesler".

    Returns:
        Mixture vapor pressure in Pa.
    """
    yi = yi if yi is not None else fuel.y0
    # Mole fraction for each compound
    xi = helpers.mass_fractions_to_mole_fractions(fuel, yi)
    # Vapor pressure for each compound (Pa)
    pvap_i = components.vapor_pressure(fuel, temp, correlation=correlation).to("Pa")
    # Mixture vapor pressure via Raoult's law
    return pvap_i @ xi


def vapor_pressure_antoine_coeffs(
    fuel: "Fuel",
    temp_vals: types.Quantity1D | None = None,
    yi: types.Quantity1D | None = None,
    *,
    units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
    correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
) -> tuple[float, float, float, float]:
    """Estimate Antoine coefficients for vapor pressure of the mixture.

    Args:
        fuel: Fuel object.
        temp_vals: Temperature range or nodes for Antoine fit in Kelvin.
            Defaults to [273.15, min(Tb_mix)].
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        units: Units for pressure in fit.
            Defaults to "mks" (Pa).
        correlation: Correlation method.
            Defaults to "Lee-Kesler".

    Returns:
        Coefficients A, B, C, D.

    Raises:
        ValueError: If units or temp_vals are invalid.
    """
    yi = yi if yi is not None else fuel.y0
    if units == "cgs":
        units = "dyne/cm^2"
    elif units == "mks":
        units = "Pa"
    elif units not in ["dyne/cm^2", "Pa"]:
        raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

    if temp_vals is not None:
        temp_vals = temp_vals.to("K")

    # Define or get temperature nodes for fit
    if temp_vals is None:
        print("temp_vals not specified, using [273.15, min(Tb_mix)] for mixture.")
        xi = helpers.mass_fractions_to_mole_fractions(fuel, yi)
        tb = arithmetic(xi, fuel.comps.tb)
        temp_vals = Units.Quantity(
            np.linspace(273.15, np.min(tb.to("K").magnitude), 20), "K"
        )
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

    temp_vals_mag = temp_vals.to("K").magnitude
    pvals = np.zeros_like(temp_vals_mag)
    for k in range(len(temp_vals)):
        pvals[k] = (
            vapor_pressure(fuel, temp_vals[k], yi, correlation=correlation)
            .to(units)
            .magnitude
        )

    log_p = np.log10(pvals)
    popt, _ = curve_fit(antoine_eq, temp_vals_mag, log_p, p0=[1, 1e3, -1])
    a, b, c = popt

    return a, b, c, d.magnitude


def surface_tension(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
    *,
    correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
) -> types.Quantity0D:
    """Calculate surface tension of the mixture.

    Uses arithmetic pseudo-property method recommended by Hugill and van
    Welsenes (1986).

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        correlation: Correlation method ("Pitzer" or "Brock-Bird").
            Defaults to "Brock-Bird".

    Returns:
        Mixture surface tension in N/m.
    """
    yi = yi if yi is not None else fuel.y0
    xi = helpers.mass_fractions_to_mole_fractions(fuel, yi)
    # Surface tension for each compound (N/m)
    sti = components.surface_tension(fuel, temp, correlation=correlation)
    # Mixture surface tension via arithmetic mean, Poling (12-5.2)
    st = arithmetic(xi, sti)
    return st.to("N/m")


def thermal_conductivity_latini(
    fuel: "Fuel",
    temp: types.Quantity0D,
    yi: types.Quantity1D | None = None,
) -> types.Quantity0D:
    """Calculate thermal conductivity of the mixture.

    Args:
        fuel: Fuel object.
        temp: Temperature to compute property.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Thermal conductivity in W/m/K.
    """
    yi = yi if yi is not None else fuel.y0
    k_i = components.thermal_conductivity_latini(fuel, temp).to("W/(m*K)").magnitude
    k_mix = np.sum(yi.magnitude * k_i ** (-2)) ** (-0.5)
    return Units.Quantity(k_mix, "W/(m*K)")


def freeze_point_boehm(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
    *,
    alpha: float = 1.0,
) -> types.Quantity0D:
    """Calculate the freeze point of the mixture using Boehm (2022) eq. (21).

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        alpha: Scaling applied to the ideal mixing-entropy term.
            Defaults to 1.0 (classical ideal-solution entropy term).

    Returns:
        Mixture freeze point in K.

    Raises:
        RuntimeError: If the root finder fails to converge.
    """
    yi = yi if yi is not None else fuel.y0
    xi = (
        helpers.mass_fractions_to_mole_fractions(fuel, yi).to("dimensionless").magnitude
    )

    r = constants.R.to("J/(mol*K)")
    tm = fuel.comps.tm.to("K")
    ds_fus = boehm_gcm.predict("ds_fus", fuel).to("J/(mol*K)")
    dh_fus = tm * ds_fus
    dcp = (
        -0.35
        * components.specific_heat_capacity_mass_ruzicka(
            fuel, Units.Quantity(298.15, "K")
        )
        * fuel.comps.mw
    ).to("J/(mol*K)")

    # Ideal mixing entropy for each compound at its mole fraction in the
    # mixture. This is the only composition-dependent term in the model.
    xi_safe = np.clip(xi, 1e-6, 1.0 - 1e-6)
    ds_mix = (
        -r
        / xi_safe
        * ((1.0 - xi_safe) * np.log(1.0 - xi_safe) + xi_safe * np.log(xi_safe))
    )

    tm_mag = tm.magnitude
    dh_fus_mag = dh_fus.magnitude
    ds_fus_mag = ds_fus.magnitude
    dcp_mag = dcp.magnitude
    ds_mix_mag = ds_mix.magnitude

    def residual(temp: types.Array1D) -> types.Array1D:
        """Residual of Boehm et al. (2022), equation 21, for each compound.

        Args:
            temp: Candidate freeze temperature of each compound in K.

        Returns:
            Residual of equation 21 for each compound.
        """
        temp_safe = np.maximum(temp, 1.0)
        lhs = temp_safe * (
            ds_fus_mag
            + xi_safe * dcp_mag * np.log(temp_safe / tm_mag)
            + alpha * ds_mix_mag
        )
        rhs = dh_fus_mag + xi_safe * dcp_mag * (temp_safe - tm_mag)
        return lhs - rhs

    sol = root(residual, tm_mag)
    if not sol.success:
        raise RuntimeError(
            f"Freeze point root-finding failed to converge: {sol.message}"
        )

    # Mark non-physical (non-positive or non-finite) solutions as -inf, and
    # only let compounds actually present in the mixture (xi > 1e-6) set the
    # freeze point, matching the reference implementation's convention.
    t_freeze_i = np.where(np.isfinite(sol.x) & (sol.x > 0), sol.x, -np.inf)
    t_freeze = np.max(np.where(xi > 1e-6, t_freeze_i, -np.inf))

    return Units.Quantity(t_freeze, "K")


def _liaw_chiu_flash_point(
    xi: types.Array1D,
    t_flash_i: types.Array1D,
    tc: types.Array1D,
    pc: types.Array1D,
    omega: types.Array1D,
) -> float:
    """Solve the ideal Liaw-Chiu mixture flash-point criterion.

    Solves for the mixture flash point T such that
    sum(xi * pvap(T) / pvap(t_flash_i)) = 1, where pvap is evaluated with the
    Lee-Kesler correlation for each compound at its own critical properties.
    A scipy root finder is used in place of the fixed-point iteration used
    in the original reference implementation.

    Args:
        xi: Mole fractions of each compound in the mixture.
        t_flash_i: Flash point of each compound in K.
        tc: Critical temperature of each compound in K.
        pc: Critical pressure of each compound.
        omega: Acentric factor of each compound.

    Returns:
        Mixture flash point in K.

    Raises:
        RuntimeError: If the root finder fails to converge.
    """

    def _pvap_lee_kesler(
        temp: types.Array1D | float,
        tc: types.Array1D,
        pc: types.Array1D,
        omega: types.Array1D,
    ) -> types.Array1D:
        """Lee-Kesler vapor pressure for each compound.

        Copies `components.vapor_pressure(..., correlation="lee_kesler")`
        in a scipy-based root-finding context.

        Args:
            temp: Temperature in K.
            tc: Critical temperature of each compound in K.
            pc: Critical pressure of each compound.
            omega: Acentric factor of each compound.

        Returns:
            Vapor pressure (units follow whatever `pc` is given in).
        """
        tr = temp / tc
        f0 = 5.92714 - (6.09648 / tr) - 1.28862 * np.log(tr) + 0.169347 * (tr**6)
        f1 = 15.2518 - (15.6875 / tr) - 13.4721 * np.log(tr) + 0.43577 * (tr**6)
        return pc * np.exp(f0 + omega * f1)

    pvap_ref = _pvap_lee_kesler(t_flash_i, tc, pc, omega)

    def residual(temp: types.Array1D) -> types.Array1D:
        """Residual of the ideal Liaw-Chiu mixture flash-point criterion.

        Args:
            temp: Candidate mixture flash point in K.

        Returns:
            Residual of the flash-point criterion.
        """
        pvap_temp = _pvap_lee_kesler(temp[0], tc, pc, omega)
        return np.array([np.sum(xi * pvap_temp / pvap_ref) - 1.0])

    t_flash_0 = np.sum(xi * t_flash_i)
    sol = root(residual, [t_flash_0])
    if not sol.success:
        raise RuntimeError(
            f"Flash point root-finding failed to converge: {sol.message}"
        )

    return sol.x[0]


def flash_point_alqaheem(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["linear", "Liaw"] = "Liaw",
) -> types.Quantity0D:
    """Calculate mixture flash point with Alqaheem-Riazi (2017) correlation eq. (18).

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        mixing_rule: Mixing rule to use.
            Defaults to "Liaw".

    Returns:
        Mixture flash point in K.
    """
    yi = yi if yi is not None else fuel.y0
    t_flash_i = components.flash_point_alqaheem(fuel)
    if mixing_rule.casefold() == "linear".casefold():
        return np.sum(yi * t_flash_i)

    # Liaw-Chiu mixing rule: solve for the mixture flash point T such that
    # sum(xi * pvap(T) / pvap(t_flash_i)) = 1, where pvap is evaluated with the
    # Lee-Kesler correlation for each compound at its own critical properties.
    xi = (
        helpers.mass_fractions_to_mole_fractions(fuel, yi).to("dimensionless").magnitude
    )
    tc = fuel.comps.tc.to("K").magnitude
    pc = fuel.comps.pc.magnitude
    omega = fuel.comps.omega.magnitude
    t_flash_i_mag = t_flash_i.to("K").magnitude

    t_flash = _liaw_chiu_flash_point(xi, t_flash_i_mag, tc, pc, omega)
    return Units.Quantity(t_flash, "K")


def flash_point_alibakhshi(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
    *,
    mixing_rule: Literal["linear", "Liaw"] = "Liaw",
) -> types.Quantity0D:
    """Calculate mixture flash point with Alibakhshi (2017) correlation eq. (5).

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        mixing_rule: Mixing rule to use.
            Defaults to "Liaw".

    Returns:
        Mixture flash point in K.
    """
    yi = yi if yi is not None else fuel.y0
    t_flash_i = components.flash_point_alibakhshi(fuel)
    if mixing_rule.casefold() == "linear".casefold():
        return np.sum(yi * t_flash_i)

    # Liaw-Chiu mixing rule: solve for the mixture flash point T such that
    # sum(xi * pvap(T) / pvap(t_flash_i)) = 1, where pvap is evaluated with the
    # Lee-Kesler correlation for each compound at its own critical properties.
    xi = (
        helpers.mass_fractions_to_mole_fractions(fuel, yi).to("dimensionless").magnitude
    )
    tc = fuel.comps.tc.to("K").magnitude
    pc = fuel.comps.pc.magnitude
    omega = fuel.comps.omega.magnitude
    t_flash_i_mag = t_flash_i.to("K").magnitude

    t_flash = _liaw_chiu_flash_point(xi, t_flash_i_mag, tc, pc, omega)
    return Units.Quantity(t_flash, "K")


def heat_of_combustion(
    fuel: "Fuel", yi: types.Quantity1D | None = None
) -> types.Quantity0D:
    """Calculate the heat of combustion of the fuel using a Hess cycle.

    This is an engineering estimate related to ASTM D4809/D3338 heating-value
    characterization, not a simulated bomb-calorimeter test.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Heat of combustion of the mixture in J/kg.
    """
    yi = yi if yi is not None else fuel.y0
    lhv_i = components.lower_heating_value(fuel).to("J/kg")
    return np.sum(yi * lhv_i)


def yield_sooting_index(
    fuel: "Fuel", yi: types.Quantity1D | None = None
) -> types.Quantity0D:
    """Calculate the yield sooting index of the fuel mixture.

    Components without a YSI value are excluded and the mole fractions of the
    remaining components are renormalized to sum to one.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Yield sooting index of the mixture.

    Raises:
        ValueError: If yield sooting index is unavailable for any component.
    """
    yi = yi if yi is not None else fuel.y0
    xi = helpers.mass_fractions_to_mole_fractions(fuel, yi).to("dimensionless")
    ysi_i = fuel.comps.ysi
    valid = ~np.isnan(ysi_i.magnitude)
    if not np.all(valid):
        msg = (
            "Yield sooting index is unavailable for components:\n"
            f"{', '.join(np.array(fuel.comps.names)[~valid])}\n"
            "Consider adding data to `ref_compounds.csv`."
        )
        raise ValueError(msg)
    if not np.any(valid):
        return Units.Quantity(np.nan, "")
    xi_valid = xi.magnitude[valid]
    ysi_valid = ysi_i.magnitude[valid]
    return Units.Quantity(np.sum(xi_valid * ysi_valid) / np.sum(xi_valid), "")


def smoke_point(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
    correlation: Literal["NJFCP", "McEnally"] = "NJFCP",
) -> types.Quantity0D:
    """Calculate the smoke point of the fuel mixture.

    Relates the mixture yield sooting index and stoichiometric air requirement
    to the smoke point SP.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        correlation: Fit coefficients to use: "McEnally" or "NJFCP".
            Defaults to "NJFCP".

    Returns:
        Smoke point of the mixture in mm.

    Raises:
        ValueError: If `correlation` is not recognized, if yield sooting index is
            unavailable for any component, if the mixture YSI is not positive, or if
            YSI / S <= a (outside the range of the correlation).

    References:
        McEnally correlation: https://doi.org/10.6084/m9.figshare.12918980
    """
    # Coefficients (a, b) of YSI / S = a + b / SP, where SP is the smoke point in mm.
    coefficients = {
        "njfcp": (0.2493, 30.2077),  # Least-squares fit to NJFCP POSF fuel smoke points
        "mcenally": (0.4034, 29.90546),
    }
    if correlation.casefold() not in coefficients:
        msg = (
            f'Unknown smoke point correlation "{correlation}". '
            'Options: "NJFCP", "McEnally".'
        )
        raise ValueError(msg)
    a, b = coefficients[correlation.casefold()]
    yi = yi if yi is not None else fuel.y0
    xi = helpers.mass_fractions_to_mole_fractions(fuel, yi)
    ysi = yield_sooting_index(fuel, yi).magnitude
    if not ysi > 0:
        msg = f"Mixture yield sooting index must be positive to compute SP, got {ysi}."
        raise ValueError(msg)
    x_o2_air = 0.209  # Mole fraction of O2 in air
    s_i = (fuel.comps.num_c + fuel.comps.num_h / 4.0) / x_o2_air
    s = np.sum(xi.magnitude * s_i) / np.sum(xi.magnitude)
    denom = ysi / s - a
    if denom <= 0:
        msg = (
            f"YSI / S = {ysi / s:.3f} must exceed {a} to compute SP; "
            "the mixture is outside the range of the correlation."
        )
        raise ValueError(msg)
    return Units.Quantity(b / denom, "mm")


def derived_cetane_number(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
    *,
    temp_ref: types.Quantity0D = constants.T_STP,
) -> types.Quantity0D:
    """Calculate the derived cetane number (DCN) of the fuel mixture.

    Combines each component's derived cetane number with a linear, liquid volume
    fraction mixing rule. Volume fractions are derived from mass fractions using
    component liquid densities at temp_ref.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).
        temp_ref: Reference temperature for liquid densities.
            Required for converting mass fractions to volume fractions.

    Returns:
        Derived cetane number of the mixture.

    Raises:
        ValueError: If derived cetane number is unavailable for any component.
    """
    yi = yi if yi is not None else fuel.y0
    dcn_i = fuel.comps.dcn
    rho_i = components.density(fuel, temp_ref).magnitude
    valid = ~np.isnan(dcn_i.magnitude)
    if not np.all(valid):
        msg = (
            "Derived cetane number is unavailable for components:\n"
            f"{', '.join(np.array(fuel.comps.names)[~valid])}"
        )
        raise ValueError(msg)
    if not np.any(valid):
        return Units.Quantity(np.nan, "")
    tot_vol = np.sum(yi / rho_i)
    vol_frac = (yi / rho_i) / tot_vol
    return Units.Quantity(np.sum(vol_frac * dcn_i.magnitude), "")


def aromatics_content(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
) -> types.Quantity0D:
    """Calculate the aromatics content of the fuel mixture.

    Sums the liquid volume fractions (evaluated at standard temperature) of all
    components classified as aromatic by `fuel.comps.hc_type`.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Aromatics content of the mixture in volume percent.
    """
    yi = yi if yi is not None else fuel.y0
    vi = helpers.mass_fractions_to_volume_fractions(fuel, yi, constants.T_STP)
    is_aromatic = np.array(fuel.comps.hc_type) == "aromatic"
    return Units.Quantity(np.sum(vi.magnitude[is_aromatic]), "").to("percent")


def naphthalenes_content(
    fuel: "Fuel",
    yi: types.Quantity1D | None = None,
) -> types.Quantity0D:
    """Calculate the naphthalenes content of the fuel mixture.

    Sums the liquid volume fractions (evaluated at standard temperature) of all
    components identified as naphthalene or alkylated naphthalenes by
    `fuellib.rdk.mol.has_naphthalene`.

    Args:
        fuel: Fuel object.
        yi: Mass fractions of each compound in the mixture.
            Defaults to `fuel.y0` (initial mass fractions).

    Returns:
        Naphthalenes content of the mixture in volume percent.
    """
    yi = yi if yi is not None else fuel.y0
    vi = helpers.mass_fractions_to_volume_fractions(fuel, yi, constants.T_STP)
    is_naphthalene = np.array([mol.has_naphthalene(m) for m in fuel.comps.rdkit_mols])
    return Units.Quantity(np.sum(vi.magnitude[is_naphthalene]), "").to("percent")
