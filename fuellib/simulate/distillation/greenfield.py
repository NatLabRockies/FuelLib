"""ASTM D86 distillation curve simulation.

Implements the D86 model of Greenfield et al. (1998). Each numbered equation N of
the paper is implemented by a private function ``_eq_N``, and variable names follow
the paper's notation. The model is documented in ``docs/distillation.rst``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from ... import constants
from ...correlate import components, helpers
from ...gcm import unifac
from ...utils import Units, types
from ...utils.types import quantity_field

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...fuel import Fuel

_R = constants.R.to("J/(mol*K)").magnitude
"""Gas constant in J/(mol*K)."""

# Greenfield et al. (1998) Table 3 lists b, c, and d multiplied by 10, 1e4, and 1e8;
# the values here are unscaled.
N2_COEFFS = np.array([31.15, -1.357e-2, 2.680e-5, -1.168e-8])
"""Coefficients of N2 for Greenfield et al. (1998) eq. (18) (Table 3)."""
O2_COEFFS = np.array([28.11, -3.680e-6, 1.746e-5, -1.065e-8])
"""Coefficients of O2 for Greenfield et al. (1998) eq. (18) (Table 3)."""

CP_AIR = 0.79 * N2_COEFFS + 0.21 * O2_COEFFS
"""Coefficients of air (79 mol% N2, 21 mol% O2), Greenfield et al. (1998) eq. (18)."""
HEATING_TIME_CONSTANT = 30.0
"""Time constant in s for adjusting the heating rate."""

_PLOT_COLORS = (
    "#2980B9",
    "#e74c3c",
    "#27ae60",
    "#8e44ad",
    "#f39c12",
    "#1abc9c",
    "#c0392b",
    "#16a085",
    "#d35400",
)
"""Line colors of successive D86 curves on the same axes."""
_PLOT_MARKERS = ("o", "s", "D", "^", "v", "p")
"""Symbols of the measured data of successive D86 curves on the same axes."""
_PLOT_T_STEP = 50.0
"""Spacing of the major temperature ticks and rounding of the temperature limits."""


@dataclass(frozen=True)
class D86Apparatus:
    """Apparatus and procedure parameters of the D86 model."""

    pressure: types.Quantity0D = quantity_field(101325.0, "Pa")
    """Ambient pressure P."""
    T_room: types.Quantity0D = quantity_field(298.15, "K")
    """Room temperature T_room, also the initial fuel temperature."""
    charge_volume: types.Quantity0D = quantity_field(100.0, "mL")
    """Volume of fuel charged to the flask at room temperature."""
    air_volume: types.Quantity0D = quantity_field(85.0, "cm^3")
    """Volume V of the air-filled CSTR."""
    cylinder_heat_capacity: types.Quantity0D = quantity_field(4.2, "J/K")
    """Effective heat capacity C_p,cylinder of the glass heated by the vapor."""
    column_length: types.Quantity0D = quantity_field(0.13, "m")
    """Length L of the vertical tube losing heat to the room."""
    column_diameter: types.Quantity0D = quantity_field(0.025, "m")
    """Outer diameter of the vertical tube losing heat to the room."""
    collection_rate_min: types.Quantity0D = quantity_field(4.0, "mL/min")
    """Lower limit of the condensate collection rate."""
    collection_rate_max: types.Quantity0D = quantity_field(5.0, "mL/min")
    """Upper limit of the condensate collection rate."""


@dataclass(frozen=True)
class D86Result:
    """Time histories from a D86 simulation."""

    time: types.Quantity1D
    """Elapsed time since the heater was switched on."""
    volume_recovered: types.Quantity1D
    """Volume fraction of the charge recovered as condensate at room temperature."""
    T_vapor: types.Quantity1D
    """D86 (thermometer) temperature T3."""
    T_liquid: types.Quantity1D
    """Temperature T1 of the liquid in the flask."""
    T_condenser: types.Quantity1D
    """Temperature T2 of the partial-condensation stage."""
    heat_input: types.Quantity1D
    """Heating rate Q1 supplied to the flask."""
    liquid_mole_fractions: types.Quantity2D
    """Mole fractions of each compound in the flask (shape (time, compounds))."""

    @overload
    def temperature_at(self, volume_fraction: float) -> types.Quantity0D: ...

    @overload
    def temperature_at(self, volume_fraction: types.Array1D) -> types.Quantity1D: ...

    def temperature_at(
        self, volume_fraction: float | types.Array1D
    ) -> types.Quantity0D | types.Quantity1D:
        """Interpolate the D86 temperature at the given recovered volume fractions.

        Args:
            volume_fraction: Recovered volume fraction(s) between 0 and 1.

        Returns:
            D86 temperature in K at each recovered volume fraction.
        """
        vol = self.volume_recovered.to("dimensionless").magnitude
        T = self.T_vapor.to("K").magnitude
        # Keep the first sample at each recovered volume (np.interp needs increasing x)
        keep = np.diff(vol, prepend=-np.inf) > 0
        return Units.Quantity(np.interp(volume_fraction, vol[keep], T[keep]), "K")

    def plot(
        self,
        ax: "Axes | None" = None,
        *,
        label: str | None = None,
        temperature_units: str = "K",
        data: tuple[types.Quantity1D, types.Quantity1D] | None = None,
    ) -> "Axes":
        """Plot the D86 curve (D86 temperature vs. volume percent recovered).

        Only the distillation period (recovered volume > 0) is plotted. Pass the
        returned axes to further calls to overlay several curves; each new curve
        takes the next color and the temperature limits cover all curves.

        Args:
            ax: Axes to plot on.
                Defaults to the axes of a new figure.
            label: Legend entry of the curve. The legend is shown if any curve on
                the axes has a label.
                Defaults to None.
            temperature_units: Units of the temperature axis (e.g. ``"degC"``).
                Defaults to ``"K"``.
            data: Measured recovered volume fractions and D86 temperatures, plotted
                as symbols in the color of the curve and labeled
                ``"Data: {label}"``.
                Defaults to None.

        Returns:
            Axes containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        vol = 100.0 * self.volume_recovered.to("dimensionless").magnitude
        T = self.T_vapor.to(temperature_units).magnitude
        recovering = vol > 0.0
        # Data are scatter collections, so the number of lines counts the curves
        n_curves = len(ax.lines)
        color = _PLOT_COLORS[n_curves % len(_PLOT_COLORS)]
        ax.plot(
            vol[recovering],
            T[recovering],
            color=color,
            linestyle="-",
            linewidth=4.0,
            label=label,
        )
        if data is not None:
            vol_data, T_data = data
            ax.scatter(
                100.0 * vol_data.to("dimensionless").magnitude,
                T_data.to(temperature_units).magnitude,
                facecolors=color,
                marker=_PLOT_MARKERS[n_curves % len(_PLOT_MARKERS)],
                s=75.0,
                zorder=5,
                clip_on=False,  # Show the symbols at 0% and 100% in full
                label=f"Data: {label}" if label is not None else None,
            )

        # Labels
        ax.set_xlabel("Volume recovered [%]", fontsize=18)
        ax.set_ylabel(f"Temperature [{Units.Unit(temperature_units):~P}]", fontsize=18)

        # Limits: full recovery range, temperatures rounded to the major tick step
        ax.set_xlim(0.0, 100.0)
        T_lo, T_hi = ax.dataLim.intervaly
        ax.set_ylim(
            _PLOT_T_STEP * np.floor(T_lo / _PLOT_T_STEP),
            _PLOT_T_STEP * (np.floor(T_hi / _PLOT_T_STEP) + 1.0),
        )

        # Ticks
        ax.xaxis.set_major_locator(MultipleLocator(20.0))
        ax.yaxis.set_major_locator(MultipleLocator(_PLOT_T_STEP))
        ax.tick_params(labelsize=18)

        # Grid and legend
        ax.grid(alpha=0.3)
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=18, ncol=2
            )
        return ax


# ---------------------------------------------------------------------------------
# Material and energy balances, Greenfield et al. (1998) eq. (1)-(14)
# ---------------------------------------------------------------------------------


def _eq_2(
    xW: types.Array1D,
    u: types.Array1D,
    R2: float,
    y: types.Array1D,
    D1: float,
    dt: float,
) -> types.Array1D:
    """Flask component balance, advanced by one explicit Euler step.

    Greenfield et al. (1998) eq. (2).

    Args:
        xW: Moles of each compound in the flask, x_i W.
        u: Mole fractions u_i of the reflux liquid.
        R2: Reflux rate from stage 2 to the flask in mol/s.
        y: Mole fractions y_i of the vapor leaving the flask.
        D1: Vapor rate leaving the flask in mol/s.
        dt: Time step in s.

    Returns:
        x_i W after the step.
    """
    # Clip negative amounts from an overshooting step
    return np.maximum(xW + dt * (u * R2 - y * D1), 0.0)


def _eq_5(
    R2: float,
    Cp2_l: float,
    T2: float,
    T1: float,
    D1: float,
    dH_vap1: float,
    Q1: float,
) -> float:
    """Flask energy balance.

    Greenfield et al. (1998) eq. (5).

    Args:
        R2: Reflux rate in mol/s.
        Cp2_l: Molar heat capacity C_p2,l of the reflux liquid in J/(mol*K).
        T2: Stage-2 (reflux) temperature in K.
        T1: Flask temperature in K.
        D1: Vapor rate leaving the flask in mol/s.
        dH_vap1: Molar enthalpy of vaporization |dH_vap1| of the flask vapor in
            J/mol.
        Q1: Heating rate of the flask in W.

    Returns:
        Heat rate in W that raises the flask temperature (left-hand side of
        eq. (5)).
    """
    return Q1 + R2 * Cp2_l * (T2 - T1) - D1 * dH_vap1


def _eq_6(D1: float, R2: float) -> float:
    """Stage-2 overall balance, solved for D2.

    Greenfield et al. (1998) eq. (6).

    Args:
        D1: Vapor rate entering stage 2 from the flask in mol/s.
        R2: Reflux rate returning to the flask in mol/s.

    Returns:
        D2, the vapor rate passing to the CSTR in mol/s.
    """
    return D1 - R2


def _eq_8(
    Cp1_v: float,
    T1: float,
    T2: float,
    dH_vap2: float,
    Q2: float,
    R2_over_D1: float,
) -> float:
    """Stage-2 energy balance, solved for D1 given the fraction condensed R2 / D1.

    Greenfield et al. (1998) eq. (8).

    Args:
        Cp1_v: Molar heat capacity C_p1,v of the flask vapor in J/(mol*K).
        T1: Flask temperature in K.
        T2: Stage-2 temperature in K.
        dH_vap2: Molar enthalpy of vaporization |dH_vap2| of the reflux in J/mol.
        Q2: Heat rate into stage 2 in W (negative, eq. (9)).
        R2_over_D1: Fraction of the flask vapor condensed in stage 2.

    Returns:
        D1, the vapor rate leaving the flask in mol/s.
    """
    return -Q2 / (Cp1_v * (T1 - T2) + R2_over_D1 * dH_vap2)


def _eq_9(h: float, A: float, T2: float, T_room: float) -> float:
    """Natural-convection heat loss of stage 2.

    Greenfield et al. (1998) eq. (9).

    Args:
        h: Heat transfer coefficient in W/(m^2*K) (``_heat_transfer_coeff``).
        A: Heat transfer area in m^2.
        T2: Stage-2 temperature in K.
        T_room: Room temperature in K.

    Returns:
        Q2 in W (negative, since heat is removed).
    """
    return -h * A * (T2 - T_room)


def _eq_10(
    ln_gamma: types.Array1D,
    ln_P_sat: types.Array1D,
    ln_phi: types.Array1D,
    ln_P: float,
) -> types.Array1D:
    """Natural log of the equilibrium ratios K_i from phase equilibrium.

    Greenfield et al. (1998) eq. (10) for stage 1 and eq. (11) for stage 2.

    Args:
        ln_gamma: ln(gamma_i), liquid activity coefficients.
        ln_P_sat: ln(P_i^sat / Pa), saturated vapor pressures.
        ln_phi: ln(phi_i), vapor fugacity coefficients (eq. (30)).
        ln_P: ln(P / Pa), total pressure.

    Returns:
        ln(K_i) for each compound.
    """
    return ln_gamma + ln_P_sat - ln_phi - ln_P


def _eq_12(
    v_out: types.Array1D | float,
    v: types.Array1D | float,
    D2: float,
    dt: float,
    VC_tot: float,
) -> types.Array1D | float:
    """CSTR species balance, integrated exactly over one time step.

    Greenfield et al. (1998) eq. (12), with D2 and v held constant over the step.

    Args:
        v_out: Mole fractions v_i^out in the CSTR.
        v: Mole fractions v_i of the vapor entering the CSTR from stage 2.
        D2: Vapor rate entering (and leaving) the CSTR in mol/s.
        dt: Time step in s.
        VC_tot: Total moles held in the CSTR, V C^tot.

    Returns:
        v_i^out after the step.
    """
    return v + (v_out - v) * np.exp(-D2 * dt / VC_tot)


def _eq_14(
    n_N2_O2: float,
    Cp_air_v: float,
    Cp_cylinder: float,
    T3_prime: float,
    n: types.Array1D,
    Cp_v: types.Array1D,
    T2: float,
) -> float:
    """D86 (thermometer) temperature T3 from the CSTR energy balance.

    Greenfield et al. (1998) eq. (14), the solution of eq. (13) for T3.

    Args:
        n_N2_O2: Moles of air in the CSTR, n_N2 + n_O2.
        Cp_air_v: Molar heat capacity of air in J/(mol*K) (eq. (18)).
        Cp_cylinder: Effective heat capacity of the glass cylinder in J/K.
        T3_prime: Thermometer temperature T3' at the previous step in K.
        n: Moles n_i of each compound entering the CSTR during the step.
        Cp_v: Molar heat capacity C_p,i,v of each compound in J/(mol*K).
        T2: Temperature of the entering vapor in K.

    Returns:
        T3 in K.
    """
    C_hold = n_N2_O2 * Cp_air_v + Cp_cylinder
    C_in = n @ Cp_v
    return (C_hold * T3_prime + C_in * T2) / (C_hold + C_in)


def _adjust_heating_rate(
    Q1: float, rate: float, rate_min: float, rate_max: float, dt: float
) -> float:
    """Adjust the heating rate to keep the collection rate within the D86 band.

    Raises Q1 when the rate is below the band and lowers it when the rate is above,
    with time constant ``HEATING_TIME_CONSTANT``.

    Args:
        Q1: Current heating rate in W.
        rate: Current collection rate in m^3/s.
        rate_min: Lower limit of the collection rate in m^3/s.
        rate_max: Upper limit of the collection rate in m^3/s.
        dt: Time step in s.

    Returns:
        Heating rate for the next step in W.
    """
    if rate < rate_min:
        return Q1 * np.exp(dt / HEATING_TIME_CONSTANT)
    if rate > rate_max:
        return Q1 * np.exp(-dt / HEATING_TIME_CONSTANT)
    return Q1


# ---------------------------------------------------------------------------------
# Physical property correlations, Greenfield et al. (1998) eq. (17)-(31)
# ---------------------------------------------------------------------------------


def _eq_17(T: float, dln_P_sat_dT: types.Array1D) -> types.Array1D:
    """Molar enthalpy of vaporization from the Clausius-Clapeyron equation.

    Greenfield et al. (1998) eq. (17).

    Args:
        T: Temperature in K.
        dln_P_sat_dT: d ln(P^sat) / dT in 1/K for each compound.

    Returns:
        dH_vap in J/mol for each compound.
    """
    return _R * T**2 * dln_P_sat_dT


def _eq_18(T: float, a: float, b: float, c: float, d: float) -> float:
    """Ideal-gas heat capacity polynomial.

    Greenfield et al. (1998) eq. (18).

    Args:
        T: Temperature in K.
        a: Constant coefficient in J/(mol*K).
        b: Linear coefficient in J/(mol*K^2).
        c: Quadratic coefficient in J/(mol*K^3).
        d: Cubic coefficient in J/(mol*K^4).

    Returns:
        C^0_p,v in J/(mol*K).
    """
    return a + b * T + c * T**2 + d * T**3


def _eq_19(Cp_l: np.ndarray, Tr: np.ndarray, omega: types.Array1D) -> np.ndarray:
    """Ideal-gas heat capacity from the liquid heat capacity.

    Greenfield et al. (1998) eq. (19), solved for the ideal-gas heat capacity.

    Args:
        Cp_l: Liquid molar heat capacity C_p,l in J/(mol*K).
        Tr: Reduced temperature T / Tc.
        omega: Acentric factor of each compound.

    Returns:
        C^0_p,v in J/(mol*K).
    """
    dCp = (
        _R * (0.5 + 2.2 * omega) * (3.67 + 11.64 * (1.0 - Tr) ** 4 + 0.634 / (1.0 - Tr))
    )
    # Floor at 4R, the ideal-gas heat capacity of a rigid nonlinear molecule
    return np.maximum(Cp_l - dCp, 4.0 * _R)


def _eq_22(omega: types.Array1D) -> types.Array2D:
    """Cross acentric factor, Greenfield et al. (1998) eq. (22).

    Args:
        omega: Acentric factor of each compound.

    Returns:
        omega_ij.
    """
    return 0.5 * (omega[:, None] + omega[None, :])


def _eq_23(Tc: types.Array1D) -> types.Array2D:
    """Cross critical temperature, Greenfield et al. (1998) eq. (23).

    Args:
        Tc: Critical temperature of each compound in K.

    Returns:
        Tc_ij in K.
    """
    return np.sqrt(Tc[:, None] * Tc[None, :])


def _eq_24(Vc: types.Array1D) -> types.Array2D:
    """Cross critical volume, Greenfield et al. (1998) eq. (24).

    Args:
        Vc: Critical molar volume of each compound in m^3/mol.

    Returns:
        Vc_ij in m^3/mol.
    """
    return ((Vc[:, None] ** (1 / 3) + Vc[None, :] ** (1 / 3)) / 2.0) ** 3


def _eq_25(Zc: types.Array1D) -> types.Array2D:
    """Cross critical compressibility, Greenfield et al. (1998) eq. (25).

    Args:
        Zc: Critical compressibility of each compound.

    Returns:
        Zc_ij.
    """
    return 0.5 * (Zc[:, None] + Zc[None, :])


def _eq_26(
    Zc_ij: types.Array2D, Tc_ij: types.Array2D, Vc_ij: types.Array2D
) -> types.Array2D:
    """Cross critical pressure, Greenfield et al. (1998) eq. (26).

    Args:
        Zc_ij: Cross critical compressibility (eq. (25)).
        Tc_ij: Cross critical temperature in K (eq. (23)).
        Vc_ij: Cross critical volume in m^3/mol (eq. (24)).

    Returns:
        Pc_ij in Pa.
    """
    return Zc_ij * _R * Tc_ij / Vc_ij


def _eq_27(
    T: float, Tc_ij: types.Array2D, Pc_ij: types.Array2D, omega_ij: types.Array2D
) -> types.Array2D:
    """Second virial coefficients of all pairs.

    Greenfield et al. (1998) eq. (27).

    Args:
        T: Temperature in K.
        Tc_ij: Cross critical temperature in K (eq. (23)).
        Pc_ij: Cross critical pressure in Pa (eq. (26)).
        omega_ij: Cross acentric factor (eq. (22)).

    Returns:
        B_ij in m^3/mol.
    """
    Tr = T / Tc_ij
    return _R * Tc_ij / Pc_ij * (_eq_28(Tr) + omega_ij * _eq_29(Tr))


def _eq_28(Tr: types.Array2D) -> types.Array2D:
    """Simple-fluid second virial term B^0, Greenfield et al. (1998) eq. (28).

    Args:
        Tr: Reduced temperature T / Tc_ij.

    Returns:
        B^0.
    """
    return 0.083 - 0.422 * Tr**-1.6


def _eq_29(Tr: types.Array2D) -> types.Array2D:
    """Acentric second virial term B^1, Greenfield et al. (1998) eq. (29).

    Args:
        Tr: Reduced temperature T / Tc_ij.

    Returns:
        B^1.
    """
    return 0.139 - 0.172 * Tr**-4.2


def _eq_30(
    B: types.Array2D,
    y: types.Array1D,
    P: float,
    T: float,
    P_sat: types.Array1D,
) -> types.Array1D:
    """Natural log of the vapor fugacity coefficients.

    Greenfield et al. (1998) eq. (30).

    Args:
        B: Second virial coefficients B_ij in m^3/mol (eq. (27)).
        y: Vapor mole fractions.
        P: Total pressure in Pa.
        T: Temperature in K.
        P_sat: Saturated vapor pressure P_k^sat of each compound in Pa.

    Returns:
        ln(phi_k) for each compound.
    """
    B_kk = np.diag(B)
    delta = _eq_31(B)
    # Double sum of eq. (30), simplified using sum(y) = 1 and the symmetry of delta
    mix = 2.0 * (delta @ y) - y @ delta @ y
    return (B_kk * (P - P_sat) + 0.5 * P * mix) / (_R * T)


def _eq_31(B: types.Array2D) -> types.Array2D:
    """Virial cross term delta_ij, Greenfield et al. (1998) eq. (31).

    Args:
        B: Second virial coefficients B_ij in m^3/mol (eq. (27)).

    Returns:
        delta_ij in m^3/mol.
    """
    B_ii = np.diag(B)
    return 2.0 * B - B_ii[:, None] - B_ii[None, :]


# ---------------------------------------------------------------------------------
# Natural convection from the vertical tube, Greenfield et al. (1998) eq. (32)-(36)
# ---------------------------------------------------------------------------------


def _eq_32(Ra_L: float, Pr: float) -> float:
    """Nusselt number of a vertical plate, used for Ra_L < 1e9.

    Greenfield et al. (1998) eq. (32), from Churchill and Chu (1975).

    Args:
        Ra_L: Rayleigh number.
        Pr: Prandtl number.

    Returns:
        Nu_L.
    """
    return 0.68 + 0.670 * Ra_L**0.25 / (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (
        4.0 / 9.0
    )


def _eq_33(Ra_L: float, Pr: float) -> float:
    """Nusselt number of a vertical plate, used for Ra_L >= 1e9.

    Greenfield et al. (1998) eq. (33), from Churchill and Chu (1975).

    Args:
        Ra_L: Rayleigh number.
        Pr: Prandtl number.

    Returns:
        Nu_L.
    """
    prandtl_fn = 1.0 + (0.492 / Pr) ** (9.0 / 16.0)
    return (0.825 + 0.387 * Ra_L ** (1.0 / 6.0) / prandtl_fn ** (8.0 / 27.0)) ** 2


def _eq_34(T: float) -> float:
    """Thermal conductivity k of air, Greenfield et al. (1998) eq. (34).

    Args:
        T: Film temperature in K.

    Returns:
        k in W/(m*K).
    """
    return -5.696e-4 + 1.025e-4 * T - 4.764e-8 * T**2 + 1.330e-11 * T**3


def _eq_35(T: float) -> float:
    """Prandtl number Pr of air, Greenfield et al. (1998) eq. (35).

    Args:
        T: Film temperature in K.

    Returns:
        Pr.
    """
    return (
        8.123e-1
        - 2.948e-4 * T
        - 9.443e-7 * T**2
        + 3.584e-9 * T**3
        - 3.827e-12 * T**4
        + 1.373e-15 * T**5
    )


def _eq_36(T: float) -> float:
    """Grashof coefficient beta g rho^2 / mu^2 of air.

    Greenfield et al. (1998) eq. (36).

    Args:
        T: Film temperature in K.

    Returns:
        beta g rho^2 / mu^2 in 1/(K*m^3).
    """
    return np.exp(43.83 - 4.4065 * np.log(T))


def _heat_transfer_coeff(T_wall: float, T_ambient: float, L: float) -> float:
    """Natural-convection heat transfer coefficient h of a vertical cylinder.

    Greenfield et al. (1998) eq. (32)-(36), with the air properties evaluated at the
    film temperature.

    Args:
        T_wall: Wall temperature in K.
        T_ambient: Ambient air temperature in K.
        L: Height of the cylinder in m.

    Returns:
        h in W/(m^2*K).
    """
    dT = T_wall - T_ambient
    if dT <= 0.0:
        return 0.0
    T_film = 0.5 * (T_wall + T_ambient)
    k = _eq_34(T_film)
    Pr = _eq_35(T_film)
    Gr = _eq_36(T_film) * L**3 * dT
    Ra_L = Gr * Pr
    Nu_L = _eq_32(Ra_L, Pr) if Ra_L < 1e9 else _eq_33(Ra_L, Pr)
    return Nu_L * k / L


# ---------------------------------------------------------------------------------
# Numerical methods
# ---------------------------------------------------------------------------------


def _rachford_rice(z: types.Array1D, K: types.Array1D) -> float:
    """Solve the Rachford-Rice equation for the vapor fraction of a flash.

    Rachford and Rice (1952).

    Args:
        z: Feed mole fractions.
        K: Equilibrium ratios K_i (eq. (10)).

    Returns:
        Molar vapor fraction V, clipped to [0, 1] outside the two-phase region.
    """

    def residual(V: float) -> float:
        """Rachford-Rice residual.

        Args:
            V: Molar vapor fraction.

        Returns:
            Residual of the Rachford-Rice equation.
        """
        return np.sum(((K - 1) * z) / ((K - 1) * V + 1.0))

    if residual(0.0) <= 0.0:
        return 0.0
    if residual(1.0) >= 0.0:
        return 1.0
    return brentq(residual, 0.0, 1.0, xtol=1e-14)


def _bubble_point_newton_step(
    S: float, y: types.Array1D, dln_P_sat_dT: types.Array1D
) -> float:
    """Newton step in temperature of the bubble-point iteration.

    Solves Greenfield et al. (1998) eq. (10) with the vapor mole fractions summing to
    one (eq. (3)), neglecting the temperature dependence of the activity and
    fugacity coefficients.

    Args:
        S: Sum of K_i x_i at the current temperature.
        y: Normalized vapor mole fractions.
        dln_P_sat_dT: d ln(P_i^sat) / dT in 1/K.

    Returns:
        Temperature correction in K.
    """
    return -np.log(S) / (y @ dln_P_sat_dT)


class Thermo:
    """Pure-component properties and vapor-liquid equilibrium on NumPy arrays."""

    def __init__(
        self,
        fuel: "Fuel",
        P: float,
        *,
        ideal_solution: bool,
        ideal_vapor: bool,
    ) -> None:
        """Tabulate pure-component properties and set up the VLE models.

        Args:
            fuel: Fuel object.
            P: Pressure in Pa.
            ideal_solution: Whether to use unit activity coefficients.
            ideal_vapor: Whether to use unit fugacity coefficients.
        """
        self.P: float = P
        self.ln_P: Any = np.log(P)
        self.N: int = fuel.num_compounds
        self.ideal_solution: bool = ideal_solution
        self.ideal_vapor: bool = ideal_vapor

        Tc = fuel.Tc.to("K").magnitude
        Pc = fuel.Pc.to("Pa").magnitude
        omega = fuel.omega.to("dimensionless").magnitude
        Vc = fuel.Vc.to("m^3/mol").magnitude

        # Tabulate the temperature-dependent properties once and interpolate them
        # with cubic splines
        T_grid = np.arange(200.0, fuel.Tb.to("K").magnitude.max() + 250.0, 1.0)
        T_col = Units.Quantity(T_grid[:, None], "K")
        ln_P_sat = np.log(
            components.saturated_vapor_pressure(fuel, T_col).to("Pa").magnitude
        )
        Cp_l = (
            components
            .molar_specific_heat_capacity(fuel, T_col)
            .to("J/(mol*K)")
            .magnitude
        )
        # Cap Tr below the singularity of eq. (19) at Tr = 1
        Tr = np.minimum(T_grid[:, None] / Tc, 0.95)
        Cp_v = _eq_19(Cp_l, Tr, omega)
        self._ln_P_sat = CubicSpline(T_grid, ln_P_sat, axis=0)
        self._dln_P_sat_dT = self._ln_P_sat.derivative()
        self._Cp_l = CubicSpline(T_grid, Cp_l, axis=0)
        self._Cp_v = CubicSpline(T_grid, Cp_v, axis=0)

        # Cross critical properties for the second virial coefficients, eq. (22)-(26)
        Zc = Pc * Vc / (_R * Tc)
        self._omega_ij = _eq_22(omega)
        self._Tc_ij = _eq_23(Tc)
        Vc_ij = _eq_24(Vc)
        Zc_ij = _eq_25(Zc)
        self._Pc_ij = _eq_26(Zc_ij, self._Tc_ij, Vc_ij)

        self._unifac = unifac.parameters(fuel)

    def ln_P_sat(self, T: float) -> types.Array1D:
        """Natural log of the saturated vapor pressure (Lee and Kesler, 1975).

        Args:
            T: Temperature in K.

        Returns:
            ln(P^sat / Pa) for each compound.
        """
        return self._ln_P_sat(T)

    def dH_vap(self, T: float) -> types.Array1D:
        """Molar enthalpy of vaporization, Greenfield et al. (1998) eq. (17).

        Args:
            T: Temperature in K.

        Returns:
            dH_vap in J/mol for each compound.
        """
        return _eq_17(T, self._dln_P_sat_dT(T))

    def Cp_l(self, T: float) -> types.Array1D:
        """Liquid molar heat capacity C_p,l (Nielsen, 1998).

        Args:
            T: Temperature in K.

        Returns:
            C_p,l in J/(mol*K) for each compound.
        """
        return self._Cp_l(T)

    def Cp_v(self, T: float) -> types.Array1D:
        """Ideal-gas molar heat capacity, Greenfield et al. (1998) eq. (19).

        Args:
            T: Temperature in K.

        Returns:
            C^0_p,v in J/(mol*K) for each compound.
        """
        return self._Cp_v(T)

    def ln_gamma(self, x: types.Array1D, T: float) -> types.Array1D:
        """Natural log of the liquid activity coefficients.

        UNIFAC 2.0 (Hayer et al., 2025).

        Args:
            x: Liquid mole fractions.
            T: Temperature in K.

        Returns:
            ln(gamma_i) for each compound.
        """
        if self.ideal_solution:
            return np.zeros(self.N)
        return unifac.ln_gamma_x(x, T, *self._unifac)

    def ln_phi(
        self, y: types.Array1D, T: float, ln_P_sat: types.Array1D
    ) -> types.Array1D:
        """Natural log of the vapor fugacity coefficients.

        Greenfield et al. (1998) eq. (27)-(31).

        Args:
            y: Vapor mole fractions.
            T: Temperature in K.
            ln_P_sat: Natural log of the saturated vapor pressures in Pa at T.

        Returns:
            ln(phi_i / phi_i^sat) for each compound.
        """
        if self.ideal_vapor:
            return np.zeros(self.N)
        B = _eq_27(T, self._Tc_ij, self._Pc_ij, self._omega_ij)
        return _eq_30(B, y, self.P, T, np.exp(ln_P_sat))

    def ln_K(self, x: types.Array1D, y: types.Array1D, T: float) -> types.Array1D:
        """Natural log of the equilibrium ratios, Greenfield et al. (1998) eq. (10).

        Args:
            x: Liquid mole fractions.
            y: Vapor mole fractions.
            T: Temperature in K.

        Returns:
            ln(K_i) for each compound.
        """
        ln_P_sat = self.ln_P_sat(T)
        return _eq_10(
            self.ln_gamma(x, T), ln_P_sat, self.ln_phi(y, T, ln_P_sat), self.ln_P
        )

    def bubble_point(self, x: types.Array1D, T: float) -> tuple[float, types.Array1D]:
        """Bubble temperature of a liquid, Greenfield et al. (1998) eq. (10).

        Args:
            x: Liquid mole fractions.
            T: Initial guess of the bubble temperature in K.

        Returns:
            Bubble temperature in K and the equilibrium vapor mole fractions.

        Raises:
            RuntimeError: If the iteration does not converge.
        """
        y = x
        for _ in range(100):
            Kx = x * np.exp(self.ln_K(x, y, T))
            S = Kx.sum()
            y = Kx / S
            dT = _bubble_point_newton_step(S, y, self._dln_P_sat_dT(T))
            # Limit the step to 20 K for robustness
            T += float(np.clip(dT, -20.0, 20.0))
            if abs(dT) < 1e-8:
                return T, y
        raise RuntimeError("Bubble point iteration did not converge.")

    def flash(
        self, y: types.Array1D, T2: float
    ) -> tuple[float, types.Array1D, types.Array1D]:
        """Isothermal flash of the flask vapor in stage 2.

        Greenfield et al. (1998) eq. (7) and (11), solved by successive substitution
        with the Rachford-Rice equation.

        Args:
            y: Mole fractions of the flask vapor fed to stage 2.
            T2: Stage-2 temperature in K.

        Returns:
            Fraction condensed R2 / D1, reflux mole fractions u, and stage-2 vapor
            mole fractions v.

        Raises:
            RuntimeError: If the successive substitution does not converge.
        """
        ln_K = self.ln_K(y, y, T2)
        for _ in range(200):
            K = np.exp(ln_K)
            D2_over_D1 = _rachford_rice(y, K)
            u = y / (1.0 + D2_over_D1 * (K - 1.0))
            u /= u.sum()
            v = K * u
            v /= v.sum()
            ln_K_new = self.ln_K(u, v, T2)
            if np.max(np.abs(ln_K_new - ln_K)) < 1e-10:
                return 1.0 - D2_over_D1, u, v
            ln_K = ln_K_new
        raise RuntimeError("Flash iteration did not converge.")


def distill(
    fuel: "Fuel",
    Yi: types.Quantity1D | None = None,
    *,
    apparatus: D86Apparatus | None = None,
    ideal_solution: bool = False,
    ideal_vapor: bool = False,
    air_cstr: bool = True,
    max_time_step: types.Quantity0D = Units.Quantity(2.0, "s"),
    residue_fraction: float = 0.01,
) -> D86Result:
    """Simulate the ASTM D86 distillation curve with the Greenfield et al. model.

    The model and the numerical method are documented in ``docs/distillation.rst``.

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the charge.
            Defaults to `fuel.Y_0` (initial mass fractions).
        apparatus: Apparatus and procedure parameters.
            Defaults to `D86Apparatus()`.
        ideal_solution: Whether to neglect liquid-phase nonideality (gamma_i = 1).
            Defaults to False.
        ideal_vapor: Whether to neglect vapor-phase nonideality (phi_i = 1).
            Defaults to False.
        air_cstr: Whether to include the air-filled CSTR, Greenfield et al. (1998)
            eq. (12)-(14). If False, the D86 temperature is the stage-2
            temperature (the "partial model").
            Defaults to True.
        max_time_step: Largest time step of the integration.
            Defaults to 2 s.
        residue_fraction: Fraction of the initial moles left in the flask at which
            the simulation stops.
            Defaults to 0.01.

    Returns:
        Time histories of the simulated D86 test.

    Raises:
        ValueError: If the fuel boils below room temperature.
        RuntimeError: If the simulation fails to reach the end condition.
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    app = apparatus if apparatus is not None else D86Apparatus()
    # Extract apparatus/simulation parameters and convert to SI units.
    P = app.pressure.to("Pa").magnitude
    T_room = app.T_room.to("K").magnitude
    L = app.column_length.to("m").magnitude
    # Heat transfer area A of eq. (9): outer surface of the vertical tube
    A = np.pi * app.column_diameter.to("m").magnitude * L
    Cp_cylinder = app.cylinder_heat_capacity.to("J/K").magnitude
    rate_min = app.collection_rate_min.to("m^3/s").magnitude
    rate_max = app.collection_rate_max.to("m^3/s").magnitude
    dt_max = max_time_step.to("s").magnitude

    # Initialize the thermodynamic model for the fuel at the given pressure.
    thermo = Thermo(fuel, P, ideal_solution=ideal_solution, ideal_vapor=ideal_vapor)

    def Q2_at(T2: float) -> float:
        """Heat rate Q2 into stage 2, Greenfield et al. (1998) eq. (9).

        Args:
            T2: Stage-2 temperature in K.

        Returns:
            Q2 in W (negative, since heat is removed).
        """
        return _eq_9(_heat_transfer_coeff(T2, T_room, L), A, T2, T_room)

    # Initial charge x_i W from the liquid molar volumes at room temperature,
    # assuming ideal mixing
    v_L = components.molar_liquid_volume(fuel, app.T_room).to("m^3/mol").magnitude
    x0 = helpers.mass_fractions_to_mole_fractions(fuel, Yi).to("").magnitude
    V0 = app.charge_volume.to("m^3").magnitude
    xW = x0 * V0 / (x0 @ v_L)
    W0 = xW.sum()

    # Initial bubble point of the charge (eq. (10))
    T1_bubble, y = thermo.bubble_point(x0, float(x0 @ fuel.Tb.to("K").magnitude))
    if T1_bubble <= T_room:
        msg = (
            f"Initial bubble point ({T1_bubble:.1f} K) is not above room temperature "
            f"({T_room:.1f} K); the D86 model does not apply."
        )
        raise ValueError(msg)

    # Initial heating rate: vaporize the middle of the collection-rate band, plus a
    # 25% margin
    rate_target = 0.5 * (rate_min + rate_max)
    Q1 = 1.25 * rate_target / (x0 @ v_L) * (y @ thermo.dH_vap(T1_bubble))

    records: list[tuple] = []

    def record(
        t: float, vol: float, T3: float, T1: float, T2: float, x: types.Array1D
    ) -> None:
        """Store the state at the current time."""
        records.append((t, vol, T3, T1, T2, Q1, x))

    # Heat the liquid from room temperature to its bubble point (eq. (5) with
    # D1 = R2 = 0)
    t = 0.0
    T1 = T_room
    record(t, 0.0, T_room, T1, T_room, x0)
    while T1 < T1_bubble:
        W_Cp1_l = W0 * (x0 @ thermo.Cp_l(T1))
        dt = min(dt_max, (T1_bubble - T1) * W_Cp1_l / Q1)
        T1 = min(T1 + Q1 * dt / W_Cp1_l, T1_bubble)
        t += dt
        record(t, 0.0, T_room, T1, T_room, x0)

    # Distill
    # V C^tot of eq. (12): ideal-gas moles in the CSTR volume at T_room and P
    VC_tot = P * app.air_volume.to("m^3").magnitude / (_R * T_room)
    v_out = np.zeros_like(xW)  # fuel mole fractions in the CSTR
    v_out_air = 1.0  # air mole fraction in the CSTR
    collected = np.zeros_like(xW)
    T3 = T_room
    D1_prev = 0.0
    while xW.sum() > residue_fraction * W0:
        if t > 1e5:
            raise RuntimeError("D86 simulation did not reach the end condition.")
        W = xW.sum()
        x = xW / W
        # Adaptive step: at most 2% of the flask contents per step
        dt = dt_max if D1_prev == 0.0 else min(dt_max, 0.02 * W / D1_prev)
        Cp1_l = x @ thermo.Cp_l(T1)
        dH_vap1 = y @ thermo.dH_vap(T1)
        Cp1_v = y @ thermo.Cp_v(T1)
        # Largest vapor rate the heater can supply
        D1_max = Q1 / dH_vap1
        # Lowest stage-2 temperature: the bubble point of the flask vapor, reached
        # when all of it condenses
        T2_min, _ = thermo.bubble_point(y, T1)

        if T1 - T2_min < 1e-3:
            # Pure or constant-boiling liquid: the vapor condenses at T2 = T1, and the
            # reflux is the vapor rate that stage 2 can condense (eq. (8))
            T2, v = T1, y
            D1 = D1_max
            R2 = min(_eq_8(Cp1_v, T1, T1, dH_vap1, Q2_at(T1), 1.0), D1)
            xW_new = _eq_2(xW, y, R2, y, D1, dt)
            T1_new, y_new = thermo.bubble_point(xW_new / xW_new.sum(), T1)
        else:
            Cp2_l = y @ thermo.Cp_l(T2_min)
            # Vapor rate that stage 2 condenses completely at T2_min (eq. (8))
            D1_total = _eq_8(
                Cp1_v, T1, T2_min, y @ thermo.dH_vap(T2_min), Q2_at(T2_min), 1.0
            )
            W_Cp1_l_dT1dt = _eq_5(D1_total, Cp2_l, T2_min, T1, D1_total, dH_vap1, Q1)
            if W_Cp1_l_dT1dt <= 0.0:
                # Total reflux: stage 2 condenses everything the heater can boil, so
                # nothing reaches the CSTR and the flask is unchanged (eq. (5) at
                # steady state with R2 = D1)
                T2, v = T2_min, y
                D1 = R2 = Q1 / (dH_vap1 + Cp2_l * (T1 - T2_min))
                xW_new, T1_new, y_new = xW, T1, y
            else:
                state: dict = {}

                def residual(T2: float) -> float:
                    """Residual of the flask energy balance for a trial T2.

                    Greenfield et al. (1998) eq. (5), with D1 and R2 from eq. (7),
                    (8), and (11) and the new flask state from eq. (2) and (10).

                    Args:
                        T2: Trial stage-2 temperature in K.

                    Returns:
                        Residual of the stage-1 energy balance in W.
                    """
                    R2_over_D1, u, v = thermo.flash(y, T2)
                    D1 = _eq_8(
                        Cp1_v, T1, T2, u @ thermo.dH_vap(T2), Q2_at(T2), R2_over_D1
                    )
                    R2 = R2_over_D1 * D1
                    W_Cp1_l_dT1dt = _eq_5(
                        R2, u @ thermo.Cp_l(T2), T2, T1, D1, dH_vap1, Q1
                    )
                    if D1 > D1_max:
                        # More vapor than the heater can supply
                        return W_Cp1_l_dT1dt
                    xW_new = _eq_2(xW, u, R2, y, D1, dt)
                    T1_new, y_new = thermo.bubble_point(xW_new / xW_new.sum(), T1)
                    state.update(D1=D1, R2=R2, v=v, xW=xW_new, T1=T1_new, y=y_new)
                    # Compare with the temperature change of the new bubble point
                    return W_Cp1_l_dT1dt - W * Cp1_l * (T1_new - T1) / dt

                T2 = brentq(residual, T2_min, T1 - 1e-6 * (T1 - T2_min), xtol=1e-9)
                residual(T2)
                D1, R2, v = state["D1"], state["R2"], state["v"]
                xW_new, T1_new, y_new = state["xW"], state["T1"], state["y"]

        D2 = _eq_6(D1, R2)
        if air_cstr:
            # Displacement of the CSTR contents by the distillate (eq. (12))
            v_out_new = _eq_12(v_out, v, D2, dt, VC_tot)
            v_out_air = float(_eq_12(v_out_air, 0.0, D2, dt, VC_tot))
            # Condensate collected: fuel entering the CSTR minus fuel held up in it
            collected += D2 * v * dt - VC_tot * (v_out_new - v_out)
            v_out = v_out_new
            # Heating of the air and glass by the incoming vapor (eq. (14))
            T3 = _eq_14(
                VC_tot * v_out_air,
                _eq_18(T3, *CP_AIR),
                Cp_cylinder,
                T3,
                v * D2 * dt,
                thermo.Cp_v(T2),
                T2,
            )
        else:
            # Without the CSTR, the thermometer reads the stage-2 temperature
            collected += D2 * v * dt
            T3 = T2

        xW, T1, y = xW_new, T1_new, y_new
        D1_prev = D1
        t += dt
        # Recovered volume at room temperature, assuming ideal mixing
        record(t, (collected @ v_L) / V0, T3, T1, T2, xW / xW.sum())

        # Keep the collection rate within the mandated band
        rate = D2 * (v @ v_L)
        Q1 = _adjust_heating_rate(Q1, rate, rate_min, rate_max, dt)

    t_arr, vol, T_vap, T_l, T_c, Q, X = (np.array(a) for a in zip(*records))
    return D86Result(
        time=Units.Quantity(t_arr, "s"),
        volume_recovered=Units.Quantity(vol, "dimensionless"),
        T_vapor=Units.Quantity(T_vap, "K"),
        T_liquid=Units.Quantity(T_l, "K"),
        T_condenser=Units.Quantity(T_c, "K"),
        heat_input=Units.Quantity(Q, "W"),
        liquid_mole_fractions=Units.Quantity(X, "dimensionless"),
    )


__all__ = ["D86Apparatus", "D86Result", "distill"]
