"""ASTM D86 distillation curve simulation.

Implements the macroscopic model of the ASTM D86 procedure from Greenfield, Lavoie,
Smith, and Curtis, "Macroscopic Model of the D86 Fuel Volatility Procedure", SAE
Technical Paper 982724 (1998).

The D86 apparatus is represented by three unit operations in series:

1. the distillation flask, an unsteady-state still pot whose liquid is held at its
   bubble point (equations 1-5 and 10),
2. an equilibrium partial-condensation stage representing the vertical neck, which
   loses heat to the room by natural convection and returns reflux to the flask
   (equations 6-9 and 11), and
3. an air-filled continuous stirred-tank reactor (CSTR) representing the air
   initially in the apparatus and the heat capacity of the glass, whose temperature
   is reported as the D86 (thermometer) temperature (equations 12-14).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .. import constants
from ..gcm import unifac
from ..utils import Units, types
from . import components, helpers

if TYPE_CHECKING:
    from ..fuel import Fuel

_R = constants.gas_constant.to("J/(mol*K)").magnitude
"""Gas constant in J/(mol*K)."""

_CP_AIR = 0.79 * np.array([31.15, -1.357e-2, 2.680e-5, -1.168e-8]) + 0.21 * np.array([
    28.11,
    -3.680e-6,
    1.746e-5,
    -1.065e-8,
])
"""Ideal-gas heat capacity coefficients of air (79% N2, 21% O2 by mole), with
Cp = a + b*T + c*T^2 + d*T^3 in J/(mol*K) (Greenfield et al., 1998, Table 3)."""

_HEATING_TIME_CONSTANT = 30.0
"""Time constant in s for adjusting the heating rate to the target collection rate."""


def _quantity_field(value: float, units: str) -> types.Quantity0D:
    """Dataclass field with a default quantity.

    Args:
        value: Default magnitude.
        units: Units of the default value.

    Returns:
        Dataclass field whose default is a new quantity.
    """
    return field(default_factory=lambda: Units.Quantity(value, units))


@dataclass(frozen=True)
class D86Apparatus:
    """Apparatus and procedure parameters of the D86 model.

    Defaults follow Greenfield et al. (1998) and ASTM D86.
    """

    pressure: types.Quantity0D = _quantity_field(101325.0, "Pa")
    """Ambient pressure."""
    T_room: types.Quantity0D = _quantity_field(298.15, "K")
    """Room temperature, also used as the initial fuel temperature."""
    charge_volume: types.Quantity0D = _quantity_field(100.0, "mL")
    """Volume of fuel charged to the flask at room temperature."""
    air_volume: types.Quantity0D = _quantity_field(85.0, "cm^3")
    """Volume of the air-filled CSTR (vertical tube and flask above the liquid)."""
    cylinder_heat_capacity: types.Quantity0D = _quantity_field(4.2, "J/K")
    """Effective heat capacity of the glass heated by the vapor."""
    column_length: types.Quantity0D = _quantity_field(0.13, "m")
    """Length of the vertical tube losing heat to the room."""
    column_diameter: types.Quantity0D = _quantity_field(0.025, "m")
    """Outer diameter of the vertical tube losing heat to the room."""
    collection_rate_min: types.Quantity0D = _quantity_field(4.0, "mL/min")
    """Lower limit of the condensate collection rate."""
    collection_rate_max: types.Quantity0D = _quantity_field(5.0, "mL/min")
    """Upper limit of the condensate collection rate."""


@dataclass(frozen=True)
class D86Result:
    """Time histories from a D86 simulation."""

    time: types.Quantity1D
    """Elapsed time since the heater was switched on."""
    volume_recovered: types.Quantity1D
    """Volume fraction of the charge recovered as condensate at room temperature."""
    T_vapor: types.Quantity1D
    """D86 (thermometer) temperature, i.e. the CSTR temperature (equation 14)."""
    T_liquid: types.Quantity1D
    """Temperature of the liquid in the flask (stage 1)."""
    T_condenser: types.Quantity1D
    """Temperature of the partial-condensation stage (stage 2)."""
    heat_input: types.Quantity1D
    """Heating rate supplied to the flask."""
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


def _heat_transfer_coefficient(T_wall: float, T_room: float, length: float) -> float:
    """Natural-convection heat transfer coefficient of a vertical cylinder.

    Uses the Churchill-Chu correlations with air properties evaluated at the film
    temperature (Greenfield et al., 1998, equations 32-36).

    Args:
        T_wall: Wall temperature in K.
        T_room: Ambient air temperature in K.
        length: Height of the cylinder in m.

    Returns:
        Heat transfer coefficient in W/(m^2*K).
    """
    dT = T_wall - T_room
    if dT <= 0.0:
        return 0.0
    T = 0.5 * (T_wall + T_room)
    k = -5.696e-4 + 1.025e-4 * T - 4.764e-8 * T**2 + 1.330e-11 * T**3
    Pr = (
        8.123e-1
        - 2.948e-4 * T
        - 9.443e-7 * T**2
        + 3.584e-9 * T**3
        - 3.827e-12 * T**4
        + 1.373e-15 * T**5
    )
    gr_coeff = np.exp(43.83 - 4.4065 * np.log(T))  # beta*g*rho^2/mu^2 in 1/(K*m^3)
    Ra = gr_coeff * length**3 * dT * Pr
    prandtl_fn = 1.0 + (0.492 / Pr) ** (9.0 / 16.0)
    if Ra < 1e9:
        Nu = 0.68 + 0.670 * Ra**0.25 / prandtl_fn ** (4.0 / 9.0)
    else:
        Nu = (0.825 + 0.387 * Ra ** (1.0 / 6.0) / prandtl_fn ** (8.0 / 27.0)) ** 2
    return Nu * k / length


def _rachford_rice(z: types.Array1D, K: types.Array1D) -> float:
    """Solve the Rachford-Rice equation for the vapor fraction of a flash.

    Args:
        z: Feed mole fractions.
        K: Equilibrium ratios y_i / x_i.

    Returns:
        Molar vapor fraction, clipped to [0, 1] outside the two-phase region.
    """
    Km1 = K - 1.0

    def residual(b: float) -> float:
        """Rachford-Rice residual.

        Args:
            b: Molar vapor fraction.

        Returns:
            Residual of the Rachford-Rice equation.
        """
        return np.sum(z * Km1 / (1.0 + b * Km1))

    if residual(0.0) <= 0.0:
        return 0.0
    if residual(1.0) >= 0.0:
        return 1.0
    return brentq(residual, 0.0, 1.0, xtol=1e-14)


class _Thermo:
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
        self.P = P
        self.ln_P = np.log(P)
        self.N = fuel.num_compounds
        self.ideal_solution = ideal_solution
        self.ideal_vapor = ideal_vapor

        Tc = fuel.Tc.to("K").magnitude
        Pc = fuel.Pc.to("Pa").magnitude
        omega = fuel.omega.to("dimensionless").magnitude
        Vc = fuel.Vc.to("m^3/mol").magnitude

        # Tabulate FuelLib's temperature-dependent correlations once and interpolate
        T_grid = np.arange(200.0, fuel.Tb.to("K").magnitude.max() + 250.0, 1.0)
        T_col = Units.Quantity(T_grid[:, None], "K")
        ln_psat = np.log(
            components.saturated_vapor_pressure(fuel, T_col).to("Pa").magnitude
        )
        cp_l = (
            components
            .molar_specific_heat_capacity(fuel, T_col)
            .to("J/(mol*K)")
            .magnitude
        )
        # Vapor heat capacity from the liquid heat capacity (equation 19)
        Tr = np.minimum(T_grid[:, None] / Tc, 0.95)
        dcp = (
            _R
            * (0.5 + 2.2 * omega)
            * (3.67 + 11.64 * (1.0 - Tr) ** 4 + 0.634 / (1.0 - Tr))
        )
        cp_v = np.maximum(cp_l - dcp, 4.0 * _R)
        self._ln_psat = CubicSpline(T_grid, ln_psat, axis=0)
        self._dln_psat = self._ln_psat.derivative()
        self._cp_l = CubicSpline(T_grid, cp_l, axis=0)
        self._cp_v = CubicSpline(T_grid, cp_v, axis=0)

        # Mixture critical properties for second virial coefficients (eqs. 22-26)
        Zc = Pc * Vc / (_R * Tc)
        self._omega_ij = 0.5 * (omega[:, None] + omega[None, :])
        self._Tc_ij = np.sqrt(Tc[:, None] * Tc[None, :])
        Vc_ij = ((Vc[:, None] ** (1 / 3) + Vc[None, :] ** (1 / 3)) / 2.0) ** 3
        Zc_ij = 0.5 * (Zc[:, None] + Zc[None, :])
        self._Pc_ij = Zc_ij * _R * self._Tc_ij / Vc_ij

        self._unifac = unifac.parameters(fuel)

    def ln_psat(self, T: float) -> types.Array1D:
        """Natural log of the saturated vapor pressure.

        Args:
            T: Temperature in K.

        Returns:
            ln(P_sat / Pa) for each compound.
        """
        return self._ln_psat(T)

    def dH_vap(self, T: float) -> types.Array1D:
        """Molar enthalpy of vaporization from Clausius-Clapeyron (equation 17).

        Args:
            T: Temperature in K.

        Returns:
            Enthalpy of vaporization in J/mol for each compound.
        """
        return _R * T**2 * self._dln_psat(T)

    def cp_l(self, T: float) -> types.Array1D:
        """Liquid molar heat capacity.

        Args:
            T: Temperature in K.

        Returns:
            Heat capacity in J/(mol*K) for each compound.
        """
        return self._cp_l(T)

    def cp_v(self, T: float) -> types.Array1D:
        """Ideal-gas molar heat capacity.

        Args:
            T: Temperature in K.

        Returns:
            Heat capacity in J/(mol*K) for each compound.
        """
        return self._cp_v(T)

    def ln_gamma(self, x: types.Array1D, T: float) -> types.Array1D:
        """Natural log of the UNIFAC liquid activity coefficients.

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
        self, y: types.Array1D, T: float, ln_psat: types.Array1D
    ) -> types.Array1D:
        """Natural log of the vapor fugacity coefficient ratio (equations 27-31).

        Args:
            y: Vapor mole fractions.
            T: Temperature in K.
            ln_psat: Natural log of the saturated vapor pressures in Pa at T.

        Returns:
            ln(phi_i / phi_i^sat) for each compound.
        """
        if self.ideal_vapor:
            return np.zeros(self.N)
        Tr = T / self._Tc_ij
        B0 = 0.083 - 0.422 * Tr**-1.6
        B1 = 0.139 - 0.172 * Tr**-4.2
        B = _R * self._Tc_ij / self._Pc_ij * (B0 + self._omega_ij * B1)
        B_ii = np.diag(B)
        delta = 2.0 * B - B_ii[:, None] - B_ii[None, :]
        mix = 2.0 * (delta @ y) - y @ delta @ y
        return (B_ii * (self.P - np.exp(ln_psat)) + 0.5 * self.P * mix) / (_R * T)

    def ln_K(self, x: types.Array1D, y: types.Array1D, T: float) -> types.Array1D:
        """Natural log of the equilibrium ratios K_i = y_i / x_i (equations 10-11).

        Args:
            x: Liquid mole fractions.
            y: Vapor mole fractions.
            T: Temperature in K.

        Returns:
            ln(K_i) for each compound.
        """
        ln_psat = self.ln_psat(T)
        return self.ln_gamma(x, T) + ln_psat - self.ln_phi(y, T, ln_psat) - self.ln_P

    def bubble_point(self, x: types.Array1D, T: float) -> tuple[float, types.Array1D]:
        """Solve for the bubble temperature of a liquid at pressure P.

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
            dT = -np.log(S) / (y @ self._dln_psat(T))
            T += float(np.clip(dT, -20.0, 20.0))
            if abs(dT) < 1e-8:
                return T, y
        raise RuntimeError("Bubble point iteration did not converge.")

    def flash(
        self, z: types.Array1D, T: float
    ) -> tuple[float, types.Array1D, types.Array1D]:
        """Isothermal flash of a feed at temperature T and pressure P.

        Args:
            z: Feed mole fractions.
            T: Temperature in K.

        Returns:
            Molar liquid fraction, liquid mole fractions, and vapor mole fractions.

        Raises:
            RuntimeError: If the successive substitution does not converge.
        """
        ln_K = self.ln_K(z, z, T)
        for _ in range(200):
            K = np.exp(ln_K)
            beta = _rachford_rice(z, K)
            u = z / (1.0 + beta * (K - 1.0))
            u /= u.sum()
            v = K * u
            v /= v.sum()
            ln_K_new = self.ln_K(u, v, T)
            if np.max(np.abs(ln_K_new - ln_K)) < 1e-10:
                return 1.0 - beta, u, v
            ln_K = ln_K_new
        raise RuntimeError("Flash iteration did not converge.")


def distillation_curve_greenfield(
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

    Implements the three-stage macroscopic model of Greenfield et al. (1998):

    * **Stage 1 (flask):** the liquid (W moles, mole fractions x) stays at its
      bubble point T1. Material and energy balances (equations 1, 4, and 5),

        dW/dt = R2 - D1,
        d(x_i W)/dt = u_i R2 - y_i D1,
        W Cp_l dT1/dt = R2 Cp_l (T2 - T1) - D1 dH_vap + Q1,

      are advanced with a semi-implicit Euler step: the flows use the vapor y in
      equilibrium with the liquid at the start of the step, while T1 at the end of
      the step is the bubble point of the updated liquid.
    * **Stage 2 (vertical tube):** the vapor D1 is partially condensed at T2 by
      natural-convection losses, -Q2 = h A (T2 - T_room), with h from the
      Churchill-Chu correlation (equations 6-9 and 32-36). The reflux R2 (mole
      fractions u) and distillate D2 (mole fractions v) are in equilibrium at T2.
      Each step solves for the T2 that satisfies the energy balances of both
      stages.
    * **Stage 3 (air-filled CSTR):** the distillate D2 displaces the air initially in
      the apparatus (equation 12) and heats the air and the glass (equation 14);
      the CSTR temperature is the reported D86 temperature.

    Phase equilibrium uses phi_i y_i P = gamma_i x_i P_i^sat with UNIFAC activity
    coefficients and second-virial fugacity coefficients (equations 10-11 and 22-31).
    Vapor pressures (Lee-Kesler), liquid heat capacities, and densities come from
    FuelLib's component correlations; the enthalpy of vaporization follows from
    the Clausius-Clapeyron equation (equation 17), and the ideal-gas heat
    capacity follows from the liquid heat capacity with equation 19.

    The fuel is heated from room temperature to its bubble point before
    distillation begins. The heating rate is then adjusted whenever the
    condensate collection rate falls outside the 4-5 mL/min band mandated by
    ASTM D86. The simulation ends when the liquid remaining in the flask falls
    below `residue_fraction` of the initial charge (in moles).

    Differences from the original model:

    * Dissolved air in the fuel (the Shair-Prausnitz correlation) is not modeled.
    * Pure-component properties come from FuelLib's correlations rather than the
      tabulated coefficients of Greenfield et al. (1998), Table 3.

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
        air_cstr: Whether to include the air-filled CSTR. If False, the D86
            temperature is the stage-2 temperature (the "partial model").
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
    P = app.pressure.to("Pa").magnitude
    T_room = app.T_room.to("K").magnitude
    length = app.column_length.to("m").magnitude
    area = np.pi * app.column_diameter.to("m").magnitude * length
    C_cyl = app.cylinder_heat_capacity.to("J/K").magnitude
    rate_min = app.collection_rate_min.to("m^3/s").magnitude
    rate_max = app.collection_rate_max.to("m^3/s").magnitude
    dt_max = max_time_step.to("s").magnitude
    thermo = _Thermo(fuel, P, ideal_solution=ideal_solution, ideal_vapor=ideal_vapor)

    def heat_loss(T2: float) -> float:
        """Natural-convection heat loss of stage 2 (equation 9).

        Args:
            T2: Stage-2 temperature in K.

        Returns:
            Heat loss in W.
        """
        return _heat_transfer_coefficient(T2, T_room, length) * area * (T2 - T_room)

    # Initial charge in moles, from liquid molar volumes at room temperature
    v_liq = (fuel.MW / components.density(fuel, app.T_room)).to("m^3/mol").magnitude
    x0 = helpers.mass_fractions_to_mole_fractions(fuel, Yi).to("").magnitude
    V0 = app.charge_volume.to("m^3").magnitude
    n = x0 * V0 / (x0 @ v_liq)
    W0 = n.sum()

    T1, y1 = thermo.bubble_point(x0, float(x0 @ fuel.Tb.to("K").magnitude))
    if T1 <= T_room:
        msg = (
            f"Initial bubble point ({T1:.1f} K) is not above room temperature "
            f"({T_room:.1f} K); the D86 model does not apply."
        )
        raise ValueError(msg)

    # Initial heating rate: vaporize the target collection rate with some margin
    rate_target = 0.5 * (rate_min + rate_max)
    Q1 = 1.25 * rate_target / (x0 @ v_liq) * (y1 @ thermo.dH_vap(T1))

    records: list[tuple] = []

    def record(
        t: float, vol: float, T3: float, T_liq: float, T2: float, x: types.Array1D
    ) -> None:
        """Store the state at the current time."""
        records.append((t, vol, T3, T_liq, T2, Q1, x))

    # Heat the liquid from room temperature to its bubble point
    t = 0.0
    T_liq = T_room
    record(t, 0.0, T_room, T_liq, T_room, x0)
    while T_liq < T1:
        heat_capacity = W0 * (x0 @ thermo.cp_l(T_liq))
        dt = min(dt_max, (T1 - T_liq) * heat_capacity / Q1)
        T_liq = min(T_liq + Q1 * dt / heat_capacity, T1)
        t += dt
        record(t, 0.0, T_room, T_liq, T_room, x0)

    # Distill
    N_cstr = P * app.air_volume.to("m^3").magnitude / (_R * T_room)
    n_air = N_cstr if air_cstr else 0.0
    n_cstr = np.zeros_like(n)
    collected = np.zeros_like(n)
    T3 = T_room
    D1_prev = 0.0
    while n.sum() > residue_fraction * W0:
        if t > 1e5:
            raise RuntimeError("D86 simulation did not reach the end condition.")
        W = n.sum()
        x = n / W
        dt = dt_max if D1_prev == 0.0 else min(dt_max, 0.02 * W / D1_prev)
        cp1 = x @ thermo.cp_l(T1)
        dH1 = y1 @ thermo.dH_vap(T1)
        cpv1 = y1 @ thermo.cp_v(T1)
        D1_max = Q1 / dH1
        Tb_y, _ = thermo.bubble_point(y1, T1)

        if T1 - Tb_y < 1e-3:
            # Pure or constant-boiling liquid: the vapor condenses at T1
            T2, v = T1, y1
            D1 = D1_max
            R2 = min(heat_loss(T1) / dH1, D1)
            n_new = np.maximum(n + dt * y1 * (R2 - D1), 0.0)
            T1_new, y1_new = thermo.bubble_point(n_new / n_new.sum(), T1)
        else:
            # Total condensation of the vapor at its bubble point Tb_y
            D1_total = heat_loss(Tb_y) / (cpv1 * (T1 - Tb_y) + y1 @ thermo.dH_vap(Tb_y))
            if Q1 + D1_total * ((y1 @ thermo.cp_l(Tb_y)) * (Tb_y - T1) - dH1) <= 0.0:
                # Total reflux: nothing passes the condenser stage
                T2, v = Tb_y, y1
                D1 = R2 = Q1 / (dH1 + (y1 @ thermo.cp_l(Tb_y)) * (T1 - Tb_y))
                n_new, T1_new, y1_new = n, T1, y1
            else:
                state: dict = {}

                def residual(T2: float) -> float:
                    """Energy balance residual of stage 1 for a stage-2 temperature.

                    Args:
                        T2: Trial stage-2 temperature in K.

                    Returns:
                        Residual of the stage-1 energy balance in W.
                    """
                    psi, u, v = thermo.flash(y1, T2)
                    # Stage-2 energy balance (equation 8) gives D1 for this T2
                    D1 = heat_loss(T2) / (
                        cpv1 * (T1 - T2) + psi * (u @ thermo.dH_vap(T2))
                    )
                    R2 = psi * D1
                    heat = Q1 + R2 * (u @ thermo.cp_l(T2)) * (T2 - T1) - D1 * dH1
                    if D1 > D1_max:
                        # More vapor than the heater can supply
                        return heat
                    n_new = np.maximum(n + dt * (u * R2 - y1 * D1), 0.0)
                    T1_new, y1_new = thermo.bubble_point(n_new / n_new.sum(), T1)
                    state.update(D1=D1, R2=R2, v=v, n=n_new, T1=T1_new, y1=y1_new)
                    return heat - W * cp1 * (T1_new - T1) / dt

                T2 = brentq(residual, Tb_y, T1 - 1e-6 * (T1 - Tb_y), xtol=1e-9)
                residual(T2)
                D1, R2, v = state["D1"], state["R2"], state["v"]
                n_new, T1_new, y1_new = state["n"], state["T1"], state["y1"]

        D2 = D1 - R2
        if air_cstr:
            # Displacement of the CSTR contents by the distillate (equation 12)
            decay = np.exp(-D2 * dt / N_cstr)
            n_cstr_new = v * N_cstr + (n_cstr - v * N_cstr) * decay
            collected += D2 * v * dt - (n_cstr_new - n_cstr)
            n_cstr = n_cstr_new
            n_air *= decay
            # Heating of the air and glass by the incoming vapor (equation 14)
            C_hold = n_air * np.polyval(_CP_AIR[::-1], T3) + C_cyl
            C_in = D2 * dt * (v @ thermo.cp_v(T2))
            T3 = (C_hold * T3 + C_in * T2) / (C_hold + C_in)
        else:
            collected += D2 * v * dt
            T3 = T2

        n, T1, y1 = n_new, T1_new, y1_new
        D1_prev = D1
        t += dt
        record(t, (collected @ v_liq) / V0, T3, T1, T2, n / n.sum())

        # Keep the collection rate within the mandated band
        rate = D2 * (v @ v_liq)
        if rate < rate_min:
            Q1 *= np.exp(dt / _HEATING_TIME_CONSTANT)
        elif rate > rate_max:
            Q1 *= np.exp(-dt / _HEATING_TIME_CONSTANT)

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


__all__ = ["D86Apparatus", "D86Result", "distillation_curve_greenfield"]
