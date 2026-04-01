import numpy as np
from scipy.optimize import bisect
from FuelLib import fuel  # noqa: E402 — FuelLib.py must be on sys.path

# ---------------------------------------------------------------------------
# Helper: unit conversions
# ---------------------------------------------------------------------------

def K2C(T):
    """Convert Kelvin to Celsius."""
    return T - 273.15


# ---------------------------------------------------------------------------
# 1. PHYSICAL PROPERTY ROUTINES (backed by FuelLib.fuel)
# ---------------------------------------------------------------------------

def calculate_K_value(fuel_obj: fuel, i: int, T: float, P: float, Xi: np.ndarray) -> float:
    """
    Vapour-liquid equilibrium K-value for component *i* using modified Raoult's law.

    K_i = γ_i * P_i^sat(T) / P

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object for the mixture.
    :param i: Component index.
    :param T: Temperature in Kelvin.
    :param P: System pressure in Pa.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :returns: K-value for component *i*.
    :rtype: float
    """
    psat_i = fuel_obj.psat(T)[i]          # Pa  (Lee-Kesler correlation)
    gamma_i = fuel_obj.activity(Xi, T)[i] # UNIFAC activity coefficient
    return gamma_i * psat_i / P


def calculate_heat_of_vaporization(fuel_obj: fuel, T: float, Xi: np.ndarray) -> float:
    """
    Mole-fraction-averaged latent heat of vaporization for the liquid mixture.

    Uses :meth:`FuelLib.fuel.latent_heat_vaporization` (Watson correlation) and
    :attr:`FuelLib.fuel.MW` to convert from J/kg to J/mol, then averages over
    components with liquid mole fractions *Xi*.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :returns: Mixture latent heat of vaporization in J/mol.
    :rtype: float
    """
    # latent_heat_vaporization returns J/kg for each component
    Lv_i = fuel_obj.latent_heat_vaporization(T)          # (num_compounds,)  J/kg
    Lv_mol_i = Lv_i * fuel_obj.MW                         # J/mol per component
    return float(np.dot(Xi, Lv_mol_i))                    # mixture-averaged J/mol


def calculate_liquid_heat_capacity(fuel_obj: fuel, T: float, Xi: np.ndarray, use_srk: bool = False, P_atm: float = 101325.0) -> float:
    """
    Mole-fraction-averaged liquid heat capacity of the mixture.

    Uses empirical :meth:`FuelLib.fuel.Cp` (J/mol/K) by default, or SRK 
    specific heat departure calculation if use_srk=True.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :param use_srk: Boolean flag to use the SRK EoS.
    :param P_atm: System pressure required for SRK (Pa).
    :returns: Mixture liquid Cp in J/mol/K.
    :rtype: float
    """
    if use_srk:
        # specific_heat_srk returns J/kg/K. Convert to J/mol/K
        Cp_kg = fuel_obj.specific_heat_srk(T, P_atm, Xi)
        MW_mix = float(np.dot(Xi, fuel_obj.MW))
        return Cp_kg * MW_mix
    else:
        Cp_i = fuel_obj.Cp(T)           # (num_compounds,) J/mol/K
        return float(np.dot(Xi, Cp_i))


def calculate_vapor_heat_capacity(fuel_obj: fuel, T: float, Yi: np.ndarray) -> float:
    """
    Mole-fraction-averaged vapour heat capacity of the mixture.

    Uses :meth:`FuelLib.fuel.Cp` (J/mol/K) — the same group-contribution
    polynomial is applied at the given temperature.  The vapour composition
    is passed as *Yi* (mole fractions of each component in the vapour phase).

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Yi: Vapour-phase mole fractions (shape: num_compounds).
    :returns: Mixture vapour Cp in J/mol/K.
    :rtype: float
    """
    Cp_i = fuel_obj.Cp(T)           # (num_compounds,) J/mol/K
    return float(np.dot(Yi, Cp_i))


def compute_h_coeff(T_wall: float, T_room: float, L: float = 0.13) -> float:
    """
    Compute the natural-convection heat-transfer coefficient *h* (W/m²/K)
    for a vertical cylinder using the Churchill-Chu correlation (Eqs. 32–36).

    Air properties (thermal conductivity *k*, Prandtl number *Pr*, and the
    Grashof group βgρ²/μ²) are polynomial fits evaluated at the film
    temperature  T_film = (T_wall + T_room) / 2.

    **Correlations**

    Eq. 34 — thermal conductivity of air (W/m/K)::

        k = -5.696e-4 + 1.025e-4·T - 4.764e-8·T² + 1.330e-11·T³

    Eq. 35 — Prandtl number::

        Pr = 8.123e-1 - 2.948e-4·T - 9.443e-7·T²
             + 3.584e-9·T³ - 3.827e-12·T⁴ + 1.373e-15·T⁵

    Eq. 36 — Grashof group::

        ln(βgρ²/μ²) = 43.83 - 4.4065·ln(T)   [units: 1/m³/K]

    Grashof / Rayleigh / Nusselt numbers::

        Gr  = (βgρ²/μ²) · L³ · ΔT
        Ra  = Gr · Pr
        Nu  = { Eq. 32  if Ra < 1e9
              { Eq. 33  otherwise
        h   = Nu · k / L

    :param T_wall: Wall (column outer surface) temperature in Kelvin.
    :param T_room: Ambient air temperature in Kelvin.
    :param L: Characteristic length of the vertical cylinder in metres (default 0.13 m).
    :returns: Heat-transfer coefficient *h* in W/m²/K.
    :rtype: float
    """
    T_film = 0.5 * (T_wall + T_room)          # film temperature (K)
    dT     = abs(T_wall - T_room)             # temperature difference (K)

    # --- Eq. 34: thermal conductivity of air (W/m/K) ---
    k_air = (
        -5.696e-4
        + 1.025e-4  * T_film
        - 4.764e-8  * T_film**2
        + 1.330e-11 * T_film**3
    )

    # --- Eq. 35: Prandtl number ---
    Pr = (
        8.123e-1
        - 2.948e-4  * T_film
        - 9.443e-7  * T_film**2
        + 3.584e-9  * T_film**3
        - 3.827e-12 * T_film**4
        + 1.373e-15 * T_film**5
    )

    # --- Eq. 36: Grashof group βgρ²/μ² (1/m³/K) ---
    ln_Gr_group = 43.83 - 4.4065 * np.log(T_film)
    Gr_group    = np.exp(ln_Gr_group)          # 1/(m³·K)

    # --- Grashof and Rayleigh numbers ---
    Gr = Gr_group * L**3 * dT                  # dimensionless
    Ra = Gr * Pr                               # Rayleigh number

    # --- Nusselt number (Churchill-Chu) ---
    psi = (0.492 / Pr) ** (9.0 / 16.0)        # inner bracket term
    if Ra < 1.0e9:
        # Eq. 32
        Nu = 0.68 + 0.670 * Ra**(1.0/4.0) / (1.0 + psi)**(4.0/9.0)
    else:
        # Eq. 33
        Nu = (0.825 + 0.387 * Ra**(1.0/6.0) / (1.0 + psi)**(8.0/27.0)) ** 2

    h = Nu * k_air / L
    return float(h)


# ---------------------------------------------------------------------------
# 2. CORE MODELING ROUTINES
# ---------------------------------------------------------------------------

def bubble_point_residual(T, fuel_obj, P, Xi, use_srk=False):
    if use_srk:
        f_L = fuel_obj.fugacities_srk(T, P, Xi)
        K = np.zeros_like(Xi)
        mask = Xi > 1e-12
        K[mask] = f_L[mask] / (Xi[mask] * P)
    else:
        # Call activity once per T to avoid $O(N^2)$ expense
        gamma = fuel_obj.activity(Xi, T)
        # Vectorized K-value calculation (K_i = γ_i * P_sat_i / P)
        Psat = fuel_obj.psat(T)
        K = gamma * Psat / P
    return np.sum(K * Xi) - 1.0

def solve_stage1_bubble_point(
    fuel_obj: fuel,
    P: float,
    Xi: np.ndarray,
    T_lo: float = 250.0,
    T_hi: float = 650.0,
    use_srk: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Routine A, Part 1 — Bubble-Point Calculation.
    """
    # Dynamically find a valid upper bound to avoid SRK dropping to
    # vapor roots at very high temperatures causing f(T_hi) < 0.
    f_lo = bubble_point_residual(T_lo, fuel_obj, P, Xi, use_srk)
    f_hi = bubble_point_residual(T_hi, fuel_obj, P, Xi, use_srk)
    
    if f_lo * f_hi > 0:
        # If they have the same sign (likely both negative), walk T_hi or T_lo.
        if f_lo < 0:
            # Need to find a T where f > 0.
            current_T = T_lo + 10.0
            found_upper = False
            for _ in range(50):
                f_test = bubble_point_residual(current_T, fuel_obj, P, Xi, use_srk)
                if f_test > 0:
                    T_hi = current_T
                    found_upper = True
                    break
                current_T += 10.0
            if not found_upper:
                # Fallback to the upper bound if monotonic search fails
                T_hi = current_T
                
    T1 = bisect(bubble_point_residual, T_lo, T_hi, args=(fuel_obj, P, Xi, use_srk), xtol=1e-4, rtol=1e-6)
    
    # Final compositions at equilibrium T1
    if use_srk:
        f_L = fuel_obj.fugacities_srk(T1, P, Xi)
        K = np.zeros_like(Xi)
        mask = Xi > 1e-12
        K[mask] = f_L[mask] / (Xi[mask] * P)
    else:
        gamma = fuel_obj.activity(Xi, T1)
        Psat = fuel_obj.psat(T1)
        K = gamma * Psat / P
        
    vapor_comp = K * Xi
    vapor_comp /= np.sum(vapor_comp)
    
    return T1, vapor_comp


def solve_stage1_energy_balance(
    fuel_obj: fuel,
    T1: float,
    Xi: np.ndarray,
    Q1: float,
    R2: float,
    T2: float,
    use_srk: bool = False,
    P_atm: float = 101325.0,
) -> float:
    """
    Routine A, Part 2 — Energy Balance.

    Calculates the vapour-generation rate *D1* (mol/s) from the pot.

    Energy balance:
        Q1 + R2 · Cp_l · (T2 − T1) = D1 · ΔH_vap

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T1: Bubble-point temperature of the liquid (K).
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :param Q1: Heat input to the pot (W = J/s).
    :param R2: Reflux return rate from Stage 2 (mol/s).
    :param T2: Temperature of the reflux stream (K).
    :param use_srk: Boolean flag to use the SRK EoS.
    :param P_atm: System pressure required for SRK (Pa).
    :returns: Vapour generation rate *D1* in mol/s.
    :rtype: float
    """
    deltaH_vap = calculate_heat_of_vaporization(fuel_obj, T1, Xi)     # J/mol
    Cp_l       = calculate_liquid_heat_capacity(fuel_obj, T1, Xi, use_srk=use_srk, P_atm=P_atm)      # J/mol/K

    D1 = (Q1 + R2 * Cp_l * (T2 - T1)) / deltaH_vap if deltaH_vap > 0 else 0.0
    return max(0.0, D1)


def solve_rachford_rice(
    fuel_obj: fuel,
    zi: np.ndarray,
    T: float,
    P: float,
    tol: float = 1e-8,
    use_srk: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Solve the Rachford-Rice isothermal flash at known temperature *T* and
    pressure *P* for a feed of composition *zi*.
    """
    zi = np.asarray(zi, dtype=float)
    Psat = fuel_obj.psat(T)
    
    # Initial guess for Ki using feed composition
    if use_srk:
        f_L = fuel_obj.fugacities_srk(T, P, zi)
        Ki = np.zeros_like(zi)
        mask = zi > 1e-12
        Ki[mask] = f_L[mask] / (zi[mask] * P)
    else:
        gamma = fuel_obj.activity(zi, T)
        Ki = gamma * Psat / P

    # --- Check trivial cases ---
    if np.all(Ki <= 1.0):
        return 0.0, zi.copy(), zi.copy()
    if np.all(Ki >= 1.0):
        return 1.0, zi.copy(), zi.copy()

    # Successive substitution for isothermal flash (outer loop)
    V = 0.5
    xi = zi.copy()
    for _iter in range(20):
        # Rachford-Rice residual function for inner bisection
        def rr_residual(v_frac):
            return np.sum(zi * (Ki - 1.0) / (1.0 + v_frac * (Ki - 1.0)))

        # Bounds for V (Whitson & Brulé)
        V_min = 1.0 / (1.0 - np.max(Ki))
        V_max = 1.0 / (1.0 - np.min(Ki))
        # Ensure bounds are within [0, 1] for the search
        V_lo = max(V_min + 1e-10, 0.0)
        V_hi = min(V_max - 1e-10, 1.0)

        try:
            V = bisect(rr_residual, V_lo, V_hi, xtol=tol)
        except ValueError:
            # Handle cases where solution is at the boundary
            V = V_lo if abs(rr_residual(V_lo)) < abs(rr_residual(V_hi)) else V_hi

        # Update liquid composition xi and then K-values
        xi = zi / (1.0 + V * (Ki - 1.0))
        xi /= np.sum(xi)
        
        if use_srk:
            f_L = fuel_obj.fugacities_srk(T, P, xi)
            Ki_new = np.zeros_like(xi)
            mask = xi > 1e-12
            Ki_new[mask] = f_L[mask] / (xi[mask] * P)
        else:
            gamma_new = fuel_obj.activity(xi, T)
            Ki_new = gamma_new * Psat / P
        
        if np.allclose(Ki, Ki_new, rtol=1e-5):
            break
        Ki = Ki_new

    yi = Ki * xi
    yi /= np.sum(yi)
    
    return float(V), xi, yi


def solve_stage2_flash(
    fuel_obj: fuel,
    D1: float,
    T1: float,
    vapor_comp_in: np.ndarray,
    h_coeff: float,
    A_area: float,
    T_room: float,
    P: float = 101325.0,
    use_srk: bool = False,
) -> tuple:
    """
    Routine B — Adiabatic Flash with Heat Loss + Full VLE.

    Finds condenser temperature *T2* via :func:`scipy.optimize.bisect` using an
    energy balance over the condenser / rectifying stage.  Once *T2* is known,
    a full isothermal Rachford-Rice flash at *(T2, P)* is solved via
    :func:`solve_rachford_rice` to obtain the equilibrium liquid (reflux) and
    vapour compositions.

    **Step 1 — Energy balance (finds T2)**::

        Q_loss = h_coeff · A_area · (T2 − T_room)
        R2     = [D1 · Cp_v(T1) · (T1 − T2) − Q_loss] / ΔH_vap(T2)
        D2     = D1 − R2

    **Step 2 — Isothermal Rachford-Rice flash at (T2, P)**::

        Σ z_i(K_i − 1) / [1 + V(K_i − 1)] = 0    (solve for V)
        x_i = z_i / [1 + V(K_i − 1)]             (reflux / liquid)
        y_i = K_i · x_i                            (forward vapour)

    where *K_i = γ_i(x, T2) · P_i^sat(T2) / P* via FuelLib.

    The energy-balance split (R2, D2) gives the *total moles* in each phase;
    the Rachford-Rice flash gives the *compositions*.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param D1: Vapour input from Stage 1 (mol/s).
    :param T1: Temperature of incoming vapour (K).
    :param vapor_comp_in: Mole fractions of incoming vapour / feed *z_i*
                          (shape: num_compounds).
    :param h_coeff: Heat-transfer coefficient (W/m²/K).
    :param A_area: Heat-transfer area of column (m²).
    :param T_room: Ambient temperature (K).
    :param P: System pressure in Pa (default 101 325 Pa).
    :returns: ``(T2, R2, reflux_comp, D2, vapor_comp_out)``
    :rtype: tuple
    """
    Cp_v_T1 = calculate_vapor_heat_capacity(fuel_obj, T1, vapor_comp_in)  # J/mol/K

    # ------------------------------------------------------------------
    # Step 1: Energy balance — find T2
    # ------------------------------------------------------------------
    # Enthalpy balance over the condenser:
    # Vapor enters at T1. It loses heat to the ambient room.
    # Total heat we CAN lose if the column wall was at T1:
    #   Q_loss_max = h_coeff * A_area * (T1 - T_room)
    #
    # Wait, the heat capacity of the vapor is Cp_v_T1.
    # To cool the vapor by 1 K takes D1 * Cp_v_T1 Joules/sec.
    #
    # Let's compute T2 strictly from an energy balance:
    # 1. Cool vapor from T1 down to its bubble point (which is T1 since it just boiled!).
    # Actually, the vapor entering is at its dew point. Any heat removal causes condensation.
    # So T2 is strongly pinned to T1 (the dew point of the mixture).
    #
    # Heat removed = Q_loss = h_coeff * A_area * (T2 - T_room)
    # Heat supplied = (D1 - R2) * Cp(T1-T2) + R2 * ΔH_vap
    # Since Q_loss is very small (~5 W) compared to latent heat (~1000 W),
    # T2 will be very close to T1.
    # 
    # Let's approximate T2 ≈ T1 for the purpose of heat loss, then compute R2:
    Q_loss_approx = h_coeff * A_area * (T1 - T_room)
    deltaH_vap_T1 = calculate_heat_of_vaporization(fuel_obj, T1, vapor_comp_in)
    
    if deltaH_vap_T1 > 0:
        R2_approx = Q_loss_approx / deltaH_vap_T1
    else:
        R2_approx = 0.0

    R2 = min(D1, max(0.0, R2_approx))
    D2 = max(0.0, D1 - R2)
    
    # Since Q_loss is barely enough to condense a fraction of the flow,
    # the remaining vapor stays at the dew point (T1).
    T2 = T1

    # ------------------------------------------------------------------
    # Step 2: Isothermal Rachford-Rice flash at (T2, P) — full VLE
    # ------------------------------------------------------------------
    # The feed to the flash is the incoming vapour composition (z_i).
    # V_flash is the vapour fraction *within the condenser stage*.
    V_flash, reflux_comp, vapor_comp_out = solve_rachford_rice(
        fuel_obj, vapor_comp_in, T2, P, use_srk=use_srk
    )

    # The energy-balance split (R2, D2) always governs mole flows.
    # The Rachford-Rice flash governs compositions only.
    # For trivial cases (total vapour or total liquid from RR), fall back
    # to the feed composition for both phases — the flow split from the
    # energy balance is still honoured so distillation continues.
    if V_flash >= 1.0 - 1e-6 or V_flash <= 1e-6:
        reflux_comp    = vapor_comp_in.copy()
        vapor_comp_out = vapor_comp_in.copy()

    return T2, R2, reflux_comp, D2, vapor_comp_out


def solve_stage3_cstr(
    fuel_obj: fuel,
    D2: float,
    T2: float,
    vapor_comp_in: np.ndarray,
    n_air_old: float,
    T_air_old: float,
    C_glass: float,
    dt: float = 1.0,
) -> tuple:
    """
    Routine C — CSTR and D86 Temperature Calculation.

    Mixes the incoming fuel vapour with the air/glass thermal mass in the
    condenser CSTR and computes the reported D86 thermocouple temperature
    via an enthalpy balance over one time step *dt*.

    **Enthalpy balance (all quantities in Joules)**::

        (n_air · Cp_air + C_glass) · T_old              ← stored enthalpy
        + n_fuel_step · Cp_v · T2                        ← fuel vapour in
        = (n_air + n_fuel_step_total + C_glass) · T_D86  ← mixed state

    **Units**:
      - *D2* is a molar flow rate (mol/s); multiplied by *dt* (s) to give
        moles entering per step.
      - *C_glass* is J/K (thermal mass, not moles).

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param D2: Vapour forward flow from Stage 2 (mol/s).
    :param T2: Temperature of the incoming vapour (K).
    :param vapor_comp_in: Mole fractions of incoming vapour (shape: num_compounds).
    :param n_air_old: Moles of air currently in the CSTR.
    :param T_air_old: Current CSTR / thermocouple temperature (K).
    :param C_glass: Effective heat capacity of the glassware (J/K).
    :param dt: Simulation time step (s). Default 1.0.
    :returns: ``(T_D86, n_air_new)``
    :rtype: tuple[float, float]
    """
    Cp_air = 29.1  # J/(mol·K) — air near 300 K

    # Per-component vapour Cp from FuelLib (J/mol/K)
    Cp_v_i = fuel_obj.Cp(T2)   # (num_compounds,)

    # --- Moles of fuel entering the CSTR this time step ---
    # D2 is mol/s; multiply by dt to get actual moles per step.
    n_fuel_in_i = D2 * dt * vapor_comp_in          # (num_compounds,)  mol
    n_fuel_in_total = np.sum(n_fuel_in_i)           # total fuel moles this step

    # --- Enthalpy balance ---
    # Stored enthalpy of air + glass at T_old
    H_stored = (n_air_old * Cp_air + C_glass) * T_air_old
    # Enthalpy brought in by fuel vapour at T2
    H_fuel_in = np.dot(n_fuel_in_i, Cp_v_i) * T2
    # Total heat capacity after mixing
    Cp_total = n_air_old * Cp_air + C_glass + np.dot(n_fuel_in_i, Cp_v_i)

    T_D86 = (H_stored + H_fuel_in) / Cp_total if Cp_total > 0 else T_air_old

    # --- Air displacement ---
    # Air is pushed out as fuel vapour enters.  Assume the CSTR volume is
    # fixed, so n_air decreases by the same number of moles of fuel that
    # enter, clamped to zero.
    n_air_new = max(0.0, n_air_old - n_fuel_in_total)

    return T_D86, n_air_new


# ---------------------------------------------------------------------------
# 3. MAIN SIMULATION DRIVER
# ---------------------------------------------------------------------------

def run_d86_simulation(
    fuel_obj: fuel,
    Xi_initial: np.ndarray,
    volume_initial_mL: float = 100.0,
    sim_params: dict = None,
) -> dict:
    """
    Main function to run the entire D86 distillation simulation.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object for the mixture.
    :param Xi_initial: Initial liquid mole fractions (shape: num_compounds).
    :param volume_initial_mL: Initial volume of liquid fuel (mL). Standard is 100 mL.
    :param sim_params: Dictionary of simulation parameters:

        * ``dt``               — time step (s)
        * ``min_volume_W_mL``  — stopping criterion (volume of liquid left in pot, mL)
        * ``P_atm``            — system pressure (Pa)
        * ``T_room``           — ambient temperature (K)
        * ``Q1``               — heat input to pot (W)
        * ``initial_moles_air``— initial moles of air in the CSTR
        * ``h_coeff``          — heat-transfer coefficient (W/m²/K)
        * ``A_area``           — heat-transfer area (m²)
        * ``C_glass``          — glassware heat capacity (J/K)
        * ``T_bubble_lo``      — lower bound for bisect bubble-point search (K)
        * ``T_bubble_hi``      — upper bound for bisect bubble-point search (K)

    :returns: Dictionary with keys ``"time"``, ``"distillate_vol"``, ``"T_D86"``.
    :rtype: dict
    """
    if sim_params is None:
        raise ValueError("sim_params dictionary is required")

    use_srk = sim_params.get("use_srk", False)

    # --- Initialisation ---
    time   = 0.0
    Xi     = Xi_initial.copy()
    distillate_vol_collected = 0.0

    # Convert initial volume (mL) to moles
    T_init = sim_params.get("T_room", 298.15)
    Yi_initial = fuel_obj.X2Y(Xi)
    if use_srk:
        rho_init = fuel_obj.density_srk(T_init, sim_params.get("P_atm", 101325.0), Xi)
    else:
        rho_init = fuel_obj.mixture_density(Yi_initial, T_init) # kg/m^3
    mass_init_kg = volume_initial_mL * 1e-6 * rho_init      # kg
    MW_avg_init = float(np.dot(Xi, fuel_obj.MW))            # kg/mol
    W_moles = mass_init_kg / MW_avg_init                    # mol

    # Stage 2 initial state
    R2 = 0.0
    T2 = sim_params["T_room"]

    # Stage 3 initial state
    n_air = sim_params["initial_moles_air"]
    T_D86 = sim_params["T_room"]

    results = {"time": [], "distillate_vol": [], "T_D86": [], "T_D86_degC": []}

    T_lo = sim_params.get("T_bubble_lo", 350.0)
    T_hi = sim_params.get("T_bubble_hi", 650.0)

    D_out = 0.025
    D_in  = 0.0175
    L     = 0.13
    sim_params["A_area"]   = np.pi * (D_out + D_in) * 0.5 * L
    # h_coeff is updated each iteration from the current column temperature;
    # give it a sensible initial value using T_room as first wall guess.
    sim_params["h_coeff"]  = compute_h_coeff(sim_params["T_room"], sim_params["T_room"], L)
    # --- PI Controller state (dt-independent) ---
    # Target distillation rate: 4.5 mL/min (within 4-5 mL/min band)
    _target_rate    = sim_params.get("target_rate_ml_min", 4.5)
    _Kp             = sim_params.get("controller_Kp",      2.0)   # W / (mL/min)
    _Ki             = sim_params.get("controller_Ki",      0.05)  # W / (mL/min · s)
    _Q_min          = sim_params.get("Q1_min",            0.0)    # W
    _Q_max          = sim_params.get("Q1_max",         1500.0)    # W
    
    _prev_error     = 0.0
    _integral_error = 0.0  # used only if using position form
    _Q1             = sim_params.get("Q1", 15.0)
    print("Starting D86 simulation...")
    V_pot_mL = volume_initial_mL

    # --- Main Simulation Loop ---
    while V_pot_mL > sim_params.get("min_volume_W_mL", 1.0) and W_moles > 0:

        # --- Routine A: Bubble Point + Energy Balance ---
        T1, vapor_comp1 = solve_stage1_bubble_point(
            fuel_obj, sim_params["P_atm"], Xi, T_lo=T_lo, T_hi=T_hi, use_srk=use_srk
        )
        D1 = solve_stage1_energy_balance(
            fuel_obj, T1, Xi, _Q1, R2, T2, use_srk=use_srk, P_atm=sim_params["P_atm"]
        )
        
        # Update h_coeff from the current Stage-2 wall temperature (T2)
        # so natural-convection heat loss tracks the column temperature.
        sim_params["h_coeff"] = compute_h_coeff(T2, sim_params["T_room"], L)

        # --- Routine B: Flash / Condensation Stage ---
        T2, R2, reflux_comp, D2, vapor_comp2 = solve_stage2_flash(
            fuel_obj,
            D1, T1, vapor_comp1,
            sim_params["h_coeff"],
            sim_params["A_area"],
            sim_params["T_room"],
            P=sim_params["P_atm"],
            use_srk=use_srk,
        )

        # --- Routine C: CSTR & D86 Temperature ---
        T_D86, n_air = solve_stage3_cstr(
            fuel_obj,
            D2, T2, vapor_comp2,
            n_air, T_D86,
            sim_params["C_glass"],
            dt=sim_params["dt"],
        )

        # --- Update System State ---
        dt = sim_params["dt"]
        moles_distilled_step = D2 * dt  # approximate (condensation in Stage 3)

        dW = (R2 - D1) * dt
        W_moles += dW

        # Component material balance in the pot
        for idx in range(fuel_obj.num_compounds):
            d_moles = (R2 * reflux_comp[idx] - D1 * vapor_comp1[idx]) * dt
            new_moles = Xi[idx] * (W_moles - dW) + d_moles
            Xi[idx] = new_moles / W_moles if W_moles > 0 else 0.0

        # Re-normalise mole fractions
        xi_sum = np.sum(Xi)
        if xi_sum > 0:
            Xi /= xi_sum

        # Update pot volume dynamically
        Yi_pot = fuel_obj.X2Y(Xi)
        if use_srk:
            rho_pot = fuel_obj.density_srk(T1, sim_params["P_atm"], Xi)
        else:
            rho_pot = fuel_obj.mixture_density(Yi_pot, T1)  # Liquid is at bubble point T1
        MW_avg_pot = float(np.dot(Xi, fuel_obj.MW))     # kg/mol
        V_pot_mL = (W_moles * MW_avg_pot / rho_pot) * 1e6

        # Accumulate distillate volume (approx. mole → volume conversion at T_room)
        MW_avg = float(np.dot(vapor_comp2, fuel_obj.MW))        # kg/mol
        # Use T_room for volume consistency with initial 100 mL charge
        T_ref = sim_params.get("T_room", 298.15)
        if use_srk:
            rho_ref = fuel_obj.density_srk(T_ref, sim_params["P_atm"], vapor_comp2)
        else:
            rho_ref = fuel_obj.mixture_density(fuel_obj.X2Y(vapor_comp2), T_ref)  # kg/m^3
        dV_mL = moles_distilled_step * MW_avg / rho_ref * 1e6   # mL
        distillate_vol_collected += dV_mL

        # --- Velocity Form PI Distillation Rate Controller ---
        current_rate_ml_min = (dV_mL / dt) * 60.0
        error = _target_rate - current_rate_ml_min
        
        # delta_Q = Kp * (e_k - e_{k-1}) + Ki * e_k * dt
        # If boiling hasn't started (rate=0), error is large; Q1 increases.
        # Once boiling begins, error drops; Q1 stabilizes.
        delta_Q = _Kp * (error - _prev_error) + _Ki * error * dt
        _Q1 += delta_Q
        _Q1 = max(_Q_min, min(_Q_max, _Q1))
        
        sim_params["Q1"] = _Q1
        _prev_error = error

        time += dt

        # --- Store Results ---
        if int(time) % 10 == 0:
            results["time"].append(time)
            results["distillate_vol"].append(distillate_vol_collected)
            results["T_D86"].append(T_D86)
            results["T_D86_degC"].append(K2C(T_D86))
            print(f"Time: {time:.0f} s | Distilled: {distillate_vol_collected:.1f} mL"
                  f" | T_D86: {T_D86:.1f} K | T1: {T1:.1f} K")

    print("Simulation finished.")
    return results


# ---------------------------------------------------------------------------
# 4. EXAMPLE USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate a fuel object (replace "my_fuel" with your mixture data file name)
    # my_fuel = fuel("my_fuel")

    # --- Example simulation parameters ---
    simulation_parameters = {
        "dt":                1.0,       # s, time step
        "min_volume_W_mL":   1.0,       # mL, end condition (stop at 1 mL left)
        "P_atm":             101325.0,  # Pa
        "T_room":            298.15,    # K
        "Q1":                150.0,     # W, heat input to pot
        "initial_moles_air": 0.008,     # mol (reduced to match D86 neck volume)
        "h_coeff":           10.0,      # W/(m^2·K)
        "A_area":            0.01,      # m^2
        "C_glass":           0.42,      # J/K (reduced to match thermocouple thermal mass)
        "T_bubble_lo":       250.0,     # K  — lower bound for bisect
        "T_bubble_hi":       650.0,     # K  — upper bound for bisect
    }

    # Example: uniform initial composition (all compounds equally present)
    # Xi_init = np.ones(my_fuel.num_compounds) / my_fuel.num_compounds
    #
    # final_results = run_d86_simulation(
    #     my_fuel,
    #     Xi_init,
    #     volume_initial_mL=100.0,
    #     sim_params=simulation_parameters,
    # )
    #
    # import matplotlib.pyplot as plt
    # plt.plot(final_results["distillate_vol"], final_results["T_D86"])
    # plt.xlabel("Distillate collected (mL)")
    # plt.ylabel("T_D86 (K)")
    # plt.title("D86 Distillation Curve")
    # plt.show()
    pass
