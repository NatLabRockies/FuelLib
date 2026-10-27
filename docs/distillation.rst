ASTM D86 Distillation Simulation
================================

The module :mod:`fuellib.simulate.distillation.greenfield` simulates the ASTM D86
distillation test with the macroscopic model of
Greenfield et al.\ :footcite:p:`greenfield_macroscopic_1998`. Given a fuel
composition, it predicts the D86 curve (thermometer temperature against the volume
fraction recovered) together with the time histories of the liquid temperature, the
heating rate, and the liquid composition in the flask.

In the code, each numbered equation :math:`N` of Greenfield et
al.\ :footcite:p:`greenfield_macroscopic_1998` is implemented by a private function
``_eq_N``, and variable names follow the notation of the paper (see
:ref:`tab-d86-nomenclature`). The equation numbers on this page are those of the
paper.

Usage
-----

.. code-block:: python

   import numpy as np
   import fuellib as fl
   from fuellib.simulate.distillation.greenfield import D86Apparatus, distill

   fuel = fl.Fuel("heptane-decane")
   result = distill(fuel)

   # D86 temperature (K) at 10%, 50%, and 90% recovered
   T10, T50, T90 = result.temperature_at(np.array([0.1, 0.5, 0.9])).magnitude

   # Variants of the model
   partial = distill(fuel, air_cstr=False)  # no air-filled CSTR
   ideal = distill(fuel, ideal_solution=True, ideal_vapor=True)

   # Different apparatus parameters
   result = distill(fuel, apparatus=D86Apparatus(air_volume=fl.Units.Quantity(100.0, "cm^3")))

   # Plot and compare D86 curves (volume % recovered vs. D86 temperature)
   ax = result.plot(label="Full model", temperature_units="degC")
   ideal.plot(ax=ax, label="Ideal", temperature_units="degC")

Model
-----

In the D86 test, 100 mL of fuel is heated in a flask, and the vapor rises through a
vertical tube to a condenser. The heating rate is adjusted so that 4-5 mL/min of
condensate is collected, and the thermometer at the top of the tube records the
D86 temperature. Greenfield et al.\ :footcite:p:`greenfield_macroscopic_1998`
represent the apparatus by three unit operations in series:

1. **Stage 1, the distillation flask:** an unsteady-state still pot whose liquid is
   held at its bubble point (eq. 1-5 and 10).
2. **Stage 2, the mixing stage:** an equilibrium partial condensation of the flask
   vapor, representing the vertical tube. It loses heat to the room by natural
   convection and returns reflux to the flask (eq. 6-9 and 11).
3. **Stage 3, the air-filled CSTR:** a continuous stirred-tank reactor holding the
   air initially in the apparatus, plus the heat capacity of the glass. Its
   temperature is reported as the D86 (thermometer) temperature (eq. 12-14).

.. _tab-d86-nomenclature:

.. list-table:: Nomenclature of the D86 model and the corresponding names in the code.
   :header-rows: 1
   :widths: auto
   :align: center

   * - Symbol
     - Code
     - Description
   * - :math:`W`
     - ``W``
     - Moles of liquid in the flask
   * - :math:`x_i W`
     - ``xW``
     - Moles of compound *i* in the flask
   * - :math:`x_i`, :math:`y_i`
     - ``x``, ``y``
     - Mole fractions of the flask liquid and flask vapor
   * - :math:`u_i`, :math:`v_i`
     - ``u``, ``v``
     - Mole fractions of the reflux and the stage-2 vapor
   * - :math:`v_i^{out}`
     - ``v_out``
     - Mole fractions in the CSTR
   * - :math:`D_1`, :math:`D_2`
     - ``D1``, ``D2``
     - Vapor rates leaving stages 1 and 2 (mol/s)
   * - :math:`R_2`
     - ``R2``
     - Reflux rate from stage 2 to the flask (mol/s)
   * - :math:`T_1`, :math:`T_2`, :math:`T_3`
     - ``T1``, ``T2``, ``T3``
     - Temperatures of stages 1, 2, and 3 (K)
   * - :math:`\dot{Q}_1`, :math:`\dot{Q}_2`
     - ``Q1``, ``Q2``
     - Heat rates into stages 1 and 2 (W)
   * - :math:`C_{p1,\ell}`, :math:`C_{p2,\ell}`
     - ``Cp1_l``, ``Cp2_l``
     - Liquid heat capacities of the flask liquid and the reflux (J/mol/K)
   * - :math:`C_{p1,v}`
     - ``Cp1_v``
     - Ideal-gas heat capacity of the flask vapor (J/mol/K)
   * - :math:`|\Delta H_{vap1}|`, :math:`|\Delta H_{vap2}|`
     - ``dH_vap1``, ``dH_vap2``
     - Enthalpies of vaporization of the flask vapor and the reflux (J/mol)
   * - :math:`V C^{tot}`
     - ``VC_tot``
     - Moles held in the CSTR
   * - :math:`C_{p,cylinder}`
     - ``Cp_cylinder``
     - Effective heat capacity of the glass (J/K)
   * - :math:`h`, :math:`A`, :math:`L`
     - ``h``, ``A``, ``L``
     - Heat transfer coefficient, area, and length of the tube
   * - :math:`P`, :math:`P_i^{sat}`
     - ``P``, ``P_sat``
     - Pressure and saturated vapor pressure (Pa)
   * - :math:`\gamma_i`, :math:`\phi_i`
     - ``ln_gamma``, ``ln_phi``
     - Activity and fugacity coefficients (natural log in the code)

Apparatus parameters
--------------------

:class:`~fuellib.simulate.distillation.greenfield.D86Apparatus` holds the apparatus
and procedure parameters. The heat transfer area of the vertical tube in equation 9
is its outer surface, :math:`A = \pi D L`.

.. list-table:: Default apparatus parameters.
   :header-rows: 1
   :widths: auto
   :align: center

   * - Field
     - Symbol
     - Default
     - Source
   * - ``pressure``
     - :math:`P`
     - 101325 Pa
     - Greenfield et al.\ :footcite:p:`greenfield_macroscopic_1998`
   * - ``T_room``
     - :math:`T_{room}`
     - 298.15 K
     - Greenfield et al. (298 K)
   * - ``charge_volume``
     -
     - 100 mL
     - ASTM D86
   * - ``air_volume``
     - :math:`V`
     - 85 cm\ :sup:`3`
     - Greenfield et al.
   * - ``cylinder_heat_capacity``
     - :math:`C_{p,cylinder}`
     - 4.2 J/K
     - Greenfield et al.
   * - ``column_length``
     - :math:`L`
     - 0.13 m
     - Greenfield et al.
   * - ``column_diameter``
     - :math:`D`
     - 0.025 m
     - Greenfield et al. (outer diameter)
   * - ``collection_rate_min``
     -
     - 4 mL/min
     - ASTM D86
   * - ``collection_rate_max``
     -
     - 5 mL/min
     - ASTM D86

Pure-compound properties
------------------------

The temperature-dependent pure-compound properties (:math:`\ln P^{sat}`,
:math:`C_{p,\ell}`, :math:`C^0_{p,v}`) are tabulated once on a 1 K grid and
interpolated with cubic splines.

Vapor pressure
^^^^^^^^^^^^^^

The saturated vapor pressure :math:`P_i^{sat}` comes from the Lee-Kesler
correlation (see :ref:`eq-GCM-correlations`).

Heat capacities
^^^^^^^^^^^^^^^

Greenfield et al.\ :footcite:p:`greenfield_macroscopic_1998` obtain the liquid heat
capacity of nonpolar compounds from the ideal-gas heat capacity with the correlation

.. math::

   \frac{C_{p,\ell} - C^0_{p,v}}{R} = (0.5 + 2.2\omega) \Big[ 3.67 + 11.64 (1 - T_r)^4 + 0.634 (1 - T_r)^{-1} \Big]. \tag{19}

FuelLib has no per-compound ideal-gas polynomials, so equation 19 is used in
reverse: the liquid heat capacity :math:`C_{p,\ell}` comes from FuelLib's
group-contribution correlation (see :ref:`eq-GCM-properties`) and
equation 19 is solved for :math:`C^0_{p,v}`. To avoid the singularity at
:math:`T_r = 1`, :math:`T_r` is capped at 0.95, and :math:`C^0_{p,v}` is floored
at :math:`4R`, the ideal-gas heat capacity of a rigid nonlinear molecule.

Cross critical properties
^^^^^^^^^^^^^^^^^^^^^^^^^

The vapor fugacity coefficients (see `Vapor fugacity coefficients`_) use second
virial coefficients from the corresponding-states correlations given by Greenfield
et al.\ :footcite:p:`greenfield_macroscopic_1998`. The cross critical properties
of each pair of compounds do not depend on temperature and are computed once,

.. math::

   \omega_{ij} = \tfrac{1}{2} (\omega_i + \omega_j) \tag{22}

.. math::

   T_{c_{ij}} = \sqrt{T_{c_i} T_{c_j}} \tag{23}

.. math::

   V_{c_{ij}} = \tfrac{1}{8} \Big( V_{c_i}^{1/3} + V_{c_j}^{1/3} \Big)^3 \tag{24}

.. math::

   Z_{c_{ij}} = \tfrac{1}{2} (Z_{c_i} + Z_{c_j}) \tag{25}

.. math::

   P_{c_{ij}} = Z_{c_{ij}} R T_{c_{ij}} / V_{c_{ij}} \tag{26}

with :math:`Z_c \equiv P_c V_c / (R T_c)`.

Initial state
-------------

Initial charge
^^^^^^^^^^^^^^

The charge volume is converted to moles of each compound, :math:`x_i W`, with the
liquid molar volumes at :math:`T_{room}` from the Rackett equation with the
Yamada-Gunn compressibility (see :ref:`eq-GCM-correlations`), assuming ideal (zero
excess volume) mixing. The same molar volumes convert the condensate to a recovered
volume fraction at room temperature.

Phase equilibrium
^^^^^^^^^^^^^^^^^

The flask liquid is at its bubble point,

.. math::

   \phi_i^{(1)} y_i P = \gamma_i^{(1)} x_i P_i^{sat}. \tag{10}

The activity coefficients :math:`\gamma_i` come from UNIFAC 2.0 (see :doc:`gcm`).
The equilibrium in stage 2 (equation 11) has the same form.

Vapor fugacity coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the cross critical properties of equations 22-26, the second virial
coefficients are

.. math::

   B_{ij} = \frac{R T_{c_{ij}}}{P_{c_{ij}}} \Big( B^0 + \omega_{ij} B^1 \Big) \tag{27}

.. math::

   B^0 = 0.083 - 0.422 \, T_r^{-1.6} \tag{28}

.. math::

   B^1 = 0.139 - 0.172 \, T_r^{-4.2} \tag{29}

with :math:`T_r = T / T_{c_{ij}}`, and the fugacity coefficients are

.. math::

   \ln \phi_k = \frac{P}{RT} \Big[ B_{kk} + \tfrac{1}{2} \sum_i \sum_\ell y_i y_\ell (2 \delta_{ik} - \delta_{i\ell}) \Big] - \frac{B_{kk} P_k^{sat}}{RT} \tag{30}

.. math::

   \delta_{ij} = 2 B_{ij} - B_{ii} - B_{jj}. \tag{31}

Because :math:`\sum_\ell y_\ell = 1` and :math:`\delta` is symmetric, the code
evaluates the double sum in equation 30 as
:math:`2 \sum_i y_i \delta_{ik} - \sum_i \sum_\ell y_i y_\ell \delta_{i\ell}`. The
last term of equation 30 is :math:`\ln \phi_k^{sat}`, so equation 30 gives
:math:`\ln(\phi_k / \phi_k^{sat})`.

Bubble point
^^^^^^^^^^^^

The bubble temperature of a liquid :math:`x_i` solves
:math:`f(T) = \ln \sum_i K_i x_i = 0`, with :math:`K_i = y_i / x_i` from
equation 10. The iteration alternates :math:`y_i = K_i x_i / S`, with
:math:`S = \sum_i K_i x_i`, and the Newton step

.. math::

   T \leftarrow T - \frac{\ln S}{\sum_i y_i \, d \ln P_i^{sat} / dT},

which neglects the temperature dependence of :math:`\gamma_i` and :math:`\phi_i`.
Steps are limited to 20 K.

Initial heating rate
^^^^^^^^^^^^^^^^^^^^

The enthalpy of vaporization follows from the Clausius-Clapeyron equation, treating
the vapor as an ideal gas,

.. math::

   \Delta H_{vap} = -R \frac{d \ln P^{sat}}{d (1/T)} = R T^2 \frac{d \ln P^{sat}}{dT}, \tag{17}

where :math:`d \ln P^{sat} / dT` is the derivative of the spline. The initial
heating rate :math:`\dot{Q}_1` is 1.25 times the rate needed to vaporize the middle
of the collection-rate band at the initial bubble point.

Warm-up
^^^^^^^

The energy balance on the flask is

.. math::

   W C_{p1,\ell} \frac{\partial T_1}{\partial t} = R_2 C_{p2,\ell} (T_2 - T_1) - D_1 |\Delta H_{vap1}| + \dot{Q}_1. \tag{5}

The charge starts at :math:`T_{room}` and is heated to its bubble point with
equation 5 at :math:`D_1 = R_2 = 0`.

Distillation
------------

Time step
^^^^^^^^^

Each step advances the state by :math:`\delta t = \min(\delta t_{max}, 0.02 W / D_1)`,
using :math:`D_1` from the previous step. The flask vapor composition :math:`y_i`
and temperature :math:`T_1` are known from the bubble point (equation 10).

Stage 1: distillation flask
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The overall and component material balances on the flask are

.. math::

   \frac{\partial W}{\partial t} = R_2 - D_1 \tag{1}

.. math::

   \frac{\partial (x_i W)}{\partial t} = u_i R_2 - y_i D_1 \tag{2}

with the mole fractions of every stream summing to one,

.. math::

   \sum_i x_i = \sum_i y_i = \sum_i u_i = \sum_i v_i = 1. \tag{3}

Equation 1 is the sum of equation 2 over all compounds, so the code integrates
only equation 2. Together with the energy balance (equation 5) and the bubble point
(equation 10), these determine the flask state.

Stage 2: mixing stage
^^^^^^^^^^^^^^^^^^^^^

The flask vapor partially condenses in stage 2. The material balances are

.. math::

   D_1 = R_2 + D_2 \tag{6}

.. math::

   y_i D_1 = u_i R_2 + v_i D_2, \tag{7}

the energy balance is

.. math::

   0 = D_1 C_{p1,v} (T_1 - T_2) + R_2 |\Delta H_{vap2}| + \dot{Q}_2, \tag{8}

the heat lost by natural convection to the room is

.. math::

   -\dot{Q}_2 = h A (T_2 - T_{room}), \tag{9}

and the reflux and vapor leaving stage 2 are in equilibrium,

.. math::

   \phi_i^{(2)} v_i P = \gamma_i^{(2)} u_i P_i^{sat}. \tag{11}

Natural convection from the vertical tube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The heat transfer coefficient :math:`h` in equation 9 comes from the Churchill-Chu
correlations for a vertical plate given by Greenfield et
al.\ :footcite:p:`greenfield_macroscopic_1998`,

.. math::

   \mathrm{Nu}_L = 0.68 + \frac{0.670 \, \mathrm{Ra}_L^{1/4}}{\Big[ 1 + (0.492 / \mathrm{Pr})^{9/16} \Big]^{4/9}}, \quad \mathrm{Ra}_L < 10^9, \tag{32}

.. math::

   \mathrm{Nu}_L = \left\{ 0.825 + \frac{0.387 \, \mathrm{Ra}_L^{1/6}}{\Big[ 1 + (0.492 / \mathrm{Pr})^{9/16} \Big]^{8/27}} \right\}^2, \quad \mathrm{Ra}_L \geq 10^9, \tag{33}

with :math:`\mathrm{Gr} = \beta g \rho^2 L^3 \Delta T / \mu^2`,
:math:`\mathrm{Ra}_L = \mathrm{Gr} \, \mathrm{Pr}`, and
:math:`h = \mathrm{Nu}_L k / L`. The properties of air are evaluated at the film
temperature :math:`(T_{wall} + T_{ambient}) / 2` with the fits of Greenfield et
al.\ :footcite:p:`greenfield_macroscopic_1998`,

.. math::

   k = -5.696 \times 10^{-4} + 1.025 \times 10^{-4} T - 4.764 \times 10^{-8} T^2 + 1.330 \times 10^{-11} T^3 \tag{34}

.. math::

   \mathrm{Pr} = 8.123 \times 10^{-1} - 2.948 \times 10^{-4} T - 9.443 \times 10^{-7} T^2 + 3.584 \times 10^{-9} T^3 - 3.827 \times 10^{-12} T^4 + 1.373 \times 10^{-15} T^5 \tag{35}

.. math::

   \ln \left( \frac{\beta g \rho^2}{\mu^2} \right) = 43.83 - 4.4065 \ln T \tag{36}

with :math:`k` in W/m/K and :math:`\beta g \rho^2 / \mu^2` in 1/(K m\ :sup:`3`).

Solution of stages 1 and 2
^^^^^^^^^^^^^^^^^^^^^^^^^^

The lowest possible stage-2 temperature, :math:`T_{2,min}`, is the bubble point of
the flask vapor, reached when all of it condenses. Three cases follow.

1. **Pure or constant-boiling liquid** (:math:`T_1 - T_{2,min} < 10^{-3}` K): the
   vapor condenses at :math:`T_2 = T_1` with :math:`u_i = v_i = y_i`. All the heat
   goes to vaporization, :math:`D_1 = \dot{Q}_1 / |\Delta H_{vap1}|`, and :math:`R_2`
   is the vapor rate that stage 2 can condense (equation 8 with :math:`T_2 = T_1`),
   capped at :math:`D_1`.
2. **Total reflux:** equation 8 with :math:`T_2 = T_{2,min}` and :math:`R_2 = D_1`
   gives the vapor rate that stage 2 can condense completely. If the right-hand side
   of equation 5 is not positive at that rate, stage 2 condenses everything the
   heater can boil, so :math:`D_2 = 0` and the flask is unchanged. :math:`D_1 = R_2`
   then follows from equation 5 with :math:`\partial T_1 / \partial t = 0`.
3. **Partial condensation:** :math:`T_2` is found in
   :math:`[T_{2,min}, T_1)` with Brent's method. For a trial :math:`T_2`,

   a. an isothermal flash of the flask vapor (equations 7 and 11) gives the fraction
      condensed :math:`R_2 / D_1` and the compositions :math:`u_i` and :math:`v_i`,
   b. equation 8 gives :math:`D_1` (and so :math:`R_2`),
   c. equation 2, integrated with an explicit Euler step, gives the new flask
      contents, and equation 10 gives the new :math:`T_1`, and
   d. the residual is the difference between the right-hand side of equation 5 and
      :math:`W C_{p1,\ell} (T_1^{new} - T_1) / \delta t`.

   Trial temperatures for which :math:`D_1` exceeds what the heater can supply
   return the right-hand side of equation 5 alone.

:math:`D_2` then follows from equation 6.

Flash
^^^^^

The isothermal flash of stage 2 uses successive substitution on
:math:`K_i`. For fixed :math:`K_i`, the
vapor fraction :math:`\beta = D_2 / D_1` solves the Rachford-Rice
equation\ :footcite:p:`rachford_procedure_1952`

.. math::

   \sum_i \frac{y_i (K_i - 1)}{1 + \beta (K_i - 1)} = 0,

clipped to :math:`[0, 1]`. Then :math:`u_i = y_i / [1 + \beta (K_i - 1)]` and
:math:`v_i = K_i u_i`, and :math:`K_i` is updated from equation 11 until it
converges.

Stage 3: air-filled CSTR
^^^^^^^^^^^^^^^^^^^^^^^^

The vapor from stage 2 displaces the air initially in the apparatus,

.. math::

   V C^{tot} \frac{\partial v_i^{out}}{\partial t} = v_i D_2 - v_i^{out} D_3, \tag{12}

with :math:`D_3 = D_2`, where :math:`V C^{tot}` is the ideal-gas moles in the CSTR
volume at :math:`T_{room}` and :math:`P`. Equation 12 is integrated exactly over
the step, holding :math:`D_2` and :math:`v_i` constant,

.. math::

   v_i^{out}(t + \delta t) = v_i + \Big( v_i^{out}(t) - v_i \Big) \exp \left( -\frac{D_2 \, \delta t}{V C^{tot}} \right),

for each fuel compound and for air (with :math:`v_{air} = 0`). The condensate
collected over the step is the fuel entering the CSTR minus the fuel accumulated in
it, :math:`v_i D_2 \delta t - V C^{tot} \Delta v_i^{out}`.

The heat lost by the entering vapor is gained by the air and the glass,

.. math::

   \Big( (n_{N_2} + n_{O_2}) C_{p,air,v} + C_{p,cylinder} \Big) (T_3 - T_3') = \Big( \sum_i n_i C_{p,i,v} \Big) (T_2 - T_3), \tag{13}

where :math:`T_3'` is the temperature at the previous time step and
:math:`n_i = v_i D_2 \delta t` are the moles entering during the step. Solving for
:math:`T_3` gives the D86 temperature,

.. math::

   T_3 = \frac{\Big( (n_{N_2} + n_{O_2}) C_{p,air,v} + C_{p,cylinder} \Big) T_3' + \Big( \sum_i n_i C_{p,i,v} \Big) T_2}{(n_{N_2} + n_{O_2}) C_{p,air,v} + C_{p,cylinder} + \sum_i n_i C_{p,i,v}}. \tag{14}

The moles of air in equation 14 are
:math:`n_{N_2} + n_{O_2} = V C^{tot} v_{air}^{out}`, and their ideal-gas heat
capacity uses the polynomial

.. math::

   C^0_{p,v} = a + b T + c T^2 + d T^3 \tag{18}

with the N\ :sub:`2` and O\ :sub:`2` coefficients of Greenfield et
al.\ :footcite:p:`greenfield_macroscopic_1998` (Table 3), mixed as 79 mol%
N\ :sub:`2` and 21 mol% O\ :sub:`2`.

Without the CSTR (``air_cstr=False``, the "partial model" of Greenfield et
al.\ :footcite:p:`greenfield_macroscopic_1998`), all the stage-2 vapor is collected
and :math:`T_3 = T_2`.

Heating-rate control
^^^^^^^^^^^^^^^^^^^^

Finally, the heating rate is adjusted to keep the collection rate in the
4-5 mL/min band: :math:`\dot{Q}_1` is multiplied by :math:`\exp(\delta t / \tau)` when
the rate is below the band and by :math:`\exp(-\delta t / \tau)` when it is above,
with :math:`\tau = 30` s. The simulation ends when :math:`W` falls below
``residue_fraction`` of its initial value.

Differences from Greenfield et al.
----------------------------------

.. list-table:: Parts of the FuelLib implementation that differ from Greenfield et al.
   :header-rows: 1
   :widths: auto
   :align: center

   * - Item
     - Greenfield et al.
     - FuelLib
   * - Vapor pressure
     - Reid et al. correlation and Antoine equation (eq. 15-16)
     - Lee-Kesler (see :ref:`eq-GCM-correlations`)
   * - Liquid heat capacity
     - Eq. 18-19 (nonpolar), Yuan-Stiel (eq. 20-21, polar)
     - Nielsen GCM (see :ref:`eq-GCM-properties`)
   * - Ideal-gas heat capacity
     - Eq. 18 with per-compound coefficients
     - Eq. 19 solved for :math:`C^0_{p,v}` (eq. 18 only for air)
   * - Activity coefficients
     - UNIFAC
     - UNIFAC 2.0 (see :doc:`gcm`)
   * - Dissolved air
     - Eq. 37-40
     - Not included; air only in the CSTR
   * - Time integration
     - 4th-order Runge-Kutta, bisection on :math:`D_1`
     - Explicit Euler, Brent's method on :math:`T_2`
   * - Bubble point
     - BUBL T algorithm of Smith and Van Ness
     - Newton iteration (see `Bubble point`_)
   * - Flash
     - Flash algorithm of Smith and Van Ness
     - Successive substitution with Rachford-Rice (see `Flash`_)
   * - Heating-rate control
     - Stepwise increase when the collection rate is too low
     - Exponential adjustment in both directions
   * - Reid vapor pressure
     - Flash at 100 °F to a 4:1 vapor-to-liquid volume ratio
     - Not implemented

Greenfield et al. also do not state some quantities that the model needs. FuelLib
uses:

- :math:`V C^{tot}` from the ideal-gas law at :math:`T_{room}` and :math:`P`,
- 79 mol% N\ :sub:`2` and 21 mol% O\ :sub:`2` for the heat capacity of air,
- the outer diameter of the tube for the heat transfer area,
  :math:`A = \pi D L`, and
- the integrated form of equation 12 given in `Stage 3: air-filled CSTR`_.

References
----------

.. footbibliography::
