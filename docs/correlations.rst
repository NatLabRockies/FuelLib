Equations for Correlations and Mixture Rules
============================================

Group contribution methods and reference data give the properties of each
compound at standard conditions (see :doc:`prediction_framework`). This page
presents the correlations that evaluate the properties of each compound at
other temperatures and pressures, and the mixture rules that combine the
compound properties into the properties of the fuel. Unless noted otherwise in
the individual correlation, all units are assumed to be SI: length (m), mass
(kg), time (s), temperature (K), mole (mol).

.. _sec-correlations-notation:

Notation
--------

A tilde denotes a molar property (e.g., :math:`\tilde{C}_{p,i}` in J/mol/K and
:math:`\Delta\tilde{H}_{v,i}` in J/mol), while the same symbol without a tilde
denotes the corresponding mass-based property (e.g., :math:`C_{p,i}` in J/kg/K
and :math:`\Delta H_{v,i}` in J/kg). The subscript *i* denotes the *i-th*
compound in a mixture.

.. _eq-GCM-correlations:

Individual component properties (``fuellib.correlate.components``)
------------------------------------------------------------------

This section presents correlations for physical properties that leverage the
individual compound properties from the :doc:`group_contribution_methods` and
the :ref:`reference data <sec-reference-properties>`. These correlations make
it possible to evaluate physical properties at non-standard temperatures and
pressures, given that group contribution properties are only defined at
standard conditions. The functions are in :mod:`fuellib.correlate.components`
and are available as methods of :attr:`Fuel.comps <fuellib.fuel.core.Fuel.comps>`.
The :ref:`tab-reduced-temps` are used throughout this section for each compound
*i*, provided :math:`T` in K unless noted otherwise.

.. _tab-correlation-qtys:

.. table:: Derived quantities and temperature corrections
   :widths: auto
   :align: center

   =============================  =====================  ===============================================================
   Property                       Units                  Description
   =============================  =====================  ===============================================================
   :math:`\nu_i`                  m\ :sup:`2`\ /s        Kinematic viscosity\ :footcite:p:`viswanath_viscosity_2007`.
   :math:`\mu_i`                  Pa s                   Dynamic viscosity from the Nannoolal GCM\ :footcite:p:`nannoolal_estimation_2009`.
   :math:`\Delta\tilde{H}_{v,i}`  J/mol                  Temperature-adjusted molar enthalpy of vaporization\ :footcite:p:`govindaraju_group_2016`.
   :math:`\Delta H_{v,i}`         J/kg                   Temperature-adjusted mass-based enthalpy of vaporization\ :footcite:p:`govindaraju_group_2016`.
   :math:`V_{m,i}`                m\ :sup:`3`\ /mol      Temperature-adjusted liquid molar volume\ :footcite:p:`rackett_equation_1970` \ :footcite:p:`yamada_saturated_1973` \ :footcite:p:`govindaraju_group_2016`.
   :math:`\rho_i`                 kg/m\ :sup:`3`         Density
   :math:`\tilde{C}_{p,i}`        J/mol/K                Molar specific heat capacity\ :footcite:p:`nielsen_molecular_1998` \ :footcite:p:`ruzicka_estimation_1993`.
   :math:`C_{p,i}`                J/kg/K                 Mass specific heat capacity\ :footcite:p:`nielsen_molecular_1998` \ :footcite:p:`ruzicka_estimation_1993`.
   :math:`p_{v,i}`                Pa                     Vapor pressure\ :footcite:p:`lee_generalized_1975` \ :footcite:p:`ambrose_vapour_1989`.
   :math:`\sigma_i`               N/m                    Surface tension\ :footcite:p:`brock_surface_1955`.
   :math:`k_i`                    W/m/K                  Thermal conductivity\ :footcite:p:`poling_properties_2001`.
   :math:`T_{\textit{flash},i}`   K                      Flash point (Alqaheem-Riazi\ :footcite:p:`alqaheem_flash_2017` or Alibakhshi et al.\ :footcite:p:`alibakhshi_modified_2015`).
   :math:`LHV_i`                  J/kg                   Lower heating value.
   =============================  =====================  ===============================================================


.. _tab-reduced-temps:

.. table:: Reduced temperature quantities
   :widths: auto
   :align: center

   =============================  =========================================  ======================================================
   Symbol                         Definition                                 Description
   =============================  =========================================  ======================================================
   :math:`T_{r,i}`                :math:`\frac{T}{T_{c,i}}`                  Reduced temperature.
   :math:`T_{r,b,i}`              :math:`\frac{T_{b,i}}{T_{c,i}}`            Reduced boiling point temperature.
   :math:`T_{r,\textit{stp},i}`   :math:`\frac{298 \text{ (K)}}{T_{c,i}}`    Reduced standard temperature.
   =============================  =========================================  ======================================================

.. _sec-kinematic-viscosity:

Kinematic viscosity
^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.nu <fuellib.fuel.components.Components.nu>`, where the
``correlation`` argument selects ``"Nannoolal"`` (default) or ``"Dutt"``.

.. autofunction:: fuellib.correlate.components.viscosity_kinematic_nannoolal
   :noindex:

.. autofunction:: fuellib.correlate.components.viscosity_kinematic_dutt
   :noindex:

The kinematic viscosity of the *i-th* compound of the fuel,

.. math::

   \nu_i = \frac{\mu_i}{\rho_i},

is calculated with the GCM of Nannoolal et al.\ :footcite:p:`nannoolal_estimation_2009`
(default) or from Dutt's equation (Eq. 4.23 in Viscosity of
Liquids\ :footcite:p:`viswanath_viscosity_2007`). Dutt's equation calculates the
kinematic viscosity directly provided :math:`T` in :math:`^{\circ}` C:

.. math::

   \nu_i = 10^{-6} \times \exp \bigg\{-3.0171 + \frac{442.78 + 1.6452 \,T_{b,i}}{T + 239 - 0.19 \,T_{b,i}} \bigg\}.

For supported hydrocarbons, the Nannoolal GCM calculates the dynamic viscosity of the *i-th* compound as:

.. math::
   \mu_i = 1.3\times10^{-3}\exp\left[-dBv_i\frac{T-T_{v,i}}{T-(T_{v,i}/16)}\right],

then obtains :math:`\nu_i = \mu_i / \rho_i`. The compound parameters
:math:`dBv_i` and :math:`T_{v,i}` are calculated from the Nannoolal group
contributions (see :ref:`eq-nannoolal-properties`).

Enthalpy of vaporization
^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.hv <fuellib.fuel.components.Components.hv>`, where the
``basis`` argument selects the molar (``"molar"``) or mass (``"mass"``, default) function.

.. autofunction:: fuellib.correlate.components.enthalpy_vaporization_molar
   :noindex:

.. autofunction:: fuellib.correlate.components.enthalpy_vaporization_mass
   :noindex:

The molar enthalpy of vaporization for each compound at standard pressure and
temperature, :math:`\Delta\tilde{H}_{v,\textit{stp},i}`, is adjusted for variations in
temperature\ :footcite:p:`govindaraju_group_2016`:

.. math::
   \Delta\tilde{H}_{v,i} = \Delta\tilde{H}_{v,\textit{stp},i} \bigg(\frac{1 - T_{r,i}}{1-T_{r,b,i}} \bigg)^{0.38}.

The mass-based enthalpy of vaporization (``basis="mass"``, default) is obtained by dividing by the
molecular weight:

.. math::
   \Delta H_{v,i} = \frac{\Delta\tilde{H}_{v,i}}{M_{w,i}}.


Liquid molar volume
^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.vm <fuellib.fuel.components.Components.vm>`.

.. autofunction:: fuellib.correlate.components.molar_liquid_volume
   :noindex:

The liquid molar volume is calculated at a specific temperature :math:`T` using
the generalized Rackett equation\ :footcite:p:`rackett_equation_1970` \ :footcite:p:`yamada_saturated_1973`
with an updated :math:`\phi_i` parameter\ :footcite:p:`govindaraju_group_2016`:

.. math::

   V_{m,i} = V_{m,\textit{stp},i} Z^{\phi_i}_{c,i},

where

.. math::

   Z_{c,i} &= 0.29056 - 0.08775 \omega_i,  \\
   \phi_i &=
   \begin{cases}
       (1 - T_{r,i})^{2/7} - (1 - T_{r,\textit{stp},i})^{2/7}, & \text{ if } T \leq T_{c,i} \\
       - (1 - T_{r,\textit{stp},i})^{2/7}, & \text{ if } T > T_{c,i}
   \end{cases}. \label{eq:phi}

Density
^^^^^^^

Wrapped by :meth:`fuel.comps.rho <fuellib.fuel.components.Components.rho>`.

.. autofunction:: fuellib.correlate.components.density
   :noindex:

The density of the *i-th* compound is given by

.. math::
   \rho_i = \frac{M_{w,i}}{V_{m,i}}.


Specific heat capacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.cp <fuellib.fuel.components.Components.cp>`, where the
``basis`` argument selects the molar or mass (default) function. The Ruzicka-Domalski
functions are not wrapped by the Fuel class and can only be called directly from
:mod:`fuellib.correlate.components`.

.. autofunction:: fuellib.correlate.components.specific_heat_capacity_molar
   :noindex:

.. autofunction:: fuellib.correlate.components.specific_heat_capacity_mass
   :noindex:

.. autofunction:: fuellib.correlate.components.specific_heat_capacity_molar_ruzicka
   :noindex:

.. autofunction:: fuellib.correlate.components.specific_heat_capacity_mass_ruzicka
   :noindex:

The molar specific heat capacity, :math:`\tilde{C}_{p,i}` (``basis="molar"``), is evaluated from the
group contribution expansion in :math:`\theta = (T - 298)/700` (see :ref:`eq-GCM-properties`).
The mass specific heat capacity (``basis="mass"``, default) is obtained by dividing by the
molecular weight:

.. math::
   C_{p,i} = \dfrac{\tilde{C}_{p,i}}{M_{w,i}}

Alternatively, the Ruzicka-Domalski\ :footcite:p:`ruzicka_estimation_1993` group additivity correlation
is available, with :math:`\theta = T / 100` (K) and the ``rd_A``, ``rd_B``, and ``rd_D``
properties included in the ``gani`` GCM (see :ref:`sec-gcm-gani`):

.. math::
   \tilde{C}_{p,i} = R \big(A_i + B_i \theta + D_i \theta^2\big).


Vapor pressure
^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.pvap <fuellib.fuel.components.Components.pvap>`, where the
``correlation`` argument selects ``"Lee-Kesler"`` (default) or ``"Ambrose-Walton"``.

.. autofunction:: fuellib.correlate.components.vapor_pressure
   :noindex:

The vapor pressure for each compound is calculated as a function of
temperature using either the Lee–Kesler method\ :footcite:p:`lee_generalized_1975`
or the Ambrose-Walton method\ :footcite:p:`ambrose_vapour_1989`.  Both methods solve

.. math::
   \ln p_{r,v,i} = f_i^{(0)} + \omega_i f_i^{(1)} + \omega_i^2 f_i^{(2)}

for the reduced vapor pressure for each compound,
:math:`p_{r,v,i} = p_{v,i}/p_{c,i}`.
The default method in `FuelLib` is the Lee-Kesler method, as it is
more stable at higher temperatures.
The Lee-Kesler\ :footcite:p:`lee_generalized_1975` method defines

.. math::

   f_i^{(0)} &= 5.92714 - \frac{6.09648}{T_{r,i}} - 1.28862 \ln T_{r,i} + 0.169347 \, T_{r,i}^6, \\
   f_i^{(1)} &= 15.2518 - \frac{15.6875}{T_{r,i}} - 13.4721 \ln T_{r,i} + 0.43577 \, T_{r,i}^6, \\
   f_i^{(2)} &= 0,

The Ambrose-Walton\ :footcite:p:`ambrose_vapour_1989` correlation sets:

.. math::

   f_i^{(0)} &= \frac{- 5.97616\tau_i + 1.29874\tau_i^{1.5} - 0.60394\tau_i^{2.5} - 1.06841\tau_i^{5}}{T_{r,i}}, \\
   f_i^{(1)} &= \frac{- 5.03365\tau_i + 1.11505\tau_i^{1.5} - 5.41217\tau_i^{2.5} - 7.46628\tau_i^{5},}{T_{r,i}}, \\
   f_i^{(2)} &= \frac{- 0.64771\tau_i + 2.41539\tau_i^{1.5} - 4.26979\tau_i^{2.5} - 3.25259\tau_i^{5}}{T_{r,i}},

with :math:`\tau_i = 1 - T_{r,i}`.

Vapor pressure (Antoine coefficients)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.pvap_antoine_coeffs <fuellib.fuel.components.Components.pvap_antoine_coeffs>`.

.. autofunction:: fuellib.correlate.components.vapor_pressure_antoine_coeffs
   :noindex:

Users also have the option to return the coefficients from an Antoine fit based on
the mixture vapor pressure calculated from Raoult's law above.  Antoine's equation is:

.. math::

   \log_{10}\Big(\frac{p_{v,i}}{D_i}\Big) = A_i - \frac{B_i}{C_i + T},

where :math:`D_i` is a conversion factor for converting :math:`p_{v,i}` from Pa to cgs-based units dyne/cm :sup:`2` (:math:`D_i = 10`).
This feature was added to provide `Pele <https://amrex-combustion.github.io>`_ users an option to estimate Antoine coefficients for use in CFD
simulations with spray. See the `PelePhysics documentation <https://amrex-combustion.github.io/PelePhysics/Spray.html>`_
for additional information.


Surface tension
^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.sigma <fuellib.fuel.components.Components.sigma>`, where the
``correlation`` argument selects ``"Brock-Bird"`` (default) or ``"Pitzer"``.

.. autofunction:: fuellib.correlate.components.surface_tension
   :noindex:

Surface tension for each compound is approximated using the relation:

.. math::
   \sigma_i = p_{c,i}^{2/3} T_{c,i}^{1/3} Q_i (1 - T_{r,i})^{11/9},

provided :math:`p_{c,i}` in bar.  The :math:`Q_i` term is defined by Brock and Bird\ :footcite:p:`brock_surface_1955` (default in FuelLib) as

.. math::
   Q_i = 0.1196 \bigg[1 + \frac{T_{r,b,i} \log(p_{c,i}/1.01325)}{1 - T_{r,b,i}}\bigg] - 0.279,

or by Curl and Pitzer\ :footcite:p:`poling_properties_2001` \ :footcite:p:`curl_volumetric_1958` \ :footcite:p:`pitzer_thermodynamics_1995` as

.. math::
   Q_i = \frac{1.86 + 1.18 \omega_i}{19.05} \bigg[ \frac{3.75 + 0.91 \omega_i}{0.291 - 0.08\omega_i} \bigg]^{2/3}.


Thermal conductivity
^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.k <fuellib.fuel.components.Components.k>`.

.. autofunction:: fuellib.correlate.components.thermal_conductivity_latini
   :noindex:

Thermal conductivity for each compound is computed according to the method of
Latini et al. as summarized in Poling's\ :footcite:p:`poling_properties_2001` book:

.. math::
   k_i = \frac{A_i(1 - T_{r,i})^{0.38}}{T_{r,i}^{1/6}}.

The constant :math:`A_i` is defined by:

.. math::
   A_i = \frac{A^\ast T_{b,i}^\alpha}{M_{w,i}^\beta T_{c,i}^{\gamma}},

provided :math:`M_{w,i}` in g/mol. The exponents vary depending on the family of
the compound as defined in :ref:`tab-thermal-conductivity-parameters`.  It is assumed
that:

* aromatics have contain aromatic group contributions (e.g. ACCH)
* cycloparaffins contain a ring (e.g. 5-membered ring) and do not contain aromatic groups
* olefins contain one or more pairs of carbon atoms linked by a double bond and do not contain aromatic groups or rings
* all other compounds are assumed to be saturated hydrocarbons.

These structural checks are performed on each compound's `RDKit <https://www.rdkit.org/docs/>`_
``Mol`` object (built from its SMILES string via
:func:`~fuellib.rdk.mol.from_smiles`) using
:func:`~fuellib.rdk.mol.has_aromatic`, :func:`~fuellib.rdk.mol.has_ring`,
and :func:`~fuellib.rdk.mol.has_double_bond` from :mod:`fuellib.rdk.mol`.


.. _tab-thermal-conductivity-parameters:

.. table:: Thermal conductivity relation parameters
   :widths: auto
   :align: center

   ===========  ==========================  ===============  ===============  ===============  ===============
   Identifier   Family                      :math:`A^\ast`   :math:`\alpha`   :math:`\beta`    :math:`\gamma`
   ===========  ==========================  ===============  ===============  ===============  ===============
   0            Saturated hydrocarbons      0.00350          1.2              0.5              0.167
   1            Aromatics                   0.0346           1.2              1.0              0.167
   2            Cycloparaffins              0.0310           1.2              1.0              0.167
   3            Olefins                     0.0361           1.2              1.0              0.167
   ===========  ==========================  ===============  ===============  ===============  ===============

Flash point
^^^^^^^^^^^

Wrapped by :meth:`fuel.comps.t_flash <fuellib.fuel.components.Components.t_flash>`, where the
``correlation`` argument selects ``"Alibakhshi"`` (default) or ``"Alqaheem"``.

.. autofunction:: fuellib.correlate.components.flash_point_alqaheem
   :noindex:

.. autofunction:: fuellib.correlate.components.flash_point_alibakhshi
   :noindex:

Two correlations are available for the flash point :math:`T_{\textit{flash},i}` of each compound,
both based on the normal boiling point :math:`T_{b,i}`.
The Alqaheem and Riazi\ :footcite:p:`alqaheem_flash_2017` correlation takes the flash point as a fixed fraction
of the boiling point:

.. math::
   T_{\textit{flash},i} = 0.70 \, T_{b,i}.

The modified group-contribution correlation of Alibakhshi et al.\ :footcite:p:`alibakhshi_modified_2015` adds a
group-contribution term:

.. math::
   T_{\textit{flash},i} = 12.14 + 0.73 \, T_{b,i} + \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} \phi_k,

where :math:`\phi_k` are the group contributions registered as the ``alibakhshi_phi``
property of the ``gani`` GCM (see :ref:`sec-gcm-gani`).

.. _sec-lower-heating-value:

Lower heating value
^^^^^^^^^^^^^^^^^^^

.. autofunction:: fuellib.correlate.components.lower_heating_value
   :noindex:

The lower (net) heating value :math:`LHV_i` is computed from a Hess cycle using the
enthalpy of formation :math:`\Delta\tilde{H}_{f,i}` and enthalpy of vaporization
:math:`\Delta\tilde{H}_{v,\textit{stp},i}`. The liquid-phase enthalpy of formation is

.. math::
   \Delta\tilde{H}_{f,\ell,i} = \Delta\tilde{H}_{f,i} - \Delta\tilde{H}_{v,\textit{stp},i},

and the molar enthalpy of combustion (an exothermic, i.e. negative, quantity in J/mol) of a compound with :math:`n_{C,i}` carbon and
:math:`n_{H,i}` hydrogen atoms (from its RDKit molecule) is

.. math::
   \Delta\tilde{H}_{c,i} = n_{C,i} \, \Delta\tilde{H}_f^{\text{CO}_2(g)}
   + \frac{n_{H,i}}{2} \, \Delta\tilde{H}_f^{\text{H}_2\text{O}(g)} - \Delta\tilde{H}_{f,\ell,i},

with :math:`\Delta\tilde{H}_f^{\text{CO}_2(g)} = -393.51` kJ/mol and
:math:`\Delta\tilde{H}_f^{\text{H}_2\text{O}(g)} = -241.83` kJ/mol at 298.15 K. The lower
heating value is then

.. math::
   LHV_i = -\frac{\Delta\tilde{H}_{c,i}}{M_{w,i}}.

This estimate is valid for hydrocarbons only and does not account for
heteroatom-containing groups.

.. _eq-mixture-properties:

Mixture properties (``fuellib.correlate.mixture``)
--------------------------------------------------

This section contains mixture rules for estimating physical properties of the
mixture from the individual compound properties defined in
:ref:`eq-GCM-correlations`. The functions are in :mod:`fuellib.correlate.mixture`
and are available as methods of :attr:`Fuel.mix <fuellib.fuel.core.Fuel.mix>`.
The :ref:`tab-mixture-properties` available in `FuelLib` are listed in
the table below.  Mass and mole fractions defined in Table :ref:`tab-mass-mole-fracs`
are used throughout this section.

.. _tab-mixture-properties:

.. table:: Mixture properties
   :widths: auto
   :align: center

   ============================  ===============  =====================
   Symbol                        Units            Description
   ============================  ===============  =====================
   :math:`\rho`                  kg/m\ :sup:`3`   Density
   :math:`\nu`                   m\ :sup:`2`/s    Kinematic viscosity
   :math:`p_v`                   Pa               Vapor pressure
   :math:`\sigma`                N/m              Surface tension
   :math:`k`                     W/m/K            Thermal conductivity
   :math:`T_{\textit{freeze}}`   K                Freeze point
   :math:`T_{\textit{flash}}`    K                Flash point
   :math:`\Delta H^\circ_c`      J/kg             Heat of combustion
   :math:`YSI`                   --               Yield sooting index
   :math:`SP`                    mm               Smoke point
   :math:`DCN`                   --               Derived cetane number
   ============================  ===============  =====================

.. _tab-mass-mole-fracs:

.. table:: Mass and mole fractions
   :widths: auto
   :align: center

   =============  ========================================  ==================================================================================
   Symbol         Definition                                Description
   =============  ========================================  ==================================================================================
   :math:`Y_i`    :math:`\frac{m_i}{\sum_{k=1}^{N_c} m_k}`   Mass fraction of compound *i*. :math:`m_i` is the mass of compound *i*.
   :math:`X_i`    :math:`\frac{n_i}{\sum_{k=1}^{N_c} n_k}`   Mole fraction of compound *i*. :math:`n_i` is the number of moles compound *i*.
   =============  ========================================  ==================================================================================

.. _conventional-mixing-rules:

Conventional mixing rules
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: fuellib.correlate.mixture.arithmetic
   :noindex:

.. autofunction:: fuellib.correlate.mixture.geometric
   :noindex:

While many of the mixture properties in FuelLib have a unique mixing rule,
FuelLib's *arithmetic* and *geometric* functions provide
general mixing rules based on the suggestions of Harstad et al\ :footcite:p:`harstad_efficient_1997`.
For a given property :math:`Q`

.. math::
   Q = \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} X_i X_j Q_{ij},

where the pseudo-property for the couple of components, :math:`Q_{ij}` is computed
using an arithmetic,

.. math::
   Q_{ij} = \frac{Q_i + Q_j}{2},

or a geometric mean,

.. math::
   Q_{ij} = \sqrt{Q_i \cdot Q_j},

where :math:`Q_i` is the property of the *i-th* compound of the multicomponent mixture.

Mixture density
^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.rho <fuellib.fuel.mixture.Mixture.rho>`.

.. autofunction:: fuellib.correlate.mixture.density
   :noindex:

The mixture's density is calculated as:

.. math::

   \rho = \sum_{i=1}^{N_c}Y_i\frac{M_{w,i}}{V_{m,i}}.


Mixture kinematic viscosity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.nu <fuellib.fuel.mixture.Mixture.nu>`, where the ``correlation``
argument selects ``"Nannoolal"`` (default) or ``"Dutt"`` for the compound viscosities and the
``mixing_rule`` argument selects ``"Kendall-Monroe"`` (default) or ``"Arrhenius"``.

.. autofunction:: fuellib.correlate.mixture.viscosity_kinematic_nannoolal
   :noindex:

.. autofunction:: fuellib.correlate.mixture.viscosity_kinematic_dutt
   :noindex:

The kinematic viscosity of the mixture is computed using the Kendall-Monroe\ :footcite:p:`kendall_viscosity_1917`
mixing rule, with an option to use the Arrhenius\ :footcite:p:`arrhenius_uber_1887`
mixing rule. The viscosity of each component. After evaluating thirty mixing rules,  Hernandez et al.\ :footcite:p:`hernandez_evaluation_2021`
found that both Kendall-Monroe and Arrhenius
were among the most effective without relying on additional data or parameter fitting.
The Kendall-Monroe rule is:

.. math::

   \nu_{KM}^{1/3} = \sum_{i=1}^{N_c} X_i \, \nu_i^{1/3}.

The Arrhenius rule is:

.. math::

   \ln \nu_{Arr} = \sum_{i=1}^{N_c} X_i\ln\nu_i .


Mixture vapor pressure
^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.pvap <fuellib.fuel.mixture.Mixture.pvap>`, where the
``correlation`` argument selects ``"Lee-Kesler"`` (default) or ``"Ambrose-Walton"``
for the compound vapor pressures.

.. autofunction:: fuellib.correlate.mixture.vapor_pressure
   :noindex:

The vapor pressure of the mixture is calculated according to Raoult's law:

.. math::

   p_{v} = \sum_{i = 1}^{N_c} X_i \, p_{v,i}.

Mixture vapor pressure (Antoine coefficients)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.pvap_antoine_coeffs <fuellib.fuel.mixture.Mixture.pvap_antoine_coeffs>`.

.. autofunction:: fuellib.correlate.mixture.vapor_pressure_antoine_coeffs
   :noindex:

Users also have the option to return the coefficients from an Antoine fit based on
the mixture vapor pressure calculated from Raoult's law above.  Antoine's equation is:

.. math::

   \log_{10}\Big(\frac{p_{v}}{D}\Big) = A - \frac{B}{C + T},

where :math:`D` is a conversion factor for converting :math:`p_v` from Pa to cgs-based units dyne/cm :sup:`2` (:math:`D_i = 10`)..
This feature was added to provide `Pele <https://amrex-combustion.github.io>`_ users an option for estimating these coefficients for use in CFD
simulations with spray. See the `PelePhysics documentation <https://amrex-combustion.github.io/PelePhysics/Spray.html>`_
for additional information.


Mixture surface tension
^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.sigma <fuellib.fuel.mixture.Mixture.sigma>`, where the
``correlation`` argument selects ``"Brock-Bird"`` (default) or ``"Pitzer"`` for the
compound surface tensions.

.. autofunction:: fuellib.correlate.mixture.surface_tension
   :noindex:

The surface tension of the mixture is calculated using the :ref:`conventional-mixing-rules`
with an arithmetic mean for the pseudo-property :math:`\sigma_{i,j}` as recommended by
Hugill and van Welsenes\ :footcite:p:`hugill_surface_1986`:

.. math::
   \sigma = \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} X_i X_j \frac{\sigma_i + \sigma_j}{2}.

Mixture thermal conductivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.k <fuellib.fuel.mixture.Mixture.k>`.

.. autofunction:: fuellib.correlate.mixture.thermal_conductivity_latini
   :noindex:

The thermal conductivity of the mixture is calculated using the power law method of
Vredeveld as described in Poling\ :footcite:p:`poling_properties_2001`:

.. math::
   k = \bigg(\sum_{i=1}^{N_c} Y_i k_i^{-2} \bigg)^{-1/2}.

.. _sec-mixture-freeze-point:

Mixture freeze point
^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.t_freeze <fuellib.fuel.mixture.Mixture.t_freeze>`.

.. autofunction:: fuellib.correlate.mixture.freeze_point_boehm
   :noindex:

The freeze point of the mixture is estimated with the solid-liquid equilibrium model
of Boehm et al.\ :footcite:p:`boehm_blend_2022`, equation 21. For each compound :math:`j` with mole fraction
:math:`X_j`, the candidate freeze temperature :math:`T_j` solves

.. math::
   T_j \Big[\Delta S_{\textit{fus},j} + X_j \Delta\tilde{C}_{p,j} \ln\frac{T_j}{T_{m,j}}
   + \alpha \, \Delta S_{\textit{mix},j}\Big]
   = \Delta H_{\textit{fus},j} + X_j \Delta\tilde{C}_{p,j} (T_{m,j} - T_j),

where

.. math::
   \Delta S_{\textit{mix},j} &= -\frac{R}{X_j}\big[(1 - X_j)\ln(1 - X_j) + X_j \ln X_j\big], \\
   \Delta H_{\textit{fus},j} &= T_{m,j} \, \Delta S_{\textit{fus},j}, \\
   \Delta\tilde{C}_{p,j} &= -0.35 \, \tilde{C}_{p,\textit{stp},j},

:math:`T_{m,j}` is the melting point from :attr:`~fuellib.fuel.components.Components.tm` (a reference value
when available, otherwise the Constantinou-Gani prediction; see :ref:`sec-reference-properties`),
:math:`\Delta S_{\textit{fus},j}`
is provided by the ``boehm`` GCM (see :ref:`sec-gcm-boehm`), and :math:`\alpha` is an optional
scaling of the ideal mixing-entropy term (default 1). The mixture freeze point is the
highest :math:`T_j` among compounds with :math:`X_j > 10^{-6}`, i.e. the temperature at
which the first crystal forms on cooling. This is an equilibrium screening model that
does not capture cooling rate, supercooling, or solid-phase nonideality.

Mixture flash point
^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.t_flash <fuellib.fuel.mixture.Mixture.t_flash>`, where the
``correlation`` argument selects ``"Alibakhshi"`` (default) or ``"Alqaheem"`` for the
compound flash points and the ``mixing_rule`` argument selects ``"Liaw"`` (default) or
``"linear"``.

.. autofunction:: fuellib.correlate.mixture.flash_point_alqaheem
   :noindex:

.. autofunction:: fuellib.correlate.mixture.flash_point_alibakhshi
   :noindex:

The mixture flash point is computed from the component flash points :math:`T_{\textit{flash},i}`
(see Flash point above) with either a linear mass-fraction mixing rule,

.. math::
   T_{\textit{flash}} = \sum_{i=1}^{N_c} Y_i T_{\textit{flash},i},

or the ideal Liaw-Chiu\ :footcite:p:`liaw_general_2006` mixing rule (default), which solves for :math:`T_{\textit{flash}}` such that

.. math::
   \sum_{i=1}^{N_c} X_i \frac{p_{v,i}(T_{\textit{flash}})}{p_{v,i}(T_{\textit{flash},i})} = 1,

where :math:`p_{v,i}` is evaluated with the Lee-Kesler correlation. This
mixing rule does not model nonideal liquid activity coefficients.

Mixture heat of combustion
^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.lhv <fuellib.fuel.mixture.Mixture.lhv>`.

.. autofunction:: fuellib.correlate.mixture.heat_of_combustion
   :noindex:

The mixture heat of combustion :math:`\Delta H^\circ_c` is reported per unit mass as the
positive magnitude of the heat released, and is computed as the mass-fraction weighted
average of the component lower heating values
:math:`LHV_i` (see :ref:`Lower heating value <sec-lower-heating-value>`). It is distinct
from the molar enthalpy of combustion :math:`\Delta\tilde{H}_{c,i}` of a single compound:

.. math::
   \Delta H^\circ_c = \sum_{i=1}^{N_c} Y_i \, LHV_i.

This is a net heating value assuming gaseous water products, and is an engineering
estimate related to ASTM D4809/D3338 rather than a simulated bomb-calorimeter test.

Mixture yield sooting index
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.ysi <fuellib.fuel.mixture.Mixture.ysi>`.

.. autofunction:: fuellib.correlate.mixture.yield_sooting_index
   :noindex:

The yield sooting index :math:`YSI` is a dimensionless measure of a fuel's tendency to soot.
The component values :math:`YSI_i` are reference data from ``ref_compounds.csv``, primarily
from the Yield Sooting Index Database of McEnally et al.\ :footcite:p:`mcenally_yield_2017`, and are
available as :attr:`~fuellib.fuel.components.Components.ysi` (see :ref:`sec-reference-properties`); they are
not predicted by FuelLib. The mixture value is the mole-fraction weighted average

.. math::
   YSI = \sum_{i=1}^{N_c} X_i \, YSI_i.

A ``ValueError`` listing the compounds is raised if the reference data are missing for any
component. Mass fractions are converted to mole fractions using the component molecular weights.

Mixture smoke point
^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.smoke_point <fuellib.fuel.mixture.Mixture.smoke_point>`.

.. autofunction:: fuellib.correlate.mixture.smoke_point
   :noindex:

The mixture smoke point :math:`SP` is estimated from the mixture
YSI using the correlation from McEnally et al.\ :footcite:p:`mcenally_soot_2020`,

.. math::
   \frac{YSI}{S} = a + \frac{b}{SP},

with the coefficients selected by the ``correlation`` argument:

.. list-table::
   :header-rows: 1

   * - ``correlation``
     - :math:`a`
     - :math:`b`
     - Basis
   * - ``"NJFCP"`` (default)
     - 0.2493
     - 30.2077
     - Least-squares fit using measured smoke points of eight NJFCP POSF fuels
   * - ``"McEnally"``
     - 0.4034
     - 29.90546
     - McEnally et al. correlation

Here :math:`S` is the moles of oxidizer (air) required to stoichiometrically
combust one mole of fuel. For a hydrocarbon :math:`\mathrm{C}_x\mathrm{H}_y`,
:math:`S = (x + y/4)/0.209`, where 0.209 is the mole fraction of :math:`\mathrm{O_2}` in air.
For a mixture, :math:`YSI` (see above) and :math:`S = \sum_i X_i S_i` are
mole-fraction weighted averages. Higher :math:`SP` indicates a lower sooting tendency, whereas
higher :math:`YSI` indicates a higher sooting tendency. The result is reported in mm. A ``ValueError``
is raised if :math:`YSI` is unavailable for any component, is not positive, or if :math:`YSI/S \le a`.

Mixture derived cetane number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.dcn <fuellib.fuel.mixture.Mixture.dcn>`.

.. autofunction:: fuellib.correlate.mixture.derived_cetane_number
   :noindex:

The derived cetane number :math:`DCN` is a dimensionless measure of ignition quality.
The component values :math:`DCN_i` are reference data from ``ref_compounds.csv`` and are
available as :attr:`~fuellib.fuel.components.Components.dcn` (see :ref:`sec-reference-properties`); they are
not predicted by FuelLib. The mixture value is computed with a linear liquid volume-fraction
mixing rule,

.. math::
   DCN = \sum_{i=1}^{N_c} \phi_i \, DCN_i, \qquad
   \phi_i = \frac{Y_i / \rho_i}{\sum_{k=1}^{N_c} Y_k / \rho_k},

where :math:`\phi_i` is the liquid volume fraction of compound *i* and :math:`\rho_i` is the
component liquid density evaluated at a reference temperature ``temp_ref`` (default: standard
temperature, ``constants.T_STP``). A ``ValueError`` listing the compounds is
raised if the reference data are missing for any component.

Linear volume-fraction blending is a first-order approximation. Cetane number can
blend nonlinearly for certain mixtures, depending on the chemical structures of the
components\ :footcite:p:`kim_designing_2023`, so the mixture :math:`DCN` should be treated as an estimate.
The graph neural network models of Kim et al. for predicting cetane numbers are available at
`BioE-KimLab/CNpred <https://github.com/BioE-KimLab/CNpred>`_.

Mixture aromatics content
^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.aromatics <fuellib.fuel.mixture.Mixture.aromatics>`.

.. autofunction:: fuellib.correlate.mixture.aromatics_content
   :noindex:

The aromatics content is the total liquid volume percent of aromatic compounds in the mixture,

.. math::
   \phi_{\textit{aro}} = 100 \sum_{i \in \mathcal{A}} \phi_i, \qquad
   \phi_i = \frac{Y_i / \rho_i}{\sum_{k=1}^{N_c} Y_k / \rho_k},

where :math:`\mathcal{A}` is the set of compounds whose
:attr:`~fuellib.fuel.components.Components.hc_type` is ``"aromatic"`` (any compound containing an aromatic
ring, including cycloaromatics and diaromatics), and :math:`\rho_i` is the component liquid
density evaluated at standard temperature (``constants.T_STP``). The result is in vol%,
directly comparable with aromatics specifications (e.g., ASTM D1319).

Mixture naphthalenes content
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrapped by :meth:`fuel.mix.naphthalenes <fuellib.fuel.mixture.Mixture.naphthalenes>`.

.. autofunction:: fuellib.correlate.mixture.naphthalenes_content
   :noindex:

The naphthalenes content is the total liquid volume percent of naphthalene and alkylated
naphthalenes in the mixture,

.. math::
   \phi_{\textit{naph}} = 100 \sum_{i \in \mathcal{N}} \phi_i, \qquad
   \phi_i = \frac{Y_i / \rho_i}{\sum_{k=1}^{N_c} Y_k / \rho_k},

where :math:`\mathcal{N}` is the set of compounds flagged by
:func:`~fuellib.rdk.mol.has_naphthalene`, and :math:`\rho_i` is the component liquid
density evaluated at standard temperature (``constants.T_STP``). A compound is flagged
when its RDKit molecule contains the
naphthalene core (two fused aromatic six-membered rings) and its fused ring system
contains exactly two rings. Larger fused systems (e.g., anthracene) and partially
saturated systems (e.g., tetralin) are not counted. Naphthalenes are a subset of the
aromatics, so :math:`\phi_{\textit{naph}} \le \phi_{\textit{aro}}`. The result is in vol%.

References
----------

.. footbibliography::
