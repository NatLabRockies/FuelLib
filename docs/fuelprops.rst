Fuel Property Prediction Model
==============================

**FuelLib** utilizes the group contribution method (GCM), as developed by Constantinou and 
Gani\ :footcite:p:`constantinou_new_1994` \ :footcite:p:`constantinou_estimation_1995` in the mid-1990s, 
to provide a systematic approach for estimating the thermodynamic properties of
pure organic compounds. The GCM decomposes molecules into structural groups, 
each contributing to a target property based on predefined group values. 
By summing these contributions, the GCM accurately predicts essential properties, 
including the acentric factor, normal boiling point, liquid molar volume at standard conditions 
(298 K) and more. This predictive capability is particularly useful for complex 
mixtures such as synthetic aviation turbine fuels (SATFs), where experimental thermodynamic data 
is limited. `FuelLib` provides SATF developers with a means to estimate 
these critical properties without extensive physical testing, thereby aiding in 
the identification of promising fuel compositions before committing to large-scale production.

`FuelLib` builds on 
`Pavan B. Govindaraju's Matlab implementation <https://github.com/gpavanb-old/GroupContribution>`_, 
and includes gas chromatography data (GC x GC) for various jet fuels from the National Jet Fuel Combustion Program\ :footcite:p:`colket_overview_2017` (NJFCP) 
Air Force Research Laboratory\ :footcite:p:`edwards_reference_2017` \ :footcite:p:`edwards_jet_2020` (AFRL) and Vozka et al.\ :footcite:p:`vozka_impact_2018`.
Additionally, `FuelLib` includes correlations for the thermodynamic properties of 
mixture such as density, viscosity, vapor pressure, surface tension, and thermal conductivity. :ref:`tab-GCM-properties` 
outlines the properties for the *i-th* compound in a mixture, which depends on 
the *k-th* first-order and *j-th* second-order group contributions.

.. _tab-GCM-properties:

Table of GCM properties
-----------------------

.. table:: GCM properties of the *i-th* component in a mixture. The subscript *stp* denotes a standard pressure assumption.
   :widths: auto
   :align: center

   ====================================  =====================  ===========================================  ====================  ===========================================================
   Property                              Units                  Group Contributions                          Units                 Description
   ====================================  =====================  ===========================================  ====================  ===========================================================
   :math:`M_{w,i}`                       kg/mol                 :math:`m_{w1k}`                              g/mol                 Molecular weight.
   :math:`T_{c,i}`                       K                      :math:`t_{c1k}`, :math:`t_{c2j}`             1                     Critical temperature\ :footcite:p:`constantinou_new_1994`.
   :math:`p_{c,i}`                       Pa                     :math:`p_{c1k}`, :math:`p_{c2j}`             bar\ :sup:`-0.5`      Critical pressure\ :footcite:p:`constantinou_new_1994`.
   :math:`V_{c,i}`                       m\ :sup:`3`\ /mol      :math:`v_{c1k}`, :math:`v_{c2j}`             m\ :sup:`3`\ /kmol    Critical volume\ :footcite:p:`constantinou_new_1994`.
   :math:`T_{b,i}`                       K                      :math:`t_{b1k}`, :math:`t_{b2j}`             1                     Normal boiling point\ :footcite:p:`constantinou_new_1994`.
   :math:`T_{m,i}`                       K                      :math:`t_{m1k}`, :math:`t_{m2j}`             1                     Normal melting point\ :footcite:p:`constantinou_new_1994`.
   :math:`\Delta H_{f,i}`                J/mol                  :math:`h_{f1k}`, :math:`h_{f2j}`             kJ/mol                Enthalpy of formation at 298 K\ :footcite:p:`constantinou_new_1994`.
   :math:`\Delta G_{f,i}`                J/mol                  :math:`g_{f1k}`, :math:`g_{f2j}`             kJ/mol                Standard Gibbs free energy at 298 K\ :footcite:p:`constantinou_new_1994`.
   :math:`\Delta H_{v,\textit{stp},i}`   J/mol                  :math:`h_{v1k}`, :math:`h_{v2j}`             kJ/mol                Enthalpy of vaporization at 298 K\ :footcite:p:`constantinou_new_1994`.
   :math:`\omega_i`                      1                      :math:`\omega_{1k}`, :math:`\omega_{2j}`     1                     Acentric factor\ :footcite:p:`constantinou_estimation_1995`.
   :math:`V_{m,\textit{stp},i}`          m\ :sup:`3`\ /mol      :math:`v_{m1k}`, :math:`v_{m2j}`             m\ :sup:`3`\ /kmol    Liquid molar volume at 298 K\ :footcite:p:`constantinou_estimation_1995`. 
   :math:`C_{p,i}`                       J/mol/K                :math:`C_{pA1_k}`, :math:`C_{pA2_k}`,...     J/mol/K               Molar specific heat capacity\ :footcite:p:`nielsen_molecular_1998` \ :footcite:p:`poling_properties_2001`.
   ====================================  =====================  ===========================================  ====================  ===========================================================

.. _eq-GCM-properties:

Equations for GCM properties
----------------------------

The properties of each compound in a mixture can be calculated as the sum of contributions 
from the first- and second-order groups that make up the compound. For a given mixture, 
let :math:`\mathbf{N}` be an :math:`N_c \times N_{g_1}` matrix that represents the 
number of first-order groups in each compound, where :math:`N_c` is the number of compounds 
in the mixture and :math:`N_{g_1}` is the total number of first-order groups as defined 
by Constantinou and Gani\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995`.  
Similarly, let :math:`\mathbf{M}` be an :math:`N_c \times N_{g_2}` matrix that specifies 
the number of second-order groups in each compound, where :math:`N_{g_2}` is the total 
number of second-order groups. The total number of groups :math:`N_g = N_{g_1} + N_{g_2} = 121`. 
The GCM properties for the *i-th* compound in the mixture are calculated as 
follows\ :footcite:p:`constantinou_new_1994` \ :footcite:p:`constantinou_estimation_1995` \ :footcite:p:`poling_properties_2001`:

.. math::

   M_{w,i} &= \bigg[\sum_{k = 1}^{N_{g_1}}\mathbf{N}_{ik}m_{w1k} \bigg] \times 10^{-3}, \\
   T_{c,i} &= 181.28 \ln  \bigg[ \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} t_{c1k} + \sum_{j=1}^{N_{g_2}}         \mathbf{M}_{ij} t_{c2j} \bigg],\\
   p_{c,i} &= \Bigg( \bigg[  \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} p_{c1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij}     p_{c2j} + 0.10022\bigg]^{-2}  + 1.3705\Bigg)\times 10^{5}, \\
   V_{c,i} &= \Bigg( \bigg[ \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} v_{c1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij}      v_{c2j} \bigg] -0.00435 \Bigg)\times 10^{-3}, \\
   T_{b,i} &= 204.359 \ln  \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} t_{b1k} + \sum_{j=1}^{N_{g_2}}  \mathbf{M}_{ij} t_{b2j}\bigg],\\
   T_{m,i} &= 102.425 \ln  \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} t_{m1k} + \sum_{j=1}^{N_{g_2}}  \mathbf{M}_{ij} t_{m2j}\bigg],\\
   \Delta H_{f,i} &= \Bigg( \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} h_{f1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} h_{f2j} \bigg] + 10.835\Bigg) \times 10^3,\\
   \Delta G_{f,i} &= \Bigg( \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} g_{f1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} g_{f2j} \bigg] -14.828 \Bigg) \times 10^3,\\
   \Delta H_{v,\textit{stp},i} &= \Bigg( \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} h_{v1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} h_{v2j} \bigg] + 6.829\Bigg) \times 10^3, \\
   \omega_i &= 0.4085 \bigg[ \ln \Big(  \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} \omega_{1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} \omega_{2j} + 1.1507\Big) \bigg]^{1/0.5050}, \\
   V_{m,\textit{stp},i} &= \Bigg( \bigg[ \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} v_{m1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} v_{m2j} \bigg] + 0.01211 \Bigg)\times 10^{-3}, \\
   C_{p,i} & =\bigg[\sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} C_{pA1_k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} C_{pA2_j} -19.7779\bigg]  \\
       & +\bigg[\sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} C_{pB1_k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} C_{pB2_j} + 22.5981\bigg] \theta \\
       & +\bigg[\sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} C_{pC1_k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} C_{pC2_j} - 10.7983\bigg] \theta^2 \\
   \theta &= \frac{T - 298.15}{700}


.. _eq-GCM-correlations:

Equations for individual compound correlations
----------------------------------------------

This section presents correlations for physical properties that leverage the individual 
compound properties defined in :ref:`eq-GCM-properties`.  These correlations make 
it possible to evaluate physical properties at non-standard temperatures and pressures, 
given that group contribution properties are only defined at standard conditions.
Unless noted otherwise in the individual correlation, all units are assumed to be SI: 
length (m), mass (kg), time (s), temperature (K), mole (mol).
The :ref:`tab-reduced-temps` are used throughout this section for each compound *i*, 
provided :math:`T` in K unless noted otherwise.

.. _tab-correlation-qtys:

.. table:: Derived quantities and temperature corrections
   :widths: auto
   :align: center

   =============================  =====================  ===============================================================
   Property                       Units                  Description
   =============================  =====================  ===============================================================
   :math:`\nu_i`                  m\ :sup:`2`\ /s        Kinematic viscosity\ :footcite:p:`viswanath_viscosity_2007`.
   :math:`L_{v,\textit{stp},i}`   J/kg                   Latent heat of vaporization at 298 K\ :footcite:p:`govindaraju_group_2016`.
   :math:`L_{v,i}`                J/kg                   Temperature-adjusted latent heat of vaporization at 298 K\ :footcite:p:`govindaraju_group_2016`.
   :math:`V_{m,i}`                m\ :sup:`3`\ /mol      Temperature-adjusted liquid molar volume\ :footcite:p:`rackett_equation_1970` \ :footcite:p:`yamada_saturated_1973` \ :footcite:p:`govindaraju_group_2016`.
   :math:`\rho_i`                 kg/m\ :sup:`3`         Density
   :math:`C_{\ell,i}`             J/kg/K                 Mass specific heat capacity. 
   :math:`p_{sat,i}`              Pa                     Saturated vapor pressure\ :footcite:p:`lee_generalized_1975` \ :footcite:p:`ambrose_vapour_1989`.
   :math:`\sigma_i`               N/m                    Surface tension\ :footcite:p:`brock_surface_1955`.
   :math:`\lambda_i`              W/m/K                  Thermal conductivity\ :footcite:p:`poling_properties_2001`.
   :math:`T_{fp,i}`               K                      Flash point (Alqaheem-Riazi or Alibakhshi et al.).
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


Kinematic viscosity
^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.viscosity_kinematic
   :noindex:

The kinematic viscosity of the *i-th* compound of the fuel, 

.. math::
   
   \nu_i = \frac{\mu_i}{\rho_i}, 

is calculated from Dutt's equation (Eq. 4.23 in Viscosity of 
Liquids\ :footcite:p:`viswanath_viscosity_2007`) provided :math:`T` in :math:`^{\circ}` C:

.. math::

   \nu_i = 10^{-6} \times \exp \bigg\{-3.0171 + \frac{442.78 + 1.6452 \,T_{b,i}}{T + 239 - 0.19 \,T_{b,i}} \bigg\}.



Latent heat of vaporization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.latent_heat_vaporization
   :noindex:

The latent heat of vaporization for each compound at standard pressure and 
temperature is calculated from the enthalpy of vaporization as:

.. math::
   L_{v,\textit{stp},i} = \frac{\Delta H_{v,\textit{stp},i}}{M_{w,i}}.

The heat of vaporization for each compound is then adjusted for variations in 
temperature\ :footcite:p:`govindaraju_group_2016`:

.. math::
   L_{v,i} = L_{v,\textit{stp},i} \bigg(\frac{1 - T_{r,i}}{1-T_{r,b,i}} \bigg)^{0.38}.


Liquid molar volume
^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.molar_liquid_vol
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

.. automethod:: fuellib.fuel.Fuel.density
   :noindex:

The density of the *i-th* compound is given by

.. math::
   \rho_i = \frac{M_{w,i}}{V_{m,i}}.


Mass specific heat capacity of the liquid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.Cl
   :noindex:

The mass specific heat capacity for each compound at standard pressure temperature is calculated from the molar specific heat capacity as:

.. math::
   C_{\ell,i} = \dfrac{C_{p,i}}{M_{w,i}} 



Saturated vapor pressure
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.psat
   :noindex:

The saturated vapor pressure for each compound is calculated as a function of 
temperature using either the Lee–Kesler method\ :footcite:p:`lee_generalized_1975` 
or the Ambrose-Walton method\ :footcite:p:`ambrose_vapour_1989`.  Both methods solve

.. math::
   \ln p_{r,\text{sat},i} = f_i^{(0)} + \omega_i f_i^{(1)} + \omega_i^2 f_i^{(2)}

for the reduced saturated vapor pressure for each compound, 
:math:`p_{r,\text{sat},i} = p_{\text{sat},i}/p_{c,i}`.  
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

.. automethod:: fuellib.fuel.Fuel.psat_antoine_coeffs
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

.. automethod:: fuellib.fuel.Fuel.surface_tension
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

.. automethod:: fuellib.fuel.Fuel.thermal_conductivity
   :noindex:

Thermal conductivity for each compound is computed according to the method of 
Latini et al. as summarized in Poling's\ :footcite:p:`poling_properties_2001` book:

.. math:: 
   \lambda_i = \frac{A_i(1 - T_{r,i})^{0.38}}{T_{r,i}^{1/6}}.

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

.. autofunction:: fuellib.correlate.components.flash_point_alqaheem
   :noindex:

.. autofunction:: fuellib.correlate.components.flash_point_alibashki
   :noindex:

Two correlations are available for the flash point :math:`T_{fp,i}` of each compound,
both based on the Constantinou-Gani normal boiling point :math:`T_{b,i}`.
The Alqaheem and Riazi (2017) correlation takes the flash point as a fixed fraction
of the boiling point:

.. math::
   T_{fp,i} = 0.70 \, T_{b,i}.

The modified group-contribution correlation of Alibakhshi et al. (2015) adds a
group-contribution term:

.. math::
   T_{fp,i} = 12.14 + 0.73 \, T_{b,i} + \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} \phi_k,

where :math:`\phi_k` are the group contributions registered as the ``alibakhshi_phi``
property of the ``gani`` GCM (:doc:`gcm`).

Lower heating value
^^^^^^^^^^^^^^^^^^^

.. autofunction:: fuellib.correlate.components.lower_heating_value
   :noindex:

The lower (net) heating value is computed from a Hess cycle using the Constantinou-Gani
enthalpy of formation :math:`\Delta H_{f,i}` and enthalpy of vaporization
:math:`\Delta H_{v,\textit{stp},i}`. The liquid-phase enthalpy of formation is

.. math::
   \Delta H_{f,\ell,i} = \Delta H_{f,i} - \Delta H_{v,\textit{stp},i},

and the enthalpy of combustion of a compound with :math:`n_{C,i}` carbon and
:math:`n_{H,i}` hydrogen atoms (from its RDKit molecule) is

.. math::
   \Delta H_{c,i} = n_{C,i} \, \Delta H_f^{\text{CO}_2(g)}
   + \frac{n_{H,i}}{2} \, \Delta H_f^{\text{H}_2\text{O}(g)} - \Delta H_{f,\ell,i},

with :math:`\Delta H_f^{\text{CO}_2(g)} = -393.51` kJ/mol and
:math:`\Delta H_f^{\text{H}_2\text{O}(g)} = -241.83` kJ/mol at 298.15 K. The lower
heating value is then

.. math::
   LHV_i = -\frac{\Delta H_{c,i}}{M_{w,i}}.

This estimate is valid for hydrocarbons only and does not account for
heteroatom-containing groups.

.. _eq-mixture-properties:

Equations for mixture properties from GCM
-----------------------------------------

This section contains correlations for estimating physical properties of the 
mixture from the individual compound and physical properties defined in 
:ref:`eq-GCM-properties` and :ref:`eq-GCM-correlations`.  These correlations make 
it possible to evaluate physical properties at non-standard temperatures and 
pressures, as the group contribution properties are only defined at standard 
conditions. The :ref:`tab-mixture-properties` available in `FuelLib` are listed in 
table below.  Mass and mole fractions defined in Table :ref:`tab-mass-mole-fracs`
are used throughout this section.

.. _tab-mixture-properties:

.. table:: Mixture properties
   :widths: auto
   :align: center
   
   ===============  ===============  =====================
   Symbol           Units            Description
   ===============  ===============  =====================
   :math:`\rho`     kg/m\ :sup:`3`   Density
   :math:`\nu`      m\ :sup:`2`/s    Kinematic viscosity
   :math:`p_v`      Pa               Vapor pressure
   :math:`\sigma`   N/m              Surface tension
   :math:`\lambda`  W/m/K            Thermal conductivity
   :math:`T_{fr}`   K                Freeze point
   :math:`T_{fp}`   K                Flash point
   :math:`LHV`      J/kg             Heat of combustion
   ===============  ===============  =====================

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

.. automethod:: fuellib.fuel.Fuel.mixture_density
   :noindex:

The mixture's density is calculated as:

.. math::
   
   \rho = \sum_{i=1}^{N_c}Y_i\frac{M_{w,i}}{V_{m,i}}.


Mixture kinematic viscosity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.mixture_kinematic_viscosity
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

.. automethod:: fuellib.fuel.Fuel.mixture_vapor_pressure
   :noindex:

The vapor pressure of the mixture is calculated according to Raoult's law:

.. math::

   p_{v} = \sum_{i = 1}^{N_c} X_i \, p_{\textit{sat},i}.

.. automethod:: fuellib.fuel.Fuel.mixture_vapor_pressure_antoine_coeffs
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

.. automethod:: fuellib.fuel.Fuel.mixture_surface_tension
   :noindex:

The surface tension of the mixture is calculated using the :ref:`conventional-mixing-rules`
with an arithmetic mean for the pseudo-property :math:`\sigma_{i,j}` as recommended by
Hugill and van Welsenes\ :footcite:p:`hugill_surface_1986`:

.. math::
   \sigma = \sum_{i=1}^{N_c} \sum_{j=1}^{N_c} X_i X_j \frac{\sigma_i + \sigma_j}{2}.

Mixture thermal conductivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: fuellib.fuel.Fuel.mixture_thermal_conductivity
   :noindex:

The thermal conductivity of the mixture is calculated using the power law method of 
Vredeveld as described in Poling\ :footcite:p:`poling_properties_2001`:

.. math::
   \lambda = \bigg(\sum_{i=1}^{N_c} Y_i \lambda_i^{-2} \bigg)^{-1/2}.

Mixture freeze point
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: fuellib.correlate.mixture.freeze_point_boehm
   :noindex:

The freeze point of the mixture is estimated with the solid-liquid equilibrium model
of Boehm et al. (2022), equation 21. For each compound :math:`j` with mole fraction
:math:`X_j`, the candidate freeze temperature :math:`T_j` solves

.. math::
   T_j \Big[\Delta S_{\textit{fus},j} + X_j \Delta C_{p,j} \ln\frac{T_j}{T_{m,j}}
   + \alpha \, \Delta S_{\textit{mix},j}\Big]
   = \Delta H_{\textit{fus},j} + X_j \Delta C_{p,j} (T_{m,j} - T_j),

where

.. math::
   \Delta S_{\textit{mix},j} &= -\frac{R}{X_j}\big[(1 - X_j)\ln(1 - X_j) + X_j \ln X_j\big], \\
   \Delta H_{\textit{fus},j} &= T_{m,j} \, \Delta S_{\textit{fus},j}, \\
   \Delta C_{p,j} &= -0.35 \, C_{p,\ell,j}(298.15 \text{ K}),

:math:`T_{m,j}` is the melting point from :attr:`~fuellib.fuel.Fuel.Tm` (a reference value
when available, otherwise the Constantinou-Gani prediction; see :ref:`sec-reference-properties`),
:math:`\Delta S_{\textit{fus},j}`
is provided by the ``boehm`` GCM (see :doc:`gcm`), and :math:`\alpha` is an optional
scaling of the ideal mixing-entropy term (default 1). The mixture freeze point is the
highest :math:`T_j` among compounds with :math:`X_j > 10^{-6}`, i.e. the temperature at
which the first crystal forms on cooling. This is an equilibrium screening model that
does not capture cooling rate, supercooling, or solid-phase nonideality.

Mixture flash point
^^^^^^^^^^^^^^^^^^^

.. autofunction:: fuellib.correlate.mixture.flash_point_alqaheem
   :noindex:

.. autofunction:: fuellib.correlate.mixture.flash_point_alibashki
   :noindex:

The mixture flash point is computed from the component flash points :math:`T_{fp,i}`
(see Flash point above) with either a linear mass-fraction mixing rule,

.. math::
   T_{fp} = \sum_{i=1}^{N_c} Y_i T_{fp,i},

or the ideal Liaw-Chiu (2006) mixing rule (default), which solves for :math:`T_{fp}` such that

.. math::
   \sum_{i=1}^{N_c} X_i \frac{p_{\textit{sat},i}(T_{fp})}{p_{\textit{sat},i}(T_{fp,i})} = 1,

where :math:`p_{\textit{sat},i}` is evaluated with the Lee-Kesler correlation. This
mixing rule does not model nonideal liquid activity coefficients.

Mixture heat of combustion
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: fuellib.correlate.mixture.heat_of_combustion
   :noindex:

The heat of combustion of the mixture is the mass-fraction weighted average of the
component lower heating values (see Lower heating value above):

.. math::
   LHV = \sum_{i=1}^{N_c} Y_i \, LHV_i.

This is a net heating value assuming gaseous water products, and is an engineering
estimate related to ASTM D4809/D3338 rather than a simulated bomb-calorimeter test.

Reference Compounds for Jet Fuels
---------------------------------

It is difficult to identify individual components of complex multicomponent jet fuels using GCxGC analysis, 
which generally provides weight percentages of a given hydrocarbon family and carbon number within a sample (e.g., 5% C10 iso-alkane, 2% C13 cycloalkane, etc.).
To address this challenge, FuelLib uses a set of reference compounds that represent the major hydrocarbon families and carbon numbers found in jet fuels.
A comprehensive list of the reference compounds used in FuelLib can be found in the 
`fuelData/refCompounds.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuelData/refCompounds.csv>`_ file, 
with associated functional group decompositions in `fuelData/groupDecompositionData/refCompounds.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuelData/groupDecompositionData/refCompounds.csv>`_.

For ease of reference, the reference compounds and keys corresponding to a PelePhysics mechanism `fuellib_posf_nonreacting <https://github.com/AMReX-Combustion/PelePhysics/tree/development/Mechanisms/fuellib_posf_nonreacting>`_ are provided in the table below.
When provided, the PelePhysics keys can be used to link the compounds in FuelLib to species in PelePhysics simulations via ``Export4Pele.py`` as described in :ref:`Exporting to PelePhysics <sec-exporting-to-pelephysics>`.

.. csv-table:: Reference compounds, chemical formulas, and corresponding PelePhysics keys by GCxGC bin.
   :file: ../fuellib/data/fuelData/refCompounds.csv
   :header-rows: 1
   :align: center
   :widths: auto

.. _sec-reference-properties:

Reference Property Data
-----------------------

Measured properties of pure compounds can replace group-contribution predictions.
These data are stored in two CSV files in ``fuellib/data`` and accessed through
:mod:`fuellib.data.references`:

- ``refCompounds.csv``: ``Common_Name``, ``InChI``, ``SMILES``, ``Num_C``, ``Family``.
- ``refProperties.csv``: ``Common_Name``, ``Property``, ``Units``, ``Value``, ``Error``, ``Source``.

Only ``Common_Name`` and ``SMILES`` are required for each compound. Missing ``InChI``,
``Num_C``, and ``Family`` values are derived from the SMILES when the data are loaded
(:func:`~fuellib.data.references.populate_missing`); the files are only modified by
calling :func:`~fuellib.data.references.update_compounds_csv`. ``Family`` is one of
``n-alkane``, ``iso-alkane``, ``alkene``, ``monocyclic``, ``dicyclic``, ``tricyclic``,
``alkylbenzene``, ``cycloaromatic``, or ``diaromatic``. ``Property`` is one of ``DCN``,
``YSI``, ``Tc``, ``Pc``, ``Vc``, ``Tm``, or ``Tb``, and ``Error`` is optional.

:func:`~fuellib.data.references.validate` checks both tables for consistency (exact headers,
unique names/InChI/SMILES, known families and properties, ``InChI`` and ``Num_C`` that
match the SMILES, numeric values, and properties that refer to listed compounds) and is
called by the loaders, so malformed data raise a ``ValueError`` listing every problem.
A compound's properties can be retrieved from a SMILES string with
:func:`~fuellib.data.references.properties_by_smiles`.

Currently, :attr:`~fuellib.fuel.Fuel.Tm` uses the reference melting point of each compound
found in the tables and the Constantinou-Gani prediction for all others. Pass
``use_references=False`` to :class:`~fuellib.fuel.Fuel` to use only the group-contribution
prediction.

Validation
----------

Single Component Fuels
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /figures/singleCompFuels.png
   :width: 600pt
   :align: center
   
   Properties of heptane, decane, and dodecane against predictive data from NIST Chemistry WebBook.

Multicomponent Fuels
^^^^^^^^^^^^^^^^^^^^^

.. figure:: /figures/multiCompFuels.png
   :width: 600pt
   :align: center

   Properties of conventional jet fuels JP-8 (POSF10264), Jet A (POSF10325), and JP-5 (POSF10289) against data from the Air Force Research Laboratory\ :footcite:p:`edwards_jet_2020`. Note that the data sets for thermal conductivity are very inconsistent, but they typically show linear decreases in thermal conductivity with temperature. 

Fuel Blends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /figures/hefaBlends.png
   :width: 250pt
   :align: center
   
   Properties of three HEFA fuels produced from different feedstocks (camelina, tallow, and mixed fat) blended with Jet-A.  Measurement and GCxGC data from Vozka et al.\ :footcite:p:`vozka_impact_2018`. 


References
----------

.. footbibliography::
