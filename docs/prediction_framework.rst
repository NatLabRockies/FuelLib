Fuel Property Prediction Framework
==================================

FuelLib predicts the properties of fuels from their composition. Fuels such as
synthetic aviation turbine fuels (SATFs) are complex mixtures, and measured
property data are often limited. FuelLib combines group contribution methods
(GCMs), measured data for pure compounds, correlations, and mixture rules to
estimate the properties of a fuel without extensive physical testing. This
helps fuel developers identify promising fuel compositions before committing to
large-scale production.

FuelLib builds on
`Pavan B. Govindaraju's Matlab implementation <https://github.com/gpavanb-old/GroupContribution>`_
of the Constantinou-Gani GCM\ :footcite:p:`govindaraju_group_2016`, and
includes gas chromatography data (GCxGC) for various jet fuels from the
National Jet Fuel Combustion Program\ :footcite:p:`colket_overview_2017`
(NJFCP), the Air Force Research
Laboratory\ :footcite:p:`edwards_reference_2017,edwards_jet_2020` (AFRL), and
Vozka et al.\ :footcite:p:`vozka_impact_2018`.

.. _sec-prediction-methodology:

FuelLib's property prediction methodology
-----------------------------------------

FuelLib predicts the properties of a fuel in four steps.

1. **Fuels are represented as a collection of individual compounds based on
   their GCxGC data bins.** GCxGC analysis reports the weight percent of each
   hydrocarbon family and carbon number in a sample (e.g., 5% C10 isoalkane).
   FuelLib assigns pseudo compounds for each bin. This is generally a single
   compound per bin, but users can choose to distribute the weight percent of a
   bin across multiple compounds. FuelLib includes a selection of common
   :doc:`reference compounds <reference_compounds>`, and users can provide
   compounds as SMILES strings as well. The weight percents of the compounds
   become the initial mass fractions, :attr:`~fuellib.fuel.core.Fuel.y0`.
2. **GCMs or measurement data predict the properties of individual compounds,
   typically at 298 K and 1 atm.** Measured properties of pure compounds are
   used when they are available (see :ref:`sec-reference-properties`).
   Otherwise, the properties are predicted from the groups that make up each
   compound with one of the :doc:`group_contribution_methods`.
3. **Correlations predict the properties of individual compounds at other
   conditions.** The standard-condition properties are inputs to correlations
   for temperature-dependent properties such as density, viscosity, vapor
   pressure, and surface tension (see :ref:`eq-GCM-correlations`).
4. **Mixture rules predict the properties of the mixture.** The compound
   properties are combined with mixture rules weighted by the mass or mole
   fraction of each compound (see :ref:`eq-mixture-properties`).

The table below summarizes the source of each compound property at standard
conditions.

.. _tab-property-sources:

.. list-table:: Sources of the compound properties at standard conditions.
   :header-rows: 1
   :widths: 55 45
   :align: center

   * - Property
     - Source
   * - Critical properties, boiling and melting points, enthalpies of formation and vaporization, liquid molar volume, and acentric factor
     - Reference data when available, otherwise the Constantinou-Gani GCM\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995`.
   * - Gibbs free energy of formation and specific heat capacity
     - Constantinou-Gani GCM\ :footcite:p:`constantinou_new_1994,nielsen_molecular_1998,poling_properties_2001`.
   * - Viscosity parameters
     - Nannoolal GCM\ :footcite:p:`nannoolal_estimation_2009`.
   * - Volume and surface parameters, activity coefficients
     - UNIFAC 2.0\ :footcite:p:`hayer_advancing_2025`.
   * - Entropy of fusion
     - Boehm GCM\ :footcite:p:`boehm_blend_2022`.
   * - Yield sooting index and derived cetane number
     - Reference data only. Yield sooting indices are primarily from the Yield Sooting Index Database\ :footcite:p:`mcenally_yield_2017`.
   * - Molecular weight, atom counts, and hydrocarbon type
     - `RDKit <https://www.rdkit.org>`_, from the SMILES string.

.. _sec-notation:

Notation
^^^^^^^^

A tilde denotes a molar property (e.g., :math:`\tilde{C}_{p,i}` in J/mol/K and
:math:`\Delta\tilde{H}_{v,i}` in J/mol), while the same symbol without a tilde
denotes the corresponding mass-based property (e.g., :math:`C_{p,i}` in J/kg/K
and :math:`\Delta H_{v,i}` in J/kg). The subscript *i* denotes the *i-th*
compound in a mixture. The group-contribution parameters (e.g.,
:math:`h_{v1k}`) follow the notation of the original publications.

.. _sec-fuel-class:

The Fuel Class
--------------

:class:`~fuellib.fuel.core.Fuel` objects wrap around multiple GCMs,
correlations, and mixture rules, and are the easiest way to access fuel
properties. A fuel is created from the name of its GCxGC data file:

.. code-block:: python

   from fuellib import Fuel, Units

   fuel = Fuel("posf10325")
   temp = Units.Quantity(300.0, "K")

   fuel.num_compounds          # number of compounds
   fuel.comps.tc               # critical temperature of each compound
   fuel.comps.rho(temp)        # density of each compound at 300 K
   fuel.mix.rho(fuel.y0, temp) # mixture density at 300 K

Pass ``use_ref_props=False`` to use only GCM predictions, or ``data_dir`` to
use custom fuels and reference compounds (see :doc:`tutorials_custom_fuels`).
All quantities carry units (see :ref:`sec-units`).

Core attributes (``Fuel.<attribute>``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core attributes hold the input data and composition of the fuel.

.. autosummary::

   ~fuellib.fuel.core.Fuel.name
   ~fuellib.fuel.core.Fuel.data_dir
   ~fuellib.fuel.core.Fuel.gc_data
   ~fuellib.fuel.core.Fuel.components_data
   ~fuellib.fuel.core.Fuel.prop_data
   ~fuellib.fuel.core.Fuel.metadata
   ~fuellib.fuel.core.Fuel.y0
   ~fuellib.fuel.core.Fuel.x0
   ~fuellib.fuel.core.Fuel.num_compounds
   ~fuellib.fuel.core.Fuel.comps
   ~fuellib.fuel.core.Fuel.mix

Individual compound properties (``Fuel.comps.<att-method>``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`Fuel.comps <fuellib.fuel.core.Fuel.comps>` is a
:class:`~fuellib.fuel.components.Components` object. Its attributes are
properties at standard conditions or structural properties, which are computed
the first time they are accessed. Its methods evaluate properties at a given
temperature with the :ref:`eq-GCM-correlations`, or convert between mass, mole,
and volume fractions. Each returns one value per compound.

**Attributes**

.. autosummary::

   ~fuellib.fuel.components.Components.tc
   ~fuellib.fuel.components.Components.pc
   ~fuellib.fuel.components.Components.vc
   ~fuellib.fuel.components.Components.tm
   ~fuellib.fuel.components.Components.tb
   ~fuellib.fuel.components.Components.hf_stp
   ~fuellib.fuel.components.Components.hv_stp
   ~fuellib.fuel.components.Components.vm_stp
   ~fuellib.fuel.components.Components.omega
   ~fuellib.fuel.components.Components.ysi
   ~fuellib.fuel.components.Components.dcn
   ~fuellib.fuel.components.Components.gf_stp
   ~fuellib.fuel.components.Components.lj_eps_by_kb
   ~fuellib.fuel.components.Components.lj_sigma
   ~fuellib.fuel.components.Components.names
   ~fuellib.fuel.components.Components.smiles
   ~fuellib.fuel.components.Components.pelephysics_keys
   ~fuellib.fuel.components.Components.rdkit_mols
   ~fuellib.fuel.components.Components.num_c
   ~fuellib.fuel.components.Components.num_h
   ~fuellib.fuel.components.Components.mw
   ~fuellib.fuel.components.Components.formulas
   ~fuellib.fuel.components.Components.hc_type
   ~fuellib.fuel.components.Components.fam

**Methods**

.. autosummary::
   :nosignatures:

   ~fuellib.fuel.components.Components.rho
   ~fuellib.fuel.components.Components.nu
   ~fuellib.fuel.components.Components.mu
   ~fuellib.fuel.components.Components.cp
   ~fuellib.fuel.components.Components.pvap
   ~fuellib.fuel.components.Components.pvap_antoine_coeffs
   ~fuellib.fuel.components.Components.vm
   ~fuellib.fuel.components.Components.hv
   ~fuellib.fuel.components.Components.d
   ~fuellib.fuel.components.Components.sigma
   ~fuellib.fuel.components.Components.k
   ~fuellib.fuel.components.Components.t_flash
   ~fuellib.fuel.components.Components.mass2y
   ~fuellib.fuel.components.Components.mass2x
   ~fuellib.fuel.components.Components.x2y
   ~fuellib.fuel.components.Components.y2x
   ~fuellib.fuel.components.Components.y2v

Mixture properties (``Fuel.mix.<method>``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`Fuel.mix <fuellib.fuel.core.Fuel.mix>` is a
:class:`~fuellib.fuel.mixture.Mixture` object. Its methods take the mass
fractions of the compounds, ``yi``, and return a single value for the mixture
with the :ref:`eq-mixture-properties`.

.. autosummary::
   :nosignatures:

   ~fuellib.fuel.mixture.Mixture.rho
   ~fuellib.fuel.mixture.Mixture.nu
   ~fuellib.fuel.mixture.Mixture.mu
   ~fuellib.fuel.mixture.Mixture.pvap
   ~fuellib.fuel.mixture.Mixture.pvap_antoine_coeffs
   ~fuellib.fuel.mixture.Mixture.sigma
   ~fuellib.fuel.mixture.Mixture.k
   ~fuellib.fuel.mixture.Mixture.mw
   ~fuellib.fuel.mixture.Mixture.t_freeze
   ~fuellib.fuel.mixture.Mixture.t_flash
   ~fuellib.fuel.mixture.Mixture.lhv
   ~fuellib.fuel.mixture.Mixture.ysi
   ~fuellib.fuel.mixture.Mixture.smoke_point
   ~fuellib.fuel.mixture.Mixture.dcn
   ~fuellib.fuel.mixture.Mixture.aromatics
   ~fuellib.fuel.mixture.Mixture.naphthalenes

References
----------

.. footbibliography::
