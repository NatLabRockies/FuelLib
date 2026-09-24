Group Contribution Method Abstraction
======================================

The :mod:`fuellib.gcm` module provides a generic, extensible framework for
registering and evaluating Group Contribution Method (GCM) property
predictions. It decouples the definition of individual property formulas
(such as the Constantinou-Gani correlations described in
:doc:`fuelprops`) from the :class:`~fuellib.fuel.Fuel` class, so new
methods or properties can be added without modifying ``fuel.py``.

Core concepts
-------------

Three pieces make up the abstraction, all defined in
:mod:`fuellib.gcm.core`:

- :class:`~fuellib.gcm.core.PropertyProtocol`: a runtime-checkable
  ``Protocol`` describing the required signature of a property function,
  ``(fuel: Fuel) -> Quantity1D``. Any callable matching this signature can
  be registered as a property predictor.
- :class:`~fuellib.gcm.core.GCM`: a named collection of property functions
  (a single group-contribution *method*, e.g. ``"gani"``). It supports
  registering, listing, retrieving, and evaluating property functions.
- :class:`~fuellib.gcm.core.GCMRegistry`: a process-wide registry of
  :class:`~fuellib.gcm.core.GCM` instances, keyed by method name.

.. code-block:: python

   from fuellib.gcm import GCMRegistry

   GCMRegistry.list_methods()
   # ['gani']

   gani = GCMRegistry.get_gcm("gani")
   gani.list_property_fns()
   # ['alibakhshi_phi', 'cp_b', 'cp_c', 'cp_stp', 'gf', 'hf', 'hv_stp', 'mw', ...]

   gani.predict("Tc", fuel)       # Quantity1D of critical temperatures
   gani.predict_all(fuel)         # {'gani': {'tc': ..., 'pc': ..., ...}}

Method and property name lookups (``GCMRegistry.get_gcm``,
``GCM.get_property``, ``GCM.predict``) are case-insensitive. Looking up an
unregistered method or property raises a ``ValueError`` listing the
available options.

The ``gani`` method
-------------------

:mod:`fuellib.gcm.gani` implements the Constantinou-Gani (and extended)
correlations from :doc:`fuelprops` on top of this abstraction. On import,
it registers a ``"gani"`` :class:`~fuellib.gcm.core.GCM` with
:class:`~fuellib.gcm.core.GCMRegistry` and populates it with property
functions for ``MW``, ``Tc``, ``Pc``, ``Vc``, ``Tb``, ``Tm``, ``Hf``,
``Gf``, ``Hv_stp``, ``omega``, ``Vm_stp``, ``Cp_stp``, ``Cp_B``, ``Cp_C``,
``rd_A``, ``rd_B``, ``rd_D``, and ``alibakhshi_phi``. Group-contribution
coefficients are read from ``fuellib/gcm/gani.csv`` into a module-level
table indexed by property name, with one column per first- or
second-order group.

.. _tab-GCM-properties-gcm:

Table of GCM properties
^^^^^^^^^^^^^^^^^^^^^^^^

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

Each of these properties maps directly onto a registered ``gani`` property
function of the same (lowercased) name (e.g. :math:`T_{c,i}` -> ``Tc``,
:math:`\Delta H_{f,i}` -> ``Hf``). See :doc:`fuelprops` for the full
description of the underlying physics and additional derived correlations
(density, viscosity, vapor pressure, surface tension, thermal conductivity)
built on top of these GCM properties.

.. _eq-GCM-properties-gcm:

Equations for GCM properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

These closed-form expressions are exactly what each corresponding ``gani``
property function evaluates, using the decomposition matrices ``nij``
(:math:`\mathbf{N}`) and, where second-order groups are involved,
``mij`` (:math:`\mathbf{M}`) returned by ``_get_decomp`` and the
coefficient rows (:math:`m_{w1k}`, :math:`t_{c1k}`, ...) returned by
``_get_row``.

Each property function follows the same pattern:

1. Build the fuel's group decomposition matrix, reindexed to match the
   table's canonical column (group) order, via the internal
   ``_get_decomp`` helper. Any group present in the table but missing from
   the fuel's decomposition data is treated as zero, so a fuel need not
   define every group as long as the ones it does define are covered by
   the table (partial group coverage).
2. Look up the property's group-contribution row from the table via the
   internal ``_get_row`` helper (raises ``KeyError`` if the property has
   no row in the table).
3. Apply the property's closed-form correlation and return the result as
   a unit-aware :data:`~fuellib.utils.types.Quantity1D` (see :doc:`units`).

Registering a new property
---------------------------

New properties can be added to an existing GCM (or a new one) with the
:meth:`~fuellib.gcm.core.GCM.register_property` decorator. A property
function must accept a :class:`~fuellib.fuel.Fuel` and return a
:data:`~fuellib.utils.types.Quantity1D` with one value per compound in
``fuel.compounds``:

.. code-block:: python

   from fuellib.gcm.gani import gani_gcm
   from fuellib.utils import Units, types


   @gani_gcm.register_property
   def my_new_property(fuel: "Fuel") -> types.Quantity1D:
       """Predict some new property from the Gani decomposition."""
       nij = fuel.gani_decomp().to_numpy()
       coeffs = ...  # derive coefficients, e.g. from a new table row
       value = ...  # apply the correlation
       return Units.Quantity(value, "K")

The function is registered under its (lowercased) ``__name__``, so it
immediately becomes available via ``gani_gcm.predict("my_new_property",
fuel)`` and through ``fuel.get_property("gani", "my_new_property")`` once
a new :class:`~fuellib.fuel.Fuel` (or its cached
:attr:`~fuellib.fuel.Fuel.gcm_properties`) is evaluated.

A completely new method can likewise be registered with
:meth:`~fuellib.gcm.core.GCMRegistry.register`:

.. code-block:: python

   from fuellib.gcm import GCMRegistry

   my_gcm = GCMRegistry.register("my_method", property_fns=[])


   @my_gcm.register_property
   def some_property(fuel: "Fuel") -> types.Quantity1D:
       ...

``Fuel`` integration
---------------------

:class:`~fuellib.fuel.Fuel` exposes three members that tie into the
registry:

- :meth:`~fuellib.fuel.Fuel.gani_decomp`: reads the fuel's group
  decomposition file and returns it restricted to ``fuel.compounds``,
  raising a ``ValueError`` if any compound in the mixture is missing from
  the file.
- :attr:`~fuellib.fuel.Fuel.gcm_properties` (a cached property):
  evaluates every registered GCM method against the fuel and returns a
  nested mapping ``{method_name: {property_name: Quantity1D}}``. Because it
  is cached, each method/property pair is only computed once per
  ``Fuel`` instance.
- :meth:`~fuellib.fuel.Fuel.get_property`: a case-insensitive lookup into
  ``gcm_properties``, raising ``KeyError`` if either the method or the
  property name is not present.

.. code-block:: python

   fuel = Fuel("decane")

   fuel.gani_decomp()                       # DataFrame, one row per compound
   fuel.gcm_properties["gani"]["tc"]        # Quantity1D of critical temperatures
   fuel.get_property("gani", "Tc")          # same values, case-insensitive lookup

``Fuel.__init__`` itself now calls ``self.get_property("gani", ...)`` to
populate its critical-property attributes (``Tc``, ``Pc``, ``Vc``, etc.)
instead of duplicating the group-contribution formulas inline, so those
attributes and the registry stay in sync automatically.

See also
--------

- :doc:`fuelprops` for the underlying Constantinou-Gani correlations and
  their physical meaning.
- :doc:`units` for details on :data:`~fuellib.utils.types.Quantity1D` and
  the unit-aware value types returned by GCM property functions.
- ``tests/test_gcm.py`` for further usage examples, including registration,
  error handling, and decomposition-matrix behavior.

References
----------

.. footbibliography::

