Source Code
===========

This page describes the units, the GCM registry, and the public API of the
source code available at `github.com/NatLabRockies/FuelLib <https://github.com/NatLabRockies/FuelLib>`_.

.. _sec-units:

Units
-----

FuelLib's public API is unit-aware: physical quantities (e.g. molecular
weight, critical temperature, vapor pressure) are returned as
`pint <https://pint.readthedocs.io>`_ ``Quantity`` objects rather than bare
``float``/``numpy.ndarray`` values. This makes the units of every quantity
explicit and lets you convert between unit systems without manually tracking
conversion factors.

All quantities are created from a single shared unit registry,
:data:`fuellib.Units <fuellib.utils.types.Units>` (equivalently
:data:`fuellib.utils.Units <fuellib.utils.types.Units>`). Quantities
created from a different ``pint.UnitRegistry`` instance are **not**
compatible with FuelLib's quantities, so always use
:data:`fl.Units <fuellib.utils.types.Units>` (or
``fl.Units.Quantity``) to build new quantities of your own, such as a
temperature to evaluate a correlation at.

Basic example
^^^^^^^^^^^^^

.. code-block:: python

   import fuellib as fl

   fuel = fl.Fuel("heptane_decane")

   # Properties on the Fuel object are pint.Quantity values and print with units
   print(f"Critical temperature: {fuel.comps.tc:~P.2f}")

   # Convert to another unit with `.to(...)`
   print(f"Critical temperature: {fuel.comps.tc.to('celsius'):~P.2f}")

   # Get the raw numeric value with `.magnitude`
   print(f"Critical temperature, magnitude only: {fuel.comps.tc.magnitude}")

   # Build your own quantity using the shared registry, then pass it to a
   # temperature-dependent correlation
   T = fl.Units.Quantity(320.0, "K")
   pvap_i = fuel.comps.pvap(T)
   print(f"Component vapor pressure at {T:~P.0f}: {pvap_i:~P.2f}")

Arithmetic between quantities automatically combines/cancels units, and pint
raises an error if you try to combine incompatible units (e.g. adding a
temperature to a pressure), which helps catch unit-mismatch bugs early.

Type annotations
^^^^^^^^^^^^^^^^

FuelLib provides type aliases from :mod:`fuellib.utils.types` for annotating
unit-aware code. :data:`~fuellib.utils.types.Quantity0D` represents a Pint
quantity with a scalar magnitude, while :data:`~fuellib.utils.types.Quantity1D`
and :data:`~fuellib.utils.types.Quantity2D` represent quantities with
one- and two-dimensional ``float64`` NumPy-array magnitudes. Use
:data:`~fuellib.utils.types.Array1D` and :data:`~fuellib.utils.types.Array2D`
for the corresponding bare NumPy arrays.

For example:

.. code-block:: python

   from fuellib.utils import Units, types

   def heat_sample(
      temperature: types.Quantity0D,
      heat_capacity: types.Quantity1D,
   ) -> types.Quantity1D:
      temperature = temperature.to("K")
      return heat_capacity * (temperature / Units.Quantity(1, "K"))

This section only covers the basics needed to work with FuelLib. For the full
set of supported units, unit conversions, formatting options, and advanced
usage (e.g. NumPy integration, uncertainty propagation), see the
`pint documentation <https://pint.readthedocs.io>`_.

.. _sec-gcm-registry:

GCM Registry
------------

The :mod:`fuellib.gcm` module provides a generic, extensible framework for
registering and evaluating the predictions of the
:doc:`group_contribution_methods`. It decouples the definition of individual
property formulas from the :class:`~fuellib.fuel.core.Fuel` class, so new
methods or properties can be added without modifying the :mod:`fuellib.fuel`
package.

Registry classes
^^^^^^^^^^^^^^^^

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
   # ['boehm', 'gani', 'nannoolal', 'unifac']

   gani = GCMRegistry.get_gcm("gani")
   gani.list_property_fns()
   # ['alibakhshi_phi', 'cp_b', 'cp_c', 'cp_stp', 'gf', 'hf', 'hv_stp', 'omega', ...]

   gani.predict("tc", fuel)       # Quantity1D of critical temperatures
   gani.predict_all(fuel)         # {'gani': {'tc': ..., 'pc': ..., ...}}

Method and property name lookups
(:meth:`GCMRegistry.get_gcm <fuellib.gcm.core.GCMRegistry.get_gcm>`,
:meth:`GCM.get_property <fuellib.gcm.core.GCM.get_property>`,
:meth:`GCM.predict <fuellib.gcm.core.GCM.predict>`) are case-insensitive. Looking up an
unregistered method or property raises a ``ValueError`` listing the
available options.

Each method module registers its :class:`~fuellib.gcm.core.GCM` with
:class:`~fuellib.gcm.core.GCMRegistry` on import. For example,
:mod:`fuellib.gcm.gani` registers property functions for ``tc``,
``pc``, ``vc``, ``tb``, ``tm``, ``hf``, ``gf``, ``hv_stp``, ``omega``,
``vm_stp``, ``cp_stp``, ``cp_b``, ``cp_c``, ``rd_a``, ``rd_b``, ``rd_d``, and
``alibakhshi_phi``. Its group-contribution coefficients are read from
``fuellib/gcm/gani.csv`` into a module-level table indexed by property name,
with one column per first- or second-order group. Each property function
follows the same pattern:

1. Build the fuel's group decomposition matrix, reindexed to match the
   table's canonical column (group) order, via the internal
   ``_get_decomp`` helper. Any group present in the table but missing from
   the fuel's decomposition data is treated as zero, so a fuel need not
   define every group as long as the ones it does define are covered by
   the table (partial group coverage).
2. Look up the property's group-contribution row from the table via the
   internal ``_get_row`` helper (raises ``KeyError`` if the property has
   no row in the table).
3. Apply the property's closed-form correlation (see :ref:`eq-GCM-properties`)
   and return the result as a unit-aware
   :data:`~fuellib.utils.types.Quantity1D` (see :ref:`sec-units`).

Registering a new property
^^^^^^^^^^^^^^^^^^^^^^^^^^

New properties can be added to an existing GCM (or a new one) with the
:meth:`~fuellib.gcm.core.GCM.register_property` decorator. A property
function must accept a :class:`~fuellib.fuel.core.Fuel` and return a
:data:`~fuellib.utils.types.Quantity1D` with one value per compound in
:attr:`fuel.comps.names <fuellib.fuel.components.Components.names>`:

.. code-block:: python

   from fuellib.gcm.gani import gani_gcm
   from fuellib.utils import Units, types


   @gani_gcm.register_property
   def my_new_property(fuel: "Fuel") -> types.Quantity1D:
       """Predict some new property from the Gani decomposition."""
       nij = fuel._gani_decomp.to_numpy()
       coeffs = ...  # derive coefficients, e.g. from a new table row
       value = ...  # apply the correlation
       return Units.Quantity(value, "K")

The function is registered under its (lowercased) ``__name__``, so it
immediately becomes available via :meth:`gani_gcm.predict("my_new_property",
fuel) <fuellib.gcm.core.GCM.predict>` and, once a corresponding cached property is added to
:class:`~fuellib.fuel.components.Components`, as an attribute of
:attr:`fuel.comps <fuellib.fuel.core.Fuel.comps>`.

A completely new method can likewise be registered with
:meth:`~fuellib.gcm.core.GCMRegistry.register`:

.. code-block:: python

   from fuellib.gcm import GCMRegistry

   my_gcm = GCMRegistry.register("my_method", property_fns=[])


   @my_gcm.register_property
   def some_property(fuel: "Fuel") -> types.Quantity1D:
       ...

``Fuel`` integration
^^^^^^^^^^^^^^^^^^^^

:class:`~fuellib.fuel.core.Fuel` ties into the registry through the following members:

- ``Fuel._gani_decomp``: the group decomposition (from
  ``ref_gani.csv``) of each compound in the mixture, in the same order as
  :attr:`fuel.comps.names <fuellib.fuel.components.Components.names>`. Each
  compound in the GCxGC data must be present in the
  reference tables, otherwise a ``ValueError`` is raised when the
  :class:`~fuellib.fuel.core.Fuel` is created.
- ``Fuel._nannoolal_decomp``: the Nannoolal group
  decomposition (from ``ref_nannoolal.csv``) of each compound, used by the
  ``nannoolal`` method. Each compound must also be present in this table.
- ``Fuel._unifac_decomp``: the UNIFAC subgroup decomposition
  (from ``ref_unifac.csv``) of each compound, used by the ``unifac`` method.
  Each compound must also be present in this table.
- :attr:`Fuel.components_data <fuellib.fuel.core.Fuel.components_data>`: the rows of ``ref_compounds.csv``
  matched to the GCxGC data, including the hydrocarbon ``Family`` used by the
  ``boehm`` method.
- Cached properties of :attr:`fuel.comps <fuellib.fuel.core.Fuel.comps>` such as
  :attr:`~fuellib.fuel.components.Components.tc`,
  :attr:`~fuellib.fuel.components.Components.pc`, and
  :attr:`~fuellib.fuel.components.Components.tb` that
  evaluate the corresponding registered property with
  :meth:`gani_gcm.predict(...) <fuellib.gcm.core.GCM.predict>`
  the first time they are accessed.
  Prediction-only coefficients (``Cp_stp``, ``Cp_B``, ``Cp_C``, ``rd_A``,
  ``rd_B``, ``rd_D``, ``alibakhshi_phi``, and ``dS_fus``) are not cached on the
  fuel; call :meth:`gani_gcm.predict(...) <fuellib.gcm.core.GCM.predict>` or
  :meth:`boehm_gcm.predict(...) <fuellib.gcm.core.GCM.predict>` directly.
  Properties that can be measured (``Tc``, ``Pc``, ``Vc``, ``Tm``,
  ``Tb``, ``Hf_stp``, ``Hv_stp``, ``Vm_stp``, ``omega``) take reference values
  from ``ref_compounds.csv`` where available unless ``use_ref_props=False``
  (see :ref:`sec-reference-properties`).

.. code-block:: python

   fuel = Fuel("decane")

   fuel._gani_decomp       # DataFrame, one row per compound
   fuel._nannoolal_decomp  # DataFrame, one row per compound
   fuel._unifac_decomp     # DataFrame, one row per compound
   fuel.components_data    # DataFrame of matched reference compounds
   fuel.comps.tc           # Quantity1D of critical temperatures
   boehm_gcm.predict("dS_fus", fuel)  # Quantity1D of fusion entropies

Previously, ``Fuel.get_property``, ``Fuel.gcm_properties``, and ``Fuel.gani_decomp``
were used to access these values. They have been replaced by the attributes above, so
the attributes and the registry stay in sync through a single ``predict`` call.

See ``tests/test_gcm_gani.py``, ``tests/test_gcm_nannoolal.py``, and
``tests/test_gcm_unifac.py`` for further usage examples, including comparisons
of automated SMILES decompositions with the reference tables and checks of the
UNIFAC activity coefficients against a reference implementation, the
pure-component limit, and the Gibbs-Duhem relation.

Public API
----------

FuelLib's public API is continuously validated in CI using ``tests/test_api.py``.
This test verifies expected public module/class signatures and runs representative
FuelLib smoke evaluations to catch unintended behavior changes.

The project aims to keep the public API stable across releases. Any intentional
breaking API change should be explicitly documented in release notes and
accompanied by updates to tests and user-facing documentation.

The API is organized as it is in the source code, by package and module, down
to the individual classes, functions, and constants. Browse the packages below
or use the sidebar, the :ref:`genindex`, and the :ref:`modindex`.

.. autosummary::
   :toctree: generated
   :recursive:

   fuellib.correlate
   fuellib.exporters
   fuellib.fuel
   fuellib.gcm
   fuellib.rdk
   fuellib.simulate
   fuellib.utils
