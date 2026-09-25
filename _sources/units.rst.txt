Units
=====

FuelLib's public API is unit-aware: physical quantities (e.g. molecular
weight, critical temperature, vapor pressure) are returned as
`pint <https://pint.readthedocs.io>`_ ``Quantity`` objects rather than bare
``float``/``numpy.ndarray`` values. This makes the units of every quantity
explicit and lets you convert between unit systems without manually tracking
conversion factors.

All quantities are created from a single shared unit registry,
``fuellib.Units`` (equivalently ``fuellib.utils.Units``). Quantities
created from a different ``pint.UnitRegistry`` instance are **not**
compatible with FuelLib's quantities, so always use ``fl.Units`` (or
``fl.Units.Quantity``) to build new quantities of your own, such as a
temperature to evaluate a correlation at.

Basic example
-------------

.. code-block:: python

   import fuellib as fl

   fuel = fl.Fuel("heptane-decane")

   # Properties on the Fuel object are pint.Quantity values and print with units
   print(f"Critical temperature: {fuel.Tc:.2f}")

   # Convert to another unit with `.to(...)`
   print(f"Critical temperature (°C): {fuel.Tc.to('celsius'):.2f}")

   # Get the raw numeric value with `.magnitude`
   print(f"Critical temperature, magnitude only: {fuel.Tc.magnitude}")

   # Build your own quantity using the shared registry, then pass it to a
   # temperature-dependent correlation
   T = fl.Units.Quantity(320.0, "K")
   p_sat_i = fuel.psat(T)
   print(f"Saturated vapor pressure at {T}: {p_sat_i:.2f}")

Arithmetic between quantities automatically combines/cancels units, and pint
raises an error if you try to combine incompatible units (e.g. adding a
temperature to a pressure), which helps catch unit-mismatch bugs early.

Type annotations
----------------

FuelLib provides type aliases from ``fuellib.utils.types`` for annotating
unit-aware code. ``Quantity0D`` represents a Pint quantity with a scalar
magnitude, while ``Quantity1D`` and ``Quantity2D`` represent quantities with
one- and two-dimensional ``float64`` NumPy-array magnitudes. Use ``Array1D``
and ``Array2D`` for the corresponding bare NumPy arrays.

For example:

.. code-block:: python

   from fuellib.utils import Units, types

   def heat_sample(
      temperature: types.Quantity0D,
      heat_capacity: types.Quantity1D,
   ) -> types.Quantity1D:
      temperature = temperature.to("K")
      return heat_capacity * (temperature / Units.Quantity(1, "K"))

Further reading
----------------

This page only covers the basics needed to work with FuelLib. For the full
set of supported units, unit conversions, formatting options, and advanced
usage (e.g. NumPy integration, uncertainty propagation), see the
`pint documentation <https://pint.readthedocs.io>`_.
