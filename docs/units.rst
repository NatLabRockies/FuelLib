Units
=====

FuelLib's public API is unit-aware: physical quantities (e.g. molecular
weight, critical temperature, vapor pressure) are returned as
`pint <https://pint.readthedocs.io>`_ ``Quantity`` objects rather than bare
``float``/``numpy.ndarray`` values. This makes the units of every quantity
explicit and lets you convert between unit systems without manually tracking
conversion factors.

All quantities are created from a single shared unit registry,
``fuellib.PintUnits`` (equivalently ``fuellib.units.PintUnits``). Quantities
created from a different ``pint.UnitRegistry`` instance are **not**
compatible with FuelLib's quantities, so always use ``fl.PintUnits`` (or
``fl.PintUnits.Quantity``) to build new quantities of your own, such as a
temperature to evaluate a correlation at.

Basic example
-------------

.. code-block:: python

   import fuellib as fl

   fuel = fl.Fuel("heptane-decane")

   # Properties on the Fuel object are pint.Quantity values and print with units
   print(f"Critical temperature: {fuel.Tc:.2f}")

   # Convert to another unit with `.to(...)`
   print(f"Critical temperature (°C): {fuel.Tc.to('degC'):.2f}")

   # Get the raw numeric value with `.magnitude`
   print(f"Critical temperature, magnitude only: {fuel.Tc.magnitude}")

   # Build your own quantity using the shared registry, then pass it to a
   # temperature-dependent correlation
   T = fl.PintUnits.Quantity(320.0, "K")
   p_sat_i = fuel.psat(T)
   print(f"Saturated vapor pressure at {T}: {p_sat_i:.2f}")

Arithmetic between quantities automatically combines/cancels units, and pint
raises an error if you try to combine incompatible units (e.g. adding a
temperature to a pressure), which helps catch unit-mismatch bugs early.

Further reading
----------------

This page only covers the basics needed to work with FuelLib. For the full
set of supported units, unit conversions, formatting options, and advanced
usage (e.g. NumPy integration, uncertainty propagation), see the
`pint documentation <https://pint.readthedocs.io>`_.
