"""Shared pint unit registry used throughout FuelLib."""

import pint

#: Single shared unit registry - all Quantities in FuelLib must use this registry.
ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)

#: Shorthand for constructing Quantities, e.g. ``Q_(300.0, "K")``.
Q_ = ureg.Quantity

__all__ = ["Q_", "ureg"]
