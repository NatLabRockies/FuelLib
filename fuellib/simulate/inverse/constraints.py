"""Constraints for inverse fuel composition modeling."""

from typing import Callable

import jax.numpy as jnp
from jax import Array
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    model_validator,
)

from ...fuel import Fuel
from ...utils import types
from . import mixing


class BaseConstraint(BaseModel):
    """Base class for all constraints."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_attribute_docstrings=True,
        arbitrary_types_allowed=True,
    )
    weight: NonNegativeFloat = 1.0
    """Penalty weight of the limit (0 disables them)."""


class DensityConstraint(BaseConstraint):
    """Density limits at a single temperature."""

    temp: types.Quantity0D
    """Temperature in Kelvin at which the limits are defined."""
    min: types.Quantity0D | None = None
    """Minimum allowed density (None disables the lower bound)."""
    max: types.Quantity0D | None = None
    """Maximum allowed density (None disables the upper bound)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "DensityConstraint":
        if self.min is None and self.max is None:
            raise ValueError("at least one of min or max must be set")
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        elif self.max is None and self.min == 0:
            raise ValueError("min must be > 0 when max is None")
        return self

    def labels(self) -> list[str]:
        """Get the names of the penalty terms.

        Returns:
            Names of the terms returned by `penalty`, in the same order.
        """
        if self.weight <= 0:
            return []
        return [
            f"{b.capitalize()} density @ {self.temp:g~P}"
            for b in ("min", "max")
            if getattr(self, b) is not None
        ]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalty for the density limit.

        Raises:
            ValueError: If neither min nor max is set for the density constraint.
        """
        rho_i = jnp.asarray(fuel.density(self.temp).m_as("kg/m^3"))
        lo = self.min.m_as("kg/m^3") if self.min is not None else None
        hi = self.max.m_as("kg/m^3") if self.max is not None else None
        scale = hi or lo
        if scale is None:
            msg = f"Min or max must be set for density constraint at {self.temp:.2f~P}"
            raise ValueError(msg)

        def penalty(Yi: Array) -> list[Array]:
            rho = mixing.density(Yi, rho_i)
            g = []
            if lo is not None:
                g.append(jnp.maximum(lo - rho, 0.0) / scale)
            if hi is not None:
                g.append(jnp.maximum(rho - hi, 0.0) / scale)
            return g

        return penalty


class ViscosityConstraint(BaseConstraint):
    """Viscosity limits at a single temperature."""

    temp: types.Quantity0D
    """Temperature in Kelvin at which the limits are defined."""
    min: types.Quantity0D | None = None
    """Minimum allowed viscosity (None disables the lower bound)."""
    max: types.Quantity0D | None = None
    """Maximum allowed viscosity (None disables the upper bound)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "ViscosityConstraint":
        if self.min is None and self.max is None:
            raise ValueError("at least one of min or max must be set")
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        elif self.max is None and self.min == 0:
            raise ValueError("min must be > 0 when max is None")
        return self

    def labels(self) -> list[str]:
        """Get the names of the penalty terms.

        Returns:
            Names of the terms returned by `penalty`, in the same order.
        """
        if self.weight <= 0:
            return []
        return [
            f"{b.capitalize()} viscosity @ {self.temp:g~P}"
            for b in ("min", "max")
            if getattr(self, b) is not None
        ]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalty for the viscosity limit.

        Raises:
            ValueError: If neither min nor max is set for the viscosity constraint.
        """
        mu_i = jnp.asarray(fuel.viscosity_kinematic(self.temp).m_as("mm^2/s"))
        mw_i = jnp.asarray(fuel.MW.m_as("g/mol"))
        lo = self.min.m_as("mm^2/s") if self.min is not None else None
        hi = self.max.m_as("mm^2/s") if self.max is not None else None
        scale = hi or lo
        if scale is None:
            msg = (
                f"Min or max must be set for viscosity constraint at {self.temp:.2f~P}"
            )
            raise ValueError(msg)

        def penalty(Yi: Array) -> list[Array]:
            mu = mixing.viscosity(Yi, mu_i, mw_i)
            g = []
            if lo is not None:
                g.append(jnp.maximum(lo - mu, 0.0) / scale)
            if hi is not None:
                g.append(jnp.maximum(mu - hi, 0.0) / scale)
            return g

        return penalty
