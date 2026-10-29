"""Constraints for inverse fuel composition modeling."""

from typing import Callable, ClassVar

import jax.numpy as jnp
import numpy as np
from jax import Array
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    model_validator,
)

from ...correlate import components
from ...fuel import Fuel
from ...utils import Units, types
from ...utils.constants import T_stp
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


class AromaticsConstraint(BaseConstraint):
    """Aromatics content limits (volume percent) for the fuel mixture."""

    min: types.Quantity0D | None = None
    """Minimum allowed aromatics content, e.g. `Units.Quantity(8, "percent")`
    (None disables the lower bound)."""
    max: types.Quantity0D | None = None
    """Maximum allowed aromatics content, e.g. `Units.Quantity(25, "percent")`
    (None disables the upper bound)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "AromaticsConstraint":
        if self.min is None and self.max is None:
            raise ValueError("at least one of min or max must be set")
        for b in (self.min, self.max):
            if b is not None and not 0 <= b.m_as("percent") <= 100:
                raise ValueError(f"aromatics limit ({b:~P}) must be in [0, 100] %")
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
            f"{b.capitalize()} aromatics content"
            for b in ("min", "max")
            if getattr(self, b) is not None
        ]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalty for the aromatics content limit.

        Raises:
            ValueError: If neither min nor max is set for the aromatics constraint.
        """
        rho_i = jnp.asarray(fuel.density(T_stp).m_as("kg/m^3"))
        is_aromatic = jnp.asarray(np.array(fuel.hc_type) == "aromatic", dtype=float)
        lo = self.min.m_as("percent") if self.min is not None else None
        hi = self.max.m_as("percent") if self.max is not None else None
        scale = hi or lo
        if scale is None:
            raise ValueError("Min or max must be set for aromatics constraint")

        def penalty(Yi: Array) -> list[Array]:
            aromatics = mixing.aromatics_content(Yi, rho_i, is_aromatic)
            g = []
            if lo is not None:
                g.append(jnp.maximum(lo - aromatics, 0.0) / scale)
            if hi is not None:
                g.append(jnp.maximum(aromatics - hi, 0.0) / scale)
            return g

        return penalty


class HeatOfCombustionConstraint(BaseConstraint):
    """Heat of combustion limits for the fuel mixture."""

    min: types.Quantity0D | None = None
    """Minimum allowed heat of combustion, e.g. `Units.Quantity(42, "MJ/kg")`
    (None disables the lower bound)."""
    max: types.Quantity0D | None = None
    """Maximum allowed heat of combustion, e.g. `Units.Quantity(45, "MJ/kg")`
    (None disables the upper bound)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "HeatOfCombustionConstraint":
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
            f"{b.capitalize()} heat of combustion"
            for b in ("min", "max")
            if getattr(self, b) is not None
        ]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalty for the heat of combustion limit.

        Raises:
            ValueError: If neither min nor max is set for the heat of combustion
                constraint.
        """
        Lhv_i = jnp.asarray(components.lower_heating_value(fuel).m_as("MJ/kg"))
        lo = self.min.m_as("MJ/kg") if self.min is not None else None
        hi = self.max.m_as("MJ/kg") if self.max is not None else None
        scale = hi or lo
        if scale is None:
            raise ValueError("Min or max must be set for heat of combustion constraint")

        def penalty(Yi: Array) -> list[Array]:
            hoc = mixing.heat_of_combustion(Yi, Lhv_i)
            g = []
            if lo is not None:
                g.append(jnp.maximum(lo - hoc, 0.0) / scale)
            if hi is not None:
                g.append(jnp.maximum(hoc - hi, 0.0) / scale)
            return g

        return penalty


class FlashPointConstraint(BaseConstraint):
    """Flash point limits for the fuel mixture."""

    min: types.Quantity0D | None = None
    """Minimum allowed flash point, e.g. `Units.Quantity(42, "celsius")`
    (None disables the lower bound)."""
    max: types.Quantity0D | None = None
    """Maximum allowed flash point, e.g. `Units.Quantity(45, "celsius")`
    (None disables the upper bound)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "FlashPointConstraint":
        if self.min is None and self.max is None:
            raise ValueError("at least one of min or max must be set")
        if self.min is not None and self.max is not None:
            if self.min.m_as("celsius") >= self.max.m_as("celsius"):
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        elif (
            self.max is None and self.min is not None and self.min.m_as("celsius") == 0
        ):
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
            f"{b.capitalize()} flash point"
            for b in ("min", "max")
            if getattr(self, b) is not None
        ]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalty for the flash point limit.

        Raises:
            ValueError: If neither min nor max is set for the flash point
                constraint.
        """
        Tfp_i = jnp.asarray(fuel.flash_point(correlation="Alibakhshi").m_as("celsius"))
        Tc_i = jnp.asarray(fuel.Tc.m_as("K"))
        Pc_i = jnp.asarray(fuel.Pc.m_as("Pa"))
        omega_i = jnp.asarray(fuel.omega)
        MW_i = jnp.asarray(fuel.MW.m_as("kg/mol"))
        lo = self.min.m_as("celsius") if self.min is not None else None
        hi = self.max.m_as("celsius") if self.max is not None else None
        scale = hi or lo
        if scale is None:
            raise ValueError("Min or max must be set for flash point constraint")

        def penalty(Yi: Array) -> list[Array]:
            fp = mixing.flash_point(Yi, Tfp_i, Tc_i, Pc_i, omega_i, MW_i)
            g = []
            if lo is not None:
                g.append(jnp.maximum(lo - fp, 0.0) / scale)
            if hi is not None:
                g.append(jnp.maximum(fp - hi, 0.0) / scale)
            return g

        return penalty


class FreezePointConstraint(BaseConstraint):
    """Freeze point limits for the fuel mixture."""

    min: types.Quantity0D | None = None
    """Minimum allowed freeze point, e.g. `Units.Quantity(42, "celsius")`
    (None disables the lower bound)."""
    max: types.Quantity0D | None = None
    """Maximum allowed freeze point, e.g. `Units.Quantity(45, "celsius")`
    (None disables the upper bound)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "FreezePointConstraint":
        if self.min is None and self.max is None:
            raise ValueError("at least one of min or max must be set")
        if self.min is not None and self.max is not None:
            if self.min.m_as("celsius") >= self.max.m_as("celsius"):
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        elif (
            self.max is None and self.min is not None and self.min.m_as("celsius") == 0
        ):
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
            f"{b.capitalize()} freeze point"
            for b in ("min", "max")
            if getattr(self, b) is not None
        ]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalty for the freeze point limit.

        Raises:
            ValueError: If neither min nor max is set for the freeze point
                constraint.
        """
        Tm_i = jnp.asarray(fuel.Tm.m_as("kelvin"))
        dS_fus_i = jnp.asarray(fuel.dS_fus.m_as("J/(mol*K)"))
        Cp_i = jnp.asarray(
            components.molar_specific_heat_capacity_ruzicka(
                fuel, Units.Quantity(298.15, "K")
            ).m_as("J/(mol*K)")
        )
        MW_i = jnp.asarray(fuel.MW.m_as("kg/mol"))
        lo = self.min.m_as("celsius") if self.min is not None else None
        hi = self.max.m_as("celsius") if self.max is not None else None
        scale = hi or lo
        if scale is None:
            raise ValueError("Min or max must be set for freeze point constraint")

        def penalty(Yi: Array) -> list[Array]:
            fp = mixing.freeze_point(Yi, Tm_i, dS_fus_i, Cp_i, MW_i)
            g = []
            if lo is not None:
                g.append(jnp.maximum(lo - fp, 0.0) / scale)
            if hi is not None:
                g.append(jnp.maximum(fp - hi, 0.0) / scale)
            return g

        return penalty


class DistillationCurveConstraint(BaseConstraint):
    """D86 distillation curve limits for the fuel mixture.

    The limits are currently fixed to the ASTM D7566 requirements:
    T10 <= 205 °C, T50 - T10 >= 15 °C, T90 - T10 >= 40 °C, and final boiling
    point <= 300 °C.
    """

    T10_max: ClassVar[float] = 205.0
    """Maximum 10% recovered temperature in Celsius."""
    T50_T10_min: ClassVar[float] = 15.0
    """Minimum difference between the 50% and 10% recovered temperatures in
    Celsius."""
    T90_T10_min: ClassVar[float] = 40.0
    """Minimum difference between the 90% and 10% recovered temperatures in
    Celsius."""
    Tf_max: ClassVar[float] = 300.0
    """Maximum final boiling point in Celsius."""

    def labels(self) -> list[str]:
        """Get the names of the penalty terms.

        Returns:
            Names of the terms returned by `penalty`, in the same order.
        """
        if self.weight <= 0:
            return []
        return ["Max T10", "Min T50 - T10", "Min T90 - T10", "Max final boiling point"]

    def bind(self, fuel: Fuel) -> Callable[[Array], list[Array]]:
        """Bind the constraint to a specific fuel mixture.

        Args:
            fuel: Object representing the fuel mixture.

        Returns:
            A callable that takes the component weight fractions vector `Yi`
            and returns the penalties for the distillation curve limits.
        """
        rho_i = jnp.asarray(
            fuel.density(Units.Quantity(60.0, "fahrenheit")).m_as("kg/m^3")
        )
        MW_i = jnp.asarray(fuel.MW.m_as("g/mol"))

        def penalty(Yi: Array) -> list[Array]:
            T10, T50, T90, Tf = mixing.distillation_curve(Yi, rho_i, MW_i)
            T10_C = T10 - 273.15
            Tf_C = Tf - 273.15
            return [
                jnp.maximum(T10_C - self.T10_max, 0.0) / self.T10_max,
                jnp.maximum(self.T50_T10_min - (T50 - T10), 0.0) / self.T50_T10_min,
                jnp.maximum(self.T90_T10_min - (T90 - T10), 0.0) / self.T90_T10_min,
                jnp.maximum(Tf_C - self.Tf_max, 0.0) / self.Tf_max,
            ]

        return penalty
