"""Define models for inverse prediction of fuel composition."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree
from optax.projections import projection_simplex
from pandas import DataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from ...fuel import Fuel
from ...utils.types import Units
from .constraints import (
    DensityConstraint,
    ViscosityConstraint,
)

__all__ = ["Config", "Problem", "Solution", "d7566_config"]


class Config(BaseModel):
    """Configuration for inverse prediction models."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", use_attribute_docstrings=True
    )

    # Property limits
    density: tuple[DensityConstraint, ...] = Field(default_factory=tuple)
    """Density limits, each at its own temperature with its own weight."""
    viscosity: tuple[ViscosityConstraint, ...] = Field(default_factory=tuple)
    """Viscosity limits, each at its own temperature with its own weight."""

    # Optimizer and convergence
    learning_rate: PositiveFloat = 1e-2
    """Adam learning rate."""
    violation_tol: NonNegativeFloat = 1e-4
    """Max normalized constraint violation counted as satisfied."""
    stall_rtol: NonNegativeFloat = 1e-6
    """Relative loss change over `stall_window` steps below which the inner loop
    is considered stalled."""
    stall_window: PositiveInt = 200
    """Number of steps over which the loss change is measured
    (< max_inner_steps)."""
    max_inner_steps: PositiveInt = 2000
    """Max optimizer steps per augmented Lagrangian iteration."""
    penalty_init: PositiveFloat = 1.0
    """Initial penalty for the augmented Lagrangian."""
    penalty_growth: float = Field(gt=1.0, default=10.0)
    """Factor by which the penalty grows when the violation doesn't decrease
    enough."""
    penalty_max: PositiveFloat = Field(gt=0.0, default=1e8)
    """Maximum penalty parameter (>= penalty_init)."""
    violation_decrease_ratio: float = Field(gt=0.0, lt=1.0, default=0.25)
    """Required ratio of the new to the previous max violation per iteration;
    the penalty grows if it isn't met."""


d7566_config = Config(
    density=(
        DensityConstraint(
            temp=Units.Quantity(15.0, "celsius"),
            min=Units.Quantity(775.0, "kg/m^3"),
            max=Units.Quantity(840.0, "kg/m^3"),
            weight=1.0,
        ),
    ),
    viscosity=(
        ViscosityConstraint(
            temp=Units.Quantity(-20.0, "celsius"),
            max=Units.Quantity(8.0, "mm^2/s"),
            weight=1.0,
        ),
        ViscosityConstraint(
            temp=Units.Quantity(-40.0, "celsius"),
            max=Units.Quantity(12.0, "mm^2/s"),
            weight=1.0,
        ),
    ),
)


class Problem:
    """Static fuel data, objective, and constraints of the inverse design problem."""

    def __init__(self, fuel: Fuel, config: Config) -> None:
        """Precompute the per-component properties of the fuel.

        Args:
            fuel: Base fuel to which an addition is blended.
            config: Optimization settings.
        """
        self.fuel: Fuel = fuel
        """Base fuel object."""
        self.config: Config = config
        """Optimization settings."""
        c = config

        self.Y_0: Array = jnp.asarray(fuel.Y_0.to("").magnitude)
        """Original fuel weight composition."""
        self._empty_base = bool(np.sum(fuel.Y_0.to("").magnitude) == 0.0)

        self._constraints = (
            *c.density,
            *c.viscosity,
        )
        self.binds = [c.bind(fuel) for c in self._constraints]
        self.constraint_names = [n for k in self._constraints for n in k.labels()]
        """Names of the active constraints."""
        self.constraint_weights = jnp.array([
            k.weight for k in self._constraints for _ in k.labels()
        ])
        """Weights of the active constraints."""

    def blend(self, params: PyTree) -> Array:
        """Blend the addition described by `params` into the base fuel.

        Args:
            params: Unconstrained parameters of the addition: composition
                (`"logits"`) and mass fraction in the blend (`"frac"`).

        Returns:
            Mass fractions of the blend.
        """
        Y_add = projection_simplex(params["logits"])
        if self._empty_base:
            # No base fuel to blend into: the blend is the addition itself
            return Y_add
        alpha = jax.nn.sigmoid(params["frac"])
        return (1 - alpha) * self.Y_0 + alpha * Y_add

    def constraints(self, Yi: Array) -> Array:
        """Evaluate the active constraints for a given fuel composition.

        Args:
            Yi: Mass fractions of the blend.

        Returns:
            Constraint values g (feasible when g <= 0), ordered as
            `constraint_names`.
        """
        g = [p for b in self.binds for p in b(Yi)]
        return jnp.array(g)

    def loss(self, params: PyTree, lam: Array, mu: Array) -> Array:
        """Compute the augmented Lagrangian of the sparsity objective.

        Uses the Powell-Hestenes-Rockafellar form for inequality constraints g <= 0.

        Args:
            params: Unconstrained parameters of the addition.
            lam: Lagrange multiplier estimates of the constraints.
            mu: Penalty parameter.

        Returns:
            Augmented Lagrangian as a scalar array.
        """
        Yi = self.blend(params)
        g = self.constraints(Yi)
        w = mu * self.constraint_weights
        penalty = jnp.sum((jax.nn.relu(lam + w * g) ** 2 - lam**2) / (2.0 * w))
        return penalty


@dataclass(frozen=True)
class Solution:
    """Result of the inverse fuel design problem."""

    blend: DataFrame
    """GCxGC data of the optimized blend (`Weight %` sums to 100)."""
    additive: DataFrame
    """GCxGC data of the additive (`Weight %` sums to 100)."""
    additive_fraction: float
    """Mass fraction of the additive in the optimized blend."""
    additive_history: np.ndarray
    """Additive mass fractions every `log_every` steps (starting at step 0) plus
    the final step, shape `(n_records, n)`."""
    additive_fraction_history: np.ndarray
    """Additive mass fraction in the blend every `log_every` steps (starting at
    step 0) plus the final step, shape `(n_records,)`."""

    def save(self, directory: Path, fuel_name: str) -> None:
        """Write the blend and additive GCxGC data and the blend ratio to files.

        Args:
            directory: Output directory (created if missing).
            fuel_name: Name of the original fuel, used in file names and labels.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.blend.to_csv(directory / f"{fuel_name}_opt.csv", index=False)
        self.additive.to_csv(directory / "additive.csv", index=False)
        add_pct = self.additive_fraction * 100.0
        (directory / "blend.txt").write_text(
            f"Original fuel ({fuel_name}): {100.0 - add_pct:.4f} %\n"
            f"Additive: {add_pct:.4f} %\n"
        )
