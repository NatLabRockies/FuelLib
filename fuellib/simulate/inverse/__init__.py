"""Predictive modeling of fuel composition from target properties."""

try:
    import jax  # noqa: F401
    import jaxtyping  # noqa: F401
    import optax  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"fuellib.simulate.inverse requires the optional '{e.name}' package. "
        'Install it with: pip install "fuellib[inverse]"'
    ) from e

from . import mixing
from .constraints import DensityConstraint, ViscosityConstraint
from .models import Config, Problem, Solution, d7566_config
from .solve import solve

__all__ = [
    "Config",
    "DensityConstraint",
    "Problem",
    "Solution",
    "ViscosityConstraint",
    "d7566_config",
    "mixing",
    "solve",
]
