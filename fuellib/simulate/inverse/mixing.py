"""JAX-based mixing rules."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array


def density(Yi: Array, rho_i: Array) -> Array:
    """Calculate the mixture density.

    Args:
        Yi: Mass fractions of the components in the mixture.
        rho_i: Densities of the individual components.

    Returns:
        The density of the mixture.
    """
    return Yi @ rho_i


def viscosity(Yi: Array, mu_i: Array, mw_i: Array) -> Array:
    """Calculate the mixture kinematic viscosity.

    Args:
        Yi: Mass fractions of the components in the mixture.
        mu_i: Kinematic viscosities of the individual components.
        mw_i: Molecular weights of the individual components.

    Returns:
        The kinematic viscosity of the mixture.
    """
    mw_bar = 1.0 / jnp.sum(Yi / mw_i)
    Xi = mw_bar * Yi / mw_i
    return jnp.sum(Xi * mu_i ** (1.0 / 3.0)) ** 3.0
