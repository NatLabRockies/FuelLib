"""JAX-based mixing rules."""

from collections.abc import Callable

import jax
from jaxtyping import ArrayLike

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array


def _Y2X(Yi: Array, mw_i: Array) -> Array:
    """Convert mass fractions to mole fractions.

    Args:
        Yi: Mass fractions of the components in the mixture.
        mw_i: Molecular weights of the individual components.

    Returns:
        Mole fractions of the components.
    """
    mw_bar = 1.0 / jnp.sum(Yi / mw_i)
    return mw_bar * Yi / mw_i


def _Y2V(Yi: Array, rho_i: Array) -> Array:
    """Convert mass fractions to volume fractions.

    Args:
        Yi: Mass fractions of the components in the mixture.
        rho_i: Densities of the individual components.

    Returns:
        Volume fractions of the components.
    """
    V_bar = 1.0 / jnp.sum(Yi / rho_i)
    return V_bar * Yi / rho_i


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
    Xi = _Y2X(Yi, mw_i)
    return jnp.sum(Xi * mu_i ** (1.0 / 3.0)) ** 3.0


def aromatics_content(Yi: Array, rho_i: Array, is_aromatic: Array) -> Array:
    """Calculate the aromatics content of the mixture.

    Args:
        Yi: Mass fractions of the components in the mixture.
        rho_i: Densities of the individual components.
        is_aromatic: 1 for aromatic components, 0 otherwise.

    Returns:
        The aromatics content in volume percent.
    """
    Vi = _Y2V(Yi, rho_i)
    return 100.0 * (Vi @ is_aromatic) / jnp.sum(Vi)


def heat_of_combustion(Yi: Array, Lhv_i: Array) -> Array:
    """Calculate the heat of combustion of the mixture.

    Args:
        Yi: Mass fractions of the components in the mixture.
        Lhv_i: Lower heating values of the individual components in MJ/kg.

    Returns:
        The heat of combustion of the mixture in MJ/kg.
    """
    return Yi @ Lhv_i


def flash_point(
    Yi: Array, Tfp_i: Array, Tc_i: Array, Pc_i: Array, omega_i: Array, MW_i: Array
) -> Array:
    """Calculate the flash point of the mixture.

    Args:
        Yi: Mass fractions of the components in the mixture.
        Tfp_i: Flash points of the individual components in degrees Celsius.
        Tc_i: Critical temperatures of the individual components in Kelvin.
        Pc_i: Critical pressures of the individual components in Pascal.
        omega_i: Acentric factors of the individual components.
        MW_i: Molecular weights of the individual components.

    Returns:
        The flash point of the mixture in degrees Celsius.
    """

    # Initialize static component properties
    def _psat_lee_kesler(T: Array) -> Array:
        Tr = T / Tc_i
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * jnp.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * jnp.log(Tr) + 0.43577 * (Tr**6)
        return Pc_i * jnp.exp(f0 + omega_i * f1)

    # Lee-Kesler requires absolute temperature; solve in Kelvin.
    Tfp_K = Tfp_i + 273.15
    psat_ref = _psat_lee_kesler(Tfp_K)
    Xi = _Y2X(Yi, MW_i)

    def _residual(T: Array) -> Array:
        return jnp.sum(Xi * _psat_lee_kesler(T) / psat_ref) - 1.0

    def _newton(f: Callable[[Array], Array], T0: Array) -> Array:
        df = jax.grad(f)
        return jax.lax.fori_loop(0, 30, lambda _, T: T - f(T) / df(T), T0)

    def _scalar_tangent_solve(g: Callable[[ArrayLike], Array], y: Array) -> Array:
        return y / g(1.0)

    T0 = jnp.sum(Xi * Tfp_K)
    T_K = jax.lax.custom_root(_residual, T0, _newton, _scalar_tangent_solve)
    return T_K - 273.15


def freeze_point(
    Yi: Array, Tm_i: Array, dS_fus_i: Array, Cp_i: Array, MW_i: Array
) -> Array:
    """Compute the freezing point of a mixture.

    Solves Boehm et al. (2022), equation 21, independently for each compound
    and returns the highest candidate temperature (first crystal on cooling).

    Args:
        Yi: Array of mass fractions of the components.
        Tm_i: Melting temperatures of the individual components in Kelvin.
        dS_fus_i: Entropies of fusion of the individual components in J/(mol*K).
        Cp_i: Heat capacities of the individual components in J/(mol*K).
        MW_i: Molecular weights of the individual components.

    Returns:
        Freezing point of the mixture in Celsius.
    """
    R = 8.314  # Universal gas constant in J/(mol*K)

    Xi = _Y2X(Yi, MW_i)
    dH_fus = Tm_i * dS_fus_i
    dCp = -0.35 * Cp_i

    # Ideal binary mixing entropy of each compound against the rest
    Xi_safe = jnp.clip(Xi, 1e-6, 1.0 - 1e-6)
    dS_mix = (
        -R
        / Xi_safe
        * ((1.0 - Xi_safe) * jnp.log1p(-Xi_safe) + Xi_safe * jnp.log(Xi_safe))
    )

    def _residual(T: Array) -> Array:
        T_safe = jnp.maximum(T, 1.0)
        lhs = T_safe * (dS_fus_i + Xi_safe * dCp * jnp.log(T_safe / Tm_i) + dS_mix)
        rhs = dH_fus + Xi_safe * dCp * (T_safe - Tm_i)
        return lhs - rhs

    # Residuals are decoupled per compound, so the Jacobian is diagonal.
    # Iterates are clipped to (10 K, Tm] so an overshooting step can't land in
    # the T < 1 region where the residual is flat (df = 0 -> inf/NaN).
    def _newton(f: Callable[[Array], Array], T0: Array) -> Array:
        df = jax.grad(lambda T: jnp.sum(f(T)))
        return jax.lax.fori_loop(
            0, 30, lambda _, T: jnp.clip(T - f(T) / df(T), 10.0, Tm_i), T0
        )

    def _diagonal_tangent_solve(g: Callable[[Array], Array], y: Array) -> Array:
        return y / g(jnp.ones_like(y))

    T = jax.lax.custom_root(_residual, Tm_i, _newton, _diagonal_tangent_solve)

    # Only converged, physical solutions for compounds present in the mixture
    # count (a compound whose residual has no root in (10 K, Tm] is left at a
    # clip bound and must be excluded).
    converged = jnp.abs(jax.lax.stop_gradient(_residual(T))) < 1e-4 * dH_fus
    valid = converged & (Xi > 1e-6)
    return jnp.max(jnp.where(valid, T, -jnp.inf)) - 273.15
