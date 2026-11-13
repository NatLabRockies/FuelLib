"""Fuel mixture attribute."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .. import correlate
from ..utils import types
from ..utils.constants import T_stp

if TYPE_CHECKING:
    from .core import Fuel


class Mixture:
    """Class representing fuel mixture."""

    def __init__(self, fuel: "Fuel") -> None:
        """Initialize the Mixture class with fuel data."""
        self._fuel: Fuel = fuel

    # -------------------------------------------------------------------------
    # Composition conversions
    # -------------------------------------------------------------------------
    def mw(self, Yi: types.Quantity1D | None = None) -> types.Quantity0D:
        """Calculate the mean molecular weight of the mixture.

        Args:
            Yi: Mass fractions of each compound.
                Defaults to `fuel.y0` (initial mass fractions).

        Returns:
            Mean molecular weight of the mixture in kg/mol.
        """
        return correlate.mixture.mean_molecular_weight(self._fuel, Yi)

    def mass2y(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Convert mass of each component to mass fractions (Yi).

        Args:
            mass: Mass of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_to_mass_fractions(self._fuel, mass)

    def mass2x(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Convert mass of each component to mole fractions (Xi).

        Args:
            mass: Mass of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_to_mole_fractions(self._fuel, mass)

    def x2y(self, Xi: types.Quantity1D) -> types.Quantity1D:
        """Convert mole fractions (Xi) to mass fractions (Yi).

        Args:
            Xi: Mole fractions of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mole_fractions_to_mass_fractions(self._fuel, Xi)

    def y2x(self, Yi: types.Quantity1D) -> types.Quantity1D:
        """Convert mass fractions (Yi) to mole fractions (Xi).

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_fractions_to_mole_fractions(self._fuel, Yi)

    def y2v(self, Yi: types.Quantity1D, T: types.Quantity0D) -> types.Quantity1D:
        """Convert mass fractions (Yi) to volume fractions (Vi).

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature at which to evaluate the mixture density.

        Returns:
            Volume fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_fractions_to_volume_fractions(self._fuel, Yi, T)

    # -------------------------------------------------------------------------
    # Temperature-dependent properties
    # -------------------------------------------------------------------------
    # NOTE: Cannot make Yi optional in mixture functions because switching the order of
    # Yi and T would break the original function signatures.
    def rho(self, Yi: types.Quantity1D, T: types.Quantity0D) -> types.Quantity0D:
        """Calculate mixture density at a given temperature.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.

        Returns:
            Mixture density in kg/m^3.
        """
        return correlate.mixture.density(self._fuel, T, Yi)

    def nu(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
        mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate kinematic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Component viscosity correlation ("Dutt" or "Nannoolal").
                Defaults to "Nannoolal".
            mixing_rule: Mixture rule ("Kendall-Monroe" or "Arrhenius").
                Defaults to "Kendall-Monroe".

        Returns:
            Mixture kinematic viscosity in m^2/s.

        Raises:
            ValueError: If `correlation` is not Dutt or Nannoolal.
        """
        if correlation.casefold() == "dutt":
            return correlate.mixture.viscosity_kinematic_dutt(
                self._fuel, T, Yi, mixing_rule=mixing_rule
            )
        if correlation.casefold() == "nannoolal":
            return correlate.mixture.viscosity_kinematic_nannoolal(
                self._fuel, T, Yi, mixing_rule=mixing_rule
            )
        raise ValueError(f"Unknown viscosity correlation: {correlation!r}")

    def mu(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
        mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate dynamic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Component viscosity correlation ("Dutt" or "Nannoolal").
                Defaults to "Nannoolal".
            mixing_rule: Mixture rule ("Kendall-Monroe" or "Arrhenius").
                Defaults to "Kendall-Monroe".

        Returns:
            Mixture dynamic viscosity in Pa*s.

        Raises:
            ValueError: If `correlation` is not Dutt or Nannoolal.
        """
        if correlation.casefold() == "dutt":
            return correlate.mixture.viscosity_dynamic_dutt(
                self._fuel, T, Yi, mixing_rule=mixing_rule
            )
        if correlation.casefold() == "nannoolal":
            return correlate.mixture.viscosity_dynamic_nannoolal(
                self._fuel, T, Yi, mixing_rule=mixing_rule
            )
        raise ValueError(f"Unknown viscosity correlation: {correlation!r}")

    def pvap(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity0D:
        """Calculate vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Mixture vapor pressure in Pa.
        """
        return correlate.mixture.vapor_pressure(
            self._fuel, T, Yi, correlation=correlation
        )

    def pvap_antoine_coeffs(
        self,
        Yi: types.Quantity1D,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """Estimate Antoine coefficients for vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            Tvals: Temperature range or nodes for Antoine fit in Kelvin.
                Defaults to [273.15, min(Tb_mix)].
            units: Units for pressure in fit.
                Defaults to "mks" (Pa).
            correlation: Correlation method.
                Defaults to "Lee-Kesler".

        Returns:
            Coefficients A, B, C, D.
        """
        return correlate.mixture.vapor_pressure_antoine_coeffs(
            self._fuel, Tvals, Yi, units=units, correlation=correlation
        )

    def sigma(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> types.Quantity0D:
        """Calculate surface tension of the mixture.

        Uses arithmetic pseudo-property method recommended by Hugill and van
        Welsenes (1986).

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Pitzer" or "Brock-Bird").
                Defaults to "Brock-Bird".

        Returns:
            Mixture surface tension in N/m.
        """
        return correlate.mixture.surface_tension(
            self._fuel, T, Yi, correlation=correlation
        )

    def k(self, Yi: types.Quantity1D, T: types.Quantity0D) -> types.Quantity0D:
        """Calculate thermal conductivity of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.

        Returns:
            Thermal conductivity in W/m/K.
        """
        return correlate.mixture.thermal_conductivity_latini(self._fuel, T, Yi)

    # -------------------------------------------------------------------------
    # Fuel properties
    # -------------------------------------------------------------------------
    def t_freeze(self, Yi: types.Quantity1D, *, alpha: float = 1.0) -> types.Quantity0D:
        """Calculate freeze point of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            alpha: Scaling applied to the ideal mixing-entropy term.
                Defaults to 1.0.

        Returns:
            Mixture freeze point in K.
        """
        return correlate.mixture.freeze_point_boehm(self._fuel, Yi, alpha=alpha)

    def t_flash(
        self,
        Yi: types.Quantity1D,
        *,
        correlation: Literal["Alqaheem", "Alibakhshi"] = "Alibakhshi",
        mixing_rule: Literal["linear", "Liaw"] = "Liaw",
    ) -> types.Quantity0D:
        """Calculate flash point of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            correlation: Correlation method ("Alqaheem" or "Alibakhshi").
                Defaults to "Alibakhshi".
            mixing_rule: Mixing rule to apply ("linear" or "Liaw"). See documentation
                for details. Defaults to "Liaw".

        Returns:
            Mixture flash point in K.

        Raises:
            ValueError: If an unknown correlation method is specified.
        """
        if correlation.casefold() == "alqaheem":
            return correlate.mixture.flash_point_alqaheem(
                self._fuel, Yi, mixing_rule=mixing_rule
            )
        if correlation.casefold() == "alibakhshi":
            return correlate.mixture.flash_point_alibakhshi(
                self._fuel, Yi, mixing_rule=mixing_rule
            )
        msg = f"Unknown correlation method: {correlation}"
        raise ValueError(msg)

    def lhv(self, Yi: types.Quantity1D) -> types.Quantity0D:
        """Calculate heat of combustion (lower heating value) of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.

        Returns:
            Mixture heat of combustion in J/kg.
        """
        return correlate.mixture.heat_of_combustion(self._fuel, Yi)

    def ysi(self, Yi: types.Quantity1D) -> types.Quantity0D:
        """Calculate yield sooting index of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.

        Returns:
            Mixture yield sooting index.
        """
        return correlate.mixture.yield_sooting_index(self._fuel, Yi)

    def smoke_point(
        self,
        Yi: types.Quantity1D,
        correlation: Literal["NJFCP", "McEnally"] = "NJFCP",
    ) -> types.Quantity0D:
        """Calculate smoke point of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            correlation: Smoke point correlation ("NJFCP" or "McEnally").
                Defaults to "NJFCP".

        Returns:
            Mixture smoke point in mm.
        """
        return correlate.mixture.smoke_point(self._fuel, Yi, correlation=correlation)

    def dcn(
        self,
        Yi: types.Quantity1D,
        *,
        T_ref: types.Quantity0D = T_stp,
    ) -> types.Quantity0D:
        """Calculate derived cetane number of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T_ref: Reference temperature for liquid densities.

        Returns:
            Mixture derived cetane number.
        """
        return correlate.mixture.derived_cetane_number(self._fuel, Yi, T_ref=T_ref)

    # -------------------------------------------------------------------------
    # Composition properties
    # -------------------------------------------------------------------------
    def aromatics(self, Yi: types.Quantity1D) -> types.Quantity0D:
        """Calculate aromatics content of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.

        Returns:
            Mixture aromatics content in volume percent.
        """
        return correlate.mixture.aromatics_content(self._fuel, Yi)

    def naphthalenes(self, Yi: types.Quantity1D) -> types.Quantity0D:
        """Calculate naphthalenes content of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.

        Returns:
            Mixture naphthalenes content in volume percent.
        """
        return correlate.mixture.naphthalenes_content(self._fuel, Yi)
