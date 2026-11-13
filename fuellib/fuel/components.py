"""Fuel components attribute."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pandas as pd
from rdkit.Chem import Mol

from .. import correlate
from ..gcm import gani_gcm
from ..rdk import mol
from ..utils import Units, types
from ..utils.constants import EpsilonByKB_gas, MW_gas, Sigma_gas

if TYPE_CHECKING:
    from .core import Fuel


class Components:
    """Class representing fuel components."""

    def __init__(self, fuel: "Fuel") -> None:
        """Initialize the Components class with fuel data."""
        self._fuel: Fuel = fuel
        self._use_ref_props: Final[bool] = fuel._use_ref_props

    def _get_ref_values(self, prop: str, output_units: str) -> types.Quantity1D:
        """Get reference values for the given compounds in the specified output units.

        Args:
            prop: Property column prefix in the reference compounds file
                (e.g. "Tc" for the "Tc_Value" and "Tc_Units" columns).
            output_units: Desired output units for the reference values.

        Returns:
            Reference values for the given compounds in the specified output units.
        """
        out = np.full(len(self._fuel.components_data), np.nan, dtype=float)
        values = self._fuel.components_data[f"{prop}_Value"]
        units = self._fuel.components_data[f"{prop}_Units"]
        for i, (value, unit) in enumerate(zip(values, units)):
            if pd.isna(value):
                continue
            quantity = Units.Quantity(float(value), "" if pd.isna(unit) else unit)
            out[i] = quantity.to(output_units).magnitude
        return Units.Quantity(out, output_units)

    def _pred_or_ref(
        self, gcm_prop: str, ref_prop: str, units: str
    ) -> types.Quantity1D:
        """Gani GCM prediction, overridden by reference values where available.

        Args:
            gcm_prop: Property name passed to the Gani GCM.
            ref_prop: Property column prefix in the reference compounds file.
            units: Output units.

        Returns:
            Property values for each component in the specified units.
        """
        preds = gani_gcm.predict(gcm_prop, self._fuel).to(units)
        if not self._use_ref_props:
            return preds
        refs = self._get_ref_values(ref_prop, units)
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), units
        )

    def _missing_names(self, missing: np.ndarray) -> str:
        """Get the names of the components flagged in `missing`.

        Args:
            missing: Boolean mask over the fuel components.

        Returns:
            Comma-separated reference compound names.
        """
        data = self._fuel.components_data
        return ", ".join(data.loc[missing, "Reference Compound"].astype(str))

    # -------------------------------------------------------------------------
    # Critical properties
    # -------------------------------------------------------------------------
    @cached_property
    def tc(self) -> types.Quantity1D:
        """Critical temperature of the fuel components in K."""
        return self._pred_or_ref("Tc", "Tc", "K")

    @cached_property
    def pc(self) -> types.Quantity1D:
        """Critical pressure of the fuel components in Pa."""
        return self._pred_or_ref("Pc", "Pc", "Pa")

    @cached_property
    def vc(self) -> types.Quantity1D:
        """Critical molar volume of the fuel components in m^3/mol."""
        return self._pred_or_ref("Vc", "Vc", "m^3/mol")

    @cached_property
    def tm(self) -> types.Quantity1D:
        """Melting point of the fuel components in K."""
        return self._pred_or_ref("Tm", "Tm", "K")

    @cached_property
    def tb(self) -> types.Quantity1D:
        """Boiling point of the fuel components in K."""
        return self._pred_or_ref("Tb", "Tb", "K")

    @cached_property
    def hf_stp(self) -> types.Quantity1D:
        """Enthalpy of formation (at STP) of the fuel components in J/mol."""
        return self._pred_or_ref("Hf", "Hf_STP", "J/mol")

    @cached_property
    def hv_stp(self) -> types.Quantity1D:
        """Enthalpy of vaporization (at STP) of the fuel components in J/mol."""
        return self._pred_or_ref("Hv_stp", "Hv_STP", "J/mol")

    @cached_property
    def vm_stp(self) -> types.Quantity1D:
        """Molar volume (at STP) of the fuel components in m^3/mol."""
        return self._pred_or_ref("Vm_stp", "Vm_STP", "m^3/mol")

    @cached_property
    def omega(self) -> types.Quantity1D:
        """Acentric factor of the fuel components (dimensionless)."""
        return self._pred_or_ref("omega", "Omega", "")

    # -------------------------------------------------------------------------
    # Reference-only properties
    # -------------------------------------------------------------------------
    @cached_property
    def ysi(self) -> types.Quantity1D:
        """Yield Sooting Index (YSI) of the fuel components (dimensionless).

        Raises:
            ValueError: If any reference values for YSI are missing.
        """  # ruff: ignore[property-docstring-starts-with-verb]
        refs = self._get_ref_values("YSI", "")
        missing = np.isnan(refs.magnitude)
        if missing.any():
            names = self._missing_names(missing)
            msg = f"Missing Yield Sooting Index (YSI) reference values for: {names}."
            raise ValueError(msg)
        return refs

    @cached_property
    def dcn(self) -> types.Quantity1D:
        """Derived cetane number (DCN) of the fuel components (dimensionless).

        Raises:
            ValueError: If any reference values for DCN are missing.
        """
        refs = self._get_ref_values("DCN", "")
        missing = np.isnan(refs.magnitude)
        if missing.any():
            names = self._missing_names(missing)
            msg = f"Missing Derived Cetane Number (DCN) reference values for: {names}."
            raise ValueError(msg)
        return refs

    # -------------------------------------------------------------------------
    # Prediction-only properties
    # -------------------------------------------------------------------------
    @cached_property
    def gf_stp(self) -> types.Quantity1D:
        """Gibbs free energy of formation (at STP) of the components in J/mol."""
        return gani_gcm.predict("Gf", self._fuel).to("J/mol")

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------
    @cached_property
    def lj_eps_by_kb(self) -> types.Quantity1D:
        """Lennard-Jones potential well depth divided by Boltzmann constant in K."""
        return (0.7915 + 0.1693 * self.omega.magnitude) * self.tc.to("K")

    @cached_property
    def lj_sigma(self) -> types.Quantity1D:
        """Lennard-Jones collision diameter of the fuel components in m."""
        sigma = (2.3551 - 0.0874 * self.omega.magnitude) * (
            self.tc.to("K").magnitude / self.pc.to("atm").magnitude
        ) ** (1.0 / 3)
        return Units.Quantity(sigma, "angstrom").to("m")

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------
    @cached_property
    def names(self) -> list[str]:
        """Reference compound names of the fuel components."""
        return self._fuel.components_data["Reference Compound"].to_list()

    @cached_property
    def smiles(self) -> list[str]:
        """SMILES strings of the fuel components."""
        return self._fuel.components_data["SMILES"].to_list()

    @cached_property
    def pelephysics_keys(self) -> list[str] | None:
        """PelePhysics species keys of the fuel components, if in the GCxGC data."""
        gc_data = self._fuel.gc_data
        if "PelePhysics Key" not in gc_data.columns:
            return None
        return gc_data["PelePhysics Key"].to_list()

    @cached_property
    def rdkit_mols(self) -> list[Mol]:
        """RDKit `Mol` objects for the fuel components."""
        return [mol.from_smiles(s) for s in self.smiles]

    @cached_property
    def num_c(self) -> np.ndarray:
        """Number of carbon atoms in each fuel component."""
        return np.array(
            [mol.atom_counts(m).get("C", 0) for m in self.rdkit_mols], dtype=int
        )

    @cached_property
    def num_h(self) -> np.ndarray:
        """Number of hydrogen atoms in each fuel component."""
        return np.array(
            [mol.atom_counts(m).get("H", 0) for m in self.rdkit_mols], dtype=int
        )

    @cached_property
    def mw(self) -> types.Quantity1D:
        """Molecular weights of the fuel components in kg/mol."""
        return Units.Quantity(
            [mol.molecular_weight(m) for m in self.rdkit_mols], "g/mol"
        ).to("kg/mol")

    @cached_property
    def formulas(self) -> list[str]:
        """Molecular formulas of the fuel components."""
        return [mol.hill_formula(m) for m in self.rdkit_mols]

    @cached_property
    def hc_type(self) -> list[str]:
        """Hydrocarbon type of each fuel component.

        Possible values (in ascending priority) are:
        * "n-alkane"
        * "iso-alkane"
        * "alkene"
        * "cyclo-alkane"
        * "aromatic"
        """
        hc_types = []
        for m in self.rdkit_mols:
            if mol.has_aromatic(m):
                hc_types.append("aromatic")
            elif mol.has_ring(m):
                hc_types.append("cyclo-alkane")
            elif mol.has_double_bond(m):
                hc_types.append("alkene")
            elif mol.has_branch(m):
                hc_types.append("iso-alkane")
            else:
                hc_types.append("n-alkane")
        return hc_types

    @cached_property
    def fam(self) -> types.Array1D:
        """Hydrocarbon family codes for thermal conductivity.

        ==== ==================
        Code Hydrocarbon Family
        ==== ==================
        0    saturated
        1    aromatics
        2    cycloparaffins
        3    olefins
        ==== ==================

        Raises:
            ValueError: If an unknown hydrocarbon type is encountered.
        """
        codes = {
            "n-alkane": 0,
            "iso-alkane": 0,
            "aromatic": 1,
            "cyclo-alkane": 2,
            "alkene": 3,
        }
        fam_codes = []
        for hc in self.hc_type:
            if hc not in codes:
                msg = f"Unknown hydrocarbon type '{hc}' encountered."
                raise ValueError(msg)
            fam_codes.append(codes[hc])
        return np.array(fam_codes, dtype=int)

    # -------------------------------------------------------------------------
    # Temperature-dependent properties
    # -------------------------------------------------------------------------
    def rho(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Calculate the density of each component at temperature T.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Density of each compound in kg/m^3.
        """
        rho_i = correlate.components.density(self._fuel, T)
        return rho_i[comp_idx] if comp_idx is not None else rho_i

    def nu(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
    ) -> types.Quantity1D:
        """Calculate component kinematic viscosity.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Dutt" or "Nannoolal").
                Defaults to "Nannoolal".

        Returns:
            Viscosity of each component in m^2/s.

        Raises:
            ValueError: If `correlation` is not Dutt or Nannoolal.
        """
        if correlation.casefold() == "dutt":
            nu_i = correlate.components.viscosity_kinematic_dutt(self._fuel, T)
        elif correlation.casefold() == "nannoolal":
            nu_i = correlate.components.viscosity_kinematic_nannoolal(self._fuel, T)
        else:
            raise ValueError(f"Unknown viscosity correlation: {correlation!r}")
        return nu_i[comp_idx] if comp_idx is not None else nu_i

    def mu(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
    ) -> types.Quantity1D:
        """Calculate component dynamic viscosity.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Dutt" or "Nannoolal").
                Defaults to "Nannoolal".

        Returns:
            Dynamic viscosity in Pa*s.

        Raises:
            ValueError: If `correlation` is not Dutt or Nannoolal.
        """
        if correlation.casefold() == "dutt":
            mu_i = correlate.components.viscosity_dynamic_dutt(self._fuel, T)
        elif correlation.casefold() == "nannoolal":
            mu_i = correlate.components.viscosity_dynamic_nannoolal(self._fuel, T)
        else:
            raise ValueError(f"Unknown viscosity correlation: {correlation!r}")
        return mu_i[comp_idx] if comp_idx is not None else mu_i

    def cp(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        basis: Literal["mass", "molar"] = "mass",
    ) -> types.Quantity1D:
        """Compute specific heat capacity at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            basis: Basis of the heat capacity ("mass" or "molar").
                Defaults to "mass".

        Returns:
            Specific heat capacity in J/kg/K (mass basis) or J/mol/K (molar basis).

        Raises:
            ValueError: If `basis` is not "mass" or "molar".
        """
        if basis.casefold() == "mass":
            cp = correlate.components.specific_heat_capacity_mass(self._fuel, T)
        elif basis.casefold() == "molar":
            cp = correlate.components.specific_heat_capacity_molar(self._fuel, T)
        else:
            raise ValueError(f"Unknown heat capacity basis: {basis!r}")
        return cp[comp_idx] if comp_idx is not None else cp

    def pvap(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity1D:
        """Compute saturated vapor pressure.

        Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
                Defaults to "Lee-Kesler".

        Returns:
            Saturated vapor pressure in Pa.
        """
        pvap = correlate.components.saturated_vapor_pressure(
            self._fuel, T, correlation=correlation
        )
        return pvap[comp_idx] if comp_idx is not None else pvap

    def pvap_antoine_coeffs(
        self,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
        """Estimate Antoine coefficients for vapor pressure of each compound.

        Args:
            Tvals: Temperature range or nodes for Antoine fit in Kelvin.
                Defaults to [273.15, Tb_i].
            units: Units for pressure in fit ("mks", "cgs", "dyne/cm^2", "Pa").
                Defaults to "mks".
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
                Defaults to "Lee-Kesler".

        Returns:
            Coefficients A, B, C, D for each compound.
        """
        return correlate.components.saturated_vapor_pressure_antoine_coeffs(
            self._fuel, Tvals=Tvals, units=units, correlation=correlation
        )

    def vm(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Compute molar liquid volume with temperature correction.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Molar liquid volume in m^3/mol.
        """
        vm_i = correlate.components.molar_liquid_volume(self._fuel, T)
        return vm_i[comp_idx] if comp_idx is not None else vm_i

    def hv(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        basis: Literal["mass", "molar"] = "mass",
    ) -> types.Quantity1D:
        """Calculate enthalpy of vaporization adjusted for temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            basis: Basis of the enthalpy of vaporization ("mass" or "molar").
                Defaults to "mass".

        Returns:
            Enthalpy of vaporization in J/kg (mass basis) or J/mol (molar basis).

        Raises:
            ValueError: If `basis` is not "mass" or "molar".
        """
        if basis.casefold() == "mass":
            hv_i = correlate.components.enthalpy_vaporization_mass(self._fuel, T)
        elif basis.casefold() == "molar":
            hv_i = correlate.components.enthalpy_vaporization_molar(self._fuel, T)
        else:
            raise ValueError(f"Unknown enthalpy of vaporization basis: {basis!r}")
        return hv_i[comp_idx] if comp_idx is not None else hv_i

    def d(
        self,
        p: types.Quantity0D,
        T: types.Quantity0D,
        sigma_gas: types.Quantity0D = Sigma_gas,
        epsilonByKB_gas: types.Quantity0D = EpsilonByKB_gas,
        MW_gas: types.Quantity0D = MW_gas,
        correlation: Literal["Tee", "Wilke"] = "Tee",
    ) -> types.Quantity1D:
        """Compute diffusion coefficients using Lennard-Jones parameters.

        Uses Wilke and Lee method (Poling, equation 11-4.1). Ambient gas
        defaults to air parameters.

        Args:
            p: Pressure to compute property.
            T: Temperature to compute property.
            sigma_gas: Collision diameter.
                Default is 3.62 Angstroms.
            epsilonByKB_gas: Well depth over Boltzmann constant.
                Default is 97.0 K.
            MW_gas: Mean molecular weight of ambient gas.
                Default is 28.97 g/mol.
            correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
                Default is "Tee".

        Returns:
            Diffusion coefficient in m^2/s.
        """
        return correlate.components.diffusion_coeffs_wilke(
            self._fuel,
            p,
            T,
            sigma_gas=sigma_gas,
            epsilonByKB_gas=epsilonByKB_gas,
            MW_gas=MW_gas,
            correlation=correlation,
        )

    def sigma(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> types.Quantity1D:
        """Calculate surface tension of each compound at a given temperature.

        Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Brock-Bird" or "Pitzer").
                Defaults to "Brock-Bird".

        Returns:
            Surface tension in N/m.
        """
        st = correlate.components.surface_tension(
            self._fuel, T, correlation=correlation
        )
        return st[comp_idx] if comp_idx is not None else st

    def k(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Calculate thermal conductivity at a given temperature.

        Uses Latini et al. method (Poling equation 10-9.1).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Thermal conductivity in W/m/K.
        """
        k_i = correlate.components.thermal_conductivity_latini(self._fuel, T)
        return k_i[comp_idx] if comp_idx is not None else k_i

    def t_flash(
        self,
        comp_idx: int | None = None,
        correlation: Literal["Alqaheem", "Alibakhshi"] = "Alibakhshi",
    ) -> types.Quantity1D:
        """Calculate flash point of each compound.

        Args:
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Alqaheem" or "Alibakhshi").
                Defaults to "Alibakhshi".

        Returns:
            Flash point in K.
        """
        if correlation.casefold() == "alqaheem":
            fp = correlate.components.flash_point_alqaheem(self._fuel)
        else:
            fp = correlate.components.flash_point_alibakhshi(self._fuel)
        return fp[comp_idx] if comp_idx is not None else fp
