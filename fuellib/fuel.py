"""Fuel class for Group Contribution Method calculations."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit.Chem import Mol

from . import correlate
from .database.database import (
    COMMON_NAME_COL,
    FAMILY_COL,
    INCHI_COL,
    PELEPHYSICS_KEY_COL,
    SMILES_COL,
    Y_COL,
    Property,
    _find_data_file,
    load_gani_database,
    load_reference_database,
    match_gc_data,
    read_gc_data,
)
from .gcm import GCMRegistry
from .rdk import mol
from .utils import FLLogger, Units, types
from .utils.constants import EpsilonByKB_gas, MW_gas, Sigma_gas


class Fuel:
    """Class for handling calculations of thermodynamic and mixture properties."""

    def __init__(self, name: str, userDataDir: str | Path | None = None) -> None:
        """Initialize a Fuel instance.

        Args:
            name: The name of the fuel (gcData/<name>.csv).
            userDataDir: Optional user database mirroring the default layout.

        Raises:
            NotADirectoryError: If `userDataDir` is not a directory.
            FileNotFoundError: If no gcData file exists for `name`.
        """
        if userDataDir is not None:
            userDataDir = Path(userDataDir)
            if not userDataDir.is_dir():
                msg = f"{userDataDir} is not a valid directory."
                raise NotADirectoryError(msg)

        self.name: str = name
        """The name of the fuel."""
        self.userDataDir: Path | None = userDataDir
        """The user database directory, if any."""

        gc_file = _find_data_file("gcData", name, userDataDir)
        if gc_file is None:
            msg = f"No gcData/{name}.csv found in the user or package database."
            raise FileNotFoundError(msg)

        self.references: pd.DataFrame = load_reference_database(userDataDir)
        """Merged reference compounds database."""
        self.gc_data: pd.DataFrame = read_gc_data(gc_file)
        """GC composition data, one row per compound in `data`."""
        self.data: pd.DataFrame = match_gc_data(self.gc_data, self.references)
        """Reference data for each GC component, with weight percent in "Y"."""

        props_file = _find_data_file("propertiesData", name, userDataDir)
        self.properties_data: pd.DataFrame | None = (
            pd.read_csv(props_file, header=0) if props_file is not None else None
        )
        """Temperature-dependent property data, if available."""

    # -------------------------------------------------------------------------
    # Parsing functions
    # -------------------------------------------------------------------------
    @property
    def Y_0(self) -> types.Quantity1D:
        """List of initial mass fractions for the compounds in the fuel mixture."""
        Y_0 = self.data[Y_COL].to_numpy().flatten().astype(float)
        return Units.Quantity(Y_0 / np.sum(Y_0), "dimensionless")

    @property
    def compounds(self) -> list[str]:
        """List of compounds in the fuel mixture."""
        return [name.strip() for name in self.data[COMMON_NAME_COL]]

    @property
    def num_compounds(self) -> int:
        """Number of compounds in the fuel mixture."""
        return len(self.compounds)

    @property
    def smiles(self) -> list[str]:
        """List of SMILES strings for the compounds in the fuel mixture."""
        return [smiles.strip() for smiles in self.data[SMILES_COL]]

    @property
    def pelephysics_keys(self) -> list[str] | None:
        """PelePhysics keys from the gcData, or None if unavailable for any compound."""
        if PELEPHYSICS_KEY_COL not in self.gc_data.columns:
            return None
        keys = self.gc_data[PELEPHYSICS_KEY_COL]
        if keys.isna().any() or (keys.astype(str).str.strip() == "").any():
            FLLogger.warning(
                f"PelePhysics keys are missing for some compounds in {self.name}."
            )
            return None
        return [str(key).strip() for key in keys]

    # -------------------------------------------------------------------------
    # Data initialization
    # -------------------------------------------------------------------------
    @cached_property
    def rdkit_mols(self) -> list[Mol]:
        """RDKit `Mol` objects for the compounds in the fuel mixture."""
        return [mol.from_smiles(smiles) for smiles in self.smiles]

    @property
    def formulas(self) -> list[str]:
        """List of chemical formulas for the compounds in the fuel mixture."""
        return [mol.hill_formula(m) for m in self.rdkit_mols]

    @property
    def inchi(self) -> list[str]:
        """List of InChI strings for the compounds in the fuel mixture."""
        return [inchi.strip() for inchi in self.data[INCHI_COL]]

    @property
    def nC(self) -> list[int]:
        """Number of carbon atoms for each compound in the fuel mixture."""
        return [mol.atom_counts(m).get("C", 0) for m in self.rdkit_mols]

    @property
    def nH(self) -> list[int]:
        """Number of hydrogen atoms for each compound in the fuel mixture."""
        return [mol.atom_counts(m).get("H", 0) for m in self.rdkit_mols]

    @property
    def families(self) -> list[str]:
        """Hydrocarbon family of each compound from the reference database.

        See `fuellib.database.database.classify_family` for the list of families.
        """
        return [str(family) for family in self.data[FAMILY_COL]]

    @property
    def hc_type(self) -> list[str]:
        """Hydrocarbon type for each compound in the fuel mixture.

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

    @property
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
        fam_codes = []
        for hc in self.hc_type:
            if hc == "n-alkane" or hc == "iso-alkane":
                fam_codes.append(0)
            elif hc == "aromatic":
                fam_codes.append(1)
            elif hc == "cyclo-alkane":
                fam_codes.append(2)
            elif hc == "alkene":
                fam_codes.append(3)
            else:
                msg = f"Unknown hydrocarbon type '{hc}' encountered."
                raise ValueError(msg)
        return np.array(fam_codes, dtype=int)

    def gani_decomp(self) -> pd.DataFrame:
        """Gani group decomposition of each compound from the reference database.

        Returns:
            Group counts indexed by compound name.
                Shape: (num_compounds, num_groups)

        Raises:
            ValueError: If any compound lacks a decomposition in
                ``referenceCompounds/gani.csv``.
        """
        table = load_gani_database(self.userDataDir)
        known = set() if table is None else set(table.index)
        missing = [
            name
            for name, inchi in zip(self.compounds, self.inchi, strict=True)
            if inchi not in known
        ]
        if table is None or missing:
            msg = (
                f"No Gani group decomposition for {missing}. "
                "Add them to referenceCompounds/gani.csv."
            )
            raise ValueError(msg)
        return (
            table.loc[self.inchi].set_axis(self.compounds).rename_axis(COMMON_NAME_COL)
        )

    @cached_property
    def gcm_properties(self) -> dict[str, dict[str, types.Quantity1D]]:
        """Pure group-contribution predictions for the compounds.

        Unlike `get_property`, these ignore literature values in the database.

        Returns:
            A dictionary mapping each GCM method name to a dictionary of (lowercase)
            property names and their predictions for each compound.
        """
        props: dict[str, dict[str, types.Quantity1D]] = {}
        for gcm in GCMRegistry.methods:
            props.update(gcm.predict_all(self))
        return props

    def get_property(
        self, field: Property | str, *, output_units: str | None = None
    ) -> types.Quantity1D:
        """Get a property of each compound from the reference database.

        Values are literature data where available and GCM predictions otherwise.
        Values stored in different units are converted to the most common unit.

        Args:
            field: The property to retrieve.
            output_units: The desired output units for the property.

        Returns:
            Quantity vector of the property for each compound.
        """
        field = Property(field)
        values = self.data[field].to_numpy(dtype=float, copy=True)
        units = self.data[field.units].dropna().astype(str)
        if units.empty:
            majority = output_units if output_units is not None else "dimensionless"
        else:
            majority = units.value_counts().idxmax()
        for i, unit in units.items():
            if unit != majority:
                values[i] = Units.Quantity(values[i], unit).to(majority).magnitude

        quantity = Units.Quantity(values, majority)
        return quantity.to(output_units) if output_units is not None else quantity

    # -------------------------------------------------------------------------
    # Component properties
    # -------------------------------------------------------------------------
    @cached_property
    def Tc(self) -> types.Quantity1D:
        """Critical temperature in K."""
        return self.get_property(Property.TC, output_units="K")

    @cached_property
    def Pc(self) -> types.Quantity1D:
        """Critical pressure in Pa."""
        return self.get_property(Property.PC, output_units="Pa")

    @cached_property
    def Vc(self) -> types.Quantity1D:
        """Critical volume in m^3/mol."""
        return self.get_property(Property.VC, output_units="m^3/mol")

    @cached_property
    def Tb(self) -> types.Quantity1D:
        """Boiling temperature in K."""
        return self.get_property(Property.TB, output_units="K")

    @cached_property
    def Tm(self) -> types.Quantity1D:
        """Melting temperature in K."""
        return self.get_property(Property.TM, output_units="K")

    @cached_property
    def Hf(self) -> types.Quantity1D:
        """Enthalpy of formation in J/mol."""
        return self.get_property(Property.DH_F_STP, output_units="J/mol")

    @cached_property
    def Gf(self) -> types.Quantity1D:
        """Gibbs free energy in J/mol."""
        return self.get_property(Property.GF, output_units="J/mol")

    @cached_property
    def Hv_stp(self) -> types.Quantity1D:
        """Enthalpy of vaporization at 298 K in J/mol."""
        return self.get_property(Property.DH_V_STP, output_units="J/mol")

    @cached_property
    def omega(self) -> types.Quantity1D:
        """Acentric factor (dimensionless)."""
        return self.get_property(Property.ACENTRIC, output_units="dimensionless")

    @cached_property
    def Vm_stp(self) -> types.Quantity1D:
        """Molar liquid volume at 298 K in m^3/mol."""
        return self.get_property(Property.VM_STP, output_units="m^3/mol")

    @cached_property
    def Cp_stp(self) -> types.Quantity1D:
        """Molar specific heat at 298 K in J/(mol*K)."""
        return self.get_property(Property.CP_STP, output_units="J/(mol*K)")

    @cached_property
    def Cp_B(self) -> types.Quantity1D:
        """Temperature-corrected specific heat (B) in J/(mol*K)."""
        return self.get_property(Property.CP_B, output_units="J/(mol*K)")

    @cached_property
    def Cp_C(self) -> types.Quantity1D:
        """Temperature-corrected specific heat (C) in J/(mol*K)."""
        return self.get_property(Property.CP_C, output_units="J/(mol*K)")

    @cached_property
    def MW(self) -> types.Quantity1D:
        """Molecular weights of the compounds in kg/mol."""
        return self.get_property(Property.MW, output_units="kg/mol")

    @cached_property
    def Lv_stp(self) -> types.Quantity1D:
        """Latent heat of vaporization at 298 K in J/kg."""
        return (self.Hv_stp / self.MW).to("J/kg")

    @cached_property
    def epsilonByKB(self) -> types.Quantity1D:
        """Lennard-Jones well depth over Boltzmann constant in K (Tee et al. 1966)."""
        omega = self.omega.magnitude
        Tc = self.Tc.to("K").magnitude
        return Units.Quantity((0.7915 + 0.1693 * omega) * Tc, "K")

    @cached_property
    def sigma(self) -> types.Quantity1D:
        """Lennard-Jones collision diameter in m (Tee et al. 1966)."""
        omega = self.omega.magnitude
        Tc = self.Tc.to("K").magnitude
        Pc = self.Pc.to("atm").magnitude
        sigma = (2.3551 - 0.0874 * omega) * (Tc / Pc) ** (1.0 / 3)
        return Units.Quantity(sigma, "angstrom").to("m")

    @cached_property
    def YSI(self) -> types.Quantity1D:
        """Yield sooting indices (NaN where unavailable)."""  # ruff: ignore[property-docstring-starts-with-verb]
        return self.get_property(Property.YSI, output_units="dimensionless")

    @cached_property
    def DCN(self) -> types.Quantity1D:
        """Derived cetane numbers (ASTM D6890 IQT scale; NaN where unavailable)."""
        return self.get_property(Property.DCN, output_units="dimensionless")

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(
        self, Yi: types.Quantity1D | None = None
    ) -> types.Quantity0D:
        """Calculate the mean molecular weight of the mixture.

        Args:
            Yi: Mass fractions of each compound.
                Defaults to `self.Y_0` (initial mass fractions).

        Returns:
            Mean molecular weight of the mixture in kg/mol.
        """
        return correlate.mixture.mean_molecular_weight(self, Yi)

    def mass2Y(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Convert mass of each component to mass fractions (Yi).

        Args:
            mass: Mass of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_to_mass_fractions(self, mass)

    def mass2X(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Convert mass of each component to mole fractions (Xi).

        Args:
            mass: Mass of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_to_mole_fractions(self, mass)

    def X2Y(self, Xi: types.Quantity1D) -> types.Quantity1D:
        """Convert mole fractions (Xi) to mass fractions (Yi).

        Args:
            Xi: Mole fractions of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mole_fractions_to_mass_fractions(self, Xi)

    def Y2X(self, Yi: types.Quantity1D) -> types.Quantity1D:
        """Convert mass fractions (Yi) to mole fractions (Xi).

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        return correlate.helpers.mass_fractions_to_mole_fractions(self, Yi)

    def density(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate the density of each component at temperature T.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Density of each compound in kg/m^3.
        """
        rho_i = correlate.components.density(self, T)
        return rho_i[comp_idx] if comp_idx is not None else rho_i

    def viscosity_kinematic(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate the viscosity using Dutt's equation.

        Uses Dutt's equation (4.23) from "Viscosity of Liquids". The equation
        predicts viscosity in mm^2/s and is converted to SI units.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Viscosity of each component in m^2/s.
        """
        nu_i = correlate.components.kinematic_viscosity_dutt(self, T)
        return nu_i[comp_idx] if comp_idx is not None else nu_i

    def viscosity_dynamic(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate liquid dynamic viscosity based on droplet temperature and density.

        Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Dynamic viscosity in Pa*s.
        """
        mu_i = correlate.components.dynamic_viscosity_dutt(self, T)
        return mu_i[comp_idx] if comp_idx is not None else mu_i

    def Cp(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Compute molar specific heat capacity at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Molar specific heat capacity in J/mol/K.
        """
        cp = correlate.components.molar_specific_heat_capacity(self, T)
        return cp[comp_idx] if comp_idx is not None else cp

    def Cl(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Mass specific heat capacity in J/kg/K.
        """
        cp = correlate.components.liquid_mass_specific_heat_capacity(self, T)
        return cp[comp_idx] if comp_idx is not None else cp

    def psat(
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
        psat = correlate.components.saturated_vapor_pressure(
            self, T, correlation=correlation
        )
        return psat[comp_idx] if comp_idx is not None else psat

    def psat_antoine_coeffs(
        self,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
        """Estimate Antoine coefficients for vapor pressure of an individual compound.

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
        A, B, C, D = correlate.components.saturated_vapor_pressure_antoine_coeffs(
            self, Tvals=Tvals, units=units, correlation=correlation
        )
        return A, B, C, D

    def molar_liquid_vol(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Compute molar liquid volume with temperature correction.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Molar liquid volume in m^3/mol.
        """
        Vmi = correlate.components.molar_liquid_volume(self, T)
        return Vmi[comp_idx] if comp_idx is not None else Vmi

    def latent_heat_vaporization(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate latent heat of vaporization adjusted for temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Latent heat of vaporization in J/kg.
        """
        Lvi = correlate.components.latent_heat_vaporization(self, T)
        return Lvi[comp_idx] if comp_idx is not None else Lvi

    def diffusion_coeff(
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
        D_AB_i = correlate.components.diffusion_coeffs_wilke(
            self,
            p,
            T,
            sigma_gas=sigma_gas,
            epsilonByKB_gas=epsilonByKB_gas,
            MW_gas=MW_gas,
            correlation=correlation,
        )
        return D_AB_i

    def surface_tension(
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
        st = correlate.components.surface_tension(self, T, correlation=correlation)
        return st[comp_idx] if comp_idx is not None else st

    def thermal_conductivity(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Calculate thermal conductivity at a given temperature.

        Uses Latini et al. method (Poling equation 10-9.1).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Thermal conductivity in W/m/K.
        """
        tc = correlate.components.thermal_conductivity_latini(self, T)
        return tc[comp_idx] if comp_idx is not None else tc

    # --- Mixture functions ---
    # NOTE: Cannot make Yi optional in mixture functions because switching the order of
    # Yi and T would break the original function signatures.
    def mixture_density(
        self, Yi: types.Quantity1D, T: types.Quantity0D
    ) -> types.Quantity1D:
        """Calculate mixture density at a given temperature.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.

        Returns:
            Mixture density in kg/m^3.
        """
        return correlate.mixture.density(self, T, Yi)

    def mixture_kinematic_viscosity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate kinematic viscosity of the mixture.

        Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
                Defaults to "Kendall-Monroe".

        Returns:
            Mixture kinematic viscosity in m^2/s.
        """
        return correlate.mixture.kinematic_viscosity_dutt(
            self, T, Yi, correlation=correlation
        )

    def mixture_dynamic_viscosity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate dynamic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").

        Returns:
            Mixture dynamic viscosity in Pa*s.
        """
        return correlate.mixture.dynamic_viscosity_dutt(
            self, T, Yi, correlation=correlation
        )

    def mixture_vapor_pressure(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity0D:
        """Calculate saturated vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Mixture saturated vapor pressure in Pa.
        """
        return correlate.mixture.saturated_vapor_pressure(
            self, T, Yi, correlation=correlation
        )

    def mixture_vapor_pressure_antoine_coeffs(
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
        return correlate.mixture.saturated_vapor_pressure_antoine_coeffs(
            self, Tvals, Yi, units=units, correlation=correlation
        )

    def mixture_surface_tension(
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
        return correlate.mixture.surface_tension(self, T, Yi, correlation=correlation)

    def mixture_thermal_conductivity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
    ) -> types.Quantity0D:
        """Calculate thermal conductivity of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.

        Returns:
            Thermal conductivity in W/m/K.
        """
        return correlate.mixture.thermal_conductivity_latini(self, T, Yi)


__all__ = ["Fuel"]
