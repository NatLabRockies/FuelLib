"""Fuel class for Group Contribution Method calculations."""

from __future__ import annotations

import os
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from scipy.optimize import curve_fit

from . import correlate
from ._data_locator import (
    get_fueldata_decomp_dir,
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_props_dir,
    get_gcmtable_dir,
    get_metadata_decomp_name,
)
from .constants import EpsilonByKB_gas, MW_gas, Sigma_gas
from .gcm import GCMRegistry
from .rdk import mol
from .utils import Units, types


class Fuel:
    """Class for handling calculations of thermodynamic and mixture properties."""

    def __init__(
        self, name: str, decompName: str | None = None, fuelDataDir: str | None = None
    ) -> None:
        """Initialize Fuel object and pre-compute GCM properties.

        Args:
            name: Name of the mixture as it appears in its gcData file.
            decompName: Name of the groupDecomposition file if different from name.
                Defaults to None.
            fuelDataDir: Directory where the fuel data is stored. If None, uses built-in
                embedded data.
        """
        self.name: str = name
        """Name of the fuel/mixture."""
        if decompName is None:
            # Try to get decomposition name from metadata
            decompName: str = get_metadata_decomp_name(name, fuelDataDir)
            """Name of the group decomposition file."""

        # Determine and set data directories for this fuel instance
        if fuelDataDir is None:
            # Use built-in embedded data
            self.fuelDataDir: str = get_fueldata_dir()
            """Directory containing the fuel data."""
            self.fuelDataGcDir: str = get_fueldata_gc_dir()
            """Directory containing the gas chromatography data."""
            self.fuelDataDecompDir: str = get_fueldata_decomp_dir()
            """Directory containing the group decomposition data."""
            self.fuelDataPropsDir: str = get_fueldata_props_dir()
            """Directory containing the fuel properties data."""
        else:
            # Validate and use custom fuel directory
            from ._data_locator import (
                _get_props_dir_for_fueldata,
                _validate_fuel_data_dir,
            )

            _validate_fuel_data_dir(fuelDataDir)
            self.fuelDataDir: str = fuelDataDir
            """Directory containing the fuel data."""
            self.fuelDataGcDir: str = os.path.join(fuelDataDir, "gcData")
            """Directory containing the gas chromatography data."""
            self.fuelDataDecompDir: str = os.path.join(
                fuelDataDir, "groupDecompositionData"
            )
            """Directory containing the group decomposition data."""
            self.fuelDataPropsDir: str = _get_props_dir_for_fueldata(fuelDataDir)
            """Directory containing the fuel properties data."""

        # Get GCM table directory (always from built-in data)
        gcmtable_dir = get_gcmtable_dir()

        self.groupDecompFile: str = os.path.join(
            self.fuelDataDecompDir, f"{decompName}.csv"
        )
        """File containing the group decomposition data for this fuel."""
        self.gcxgcFile: str = os.path.join(self.fuelDataGcDir, f"{name}_init.csv")
        """File containing the GCxGC compositional data for this fuel."""
        self.gcmTableFile: str = os.path.join(gcmtable_dir, "gcmTable.csv")
        """File containing the GCM table data."""

        # --- Compute critical properties at standard temp (num_compounds,)
        self.Tc: types.Quantity1D = self.get_property("gani", "Tc").to("K")
        """Critical temperature in K."""
        self.Pc: types.Quantity1D = self.get_property("gani", "Pc").to("Pa")
        """Critical pressure in Pa."""
        self.Vc: types.Quantity1D = self.get_property("gani", "Vc").to("m^3/mol")
        """Critical volume in m^3/mol."""
        self.Tb: types.Quantity1D = self.get_property("gani", "Tb").to("K")
        """Boiling temperature in K."""
        self.Tm: types.Quantity1D = self.get_property("gani", "Tm").to("K")
        """Melting temperature in K."""
        self.Hf: types.Quantity1D = self.get_property("gani", "Hf").to("J/mol")
        """Enthalpy of formation in J/mol."""
        self.Gf: types.Quantity1D = self.get_property("gani", "Gf").to("J/mol")
        """Gibbs free energy in J/mol."""
        self.Hv_stp: types.Quantity1D = self.get_property("gani", "Hv_stp").to("J/mol")
        """Enthalpy of vaporization at 298 K in J/mol."""
        self.omega: types.Quantity1D = self.get_property("gani", "omega")
        """Accentric factor (dimensionless)."""
        self.Vm_stp: types.Quantity1D = self.get_property("gani", "Vm_stp").to(
            "m^3/mol"
        )
        """Molar liquid volume at 298 K in m^3/mol."""
        self.Cp_stp: types.Quantity1D = self.get_property("gani", "Cp_stp").to(
            "J/(mol*K)"
        )
        """Molar specific heat at 298 K in J/(mol*K)."""
        self.Cp_B: types.Quantity1D = self.get_property("gani", "Cp_B").to("J/(mol*K)")
        """Temperature-corrected specific heat (B) in J/(mol*K)."""
        self.Cp_C: types.Quantity1D = self.get_property("gani", "Cp_C").to("J/(mol*K)")
        """Temperature-corrected specific heat (C) in J/(mol*K)."""
        # L_v,stp (latent heat of vaporization at 298 K)
        self.Lv_stp: types.Quantity1D = (self.Hv_stp / self.MW).to("J/kg")
        """Latent heat of vaporization at 298 K in J/kg."""

        # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
        _lj_w = self.omega.magnitude
        _lj_tc = self.Tc.to("K").magnitude
        _lj_pc = self.Pc.to("atm").magnitude
        _epsilon_by_kb = (0.7915 + 0.1693 * _lj_w) * _lj_tc
        self.epsilonByKB: types.Quantity1D = Units.Quantity(_epsilon_by_kb, "K")
        """Lennard-Jones well depth over Boltzmann constant in K."""

        _sigma = (2.3551 - 0.0874 * _lj_w) * (_lj_tc / _lj_pc) ** (1.0 / 3)
        self.sigma: types.Quantity1D = Units.Quantity(_sigma, "angstrom").to("m")
        """Lennard-Jones collision diameter in m."""

    # -------------------------------------------------------------------------
    # Parsing functions
    # -------------------------------------------------------------------------
    @cached_property
    def gcxgc_data(self) -> pd.DataFrame:
        """GCxGC data in a pandas DataFrame.

        Returns:
            pandas DataFrame representing the GCxGC data.
                Shape: (num_compounds, num_columns)
        """
        return pd.read_csv(self.gcxgcFile, header=0, index_col=0)

    @property
    def Y_0(self) -> types.Quantity1D:
        """List of initial mass fractions for the compounds in the fuel mixture."""
        if "Weight %" not in self.gcxgc_data.columns:
            return Units.Quantity([], "dimensionless")
        Y_0 = self.gcxgc_data["Weight %"].to_numpy().flatten().astype(float)
        return Units.Quantity(Y_0 / np.sum(Y_0), "dimensionless")

    @property
    def compounds(self) -> list[str]:
        """List of compounds in the fuel mixture."""
        return list(self.gcxgc_data.index)

    @property
    def num_compounds(self) -> int:
        """Number of compounds in the fuel mixture."""
        return len(self.compounds)

    @property
    def smiles(self) -> list[str]:
        """List of SMILES strings for the compounds in the fuel mixture.

        Raises:
            ValueError: If the SMILES column is missing from the GCxGC data.
        """
        if "SMILES" not in self.gcxgc_data.columns:
            msg = "SMILES column is missing from the GCxGC data."
            raise ValueError(msg)
        return [smiles.strip() for smiles in self.gcxgc_data["SMILES"].to_list()]

    @property
    def pelephysics_keys(self) -> list[str] | None:
        """List of PelePhysics keys for the compounds in the fuel mixture."""
        if "PelePhysics Key" not in self.gcxgc_data.columns:
            return None
        return [key.strip() for key in self.gcxgc_data["PelePhysics Key"].to_list()]

    def gani_decomp(self) -> pd.DataFrame:
        """Parse the Gani decomposition matrix into a DataFrame.

        Returns:
            A pandas DataFrame representing the Gani decomposition matrix.
                Shape: (num_compounds, num_groups)

        Raises:
            ValueError: If any compounds in the fuel mixture are missing from the Gani
                decomposition file.
        """
        df = pd.read_csv(self.groupDecompFile, header=0, index_col=0)
        missing = set(self.compounds) - set(df.index)
        if missing:
            msg = (
                f"Gani decomposition file ({self.groupDecompFile}) is missing compounds"
                f" present in the fuel mixture: {sorted(missing)}."
            )
            raise ValueError(msg)
        return df.loc[self.compounds]

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
        return [mol.inchi(m) for m in self.rdkit_mols]

    @property
    def nC(self) -> list[int]:
        """Number of carbon atoms for each compound in the fuel mixture."""
        return [mol.atom_counts(m).get("C", 0) for m in self.rdkit_mols]

    @property
    def nH(self) -> list[int]:
        """Number of hydrogen atoms for each compound in the fuel mixture."""
        return [mol.atom_counts(m).get("H", 0) for m in self.rdkit_mols]

    @property
    def MW(self) -> types.Quantity1D:
        """Molecular weights of the compounds in the fuel mixture in kg/mol."""
        return Units.Q([mol.molecular_weight(m) for m in self.rdkit_mols], "g/mol").to(
            "kg/mol"
        )

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

    @cached_property
    def gcm_properties(self) -> dict[str, dict[str, types.Quantity1D]]:
        """Pre-computed GCM properties for the compounds.

        Returns:
            A dictionary containing the pre-computed GCM properties for the compounds.
            The keys are the GCM method names, and the values are dictionaries mapping
            property names to 1D numpy arrays of the property values for each compound.
        """
        props: dict[str, dict[str, types.Quantity1D]] = {}
        for gcm in GCMRegistry.methods:
            props.update(gcm.predict_all(self))
        return props

    def get_property(self, method: str, property_name: str) -> types.Quantity1D:
        """Get a specific property prediction from the GCM for each compound.

        Args:
            method: The GCM method to use.
            property_name: The name of the property to retrieve.

        Returns:
            Quantity vector of the requested predictions for each compound.

        Raises:
            KeyError: If the GCM method or property is not found.
        """
        method = method.lower()
        if method not in self.gcm_properties:
            msg = f"Method '{method}' not found in computed GCM properties."
            raise KeyError(msg)

        property_name = property_name.lower()
        if property_name not in self.gcm_properties[method]:
            msg = f"Property '{property_name}' not found in computed GCM properties."
            raise KeyError(msg)

        return self.gcm_properties[method][property_name]

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
        cp = correlate.components.molar_specific_heat(self, T)
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
        cp = correlate.components.liquid_mass_specific_heat(self, T)
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
