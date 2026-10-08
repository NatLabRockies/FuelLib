"""Fuel class for Group Contribution Method calculations."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from rdkit.Chem import Mol

from . import correlate
from .constants import EpsilonByKB_gas, MW_gas, Sigma_gas
from .gcm import boehm_gcm, gani_gcm
from .rdk import mol
from .utils import Units, types

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _resolve_ref_path(path: str | Path | None, dataDir: Path, fileName: str) -> Path:
    """Resolve the path to a reference data file, falling back to the default.

    Args:
        path: User-specified path to the file, or None.
        dataDir: Fuel data directory to look in when `path` is None.
        fileName: Name of the reference file.

    Returns:
        Path to an existing reference data file.

    Raises:
        FileNotFoundError: If neither the requested nor default file exists.
    """
    refPath = Path(path) if path is not None else dataDir / fileName
    if not refPath.exists():
        print(f"Reference file not found at {refPath}, using default location.")
        refPath = DEFAULT_DATA_DIR / fileName
        if not refPath.exists():
            msg = f"Default reference file not found: {refPath}"
            raise FileNotFoundError(msg)
    return refPath


def _match_gc_to_ref(gcData: DataFrame, refData: DataFrame, refName: str) -> DataFrame:
    """Match GCxGC data to a reference data table.

    Rows are matched first by 'Reference Compound' (case-insensitive), then any
    remaining rows are matched by comparing the InChI of their 'SMILES'
    (if the reference table has a 'SMILES' column).

    Args:
        gcData: GCxGC data as a DataFrame.
        refData: Reference data as a DataFrame.
        refName: Name of the reference data (used in error messages).

    Returns:
        Subset of `refData` with one row per GCxGC row, in GCxGC order.

    Raises:
        ValueError: If any GCxGC row cannot be matched to a reference row.
    """
    refIdx = Series(pd.NA, index=gcData.index, dtype="object")

    def _normalize_names(names: Series) -> Series:
        return names.astype("string").str.strip().str.lower()

    def _smiles_to_inchi(smiles: Series) -> Series:
        return (
            smiles
            .astype("string")
            .str.strip()
            .map(
                lambda s: (
                    mol.inchi(mol.from_smiles(s)) if not pd.isna(s) and s else pd.NA
                )
            )
        )

    if "Reference Compound" in gcData.columns:
        refNames = _normalize_names(refData["Reference Compound"]).dropna()
        nameLookup = dict(zip(refNames, refNames.index))
        refIdx = _normalize_names(gcData["Reference Compound"]).map(nameLookup)

    unmatched = refIdx.isna()
    if unmatched.any() and "SMILES" in gcData.columns and "SMILES" in refData.columns:
        refInchi = _smiles_to_inchi(refData["SMILES"]).dropna()
        inchiLookup = dict(zip(refInchi, refInchi.index))
        refIdx[unmatched] = _smiles_to_inchi(gcData.loc[unmatched, "SMILES"]).map(
            inchiLookup
        )

    unmatched = refIdx.isna()
    if unmatched.any():
        idCols = [c for c in ("Reference Compound", "SMILES") if c in gcData.columns]
        msg = (
            f"GCxGC compounds not found in reference data '{refName}':\n"
            f"{gcData.loc[unmatched, idCols].to_string()}"
        )
        raise ValueError(msg)

    return refData.loc[refIdx.astype(int)].reset_index(drop=True)


class Fuel:
    """Class for handling calculations of thermodynamic and mixture properties."""

    def __init__(
        self,
        name: str,
        fuelDataDir: str | Path | None = None,
        refCompoundsPath: str | Path | None = None,
        refGaniPath: str | Path | None = None,
        *,
        useRefProperties: bool = True,
    ) -> None:
        """Initialize a Fuel instance.

        Args:
            name: Name of the fuel.
            fuelDataDir: Directory containing fuel data.
                Defaults to `fuelLib/data`.
            refCompoundsPath: Path to the reference compounds CSV file.
                Defaults to `fuellib/data/referenceCompounds.csv`.
            refGaniPath: Path to the reference Gani decomposition CSV file.
                Defaults to `fuellib/data/referenceGani.csv`.
            useRefProperties: Whether to use reference compound properties for
                calculations. Defaults to `True`.

        Raises:
            FileNotFoundError: If the fuel data directory, reference CSV files,
                or gcxgc data file do not exist.
            ValueError: If the GCxGC data does not contain either 'Reference Compound'
                or 'SMILES' columns or 'Weight %' column is missing.
        """
        self.name: str = name
        """Name of the fuel/mixture."""
        self.useRefProperties: bool = useRefProperties
        """Whether to use reference compound properties for calculations."""
        self.fuelDataDir: Path = (
            Path(fuelDataDir) if fuelDataDir is not None else DEFAULT_DATA_DIR
        )
        """Directory containing fuel data."""
        if not self.fuelDataDir.exists():
            msg = f"Fuel data directory does not exist: {self.fuelDataDir}"
            raise FileNotFoundError(msg)

        gcFile: Path = self.fuelDataDir / "gcData" / f"{name}.csv"
        if not gcFile.exists():
            msg = f"GCxGC data file does not exist: {gcFile}"
            raise FileNotFoundError(msg)

        self.gcData: DataFrame = pd.read_csv(gcFile)
        """GCxGC data as a DataFrame."""
        if (
            "Reference Compound" not in self.gcData.columns
            and "SMILES" not in self.gcData.columns
        ):
            msg = (
                "GCxGC data must contain either 'Reference Compound' or 'SMILES' "
                "columns."
            )
            raise ValueError(msg)
        if "Weight %" not in self.gcData.columns:
            msg = "GCxGC data must contain 'Weight %' column."
            raise ValueError(msg)
        Y_0 = self.gcData["Weight %"].to_numpy(float)
        self.Y_0: types.Quantity1D = (
            Units.Quantity(Y_0 / sum(Y_0), "")
            if sum(Y_0) > 0
            else Units.Quantity(np.zeros_like(Y_0), "")
        )
        """Initial weight fractions from the GCxGC data."""

        self.refCompounds: DataFrame = pd.read_csv(
            _resolve_ref_path(refCompoundsPath, self.fuelDataDir, "refCompounds.csv")
        )
        """Reference compounds and their properties."""
        self.refGani: DataFrame = pd.read_csv(
            _resolve_ref_path(refGaniPath, self.fuelDataDir, "refGani.csv")
        ).rename(columns={"Common_Name": "Reference Compound"})
        """Reference Gani group decompositions."""
        self.compoundsData: DataFrame = _match_gc_to_ref(
            self.gcData, self.refCompounds, "referenceCompounds"
        )
        """Matched GCxGC data to reference compounds."""
        self.compounds = self.compoundsData["Reference Compound"].to_list()
        """List of common names for each matched compound."""
        self.smiles = self.compoundsData["SMILES"].to_list()
        """List of SMILES strings for each matched compound."""
        self.pelephysics_keys: list[str] | None = (
            self.gcData["PelePhysics Key"].tolist()
            if "PelePhysics Key" in self.gcData.columns
            else None
        )

        self.ganiDecomp: DataFrame = (
            _match_gc_to_ref(self.compoundsData, self.refGani, "referenceGani")
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """Gani group decompositions for each matched compound."""

        propFilePath = self.fuelDataDir / "propertiesData" / f"{name}.csv"
        self.propData: DataFrame | None = (
            pd.read_csv(propFilePath) if propFilePath.exists() else None
        )
        """Property data for the fuel/mixture, if available."""

    def _get_ref_values(self, prop: str, output_units: str) -> types.Quantity1D:
        """Get reference values for the given compounds in the specified output units.

        Args:
            prop: Property column prefix in the reference compounds file
                (e.g. "Tc" for the "Tc_Value" and "Tc_Units" columns).
            output_units: Desired output units for the reference values.

        Returns:
            Reference values for the given compounds in the specified output units.
        """
        out = np.full(len(self.compoundsData), np.nan, dtype=float)
        values = self.compoundsData[f"{prop}_Value"]
        units = self.compoundsData[f"{prop}_Units"]
        for i, (value, unit) in enumerate(zip(values, units)):
            if pd.isna(value):
                continue
            quantity = Units.Quantity(float(value), "" if pd.isna(unit) else unit)
            out[i] = quantity.to(output_units).magnitude
        return Units.Quantity(out, output_units)

    # -------------------------------------------------------------------------
    # Critical properties
    # -------------------------------------------------------------------------
    @cached_property
    def Tc(self) -> types.Quantity1D:
        """Critical temperature for each compound in K."""
        preds = gani_gcm.predict("Tc", self).to("K")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Tc", "K")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), "K"
        )

    @cached_property
    def Pc(self) -> types.Quantity1D:
        """Critical pressure for each compound in Pa."""
        preds = gani_gcm.predict("Pc", self).to("Pa")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Pc", "Pa")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), "Pa"
        )

    @cached_property
    def Vc(self) -> types.Quantity1D:
        """Critical volume for each compound in m^3/mol."""
        preds = gani_gcm.predict("Vc", self).to("m^3/mol")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Vc", "m^3/mol")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude),
            "m^3/mol",
        )

    @cached_property
    def Tm(self) -> types.Quantity1D:
        """Melting point for each compound in K."""
        preds = gani_gcm.predict("Tm", self).to("K")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Tm", "K")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), "K"
        )

    @cached_property
    def Tb(self) -> types.Quantity1D:
        """Boiling point for each compound in K."""
        preds = gani_gcm.predict("Tb", self).to("K")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Tb", "K")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), "K"
        )

    @cached_property
    def Hf_stp(self) -> types.Quantity1D:
        """Enthalpy of formation (at STP) for each compound in J/mol."""
        preds = gani_gcm.predict("Hf", self).to("J/mol")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Hf_STP", "J/mol")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), "J/mol"
        )

    @cached_property
    def Hv_stp(self) -> types.Quantity1D:
        """Enthalpy of vaporization (at STP) for each compound in J/mol."""
        preds = gani_gcm.predict("Hv_stp", self).to("J/mol")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Hv_STP", "J/mol")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), "J/mol"
        )

    @cached_property
    def Vm_stp(self) -> types.Quantity1D:
        """Molar volume (at STP) for each compound in m^3/mol."""
        preds = gani_gcm.predict("Vm_stp", self).to("m^3/mol")
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Vm_STP", "m^3/mol")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude),
            "m^3/mol",
        )

    @cached_property
    def omega(self) -> types.Quantity1D:
        """Acentric factor for each compound (dimensionless)."""
        preds = gani_gcm.predict("omega", self)
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values("Omega", "")
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), ""
        )

    # -------------------------------------------------------------------------
    # Reference-only properties
    # -------------------------------------------------------------------------
    @cached_property
    def YSI(self) -> types.Quantity1D:
        """Yield Sooting Index (YSI) for each compound (dimensionless).

        Raises:
            ValueError: If any reference values for YSI are missing.
        """  # ruff: ignore[property-docstring-starts-with-verb]
        refs = self._get_ref_values("YSI", "")
        missing = np.isnan(refs.magnitude)
        if missing.any():
            names = ", ".join(
                self.compoundsData.loc[missing, "Reference Compound"].astype(str)
            )
            msg = f"Missing Yield Sooting Index (YSI) reference values for: {names}."
            raise ValueError(msg)
        return refs

    @cached_property
    def DCN(self) -> types.Quantity1D:
        """Derived cetane number (DCN) for each compound (dimensionless).

        Raises:
            ValueError: If any reference values for DCN are missing.
        """
        refs = self._get_ref_values("DCN", "")
        missing = np.isnan(refs.magnitude)
        if missing.any():
            names = ", ".join(
                self.compoundsData.loc[missing, "Reference Compound"].astype(str)
            )
            msg = f"Missing Derived Cetane Number (DCN) reference values for: {names}."
            raise ValueError(msg)
        return refs

    # -------------------------------------------------------------------------
    # Prediction-only properties
    # -------------------------------------------------------------------------
    @cached_property
    def Gf_stp(self) -> types.Quantity1D:
        """Gibbs free energy of formation (at STP) for each compound in J/mol."""
        return gani_gcm.predict("Gf", self).to("J/mol")

    @cached_property
    def Cp_stp(self) -> types.Quantity1D:
        """Heat capacity at constant pressure for each compound in J/(mol*K)."""
        return gani_gcm.predict("Cp_stp", self).to("J/(mol*K)")

    @cached_property
    def Cp_B(self) -> types.Quantity1D:
        """Temperature correction term B for heat capacity in J/(mol*K)."""
        return gani_gcm.predict("Cp_B", self).to("J/(mol*K)")

    @cached_property
    def Cp_C(self) -> types.Quantity1D:
        """Temperature correction term C for heat capacity in J/(mol*K)."""
        return gani_gcm.predict("Cp_C", self).to("J/(mol*K)")

    @cached_property
    def dS_fus(self) -> types.Quantity1D:
        """Fusion entropy for each compound in the fuel mixture in J/(mol*K)."""
        return boehm_gcm.predict("dS_fus", self).to("J/(mol*K)")

    @cached_property
    def RD_coeffs(self) -> tuple[types.Quantity1D, types.Quantity1D, types.Quantity1D]:
        """Ruzicka-Domalski correlation coefficients for heat capacity."""
        rd_A = gani_gcm.predict("rd_A", self)
        rd_B = gani_gcm.predict("rd_B", self)
        rd_D = gani_gcm.predict("rd_D", self)
        return rd_A, rd_B, rd_D

    @cached_property
    def phi(self) -> types.Quantity1D:
        """Alibakhshi phi terms for each compound in the fuel mixture in K."""
        return gani_gcm.predict("alibakhshi_phi", self).to("K")

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------
    @cached_property
    def Lv_stp(self) -> types.Quantity1D:
        """Latent heat of vaporization (at STP) for each compound in J/kg."""
        return (self.Hv_stp / self.MW).to("J/kg")

    @cached_property
    def epsilonByKB(self) -> types.Quantity1D:
        """Depth of the Lennard-Jones potential well divided by Boltzmann constant."""
        return (0.7915 + 0.1693 * self.omega.magnitude) * self.Tc.to("K")

    @cached_property
    def sigma(self) -> types.Quantity1D:
        """Lennard-Jones collision diameter for each compound in m."""
        sigma = (2.3551 - 0.0874 * self.omega.magnitude) * (
            self.Tc.to("K").magnitude / self.Pc.to("atm").magnitude
        ) ** (1.0 / 3)
        return Units.Quantity(sigma, "angstrom").to("m")

    # -------------------------------------------------------------------------
    # Data initialization
    # -------------------------------------------------------------------------
    @cached_property
    def rdkit_mols(self) -> list[Mol]:
        """RDKit `Mol` objects for the compounds in the fuel mixture."""
        return [mol.from_smiles(smiles) for smiles in self.smiles]

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
    def num_compounds(self) -> int:
        """Number of compounds in the fuel mixture."""
        return len(self.compounds)

    @property
    def formulas(self) -> list[str]:
        """Molecular formulas for each compound in the fuel mixture."""
        return [mol.hill_formula(m) for m in self.rdkit_mols]

    @cached_property
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
