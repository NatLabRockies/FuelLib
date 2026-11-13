"""Core Fuel class."""

import warnings
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
import pandas as pd
import yaml
from rdkit.Chem import Mol

from ..data._reference_data import (
    DEFAULT_DATA_DIR,
    load_ref_compounds,
    load_ref_decomp,
    smiles_series_to_inchi,
)
from ..gcm import gani_gcm
from ..utils import Units, types
from ..utils.constants import EpsilonByKB_gas, MW_gas, Sigma_gas, T_stp
from .components import Components
from .mixture import Mixture


def _warn_deprecated(old: str, new: str) -> None:
    """Warn that `Fuel.<old>` is deprecated in favor of `Fuel.<new>`."""
    warnings.warn(
        f"Fuel.{old} is deprecated. Use Fuel.{new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _match_gc_to_ref(
    gc_data: pd.DataFrame, ref_data: pd.DataFrame, ref_name: str
) -> pd.DataFrame:
    """Match GCxGC data to a reference data table.

    Rows are matched first by 'Reference Compound' (case-insensitive), then any
    remaining rows are matched by comparing the InChI of their 'SMILES'
    (if the reference table has a 'SMILES' column).

    Args:
        gc_data: GCxGC data as a DataFrame.
        ref_data: Reference data as a DataFrame.
        ref_name: Name of the reference data (used in error messages).

    Returns:
        Subset of `ref_data` with one row per GCxGC row, in GCxGC order.

    Raises:
        ValueError: If any GCxGC row cannot be matched to a reference row.
    """
    refIdx = pd.Series(pd.NA, index=gc_data.index, dtype="object")

    def _normalize_names(names: pd.Series) -> pd.Series:
        return names.astype("string").str.strip().str.lower()

    if "Reference Compound" in gc_data.columns:
        refNames = _normalize_names(ref_data["Reference Compound"]).dropna()
        nameLookup = dict(zip(refNames, refNames.index))
        refIdx = _normalize_names(gc_data["Reference Compound"]).map(nameLookup)

    unmatched = refIdx.isna()
    if unmatched.any() and "SMILES" in gc_data.columns and "SMILES" in ref_data.columns:
        refInchi = smiles_series_to_inchi(ref_data["SMILES"]).dropna()
        inchiLookup = dict(zip(refInchi, refInchi.index))
        refIdx[unmatched] = smiles_series_to_inchi(
            gc_data.loc[unmatched, "SMILES"]
        ).map(inchiLookup)

    unmatched = refIdx.isna()
    if unmatched.any():
        idCols = [c for c in ("Reference Compound", "SMILES") if c in gc_data.columns]
        msg = (
            f"GCxGC compounds not found in reference data '{ref_name}':\n"
            f"{gc_data.loc[unmatched, idCols].to_string()}"
        )
        raise ValueError(msg)

    return ref_data.loc[refIdx.astype(int)].reset_index(drop=True)


class Fuel:
    """Class for handling calculations of thermodynamic and mixture properties."""

    def __init__(
        self,
        name: str,
        data_dir: str | Path | None = None,
        *,
        use_ref_props: bool = True,
        fuelDataDir: str | Path | None = None,
        useRefProperties: bool | None = None,
    ) -> None:
        """Initialize a Fuel instance.

        Args:
            name: Name of the fuel.
            data_dir: Directory containing fuel data.
                Defaults to `fuelLib/data`.
            use_ref_props: Whether to use reference compound properties for
                calculations. Defaults to `True`.
            fuelDataDir: Deprecated alias for `data_dir`.
            useRefProperties: Deprecated alias for `use_ref_props`.

        Raises:
            FileNotFoundError: If the fuel data directory does not exist.
            ValueError: If the GCxGC data file is missing required columns.
        """
        if fuelDataDir is not None:
            warnings.warn(
                "Fuel(fuelDataDir=...) is deprecated. Use Fuel(data_dir=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data_dir = data_dir if data_dir is not None else fuelDataDir
        if useRefProperties is not None:
            warnings.warn(
                "Fuel(useRefProperties=...) is deprecated. "
                "Use Fuel(use_ref_props=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_ref_props = useRefProperties

        self.name: str = name
        """Name of the fuel."""
        self._use_ref_props: Final[bool] = use_ref_props
        """Whether to use external reference compound properties for calculations."""

        # Load data
        self.data_dir: Path = (
            Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        )
        """Directory containing fuel data."""
        if not self.data_dir.exists():
            msg = f"Fuel data directory does not exist: {self.data_dir}"
            raise FileNotFoundError(msg)

        gc_file: Path = self.data_dir / "gc_data" / f"{name}.csv"
        if not gc_file.exists():
            msg = f"GCxGC data file does not exist: {gc_file}"
            raise FileNotFoundError(msg)

        self.gc_data: pd.DataFrame = pd.read_csv(gc_file)
        """GCxGC data as a DataFrame."""
        if (
            "Reference Compound" not in self.gc_data.columns
            and "SMILES" not in self.gc_data.columns
        ):
            msg = (
                f"GCxGC data must contain either 'Reference Compound' or 'SMILES' "
                f"columns. File: {gc_file}"
            )
            raise ValueError(msg)
        if "Weight %" not in self.gc_data.columns:
            msg = f"GCxGC data must contain 'Weight %' column. File: {gc_file}"
            raise ValueError(msg)

        y0 = self.gc_data["Weight %"].to_numpy(float)
        self.y0: types.Quantity1D = (
            Units.Quantity(y0 / sum(y0), "")
            if sum(y0) > 0
            else Units.Quantity(np.zeros_like(y0), "")
        )
        """Normalized weight fractions from the GCxGC data."""

        self._ref_comps: pd.DataFrame = load_ref_compounds(self.data_dir)
        """Reference compounds and their properties (bundled plus custom)."""
        self.components_data: pd.DataFrame = _match_gc_to_ref(
            self.gc_data, self._ref_comps, "reference compounds"
        )
        """Matched components data with reference compounds."""

        self._ref_gani: pd.DataFrame = load_ref_decomp(self.data_dir, "Gani")
        """Reference Gani group decompositions (bundled plus custom)."""
        self._gani_decomp: pd.DataFrame = (
            _match_gc_to_ref(self.components_data, self._ref_gani, "referenceGani")
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """Gani group decompositions for each matched compound."""
        self._ref_nannoolal: pd.DataFrame = load_ref_decomp(self.data_dir, "Nannoolal")
        """Reference Nannoolal group decompositions (bundled plus custom)."""
        self._nannoolal_decomp: pd.DataFrame = (
            _match_gc_to_ref(
                self.components_data, self._ref_nannoolal, "referenceNannoolal"
            )
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """Nannoolal group decompositions for each matched compound."""
        self._ref_unifac: pd.DataFrame = load_ref_decomp(self.data_dir, "UNIFAC")
        """Reference UNIFAC group decompositions (bundled plus custom)."""
        self._unifac_decomp: pd.DataFrame = (
            _match_gc_to_ref(self.components_data, self._ref_unifac, "referenceUNIFAC")
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """UNIFAC group decompositions for each matched compound."""

        prop_file = self.data_dir / "properties_data" / f"{name}.csv"
        self.prop_data: pd.DataFrame | None = (
            pd.read_csv(prop_file) if prop_file.exists() else None
        )
        """Property data for the fuel/mixture, if available."""

        metadata_file = self.data_dir / "fuel_metadata.yaml"
        self.metadata: dict | None = (
            (yaml.safe_load(metadata_file.read_text()) or {}).get("fuels", {}).get(name)
            if metadata_file.exists()
            else None
        )
        """Fuel metadata, if available."""

        self.comps: Components = Components(self)
        """Components attribute for the fuel."""

        self.mix: Mixture = Mixture(self)
        """Mixture attribute for the fuel."""

    @property
    def num_compounds(self: Self) -> int:
        """Number of components in the fuel."""
        return len(self.components_data)

    # Deprecated properties for backward compatibility
    @property
    def useRefProperties(self: Self) -> bool:
        """Deprecated. Use `_use_ref_props` instead."""
        warnings.warn(
            "Fuel.useRefProperties is deprecated. Use Fuel._use_ref_props instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_ref_props

    @property
    def fuelDataDir(self: Self) -> Path:
        """Deprecated. Use `data_dir` instead."""
        warnings.warn(
            "Fuel.fuelDataDir is deprecated. Use Fuel.data_dir instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.data_dir

    @property
    def gcData(self: Self) -> pd.DataFrame:
        """Deprecated. Use `gc_data` instead."""
        warnings.warn(
            "Fuel.gcData is deprecated. Use Fuel.gc_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gc_data

    @property
    def Y_0(self: Self) -> types.Quantity1D:
        """Deprecated. Use `y0` instead."""
        warnings.warn(
            "Fuel.Y_0 is deprecated. Use Fuel.y0 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.y0

    @property
    def compoundsData(self: Self) -> pd.DataFrame:
        """Deprecated. Use `components_data` instead."""
        warnings.warn(
            "Fuel.compoundsData is deprecated. Use Fuel.components_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.components_data

    @property
    def refGani(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_ref_gani` instead."""
        warnings.warn(
            "Fuel.refGani is deprecated. Use Fuel._ref_gani instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ref_gani

    @property
    def ganiDecomp(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_gani_decomp` instead."""
        warnings.warn(
            "Fuel.ganiDecomp is deprecated. Use Fuel._gani_decomp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._gani_decomp

    @property
    def refNannoolal(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_ref_nannoolal` instead."""
        warnings.warn(
            "Fuel.refNannoolal is deprecated. Use Fuel._ref_nannoolal instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ref_nannoolal

    @property
    def nannoolalDecomp(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_nannoolal_decomp` instead."""
        warnings.warn(
            "Fuel.nannoolalDecomp is deprecated. Use Fuel._nannoolal_decomp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._nannoolal_decomp

    @property
    def refUnifac(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_ref_unifac` instead."""
        warnings.warn(
            "Fuel.refUnifac is deprecated. Use Fuel._ref_unifac instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ref_unifac

    @property
    def unifacDecomp(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_unifac_decomp` instead."""
        warnings.warn(
            "Fuel.unifacDecomp is deprecated. Use Fuel._unifac_decomp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._unifac_decomp

    @property
    def refCompounds(self: Self) -> pd.DataFrame:
        """Deprecated. Use `_ref_comps` instead."""
        warnings.warn(
            "Fuel.refCompounds is deprecated. Use Fuel._ref_comps instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ref_comps

    @property
    def compounds(self: Self) -> list[str]:
        """Deprecated. Use `comps.names` instead."""
        warnings.warn(
            "Fuel.compounds is deprecated. Use Fuel.comps.names instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.names

    @property
    def smiles(self: Self) -> list[str]:
        """Deprecated. Use `comps.smiles` instead."""
        warnings.warn(
            "Fuel.smiles is deprecated. Use Fuel.comps.smiles instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.smiles

    @property
    def pelephysics_keys(self: Self) -> list[str] | None:
        """Deprecated. Use `comps.pelephysics_keys` instead."""
        warnings.warn(
            "Fuel.pelephysics_keys is deprecated. "
            "Use Fuel.comps.pelephysics_keys instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.pelephysics_keys

    @property
    def propData(self: Self) -> pd.DataFrame | None:
        """Deprecated. Use `prop_data` instead."""
        warnings.warn(
            "Fuel.propData is deprecated. Use Fuel.prop_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.prop_data

    @property
    def Tc(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.tc` instead."""
        warnings.warn(
            "Fuel.Tc is deprecated. Use Fuel.comps.tc instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.tc

    @property
    def Pc(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.pc` instead."""
        warnings.warn(
            "Fuel.Pc is deprecated. Use Fuel.comps.pc instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.pc

    @property
    def Vc(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.vc` instead."""
        warnings.warn(
            "Fuel.Vc is deprecated. Use Fuel.comps.vc instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.vc

    @property
    def omega(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.omega` instead."""
        warnings.warn(
            "Fuel.omega is deprecated. Use Fuel.comps.omega instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.omega

    @property
    def Tm(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.tm` instead."""
        warnings.warn(
            "Fuel.Tm is deprecated. Use Fuel.comps.tm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.tm

    @property
    def Tb(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.tb` instead."""
        warnings.warn(
            "Fuel.Tb is deprecated. Use Fuel.comps.tb instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.tb

    @property
    def Hf_stp(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.hf_stp` instead."""
        warnings.warn(
            "Fuel.Hf_stp is deprecated. Use Fuel.comps.hf_stp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.hf_stp

    @property
    def Hv_stp(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.hv_stp` instead."""
        warnings.warn(
            "Fuel.Hv_stp is deprecated. Use Fuel.comps.hv_stp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.hv_stp

    @property
    def Vm_stp(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.vm_stp` instead."""
        warnings.warn(
            "Fuel.Vm_stp is deprecated. Use Fuel.comps.vm_stp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.vm_stp

    @property
    def YSI(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.ysi` instead."""
        warnings.warn(
            "Fuel.YSI is deprecated. Use Fuel.comps.ysi instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.ysi

    @property
    def DCN(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.dcn` instead."""
        warnings.warn(
            "Fuel.DCN is deprecated. Use Fuel.comps.dcn instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.dcn

    @property
    def Gf_stp(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.gf_stp` instead."""
        warnings.warn(
            "Fuel.Gf_stp is deprecated. Use Fuel.comps.gf_stp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.gf_stp

    @property
    def Cp_stp(self: Self) -> types.Quantity1D:
        """Deprecated. Use `gani_gcm.predict("Cp_stp", fuel)` instead."""
        warnings.warn(
            "Fuel.Cp_stp is deprecated. "
            'Use fuellib.gcm.gani_gcm.predict("Cp_stp", fuel) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return gani_gcm.predict("Cp_stp", self).to("J/(mol*K)")

    @property
    def Lv_stp(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.hv_stp / comps.mw` instead."""
        warnings.warn(
            "Fuel.Lv_stp is deprecated. Use Fuel.comps.hv_stp / Fuel.comps.mw instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self.comps.hv_stp / self.comps.mw).to("J/kg")

    @property
    def epsilonByKB(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.lj_eps_by_kb` instead."""
        warnings.warn(
            "Fuel.epsilonByKB is deprecated. Use Fuel.comps.lj_eps_by_kb instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.lj_eps_by_kb

    @property
    def sigma(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.lj_sigma` instead."""
        warnings.warn(
            "Fuel.sigma is deprecated. Use Fuel.comps.lj_sigma instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.lj_sigma

    @property
    def rdkit_mols(self: Self) -> list[Mol]:
        """Deprecated. Use `comps.rdkit_mols` instead."""
        warnings.warn(
            "Fuel.rdkit_mols is deprecated. Use Fuel.comps.rdkit_mols instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.rdkit_mols

    @property
    def nC(self: Self) -> np.ndarray:
        """Deprecated. Use `comps.num_c` instead."""
        warnings.warn(
            "Fuel.nC is deprecated. Use Fuel.comps.num_c instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.num_c

    @property
    def nH(self: Self) -> np.ndarray:
        """Deprecated. Use `comps.num_h` instead."""
        warnings.warn(
            "Fuel.nH is deprecated. Use Fuel.comps.num_h instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.num_h

    @property
    def MW(self: Self) -> types.Quantity1D:
        """Deprecated. Use `comps.mw` instead."""
        warnings.warn(
            "Fuel.MW is deprecated. Use Fuel.comps.mw instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.mw

    @property
    def formulas(self: Self) -> list[str]:
        """Deprecated. Use `comps.formulas` instead."""
        warnings.warn(
            "Fuel.formulas is deprecated. Use Fuel.comps.formulas instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.formulas

    @property
    def hc_type(self: Self) -> list[str]:
        """Deprecated. Use `comps.hc_type` instead."""
        warnings.warn(
            "Fuel.hc_type is deprecated. Use Fuel.comps.hc_type instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.hc_type

    @property
    def fam(self: Self) -> types.Array1D:
        """Deprecated. Use `comps.fam` instead."""
        warnings.warn(
            "Fuel.fam is deprecated. Use Fuel.comps.fam instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.comps.fam

    # Deprecated methods for backward compatibility
    def density(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.rho` instead.

        Returns:
            Density of each compound in kg/m^3.
        """
        _warn_deprecated("density", "comps.rho")
        return self.comps.rho(T, comp_idx)

    def viscosity_kinematic(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.nu` instead.

        Returns:
            Viscosity of each component in m^2/s.
        """
        _warn_deprecated("viscosity_kinematic", "comps.nu")
        return self.comps.nu(T, comp_idx, correlation)

    def viscosity_dynamic(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.mu` instead.

        Returns:
            Dynamic viscosity in Pa*s.
        """
        _warn_deprecated("viscosity_dynamic", "comps.mu")
        return self.comps.mu(T, comp_idx, correlation)

    def Cp(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.cp(basis="molar")` instead.

        Returns:
            Molar specific heat capacity in J/mol/K.
        """
        _warn_deprecated("Cp", 'comps.cp(basis="molar")')
        return self.comps.cp(T, comp_idx, basis="molar")

    def Cl(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.cp(basis="mass")` instead.

        Returns:
            Mass specific heat capacity in J/kg/K.
        """
        _warn_deprecated("Cl", 'comps.cp(basis="mass")')
        return self.comps.cp(T, comp_idx, basis="mass")

    def psat(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.pvap` instead.

        Returns:
            Vapor pressure in Pa.
        """
        _warn_deprecated("psat", "comps.pvap")
        return self.comps.pvap(T, comp_idx, correlation)

    def psat_antoine_coeffs(
        self: Self,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
        """Deprecated. Use `comps.pvap_antoine_coeffs` instead.

        Returns:
            Coefficients A, B, C, D for each compound.
        """
        _warn_deprecated("psat_antoine_coeffs", "comps.pvap_antoine_coeffs")
        return self.comps.pvap_antoine_coeffs(Tvals, units, correlation)

    def molar_liquid_vol(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.vm` instead.

        Returns:
            Molar liquid volume in m^3/mol.
        """
        _warn_deprecated("molar_liquid_vol", "comps.vm")
        return self.comps.vm(T, comp_idx)

    def latent_heat_vaporization(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.hv` instead.

        Returns:
            Latent heat of vaporization in J/kg.
        """
        _warn_deprecated("latent_heat_vaporization", "comps.hv")
        return self.comps.hv(T, comp_idx)

    def diffusion_coeff(
        self: Self,
        p: types.Quantity0D,
        T: types.Quantity0D,
        sigma_gas: types.Quantity0D = Sigma_gas,
        epsilonByKB_gas: types.Quantity0D = EpsilonByKB_gas,
        MW_gas: types.Quantity0D = MW_gas,
        correlation: Literal["Tee", "Wilke"] = "Tee",
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.d` instead.

        Returns:
            Diffusion coefficient in m^2/s.
        """
        _warn_deprecated("diffusion_coeff", "comps.d")
        return self.comps.d(p, T, sigma_gas, epsilonByKB_gas, MW_gas, correlation)

    def surface_tension(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.sigma` instead.

        Returns:
            Surface tension in N/m.
        """
        _warn_deprecated("surface_tension", "comps.sigma")
        return self.comps.sigma(T, comp_idx, correlation)

    def thermal_conductivity(
        self: Self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.k` instead.

        Returns:
            Thermal conductivity in W/m/K.
        """
        _warn_deprecated("thermal_conductivity", "comps.k")
        return self.comps.k(T, comp_idx)

    def flash_point(
        self: Self,
        comp_idx: int | None = None,
        correlation: Literal["Alqaheem", "Alibakhshi"] = "Alibakhshi",
    ) -> types.Quantity1D:
        """Deprecated. Use `comps.t_flash` instead.

        Returns:
            Flash point in K.
        """
        _warn_deprecated("flash_point", "comps.t_flash")
        return self.comps.t_flash(comp_idx, correlation)

    # Deprecated composition conversions for backward compatibility
    def mean_molecular_weight(
        self: Self, Yi: types.Quantity1D | None = None
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.mw` instead.

        Returns:
            Mean molecular weight of the mixture in kg/mol.
        """
        _warn_deprecated("mean_molecular_weight", "mix.mw")
        return self.mix.mw(Yi)

    def mass2Y(self: Self, mass: types.Quantity1D) -> types.Quantity1D:
        """Deprecated. Use `mix.mass2y` instead.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        _warn_deprecated("mass2Y", "mix.mass2y")
        return self.mix.mass2y(mass)

    def mass2X(self: Self, mass: types.Quantity1D) -> types.Quantity1D:
        """Deprecated. Use `mix.mass2x` instead.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        _warn_deprecated("mass2X", "mix.mass2x")
        return self.mix.mass2x(mass)

    def X2Y(self: Self, Xi: types.Quantity1D) -> types.Quantity1D:
        """Deprecated. Use `mix.x2y` instead.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        _warn_deprecated("X2Y", "mix.x2y")
        return self.mix.x2y(Xi)

    def Y2X(self: Self, Yi: types.Quantity1D) -> types.Quantity1D:
        """Deprecated. Use `mix.y2x` instead.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        _warn_deprecated("Y2X", "mix.y2x")
        return self.mix.y2x(Yi)

    def Y2V(self: Self, Yi: types.Quantity1D, T: types.Quantity0D) -> types.Quantity1D:
        """Deprecated. Use `mix.y2v` instead.

        Returns:
            Volume fractions of the compounds (shape: num_compounds,).
        """
        _warn_deprecated("Y2V", "mix.y2v")
        return self.mix.y2v(Yi, T)

    # Deprecated mixture methods for backward compatibility
    def mixture_density(
        self: Self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.rho` instead.

        Returns:
            Mixture density in kg/m^3.
        """
        _warn_deprecated("mixture_density", "mix.rho")
        return self.mix.rho(Yi, T)

    def mixture_kinematic_viscosity(
        self: Self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
        mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.nu` instead.

        Returns:
            Mixture kinematic viscosity in m^2/s.
        """
        _warn_deprecated("mixture_kinematic_viscosity", "mix.nu")
        return self.mix.nu(Yi, T, correlation, mixing_rule)

    def mixture_dynamic_viscosity(
        self: Self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Dutt", "Nannoolal"] = "Nannoolal",
        mixing_rule: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.mu` instead.

        Returns:
            Mixture dynamic viscosity in Pa*s.
        """
        _warn_deprecated("mixture_dynamic_viscosity", "mix.mu")
        return self.mix.mu(Yi, T, correlation, mixing_rule)

    def mixture_vapor_pressure(
        self: Self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.pvap` instead.

        Returns:
            Mixture vapor pressure in Pa.
        """
        _warn_deprecated("mixture_vapor_pressure", "mix.pvap")
        return self.mix.pvap(Yi, T, correlation)

    def mixture_vapor_pressure_antoine_coeffs(
        self: Self,
        Yi: types.Quantity1D,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """Deprecated. Use `mix.pvap_antoine_coeffs` instead.

        Returns:
            Coefficients A, B, C, D.
        """
        _warn_deprecated(
            "mixture_vapor_pressure_antoine_coeffs", "mix.pvap_antoine_coeffs"
        )
        return self.mix.pvap_antoine_coeffs(Yi, Tvals, units, correlation)

    def mixture_surface_tension(
        self: Self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.sigma` instead.

        Returns:
            Mixture surface tension in N/m.
        """
        _warn_deprecated("mixture_surface_tension", "mix.sigma")
        return self.mix.sigma(Yi, T, correlation)

    def mixture_thermal_conductivity(
        self: Self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.k` instead.

        Returns:
            Thermal conductivity in W/m/K.
        """
        _warn_deprecated("mixture_thermal_conductivity", "mix.k")
        return self.mix.k(Yi, T)

    def mixture_freeze_point(
        self: Self,
        Yi: types.Quantity1D,
        *,
        alpha: float = 1.0,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.t_freeze` instead.

        Returns:
            Mixture freeze point in K.
        """
        _warn_deprecated("mixture_freeze_point", "mix.t_freeze")
        return self.mix.t_freeze(Yi, alpha=alpha)

    def mixture_flash_point(
        self: Self,
        Yi: types.Quantity1D,
        *,
        correlation: Literal["Alqaheem", "Alibakhshi"] = "Alibakhshi",
        mixing_rule: Literal["linear", "Liaw"] = "Liaw",
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.t_flash` instead.

        Returns:
            Mixture flash point in K.
        """
        _warn_deprecated("mixture_flash_point", "mix.t_flash")
        return self.mix.t_flash(Yi, correlation=correlation, mixing_rule=mixing_rule)

    def mixture_heat_of_combustion(
        self: Self,
        Yi: types.Quantity1D,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.lhv` instead.

        Returns:
            Mixture heat of combustion in J/kg.
        """
        _warn_deprecated("mixture_heat_of_combustion", "mix.lhv")
        return self.mix.lhv(Yi)

    def mixture_yield_sooting_index(
        self: Self,
        Yi: types.Quantity1D,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.ysi` instead.

        Returns:
            Mixture yield sooting index.
        """
        _warn_deprecated("mixture_yield_sooting_index", "mix.ysi")
        return self.mix.ysi(Yi)

    def mixture_smoke_point(
        self: Self,
        Yi: types.Quantity1D,
        correlation: Literal["NJFCP", "McEnally"] = "NJFCP",
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.smoke_point` instead.

        Returns:
            Mixture smoke point in mm.
        """
        _warn_deprecated("mixture_smoke_point", "mix.smoke_point")
        return self.mix.smoke_point(Yi, correlation)

    def mixture_derived_cetane_number(
        self: Self,
        Yi: types.Quantity1D,
        *,
        T_ref: types.Quantity0D = T_stp,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.dcn` instead.

        Returns:
            Mixture derived cetane number.
        """
        _warn_deprecated("mixture_derived_cetane_number", "mix.dcn")
        return self.mix.dcn(Yi, T_ref=T_ref)

    def mixture_aromatics_content(
        self: Self,
        Yi: types.Quantity1D,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.aromatics` instead.

        Returns:
            Mixture aromatics content in volume percent.
        """
        _warn_deprecated("mixture_aromatics_content", "mix.aromatics")
        return self.mix.aromatics(Yi)

    def mixture_naphthalenes_content(
        self: Self,
        Yi: types.Quantity1D,
    ) -> types.Quantity0D:
        """Deprecated. Use `mix.naphthalenes` instead.

        Returns:
            Mixture naphthalenes content in volume percent.
        """
        _warn_deprecated("mixture_naphthalenes_content", "mix.naphthalenes")
        return self.mix.naphthalenes(Yi)
