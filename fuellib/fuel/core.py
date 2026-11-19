"""Core Fuel class."""

from pathlib import Path
from typing import Final, Self

import numpy as np
import pandas as pd
import yaml

from ..data._reference_data import (
    DEFAULT_DATA_DIR,
    load_ref_compounds,
    load_ref_decomp,
    smiles_series_to_inchi,
)
from ..utils import Units, types
from .components import Components
from .mixture import Mixture


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
    ref_idx = pd.Series(pd.NA, index=gc_data.index, dtype="object")

    def _normalize_names(names: pd.Series) -> pd.Series:
        return names.astype("string").str.strip().str.lower()

    if "Reference Compound" in gc_data.columns:
        ref_names = _normalize_names(ref_data["Reference Compound"]).dropna()
        name_lookup = dict(zip(ref_names, ref_names.index))
        ref_idx = _normalize_names(gc_data["Reference Compound"]).map(name_lookup)

    unmatched = ref_idx.isna()
    if unmatched.any() and "SMILES" in gc_data.columns and "SMILES" in ref_data.columns:
        ref_inchi = smiles_series_to_inchi(ref_data["SMILES"]).dropna()
        inchi_lookup = dict(zip(ref_inchi, ref_inchi.index))
        ref_idx[unmatched] = smiles_series_to_inchi(
            gc_data.loc[unmatched, "SMILES"]
        ).map(inchi_lookup)

    unmatched = ref_idx.isna()
    if unmatched.any():
        id_cols = [c for c in ("Reference Compound", "SMILES") if c in gc_data.columns]
        msg = (
            f"GCxGC compounds not found in reference data '{ref_name}':\n"
            f"{gc_data.loc[unmatched, id_cols].to_string()}"
        )
        raise ValueError(msg)

    return ref_data.loc[ref_idx.astype(int)].reset_index(drop=True)


class Fuel:
    """Class for handling calculations of thermodynamic and mixture properties."""

    def __init__(
        self,
        name: str,
        data_dir: str | Path | None = None,
        *,
        use_ref_props: bool = True,
    ) -> None:
        """Initialize a Fuel instance.

        Args:
            name: Name of the fuel.
            data_dir: Directory containing fuel data.
                Defaults to `fuellib/data`.
            use_ref_props: Whether to use reference compound properties for
                calculations. Defaults to `True`.

        Raises:
            FileNotFoundError: If the fuel data directory does not exist.
            ValueError: If the GCxGC data file is missing required columns.
        """
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

        self._ref_comps: pd.DataFrame = load_ref_compounds(self.data_dir)
        """Reference compounds and their properties (bundled plus custom)."""
        self.components_data: pd.DataFrame = _match_gc_to_ref(
            self.gc_data, self._ref_comps, "reference compounds"
        )
        """Matched components data with reference compounds."""

        self._ref_gani: pd.DataFrame = load_ref_decomp(self.data_dir, "Gani")
        """Reference Gani group decompositions (bundled plus custom)."""
        self._gani_decomp: pd.DataFrame = (
            _match_gc_to_ref(self.components_data, self._ref_gani, "ref_gani")
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """Gani group decompositions for each matched compound."""
        self._ref_nannoolal: pd.DataFrame = load_ref_decomp(self.data_dir, "Nannoolal")
        """Reference Nannoolal group decompositions (bundled plus custom)."""
        self._nannoolal_decomp: pd.DataFrame = (
            _match_gc_to_ref(self.components_data, self._ref_nannoolal, "ref_nannoolal")
            .drop("Reference Compound", axis=1)
            .fillna(0)
        )
        """Nannoolal group decompositions for each matched compound."""
        self._ref_unifac: pd.DataFrame = load_ref_decomp(self.data_dir, "UNIFAC")
        """Reference UNIFAC group decompositions (bundled plus custom)."""
        self._unifac_decomp: pd.DataFrame = (
            _match_gc_to_ref(self.components_data, self._ref_unifac, "ref_unifac")
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

        y0 = self.gc_data["Weight %"].to_numpy(float)
        self.y0: types.Quantity1D = (
            Units.Quantity(y0 / sum(y0), "")
            if sum(y0) > 0
            else Units.Quantity(np.zeros_like(y0), "")
        )
        """Normalized weight fractions from the GCxGC data."""

        self.x0: types.Quantity1D = self.comps.y2x(self.y0)
        """Normalized mole fractions from the GCxGC data."""

    @property
    def num_compounds(self: Self) -> int:
        """Number of components in the fuel."""
        return len(self.components_data)
