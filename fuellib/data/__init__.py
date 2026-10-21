"""Bundled fuel data for the FuelLib library."""

from ._reference_data import (
    DEFAULT_DATA_DIR,
    load_ref_compounds,
    load_ref_decomp,
    smiles_series_to_inchi,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "load_ref_compounds",
    "load_ref_decomp",
    "smiles_series_to_inchi",
]
