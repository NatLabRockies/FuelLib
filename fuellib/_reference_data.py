"""Reference-compound and GCM decomposition data loading."""

from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

from .rdk.mol import smiles_to_inchi

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _resolve_ref_path(path: str | Path | None, data_dir: Path, file_name: str) -> Path:
    """Resolve a custom reference-table path, falling back to bundled data.

    Args:
        path: Explicit custom reference-table path, if provided.
        data_dir: Directory containing custom fuel data.
        file_name: Reference-table file name.

    Returns:
        Resolved path to the custom or bundled reference table.

    Raises:
        FileNotFoundError: If neither the custom nor bundled table exists.
    """
    ref_path = Path(path) if path is not None else data_dir / file_name
    if not ref_path.exists():
        print(f"Reference file not found at {ref_path}, using default location.")
        ref_path = DEFAULT_DATA_DIR / file_name
        if not ref_path.exists():
            msg = f"Default reference file not found: {ref_path}"
            raise FileNotFoundError(msg)
    return ref_path


def smiles_series_to_inchi(smiles: Series) -> Series:
    """Convert non-empty SMILES strings in a Series to InChI identifiers.

    Args:
        smiles: SMILES strings to convert.

    Returns:
        InChI identifiers, with missing values for blank SMILES strings.
    """
    return (
        smiles
        .astype("string")
        .str.strip()
        .map(lambda s: smiles_to_inchi(s) if not pd.isna(s) and s else pd.NA)
    )


def load_ref_compounds(path: str | Path | None, data_dir: Path) -> DataFrame:
    """Load bundled reference compounds and apply custom overrides.

    Custom rows replace bundled rows when their SMILES strings resolve to the
    same InChI, allowing alternative valid SMILES representations for the same
    compound.

    Args:
        path: Explicit custom reference-compound file, if provided.
        data_dir: Directory containing custom fuel data.

    Returns:
        The bundled reference-compound table with custom additions and overrides.
    """
    default_path = DEFAULT_DATA_DIR / "refCompounds.csv"
    custom_path = _resolve_ref_path(path, data_dir, "refCompounds.csv")
    default_data = pd.read_csv(default_path)

    if custom_path.resolve() == default_path.resolve():
        return default_data

    custom_data = pd.read_csv(custom_path)
    if "SMILES" not in custom_data.columns:
        return pd.concat([default_data, custom_data], ignore_index=True, sort=False)

    custom_inchi = smiles_series_to_inchi(custom_data["SMILES"])
    custom_with_inchi = custom_data.assign(_InChI=custom_inchi)
    custom_with_inchi = pd.concat([
        custom_with_inchi[custom_with_inchi["_InChI"].isna()],
        custom_with_inchi[custom_with_inchi["_InChI"].notna()].drop_duplicates(
            "_InChI", keep="last"
        ),
    ])
    custom_inchis = custom_with_inchi["_InChI"].dropna()
    default_inchi = smiles_series_to_inchi(default_data["SMILES"])
    bundled_rows = default_data.loc[~default_inchi.isin(custom_inchis)]

    return pd.concat(
        [bundled_rows, custom_with_inchi.drop(columns="_InChI")],
        ignore_index=True,
        sort=False,
    )


def load_ref_gani(path: str | Path | None, data_dir: Path) -> DataFrame:
    """Load bundled Gani decompositions and apply custom name-based overrides.

    Args:
        path: Explicit custom Gani decomposition file, if provided.
        data_dir: Directory containing custom fuel data.

    Returns:
        The bundled Gani decomposition table with custom additions and overrides.
    """
    default_path = DEFAULT_DATA_DIR / "refGani.csv"
    custom_path = _resolve_ref_path(path, data_dir, "refGani.csv")
    default_data = pd.read_csv(default_path)

    if custom_path.resolve() == default_path.resolve():
        return default_data

    custom_data = pd.read_csv(custom_path)
    if "Reference Compound" not in custom_data.columns:
        return pd.concat([default_data, custom_data], ignore_index=True, sort=False)

    bundled_rows = default_data.loc[
        ~default_data["Reference Compound"].isin(custom_data["Reference Compound"])
    ]
    return pd.concat([bundled_rows, custom_data], ignore_index=True, sort=False)


__all__ = [
    "DEFAULT_DATA_DIR",
    "load_ref_compounds",
    "load_ref_gani",
    "smiles_series_to_inchi",
]
