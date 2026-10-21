"""Reference-compound and GCM decomposition data loading."""

from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame, Series

from ..rdk.mol import smiles_to_inchi

DEFAULT_DATA_DIR = Path(__file__).parent


def _find_custom_ref_path(data_dir: Path, file_name: str) -> Path | None:
    """Locate a custom reference table in a fuel data directory.

    Args:
        data_dir: Directory containing custom fuel data.
        file_name: Reference-table file name.

    Returns:
        Path to the custom reference table, or `None` if `data_dir` has no such
        file or it is the bundled table.
    """
    custom_path = data_dir / file_name
    if not custom_path.is_file():
        return None
    if custom_path.resolve() == (DEFAULT_DATA_DIR / file_name).resolve():
        return None
    return custom_path


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


def load_ref_compounds(data_dir: Path) -> DataFrame:
    """Load bundled reference compounds and apply custom overrides.

    If `data_dir` contains a `refCompounds.csv`, its rows are added to the bundled
    table. Custom rows replace bundled rows when their SMILES strings resolve to
    the same InChI, allowing alternative valid SMILES representations for the same
    compound.

    Args:
        data_dir: Directory containing custom fuel data.

    Returns:
        The bundled reference-compound table with custom additions and overrides.
    """
    default_data = pd.read_csv(DEFAULT_DATA_DIR / "refCompounds.csv")
    custom_path = _find_custom_ref_path(data_dir, "refCompounds.csv")
    if custom_path is None:
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


def load_ref_decomp(data_dir: Path, method: Literal["Gani", "Nannoolal"]) -> DataFrame:
    """Load bundled Gani or Nannoolal decompositions and apply name-based overrides.

    If `data_dir` contains a `ref{method}.csv`, its rows are added to the bundled
    table, replacing bundled rows with the same `Reference Compound`.

    Args:
        data_dir: Directory containing custom fuel data.
        method: Decomposition method, either "Gani" or "Nannoolal".

    Returns:
        The bundled Gani or Nannoolal decomposition table with custom additions and
        overrides.
    """
    file_name = f"ref{method}.csv"
    default_data = pd.read_csv(DEFAULT_DATA_DIR / file_name)
    custom_path = _find_custom_ref_path(data_dir, file_name)
    if custom_path is None:
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
    "load_ref_decomp",
    "smiles_series_to_inchi",
]
