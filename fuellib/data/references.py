"""Reference compounds with known properties for fuel analysis.

Data are stored in two CSV files next to this module:

- ``refCompounds.csv``: compound names, identifiers, and families.
- ``refProperties.csv``: measured property values for those compounds.

Only ``Common_Name`` and ``SMILES`` are required for each compound. Missing
``InChI``, ``Num_C``, and ``Family`` values are derived from the SMILES by
:func:`populate_missing`, which :func:`load_compounds` applies automatically.
Loading never modifies the files; call :func:`update_compounds_csv` to write the
populated values back. Use :func:`validate` to check the two tables for
consistency; the loaders do so automatically.
"""

import csv
from pathlib import Path

import pandas as pd
from rdkit.Chem import Mol

from fuellib.rdk.mol import (
    atom_counts,
    count_aromatic_rings,
    from_smiles,
    has_branch,
    has_double_bond,
    inchi,
)

DATA_DIR = Path(__file__).parent
COMPOUNDS_PATH = DATA_DIR / "refCompounds.csv"
PROPERTIES_PATH = DATA_DIR / "refProperties.csv"

COMPOUND_HEADERS = ["Common_Name", "InChI", "SMILES", "Num_C", "Family"]
PROPERTY_HEADERS = ["Common_Name", "Property", "Units", "Value", "Error", "Source"]

FAMILIES = frozenset({
    "n-alkane",
    "iso-alkane",
    "alkene",
    "monocyclic",
    "dicyclic",
    "tricyclic",
    "alkylbenzene",
    "cycloaromatic",
    "diaromatic",
})
PROPERTIES = frozenset({"DCN", "YSI", "Tc", "Pc", "Vc", "Tm", "Tb"})


def validate(compounds: pd.DataFrame, properties: pd.DataFrame) -> None:
    """Validate the reference compound and property tables.

    All problems found are collected and reported together.

    Args:
        compounds: Table of reference compounds.
        properties: Table of reference property values.

    Raises:
        ValueError: If either table is malformed or the tables are inconsistent.
    """
    errors: list[str] = []

    # Headers must be exact; further checks are meaningless otherwise.
    for name, df, expected in (
        ("compounds", compounds, COMPOUND_HEADERS),
        ("properties", properties, PROPERTY_HEADERS),
    ):
        if list(df.columns) != expected:
            errors.append(f"{name}: columns {list(df.columns)} != {expected}")
    if errors:
        raise ValueError("Invalid reference data:\n  " + "\n  ".join(errors))

    errors += _compound_errors(compounds)
    errors += _property_errors(properties, set(compounds["Common_Name"]))
    if errors:
        raise ValueError("Invalid reference data:\n  " + "\n  ".join(errors))


def _read_csv(path: Path) -> pd.DataFrame:
    # Spreadsheet exports often append empty, unnamed columns and pad cells
    # with whitespace; drop the former and strip the latter.
    df = pd.read_csv(path, skipinitialspace=True)
    extra = [c for c in df.columns if str(c).startswith("Unnamed:")]
    df = df.drop(columns=[c for c in extra if df[c].isna().all()])
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _check_headers(df: pd.DataFrame) -> None:
    if list(df.columns) != COMPOUND_HEADERS:
        raise ValueError(
            f"Invalid reference data:\n  compounds: columns {list(df.columns)} "
            f"!= {COMPOUND_HEADERS}"
        )


def classify_family(mol: Mol) -> str:
    """Classify a hydrocarbon into one of :data:`FAMILIES`.

    Args:
        mol: RDKit Mol object (hydrocarbon).

    Returns:
        The family name.

    Raises:
        ValueError: If `mol` is not a hydrocarbon or fits no family.
    """
    if not set(atom_counts(mol)) <= {"C", "H"}:
        raise ValueError("only hydrocarbons can be classified")
    n_rings = mol.GetRingInfo().NumRings()
    n_aromatic = count_aromatic_rings(mol)
    if n_aromatic >= 2:
        return "diaromatic"
    if n_aromatic == 1:
        return "alkylbenzene" if n_rings == 1 else "cycloaromatic"
    if n_rings == 1:
        return "monocyclic"
    if n_rings == 2:
        return "dicyclic"
    if n_rings >= 3:
        return "tricyclic"
    if has_double_bond(mol):
        return "alkene"
    return "iso-alkane" if has_branch(mol) else "n-alkane"


def populate_missing(compounds: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``InChI``, ``Num_C``, and ``Family`` values from the SMILES.

    Values that are already present are left untouched.

    Args:
        compounds: Table of reference compounds. ``Common_Name`` and ``SMILES``
            must be present in every row.

    Returns:
        A populated copy of `compounds`.

    Raises:
        ValueError: If the columns are wrong, a required value is missing, or a
            SMILES string cannot be parsed or classified.
    """
    _check_headers(compounds)
    df = compounds.copy()
    errors = [
        f"compounds row {i}: '{col}' is required"
        for col in ("Common_Name", "SMILES")
        for i in df.index[_is_missing(df[col])]
    ]
    if errors:
        raise ValueError("Invalid reference data:\n  " + "\n  ".join(errors))

    for col in ("InChI", "Num_C", "Family"):
        df[col] = df[col].astype(object)
    for i in df.index:
        needed = [
            c
            for c in ("InChI", "Num_C", "Family")
            if _is_missing(df.loc[[i], c]).item()
        ]
        if not needed:
            continue
        try:
            mol = from_smiles(str(df.at[i, "SMILES"]))
            if "InChI" in needed:
                df.at[i, "InChI"] = inchi(mol)
            if "Num_C" in needed:
                df.at[i, "Num_C"] = atom_counts(mol).get("C", 0)
            if "Family" in needed:
                df.at[i, "Family"] = classify_family(mol)
        except ValueError as exc:
            errors.append(f"compounds row {i}: {exc}")
    if errors:
        raise ValueError("Invalid reference data:\n  " + "\n  ".join(errors))

    num_c = pd.to_numeric(df["Num_C"], errors="coerce")
    if num_c.notna().all() and (num_c % 1 == 0).all():
        df["Num_C"] = num_c.astype(int)
    return df


def _compound_errors(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col in ("Common_Name", "InChI", "SMILES", "Family"):
        empty = _is_missing(df[col])
        errors += [f"compounds row {i}: '{col}' is empty" for i in df.index[empty]]

    for col in ("Common_Name", "InChI", "SMILES"):
        dup = df[col].notna() & df[col].duplicated(keep=False)
        errors += [
            f"compounds row {i}: duplicate {col} '{df.at[i, col]}'"
            for i in df.index[dup]
        ]

    bad_family = df["Family"].notna() & ~df["Family"].isin(FAMILIES)
    errors += [
        f"compounds row {i}: unknown Family '{df.at[i, 'Family']}' "
        f"(allowed: {sorted(FAMILIES)})"
        for i in df.index[bad_family]
    ]

    num_c = pd.to_numeric(df["Num_C"], errors="coerce")
    bad_num_c = num_c.isna() | (num_c < 1) | (num_c % 1 != 0)
    errors += [
        f"compounds row {i}: Num_C '{df.at[i, 'Num_C']}' is not a positive integer"
        for i in df.index[bad_num_c]
    ]

    bad_inchi = df["InChI"].notna() & ~df["InChI"].astype(str).str.startswith("InChI=")
    errors += [
        f"compounds row {i}: InChI must start with 'InChI='"
        for i in df.index[bad_inchi]
    ]
    errors += _consistency_errors(df)
    return errors


def _consistency_errors(df: pd.DataFrame) -> list[str]:
    # Check InChI and Num_C agree with the SMILES.
    errors: list[str] = []
    for i in df.index[~_is_missing(df["SMILES"])]:
        try:
            mol = from_smiles(str(df.at[i, "SMILES"]))
        except ValueError as exc:
            errors.append(f"compounds row {i}: {exc}")
            continue
        if not _is_missing(df.loc[[i], "InChI"]).item() and df.at[i, "InChI"] != inchi(
            mol
        ):
            errors.append(f"compounds row {i}: InChI does not match SMILES")
        num_c = pd.to_numeric(df.at[i, "Num_C"], errors="coerce")
        if pd.notna(num_c) and num_c != atom_counts(mol).get("C", 0):
            errors.append(f"compounds row {i}: Num_C does not match SMILES")
    return errors


def _property_errors(df: pd.DataFrame, known_compounds: set[str]) -> list[str]:
    errors: list[str] = []
    for col in ("Common_Name", "Property", "Source"):
        errors += [
            f"properties row {i}: '{col}' is empty" for i in df.index[df[col].isna()]
        ]

    unknown = df["Common_Name"].notna() & ~df["Common_Name"].isin(known_compounds)
    errors += [
        f"properties row {i}: Common_Name '{df.at[i, 'Common_Name']}' "
        "is not in the compounds table"
        for i in df.index[unknown]
    ]

    bad_property = df["Property"].notna() & ~df["Property"].isin(PROPERTIES)
    errors += [
        f"properties row {i}: unknown Property '{df.at[i, 'Property']}' "
        f"(allowed: {sorted(PROPERTIES)})"
        for i in df.index[bad_property]
    ]

    dup = df.duplicated(["Common_Name", "Property"], keep=False)
    errors += [
        f"properties row {i}: duplicate entry for "
        f"({df.at[i, 'Common_Name']}, {df.at[i, 'Property']})"
        for i in df.index[dup]
    ]

    value = pd.to_numeric(df["Value"], errors="coerce")
    errors += [
        f"properties row {i}: Value '{df.at[i, 'Value']}' is not numeric"
        for i in df.index[value.isna()]
    ]

    # Error is optional but, if given, must be a non-negative number.
    error = pd.to_numeric(df["Error"], errors="coerce")
    bad_error = df["Error"].notna() & (error.isna() | (error < 0))
    errors += [
        f"properties row {i}: Error '{df.at[i, 'Error']}' is not a non-negative number"
        for i in df.index[bad_error]
    ]
    return errors


def load_compounds() -> pd.DataFrame:
    """Load and validate the reference compounds table.

    Missing ``InChI``, ``Num_C``, and ``Family`` values are populated from the
    SMILES (in memory only).

    Returns:
        DataFrame with columns :data:`COMPOUND_HEADERS`.
    """
    compounds = populate_missing(_read_csv(COMPOUNDS_PATH))
    validate(compounds, _read_csv(PROPERTIES_PATH))
    return compounds


def update_compounds_csv() -> pd.DataFrame:
    """Populate missing values and write them back to the compounds CSV.

    Returns:
        The populated and validated compounds table.
    """
    compounds = load_compounds()
    # InChI strings contain commas, so quote every non-numeric field.
    compounds.to_csv(COMPOUNDS_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return compounds


def load_properties() -> pd.DataFrame:
    """Load and validate the reference properties table.

    Returns:
        DataFrame with columns :data:`PROPERTY_HEADERS`.
    """
    properties = _read_csv(PROPERTIES_PATH)
    validate(populate_missing(_read_csv(COMPOUNDS_PATH)), properties)
    return properties


def properties_by_smiles(smiles: str) -> pd.DataFrame | None:
    """Lookup reference properties by their SMILES string.

    Converts SMILES to InChI internally to match with the reference properties.

    Args:
        smiles: The SMILES string to search for.

    Returns:
        DataFrame containing the matching reference properties, or ``None`` if no match.
    """
    mol = from_smiles(smiles)
    inchi_str = inchi(mol)
    compounds = load_compounds()
    properties = load_properties()
    common_names = compounds.loc[compounds["InChI"] == inchi_str, "Common_Name"]
    if common_names.empty:
        return None
    return properties[properties["Common_Name"].isin(common_names)]


if __name__ == "__main__":
    update_compounds_csv()
