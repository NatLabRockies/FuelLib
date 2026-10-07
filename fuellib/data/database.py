"""FuelLib database implementation."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from rdkit.Chem import Mol

from fuellib.gcm import boehm_gcm, gani, gani_gcm
from fuellib.rdk import mol
from fuellib.utils import FLLogger, Units

if TYPE_CHECKING:
    from fuellib.fuel import Fuel

DEFAULT_DIR = Path(__file__).resolve().parent


# Non-property column names in the database DataFrames (`Fuel.data`, `Fuel.gc_data`).
FAMILY_COL = "Family"
NUM_C_COL = "Num_C"
COMMON_NAME_COL = "Common_Name"
SMILES_COL = "SMILES"
INCHI_COL = "InChI"
Y_COL = "Y"
WEIGHT_PERCENT_COL = "Weight %"
PELEPHYSICS_KEY_COL = "PelePhysics_Key"


class Property(StrEnum):
    """Property column names in the database DataFrames (e.g. `Fuel.data`).

    Members are strings, so they index DataFrames directly, e.g.
    ``fuel.data[Property.TC]``. Companion columns are available via `units`,
    `err`, and `source`, e.g. ``fuel.data[Property.TC.units]``.
    """

    # Properties stored in referenceCompounds/compounds.csv
    TC = "Tc"
    PC = "Pc"
    VC = "Vc"
    TB = "Tb"
    TM = "Tm"
    DH_F_STP = "dH_f_stp"
    DH_V_STP = "dH_v_stp"
    ACENTRIC = "acentric"
    VM_STP = "Vm_stp"
    YSI = "YSI"
    DCN = "DCN"

    # Software-only properties (never written to CSV)
    GF = "Gf"
    CP_STP = "Cp_stp"
    CP_B = "Cp_B"
    CP_C = "Cp_C"
    RD_A = "rd_A"
    RD_B = "rd_B"
    RD_D = "rd_D"
    ALIBAKHSHI_PHI = "alibakhshi_phi"
    DS_FUS = "dS_fus"
    MW = "MW"

    @property
    def units(self) -> str:
        """Name of the units column for this property."""
        return f"{self.value}_units"

    @property
    def err(self) -> str:
        """Name of the uncertainty column for this property."""
        return f"{self.value}_err"

    @property
    def source(self) -> str:
        """Name of the source column for this property."""
        return f"{self.value}_source"


def _names(*props: Property) -> tuple[str, ...]:
    return tuple(prop.value for prop in props)


ID_COLUMNS = (FAMILY_COL, NUM_C_COL, COMMON_NAME_COL, SMILES_COL, INCHI_COL)
REQUIRED_COLUMNS = (COMMON_NAME_COL, SMILES_COL)
PROPERTIES = _names(
    Property.TC,
    Property.PC,
    Property.VC,
    Property.TB,
    Property.TM,
    Property.DH_F_STP,
    Property.DH_V_STP,
    Property.ACENTRIC,
    Property.VM_STP,
    Property.YSI,
    Property.DCN,
)
SUFFIXES = ("", "_units", "_err", "_source")
REF_COMPOUND_COLUMNS: tuple[str, ...] = ID_COLUMNS + tuple(
    f"{prop}{suffix}" for prop in PROPERTIES for suffix in SUFFIXES
)
TEXT_COLUMNS: tuple[str, ...] = (
    FAMILY_COL,
    COMMON_NAME_COL,
    SMILES_COL,
    INCHI_COL,
) + tuple(f"{prop}{suffix}" for prop in PROPERTIES for suffix in ("_units", "_source"))

#: Reference-compound property column -> Gani GCM property function name.
GANI_PROPERTY_MAP = {
    Property.TC.value: "Tc",
    Property.PC.value: "Pc",
    Property.VC.value: "Vc",
    Property.TB.value: "Tb",
    Property.TM.value: "Tm",
    Property.DH_F_STP.value: "Hf",
    Property.DH_V_STP.value: "Hv_stp",
    Property.ACENTRIC.value: "omega",
    Property.VM_STP.value: "Vm_stp",
}
GANI_SOURCE = "Constantinou-Gani GCM (FuelLib)"

#: Software-only properties: added to the loaded DataFrame, never written to CSV.
#: Column names match the Gani/Boehm GCM property function names.
GANI_DERIVED_PROPERTIES = _names(
    Property.GF,
    Property.CP_STP,
    Property.CP_B,
    Property.CP_C,
    Property.RD_A,
    Property.RD_B,
    Property.RD_D,
    Property.ALIBAKHSHI_PHI,
)
BOEHM_DERIVED_PROPERTIES = _names(Property.DS_FUS)
BOEHM_SOURCE = "Boehm GCM (FuelLib)"
RDKIT_SOURCE = "RDKit (FuelLib)"

SUPPORTED_FAMILIES = (
    "n-alkane",
    "iso-alkane",
    "alkene",
    "monocyclic",
    "dicyclic",
    "tricyclic",
    "alkylbenzene",
    "cycloaromatic",
    "diaromatic",
)


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------
def read_reference_compounds(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a referenceCompounds/compounds.csv file.

    Args:
        path: Path to the compounds CSV file.

    Returns:
        The compounds conformed to `REF_COMPOUND_COLUMNS`, and a DataFrame of any
        unrecognized columns (kept so they survive write-back).

    Raises:
        ValueError: If required columns or values are missing.
    """
    raw = pd.read_csv(path, header=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        msg = f"Required columns {missing} are missing from {path}."
        raise ValueError(msg)
    blank = raw[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if blank.any():
        msg = (
            f"Rows {[i + 2 for i in raw.index[blank]]} in {path} are missing a "
            f"value for one of {list(REQUIRED_COLUMNS)}."
        )
        raise ValueError(msg)
    unknown = [c for c in raw.columns if c not in REF_COMPOUND_COLUMNS]
    if unknown:
        FLLogger.warning(f"Ignoring unrecognized columns in {path}: {unknown}")

    df = raw.reindex(columns=list(REF_COMPOUND_COLUMNS))
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(object)
    df[NUM_C_COL] = df[NUM_C_COL].astype("Int64")
    return df, raw[unknown]


def write_reference_compounds(
    path: Path, compounds: pd.DataFrame, extras: pd.DataFrame
) -> None:
    """Write reference compounds back to disk, preserving unrecognized columns.

    Args:
        path: Path to the compounds CSV file.
        compounds: The compounds DataFrame.
        extras: Unrecognized columns originally present in the file.
    """
    pd.concat([compounds, extras], axis=1).to_csv(path, index=False)
    FLLogger.info(f"Wrote auto-populated reference data to {path}")


def read_gani(path: Path) -> pd.DataFrame:
    """Read a referenceCompounds/gani.csv group decomposition file.

    Args:
        path: Path to the Gani decomposition CSV file.

    Returns:
        Group counts indexed by InChI, with columns matching the Gani GCM table.
        Groups absent from the file are filled with 0.

    Raises:
        ValueError: If the first column is not "InChI".
    """
    df = pd.read_csv(path, header=0, index_col=0)
    if df.index.name != "InChI":
        msg = f"The first column of {path} must be 'InChI'."
        raise ValueError(msg)
    unknown = [c for c in df.columns if c not in gani.TABLE.columns]
    if unknown:
        FLLogger.warning(f"Ignoring unrecognized Gani groups in {path}: {unknown}")
    df = df.reindex(columns=gani.TABLE.columns, fill_value=0)
    # Header-only files (e.g. from `write_template`) are read as object columns.
    return df.astype(int) if df.empty else df


def write_template(path: Path | str, fuel_name: str = "template") -> Path:
    """Write a blank user database directory with CSV headers filled in.

    Creates::

        <path>/referenceCompounds/compounds.csv
        <path>/referenceCompounds/gani.csv
        <path>/gcData/<fuel_name>.csv
        <path>/propertiesData/<fuel_name>.csv

    Args:
        path: Directory to write the template into (created if needed).
        fuel_name: File stem for the gcData and propertiesData templates.

    Returns:
        The template directory.

    Raises:
        FileExistsError: If any of the template files already exist.
    """
    root = Path(path)
    templates = {
        root / "referenceCompounds" / "compounds.csv": list(REF_COMPOUND_COLUMNS),
        root / "referenceCompounds" / "gani.csv": [INCHI_COL, *gani.TABLE.columns],
        root / "gcData" / f"{fuel_name}.csv": [
            COMMON_NAME_COL,
            SMILES_COL,
            WEIGHT_PERCENT_COL,
        ],
        root / "propertiesData" / f"{fuel_name}.csv": [
            "Temp",
            "Temp_Units",
            "Property",
            "Property_Units",
            "Property_Value",
        ],
    }
    existing = [str(p) for p in templates if p.exists()]
    if existing:
        msg = f"Refusing to overwrite existing files: {existing}"
        raise FileExistsError(msg)
    for file, columns in templates.items():
        file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(file, index=False)
    FLLogger.info(f"Wrote blank database template to {root}")
    return root


# -----------------------------------------------------------------------------
# Auto-population
# -----------------------------------------------------------------------------
def classify_family(m: Mol) -> str:
    """Classify a hydrocarbon into one of `SUPPORTED_FAMILIES`.

    Families are determined from the structure:

    =============== ==========================================
    Family          Description
    =============== ==========================================
    n-alkane        Straight-chain alkanes
    iso-alkane      Branched-chain alkanes
    alkene          Unsaturated hydrocarbons with double bonds
    monocyclic      Single-ring hydrocarbons
    dicyclic        Fused two-ring hydrocarbons
    tricyclic       Fused three-ring hydrocarbons
    alkylbenzene    Alkyl-substituted benzene derivatives
    cycloaromatic   Fused cyclic-aromatic hydrocarbons
    diaromatic      Fused two-ring aromatic hydrocarbons
    =============== ==========================================

    Args:
        m: RDKit Mol object.

    Returns:
        The family name.

    Raises:
        ValueError: If the family cannot be identified.
    """
    aromatic = mol.has_aromatic(m)
    fused = mol.has_fused_rings(m)
    fused_two = mol.has_fused_rings(m, number_of_rings=2)
    fused_three = mol.has_fused_rings(m, number_of_rings=3)
    n_aromatic_rings = mol.count_aromatic_rings(m)

    if fused_two and n_aromatic_rings >= 2:
        return "diaromatic"
    if fused_two and n_aromatic_rings == 1:
        return "cycloaromatic"
    if aromatic and not fused:
        return "alkylbenzene"
    if fused_three and not aromatic:
        return "tricyclic"
    if fused_two and not aromatic:
        return "dicyclic"
    if fused:
        msg = f"Cannot identify family for molecule: {mol.smiles(m)}"
        raise ValueError(msg)
    if mol.has_ring(m):
        return "monocyclic"
    if mol.has_double_bond(m):
        return "alkene"
    if mol.has_branch(m):
        return "iso-alkane"
    return "n-alkane"


def autofill_identity(compounds: pd.DataFrame, source: Path) -> bool:
    """Populate missing Family, Num_C, and InChI values from SMILES (in place).

    Args:
        compounds: The compounds DataFrame.
        source: File the compounds were read from, used in error messages.

    Returns:
        Whether any values were populated.

    Raises:
        ValueError: If a Family is not one of `SUPPORTED_FAMILIES`.
    """
    changed = False
    identity = [FAMILY_COL, NUM_C_COL, INCHI_COL]
    for idx, row in compounds.iterrows():
        if not row[identity].isna().any():
            continue
        m = mol.from_smiles(row[SMILES_COL].strip())
        if pd.isna(row[INCHI_COL]):
            compounds.at[idx, INCHI_COL] = mol.inchi(m)
        if pd.isna(row[NUM_C_COL]):
            compounds.at[idx, NUM_C_COL] = mol.atom_counts(m).get("C", 0)
        if pd.isna(row[FAMILY_COL]):
            compounds.at[idx, FAMILY_COL] = classify_family(m)
        changed = True

    invalid = sorted(set(compounds[FAMILY_COL]) - set(SUPPORTED_FAMILIES))
    if invalid:
        msg = (
            f"Unsupported families {invalid} in {source}. "
            f"Supported families: {list(SUPPORTED_FAMILIES)}"
        )
        raise ValueError(msg)
    return changed


class _GaniDecomposition:
    """Minimal stand-in for `Fuel` exposing the interface Gani GCM needs."""

    def __init__(self, decomp: pd.DataFrame) -> None:
        self._decomp = decomp

    def gani_decomp(self) -> pd.DataFrame:
        return self._decomp


def autofill_gcm(
    compounds: pd.DataFrame, gani_table: pd.DataFrame | None, source: Path
) -> bool:
    """Populate missing GCM-predictable properties using the Gani method (in place).

    Compounds without a Gani decomposition are skipped with a warning; their
    missing properties remain NaN.

    Args:
        compounds: The compounds DataFrame (InChI must already be populated).
        gani_table: Gani group counts indexed by InChI, or None if unavailable.
        source: File the compounds were read from, used in warning messages.

    Returns:
        Whether any values were populated.
    """
    props = list(GANI_PROPERTY_MAP)
    missing = compounds[props].isna()
    needs_gcm = missing.any(axis=1)
    if not needs_gcm.any():
        return False

    inchis = compounds.loc[needs_gcm, INCHI_COL]
    absent = (
        ~inchis.isin(gani_table.index)
        if gani_table is not None
        else pd.Series(True, index=inchis.index)
    )
    if absent.any():
        names = compounds.loc[inchis.index[absent], COMMON_NAME_COL].to_list()
        FLLogger.warning(
            f"No Gani group decomposition for {names} in {source}; their missing "
            "properties are left NaN. Add them to referenceCompounds/gani.csv."
        )
        inchis = inchis[~absent]
    if gani_table is None or inchis.empty:
        return False

    # GCM properties only require `gani_decomp()`, not a full Fuel.
    decomp = cast("Fuel", _GaniDecomposition(gani_table.loc[inchis.to_list()]))
    for col, fn in GANI_PROPERTY_MAP.items():
        mask = missing.loc[inchis.index, col].to_numpy()
        if not mask.any():
            continue
        prediction = gani_gcm.predict(fn, decomp)
        rows = inchis.index[mask]
        compounds.loc[rows, col] = np.asarray(prediction.magnitude)[mask]
        compounds.loc[rows, f"{col}_units"] = str(prediction.units)
        compounds.loc[rows, f"{col}_err"] = np.nan
        compounds.loc[rows, f"{col}_source"] = GANI_SOURCE
    return True


class _BoehmInput:
    """Minimal stand-in for `Fuel` exposing the interface Boehm GCM needs."""

    def __init__(self, compounds: pd.DataFrame) -> None:
        self.smiles = [s.strip() for s in compounds[SMILES_COL]]
        self.rdkit_mols = [mol.from_smiles(s) for s in self.smiles]
        self.families = compounds[FAMILY_COL].to_list()
        self.nC = compounds[NUM_C_COL].astype(int).to_list()
        self.num_compounds = len(self.smiles)


def add_derived_properties(
    compounds: pd.DataFrame, gani_table: pd.DataFrame | None
) -> pd.DataFrame:
    """Add software-only Gani and Boehm properties (never written to CSV).

    Compounds without a Gani decomposition get NaN for the Gani-derived
    properties and a warning is logged.

    Args:
        compounds: The merged compounds DataFrame.
        gani_table: Gani group counts indexed by InChI, or None if unavailable.

    Returns:
        A copy of `compounds` with value, units, err, and source columns for each
        property in `GANI_DERIVED_PROPERTIES` and `BOEHM_DERIVED_PROPERTIES`, and
        for the RDKit molecular weight ("MW").
    """
    inchis = compounds[INCHI_COL]
    has_decomp = (
        inchis.isin(gani_table.index).to_numpy()
        if gani_table is not None
        else np.zeros(len(compounds), dtype=bool)
    )
    if not has_decomp.all():
        FLLogger.warning(
            "No Gani group decomposition for "
            f"{compounds.loc[~has_decomp, COMMON_NAME_COL].to_list()}; "
            f"{list(GANI_DERIVED_PROPERTIES)} are NaN for these compounds."
        )
    decomp_table = (
        gani_table.loc[inchis[has_decomp].to_list()]
        if gani_table is not None and has_decomp.any()
        else gani.TABLE.iloc[0:0]
    )
    # GCM properties only require the attributes provided by the stand-ins.
    decomp = cast("Fuel", _GaniDecomposition(decomp_table))

    new_columns: dict[str, object] = {}
    for prop in GANI_DERIVED_PROPERTIES:
        prediction = gani_gcm.predict(prop, decomp)
        values = np.full(len(compounds), np.nan)
        values[has_decomp] = np.asarray(prediction.magnitude)
        new_columns[prop] = values
        new_columns[f"{prop}_units"] = str(prediction.units)
        new_columns[f"{prop}_err"] = np.nan
        new_columns[f"{prop}_source"] = [
            GANI_SOURCE if found else np.nan for found in has_decomp
        ]

    rdkit_input = _BoehmInput(compounds)
    boehm_input = cast("Fuel", rdkit_input)
    for prop in BOEHM_DERIVED_PROPERTIES:
        prediction = boehm_gcm.predict(prop, boehm_input)
        new_columns[prop] = np.asarray(prediction.magnitude)
        new_columns[f"{prop}_units"] = str(prediction.units)
        new_columns[f"{prop}_err"] = np.nan
        new_columns[f"{prop}_source"] = BOEHM_SOURCE

    mw = Units.Quantity(
        [mol.molecular_weight(m) for m in rdkit_input.rdkit_mols], "g/mol"
    )
    new_columns[Property.MW.value] = np.asarray(mw.magnitude)
    new_columns[Property.MW.units] = str(mw.units)
    new_columns[Property.MW.err] = np.nan
    new_columns[Property.MW.source] = RDKIT_SOURCE

    return pd.concat(
        [compounds, pd.DataFrame(new_columns, index=compounds.index)], axis=1
    )


# -----------------------------------------------------------------------------
# Database assembly
# -----------------------------------------------------------------------------
@dataclass
class _ReferenceSource:
    """Reference data loaded from a single database directory."""

    compounds_file: Path
    compounds: pd.DataFrame
    extras: pd.DataFrame
    gani: pd.DataFrame | None


def _load_source(data_dir: Path) -> _ReferenceSource | None:
    """Load the referenceCompounds folder of a database directory.

    Args:
        data_dir: Database root directory.

    Returns:
        The loaded reference data, or None if compounds.csv is absent.
    """
    ref_dir = data_dir / "referenceCompounds"
    compounds_file = ref_dir / "compounds.csv"
    gani_file = ref_dir / "gani.csv"
    if not compounds_file.is_file():
        if gani_file.is_file():
            FLLogger.warning(f"Ignoring {gani_file}: no compounds.csv beside it.")
        return None
    compounds, extras = read_reference_compounds(compounds_file)
    gani_table = read_gani(gani_file) if gani_file.is_file() else None
    return _ReferenceSource(compounds_file, compounds, extras, gani_table)


def _merge_gani(
    default: pd.DataFrame | None, user: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Merge Gani tables, with user rows replacing default rows of the same InChI.

    Args:
        default: Default Gani table, if any.
        user: User Gani table, if any.

    Returns:
        The merged Gani table, or None if neither table exists.
    """
    if default is None or user is None:
        return user if default is None else default
    return pd.concat([default[~default.index.isin(user.index)], user])


def _normalize_names(names: pd.Series) -> pd.Series:
    return names.astype(str).str.strip().str.lower()


def _merge_compounds(default: pd.DataFrame, user: pd.DataFrame) -> pd.DataFrame:
    """Merge compounds, with user rows replacing default rows sharing InChI/name.

    Args:
        default: Default compounds DataFrame.
        user: User compounds DataFrame.

    Returns:
        The merged compounds DataFrame.
    """
    names = COMMON_NAME_COL
    overridden = default[INCHI_COL].isin(user[INCHI_COL]) | _normalize_names(
        default[names]
    ).isin(_normalize_names(user[names]))
    if overridden.any():
        FLLogger.info(
            "User reference compounds override defaults: "
            f"{default.loc[overridden, names].to_list()}"
        )
    return pd.concat([default[~overridden], user], ignore_index=True)


def load_reference_database(userDataDir: Path | None = None) -> pd.DataFrame:
    """Load, auto-populate, write back, and merge default and user reference data.

    Default rows are only populated from default data, so user inputs never
    modify the default database. Software-only derived properties are added to
    the merged result after write-back.

    Args:
        userDataDir: Optional user database directory.

    Returns:
        The merged reference compounds DataFrame.

    Raises:
        FileNotFoundError: If the default compounds.csv is missing.
    """
    default = _load_source(DEFAULT_DIR)
    if default is None:
        msg = f"Default reference compounds are missing from {DEFAULT_DIR}."
        raise FileNotFoundError(msg)
    user = _load_source(userDataDir) if userDataDir else None

    merged_gani = _merge_gani(default.gani, user.gani if user else None)
    sources = [(default, default.gani)]
    if user is not None:
        sources.append((user, merged_gani))

    for src, gani_table in sources:
        changed = autofill_identity(src.compounds, src.compounds_file)
        changed |= autofill_gcm(src.compounds, gani_table, src.compounds_file)
        if changed:
            write_reference_compounds(src.compounds_file, src.compounds, src.extras)

    merged = (
        default.compounds
        if user is None
        else _merge_compounds(default.compounds, user.compounds)
    )
    return add_derived_properties(merged, merged_gani)


def load_gani_database(userDataDir: Path | None = None) -> pd.DataFrame | None:
    """Load and merge the default and user Gani group decompositions.

    User rows replace default rows with the same InChI. As in
    `load_reference_database`, a user gani.csv is only used alongside a user
    compounds.csv.

    Args:
        userDataDir: Optional user database directory.

    Returns:
        Group counts indexed by InChI, or None if no gani.csv exists.
    """
    tables: list[pd.DataFrame | None] = []
    for root in (DEFAULT_DIR, userDataDir):
        ref_dir = root / "referenceCompounds" if root is not None else None
        if (
            ref_dir is not None
            and (ref_dir / "compounds.csv").is_file()
            and (ref_dir / "gani.csv").is_file()
        ):
            tables.append(read_gani(ref_dir / "gani.csv"))
        else:
            tables.append(None)
    return _merge_gani(*tables)


def _find_data_file(subdir: str, name: str, userDataDir: Path | None) -> Path | None:
    """Locate `<subdir>/<name>.csv`, preferring the user database.

    Args:
        subdir: Database subdirectory (e.g. "gcData").
        name: File stem to look for.
        userDataDir: Optional user database directory.

    Returns:
        Path to the file, or None if it exists in neither database.
    """
    for root in (userDataDir, DEFAULT_DIR):
        if root is None:
            continue
        path = root / subdir / f"{name}.csv"
        if path.is_file():
            return path
    return None


def _strip_or_none(value: object) -> str | None:
    """Return `value` as a stripped string, or None if it is missing or blank."""
    if pd.isna(value):
        return None
    return str(value).strip() or None


def read_gc_data(path: Path) -> pd.DataFrame:
    """Read a gcData file.

    Header variants ("Common Name", "PelePhysics Key") are normalized to use
    underscores, identifier values are stripped, and rows without a SMILES or
    Common_Name are dropped so each remaining row corresponds to one compound.

    Args:
        path: Path to the gcData CSV file.

    Returns:
        The GC composition data.

    Raises:
        ValueError: If required columns are missing.
    """
    gc_data = pd.read_csv(path, header=0).rename(
        columns={
            "Common Name": COMMON_NAME_COL,
            "PelePhysics Key": PELEPHYSICS_KEY_COL,
        }
    )
    id_columns = [c for c in (SMILES_COL, COMMON_NAME_COL) if c in gc_data.columns]
    if WEIGHT_PERCENT_COL not in gc_data.columns or not id_columns:
        msg = (
            f"{path} requires a 'Weight %' column and a 'SMILES' or "
            "'Common_Name' column."
        )
        raise ValueError(msg)
    for col in id_columns:
        gc_data[col] = [_strip_or_none(v) for v in gc_data[col]]
    identified = gc_data[id_columns].notna().any(axis=1)
    return gc_data[identified].reset_index(drop=True)


def match_gc_data(gc_data: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Match each GC row to a reference compound by SMILES (InChI) or common name.

    Args:
        gc_data: GC composition data from `read_gc_data`.
        references: Merged reference compounds DataFrame.

    Returns:
        Reference data for each GC row (same order) with mass fraction in "Y".

    Raises:
        ValueError: If a compound is unknown.
    """
    ref_names = _normalize_names(references[COMMON_NAME_COL])
    matches, weights = [], []
    for _, row in gc_data.iterrows():
        smiles = _strip_or_none(row.get(SMILES_COL))
        name = _strip_or_none(row.get(COMMON_NAME_COL))
        if smiles is not None:
            hits = references.index[
                references[INCHI_COL] == mol.inchi(mol.from_smiles(smiles))
            ]
            label = f"SMILES '{smiles}'"
        elif name is not None:
            hits = references.index[ref_names == name.lower()]
            label = f"common name '{name}'"
        else:
            continue
        if hits.empty:
            msg = (
                f"Reference compound for {label} is missing.\n"
                "Please provide a user referenceCompounds/compounds.csv containing it."
            )
            raise ValueError(msg)
        matches.append(hits[0])
        weights.append(row[WEIGHT_PERCENT_COL])

    data = references.loc[matches].reset_index(drop=True)
    data[Y_COL] = np.array(weights) / np.sum(weights)
    return data
