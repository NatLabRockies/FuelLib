"""Tests for the reference compounds data and its validator."""

import pandas as pd
import pytest

from fuellib.data import references
from fuellib.rdk.mol import atom_counts, from_smiles, inchi


@pytest.fixture
def shipped_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(references.COMPOUNDS_PATH).astype(object),
        pd.read_csv(references.PROPERTIES_PATH).astype(object),
    )


@pytest.fixture
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valid synthetic tables, independent of the size of the shipped data."""
    names = ["n-heptane", "n-octane", "toluene", "decalin", "1-pentene", "naphthalene"]
    smiles = [
        "CCCCCCC",
        "CCCCCCCC",
        "Cc1ccccc1",
        "C1CCC2CCCCC2C1",
        "CCCC=C",
        "c1ccc2ccccc2c1",
    ]
    blank = pd.DataFrame({
        "Common_Name": names,
        "InChI": None,
        "SMILES": smiles,
        "Num_C": None,
        "Family": None,
    })
    compounds = references.populate_missing(blank).astype(object)
    properties = pd.DataFrame({
        "Common_Name": names[:5],
        "Property": "Tm",
        "Units": "celsius",
        "Value": [-90.6, -56.8, -95.0, -30.4, -165.2],
        "Error": None,
        "Source": "test",
    }).astype(object)
    return compounds, properties


def test_shipped_data_is_valid() -> None:
    assert not references.load_compounds().empty
    assert not references.load_properties().empty


def test_chemistry_matches_smiles(
    shipped_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    compounds, _ = shipped_tables
    for row in compounds.to_dict("records"):
        name = row["Common_Name"]
        mol = from_smiles(row["SMILES"])
        assert row["InChI"] == inchi(mol), f"InChI mismatch for {name}"
        assert row["Num_C"] == atom_counts(mol)["C"], f"Num_C mismatch: {name}"


def test_rejects_wrong_columns(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    compounds, properties = tables
    with pytest.raises(ValueError, match="columns"):
        references.validate(compounds.rename(columns={"SMILES": "Smiles"}), properties)


def test_rejects_bad_compound_rows(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    compounds, properties = tables
    compounds = compounds.copy()
    compounds.loc[0, "Family"] = "not-a-family"
    compounds.loc[1, "Num_C"] = 2.5
    compounds.loc[2, "InChI"] = compounds.loc[3, "InChI"]
    compounds.loc[4, "Common_Name"] = compounds.loc[5, "Common_Name"]
    with pytest.raises(ValueError) as exc:
        references.validate(compounds, properties)
    msg = str(exc.value)
    assert "unknown Family" in msg
    assert "Num_C" in msg
    assert "duplicate InChI" in msg
    assert "duplicate Common_Name" in msg


def test_rejects_bad_property_rows(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    compounds, properties = tables
    properties = properties.copy()
    properties.loc[0, "Common_Name"] = "unobtainium"
    properties.loc[1, "Property"] = "XYZ"
    properties.loc[2, "Value"] = "abc"
    properties.loc[3, "Error"] = -1.0
    properties.loc[4, ["Common_Name", "Property"]] = properties.loc[
        3, ["Common_Name", "Property"]
    ]
    with pytest.raises(ValueError) as exc:
        references.validate(compounds, properties)
    msg = str(exc.value)
    assert "not in the compounds table" in msg
    assert "unknown Property" in msg
    assert "not numeric" in msg
    assert "non-negative" in msg
    assert "duplicate entry" in msg


@pytest.mark.parametrize(
    ("smiles", "family"),
    [
        ("CCCCCCC", "n-alkane"),
        ("CC(C)CCC", "iso-alkane"),
        ("CCCC=C", "alkene"),
        ("C1CCCCC1", "monocyclic"),
        ("C1CCC2CCCCC2C1", "dicyclic"),
        ("C1C2CC3CC1CC(C2)C3", "tricyclic"),
        ("Cc1ccccc1", "alkylbenzene"),
        ("c1ccc2CCCCc2c1", "cycloaromatic"),
        ("c1ccc2ccccc2c1", "diaromatic"),
    ],
)
def test_classify_family(smiles: str, family: str) -> None:
    assert references.classify_family(from_smiles(smiles)) == family


def test_classify_family_rejects_non_hydrocarbon() -> None:
    with pytest.raises(ValueError, match="hydrocarbons"):
        references.classify_family(from_smiles("CCO"))


def test_populate_missing_fills_blanks(
    shipped_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    compounds, _ = shipped_tables
    blanked = compounds.copy()
    blanked[["InChI", "Num_C", "Family"]] = None
    populated = references.populate_missing(blanked)
    pd.testing.assert_frame_equal(populated, compounds.astype(populated.dtypes))


def test_populate_missing_keeps_existing_values(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    compounds, _ = tables
    compounds = compounds.copy()
    compounds.loc[0, "Family"] = "alkene"
    compounds.loc[1, "InChI"] = None
    populated = references.populate_missing(compounds)
    assert populated.loc[0, "Family"] == "alkene"
    assert populated.loc[1, "InChI"].startswith("InChI=")


@pytest.mark.parametrize("col", ["Common_Name", "SMILES"])
def test_populate_missing_requires_name_and_smiles(
    tables: tuple[pd.DataFrame, pd.DataFrame], col: str
) -> None:
    compounds, _ = tables
    compounds = compounds.copy()
    compounds.loc[0, col] = None
    with pytest.raises(ValueError, match=f"'{col}' is required"):
        references.populate_missing(compounds)


def test_populate_missing_rejects_invalid_smiles(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    compounds, _ = tables
    compounds = compounds.copy()
    compounds.loc[0, "SMILES"] = "not a smiles"
    compounds.loc[0, "InChI"] = None
    with pytest.raises(ValueError, match="Invalid SMILES"):
        references.populate_missing(compounds)


def test_validate_rejects_inconsistent_inchi_and_num_c(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    compounds, properties = tables
    compounds = compounds.copy()
    compounds.loc[0, "InChI"] = compounds.loc[1, "InChI"]
    compounds.loc[2, "Num_C"] = 99
    with pytest.raises(ValueError) as exc:
        references.validate(compounds, properties)
    assert "InChI does not match SMILES" in str(exc.value)
    assert "Num_C does not match SMILES" in str(exc.value)
