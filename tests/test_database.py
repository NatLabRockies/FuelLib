"""Tests for the reference compound database (`fuellib.database`)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuellib as fl
from fuellib.database import Property, database
from fuellib.database.database import (
    COMMON_NAME_COL,
    DEFAULT_DIR,
    FAMILY_COL,
    INCHI_COL,
    NUM_C_COL,
    SMILES_COL,
)
from fuellib.rdk.mol import atom_counts, from_smiles, inchi

COMPOUNDS_FILE = DEFAULT_DIR / "referenceCompounds" / "compounds.csv"
GANI_FILE = DEFAULT_DIR / "referenceCompounds" / "gani.csv"


def _write_compounds(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_shipped_reference_data_is_valid() -> None:
    compounds, _ = database.read_reference_compounds(COMPOUNDS_FILE)
    assert not compounds.empty
    assert not compounds[list(database.ID_COLUMNS)].isna().any().any()
    assert set(compounds[FAMILY_COL]) <= set(database.SUPPORTED_FAMILIES)
    assert compounds[INCHI_COL].is_unique
    assert set(compounds[INCHI_COL]) <= set(database.read_gani(GANI_FILE).index)


def test_shipped_chemistry_matches_smiles() -> None:
    compounds, _ = database.read_reference_compounds(COMPOUNDS_FILE)
    for row in compounds.to_dict("records"):
        name = row[COMMON_NAME_COL]
        mol = from_smiles(row[SMILES_COL])
        assert row[INCHI_COL] == inchi(mol), f"InChI mismatch for {name}"
        assert row[NUM_C_COL] == atom_counts(mol)["C"], f"Num_C mismatch: {name}"


@pytest.mark.parametrize(
    ("smiles", "family"),
    [
        ("CCCCCCC", "n-alkane"),
        ("CC(C)CCC", "iso-alkane"),
        ("CCCC=C", "alkene"),
        ("C1CCCCC1", "monocyclic"),
        ("C1CCC2CCCCC2C1", "dicyclic"),
        ("C1CC2CC3CCCC3C2C1", "tricyclic"),
        ("Cc1ccccc1", "alkylbenzene"),
        ("c1ccc2CCCCc2c1", "cycloaromatic"),
        ("c1ccc2ccccc2c1", "diaromatic"),
    ],
)
def test_classify_family(smiles: str, family: str) -> None:
    assert database.classify_family(from_smiles(smiles)) == family


def test_read_reference_compounds_requires_columns(tmp_path: Path) -> None:
    path = _write_compounds(tmp_path / "compounds.csv", [{COMMON_NAME_COL: "x"}])
    with pytest.raises(ValueError, match="Required columns"):
        database.read_reference_compounds(path)


def test_read_reference_compounds_requires_values(tmp_path: Path) -> None:
    path = _write_compounds(
        tmp_path / "compounds.csv",
        [{COMMON_NAME_COL: "n-heptane", SMILES_COL: None}],
    )
    with pytest.raises(ValueError, match="missing a value"):
        database.read_reference_compounds(path)


def test_read_reference_compounds_keeps_unknown_columns(tmp_path: Path) -> None:
    path = _write_compounds(
        tmp_path / "compounds.csv",
        [{COMMON_NAME_COL: "n-heptane", SMILES_COL: "CCCCCCC", "Notes": "hi"}],
    )
    compounds, extras = database.read_reference_compounds(path)
    assert list(compounds.columns) == list(database.REF_COMPOUND_COLUMNS)
    assert extras["Notes"].tolist() == ["hi"]


def test_autofill_identity(tmp_path: Path) -> None:
    path = _write_compounds(
        tmp_path / "compounds.csv",
        [
            {COMMON_NAME_COL: "toluene", SMILES_COL: "Cc1ccccc1"},
            {COMMON_NAME_COL: "decalin", SMILES_COL: "C1CCC2CCCCC2C1"},
        ],
    )
    compounds, _ = database.read_reference_compounds(path)
    assert database.autofill_identity(compounds, path)
    assert compounds[FAMILY_COL].tolist() == ["alkylbenzene", "dicyclic"]
    assert compounds[NUM_C_COL].tolist() == [7, 10]
    assert compounds[INCHI_COL].str.startswith("InChI=").all()
    assert not database.autofill_identity(compounds, path)


def test_autofill_identity_rejects_unsupported_family(tmp_path: Path) -> None:
    path = _write_compounds(
        tmp_path / "compounds.csv",
        [{COMMON_NAME_COL: "x", SMILES_COL: "CCCCCCC", FAMILY_COL: "not-a-family"}],
    )
    compounds, _ = database.read_reference_compounds(path)
    with pytest.raises(ValueError, match="Unsupported families"):
        database.autofill_identity(compounds, path)


def test_write_template(tmp_path: Path) -> None:
    root = database.write_template(tmp_path, "mix")
    compounds, _ = database.read_reference_compounds(
        root / "referenceCompounds" / "compounds.csv"
    )
    assert compounds.empty
    assert database.read_gani(root / "referenceCompounds" / "gani.csv").empty
    assert database.read_gc_data(root / "gcData" / "mix.csv").empty
    with pytest.raises(FileExistsError):
        database.write_template(tmp_path, "mix")


def test_user_database_overrides_and_autofills(tmp_path: Path) -> None:
    database.write_template(tmp_path, "mix")
    user_compounds = tmp_path / "referenceCompounds" / "compounds.csv"
    _write_compounds(
        user_compounds,
        [
            {
                COMMON_NAME_COL: "n-heptane",
                SMILES_COL: "CCCCCCC",
                Property.TC.value: 540.2,
                Property.TC.units: "kelvin",
                Property.TC.source: "test",
            }
        ],
    )
    (tmp_path / "gcData" / "mix.csv").write_text(
        "Common_Name,Weight %\n N-Heptane ,100\n"
    )

    fuel = fl.Fuel("mix", userDataDir=tmp_path)
    assert fuel.compounds == ["n-heptane"]
    assert np.isclose(fuel.Tc.magnitude[0], 540.2)

    # The user file is auto-populated from the (merged) Gani decompositions.
    written, _ = database.read_reference_compounds(user_compounds)
    assert written[FAMILY_COL].tolist() == ["n-alkane"]
    assert written[Property.PC.source].tolist() == [database.GANI_SOURCE]
    assert written[Property.TC.source].tolist() == ["test"]


def test_unknown_gc_compound_raises(tmp_path: Path) -> None:
    (tmp_path / "gcData").mkdir()
    (tmp_path / "gcData" / "bad.csv").write_text(
        "Common_Name,Weight %\nunobtainium,1\n"
    )
    with pytest.raises(ValueError, match="unobtainium"):
        fl.Fuel("bad", userDataDir=tmp_path)


def test_missing_gc_data_raises() -> None:
    with pytest.raises(FileNotFoundError):
        fl.Fuel("not-a-fuel")


def test_get_property_converts_mixed_units() -> None:
    fuel = fl.Fuel("posf10325")
    expected = fuel.get_property(Property.TC, output_units="K").magnitude
    fuel.data.loc[0, Property.TC.value] = expected[0] - 273.15
    fuel.data.loc[0, Property.TC.units] = "degC"
    tc = fuel.get_property(Property.TC)
    assert str(tc.units) == "kelvin"
    assert np.allclose(tc.magnitude, expected)
    assert np.allclose(fuel.get_property("Tc", output_units="K").magnitude, expected)


def test_gani_decomp_matches_database() -> None:
    fuel = fl.Fuel("heptane-decane")
    decomp = fuel.gani_decomp()
    assert decomp.index.tolist() == fuel.compounds
    assert decomp.shape[0] == fuel.num_compounds
    tc = fuel.gcm_properties["gani"]["tc"].to("K").magnitude
    assert np.allclose(tc, fuel.Tc.magnitude)
