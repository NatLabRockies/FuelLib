"""Test the error handling shared by every GCM ``decompose`` function."""

import pandas as pd
import pytest

from fuellib.gcm import gani, nannoolal, unifac
from fuellib.gcm.core import Decomposer

DECOMPOSERS = [
    pytest.param(gani.decompose, "Gani", id="gani"),
    pytest.param(nannoolal.decompose, "Nannoolal", id="nannoolal"),
    pytest.param(unifac.decompose, "UNIFAC", id="unifac"),
]


@pytest.mark.parametrize(("decompose", "method"), DECOMPOSERS)
@pytest.mark.parametrize("smiles", ["", "   ", float("nan"), None])
def test_decompose_rejects_empty_smiles(decompose, method, smiles):
    """Empty or non-string SMILES should raise instead of returning a zero row."""
    with pytest.raises(ValueError) as error:
        decompose(smiles)
    assert str(error.value) == (
        f"{method} decomposition failed: SMILES must be a non-empty string."
    )


@pytest.mark.parametrize(("decompose", "method"), DECOMPOSERS)
def test_decompose_reports_unparsable_smiles(decompose, method):
    """Unparsable SMILES should name the method and the input."""
    with pytest.raises(ValueError) as error:
        decompose("C1CC")
    assert str(error.value) == (
        f"{method} decomposition failed for SMILES 'C1CC': "
        "RDKit could not parse the input."
    )


@pytest.mark.parametrize(("decompose", "method"), DECOMPOSERS)
def test_decompose_reports_unmatched_atoms(decompose, method):
    """Unsupported atoms should report how many atoms the groups covered."""
    with pytest.raises(ValueError) as error:
        decompose("CCCCCCCO")
    assert str(error.value) == (
        f"{method} decomposition failed for SMILES 'CCCCCCCO': "
        "first-order groups matched 7 of 8 atoms."
    )


@pytest.fixture
def decomposer():
    """A minimal decomposer whose table lacks the 'CH2' group."""
    return Decomposer("Test", pd.Index(["CH3"]), [("CH3", "[CH3]"), ("CH2", "[CH2]")])


def test_decomposer_rejects_invalid_smarts():
    """An invalid SMARTS pattern should name the method, SMILES, and pattern."""
    decomposer = Decomposer("Test", pd.Index(["CH3"]), [("CH3", "[C")])
    with pytest.raises(ValueError) as error:
        decomposer("CCC")
    assert str(error.value) == (
        "Test decomposition failed for SMILES 'CCC': invalid SMARTS pattern '[C'."
    )


def test_decomposer_reports_groups_missing_from_table(decomposer):
    """Matched groups absent from the table should raise a KeyError listing them."""
    with pytest.raises(KeyError) as error:
        decomposer("CCCCCC")
    assert error.value.args[0] == (
        "Test decomposition failed for SMILES 'CCCCCC': "
        "groups not found in Test table: ['CH2']"
    )


def test_decomposer_second_order_counts_are_added():
    """Counts from second_order_counts should be added to the first order counts."""

    class Extra(Decomposer):
        def second_order_counts(self, mol, smiles):
            return {"CH3": 10}

    decomposer = Extra(
        "Test", pd.Index(["CH3", "CH2"]), [("CH3", "[CH3]"), ("CH2", "[CH2]")]
    )
    row = decomposer("CCC").iloc[0]
    assert row.to_dict() == {"CH3": 12, "CH2": 1}
