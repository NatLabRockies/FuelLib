"""Test suite for RDKit-related functionality in the FuelLib package."""

import pytest
from fuellib.rdk import mol


class TestMol:
    """Test suite for the `fuellib.rdk.mol` module."""

    def test_smiles_round_trip(self) -> None:
        """Test that a valid SMILES string can be converted to an RDKit Mol object and back to the same SMILES string."""
        original_smiles = "C"
        result = mol.from_smiles(original_smiles)
        assert result is not None
        round_trip_smiles = mol.smiles(result)
        assert round_trip_smiles == original_smiles

    def test_invalid_smiles(self) -> None:
        """Test that an invalid SMILES string raises a ValueError."""
        with pytest.raises(ValueError):
            mol.from_smiles("Invalid smiles")

    def test_inchi_round_trip(self) -> None:
        """Test that a valid InChI string can be converted to an RDKit Mol object and back to the same InChI string."""
        original_inchi = "InChI=1S/CH4/h1H4"
        result = mol.from_inchi(original_inchi)
        assert result is not None
        round_trip_inchi = mol.inchi(result)
        assert round_trip_inchi == original_inchi

    def test_invalid_inchi(self) -> None:
        """Test that an invalid InChI string raises a ValueError."""
        with pytest.raises(ValueError):
            mol.from_inchi("Invalid InChI")

    def test_hill_formula(self) -> None:
        """Test that the Hill formula is correctly generated for a molecule."""
        mol_obj = mol.from_smiles("C")
        formula = mol.hill_formula(mol_obj)
        assert formula == "CH4"

    @pytest.mark.parametrize(
        "smiles, expected_counts",
        [
            ("C", {"C": 1, "H": 4}),
            ("CC1=CC=CC=C1", {"C": 7, "H": 8}),
            ("CC(O)C#N", {"C": 3, "O": 1, "H": 5, "N": 1}),
        ],
    )
    def test_atom_counts(self, smiles: str, expected_counts: dict[str, int]) -> None:
        """Test that the atom counts are correctly calculated for a molecule."""
        mol_obj = mol.from_smiles(smiles)
        counts = mol.atom_counts(mol_obj)
        assert counts == expected_counts

    @pytest.mark.parametrize(
        "smiles, expected_has_aromatic",
        [
            ("C=CC1CCCCC1", False),
            ("C(C)CC1=CC=CC=C1", True),
            ("CC(O)C#N", False),
            ("C1=CC=CC2=C1CC=CC2", True),
            ("CC(C)C", False),
        ],
    )
    def test_has_aromatic(self, smiles: str, expected_has_aromatic: bool) -> None:
        """Test that the presence of aromatic atoms is correctly detected for a molecule."""
        mol_obj = mol.from_smiles(smiles)
        result = mol.has_aromatic(mol_obj)
        assert result == expected_has_aromatic

    @pytest.mark.parametrize(
        "smiles, expected_has_ring",
        [
            ("C=CC1CCCCC1", True),
            ("C(C)CC1=CC=CC=C1", True),
            ("CC(O)C#N", False),
            ("C1=CC=CC2=C1CC=CC2", True),
            ("CC(C)C", False),
        ],
    )
    def test_has_ring(self, smiles: str, expected_has_ring: bool) -> None:
        """Test that the presence of ring structures is correctly detected for a molecule."""
        mol_obj = mol.from_smiles(smiles)
        result = mol.has_ring(mol_obj)
        assert result == expected_has_ring

    @pytest.mark.parametrize(
        "smiles, expected_has_double_bond",
        [
            ("C=CC1CCCCC1", True),
            ("C(C)CC1=CC=CC=C1", False),
            ("CC(O)C#N", False),
            ("C1=CC=CC2=C1CC=CC2", True),
            ("CC(C)C", False),
        ],
    )
    def test_has_double_bond(self, smiles: str, expected_has_double_bond: bool) -> None:
        """Test that the presence of double bonds is correctly detected for a molecule."""
        mol_obj = mol.from_smiles(smiles)
        result = mol.has_double_bond(mol_obj)
        assert result == expected_has_double_bond

    @pytest.mark.parametrize(
        "smiles, expected_has_branch",
        [
            ("C=CC1CCCCC1", True),
            ("C(C)CC1=CC=CC=C1", True),
            ("CC(O)C#N", True),
            ("C1=CC=CC2=C1CC=CC2", False),
            ("CC(C)C", True),
        ],
    )
    def test_has_branch(self, smiles: str, expected_has_branch: bool) -> None:
        """Test that the presence of branches is correctly detected for a molecule."""
        mol_obj = mol.from_smiles(smiles)
        result = mol.has_branch(mol_obj)
        assert result == expected_has_branch

    @pytest.mark.parametrize(
        "smiles, expected_molecular_weight, exact",
        [
            ("[C]", 12.011, False),  # Average C mass
            ("[C]", 12.0, True),  # Exact C12 mass
            ("[H]", 1.008, False),  # Average H mass
            ("[H]", 1.007825, True),  # Exact H1 mass
            ("C", 16.04, False),  # Average CH4 mass
            ("C", 16.0313, True),  # Exact [C12][H1]4 mass
        ],
    )
    def test_molecular_weight(
        self, smiles: str, expected_molecular_weight: float, exact: bool
    ) -> None:
        """Test that the molecular weight is correctly calculated for a molecule."""
        mol_obj = mol.from_smiles(smiles)
        result = mol.molecular_weight(mol_obj, exact=exact)
        assert pytest.approx(result, 0.01) == expected_molecular_weight
