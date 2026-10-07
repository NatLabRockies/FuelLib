"""Test suite for the Fuel class in the fuellib package."""

import pytest

from fuellib import Fuel, Property, Units

import numpy as np


class TestParsing:
    """Unit tests for the parsing functions of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("decane", 1), ("posf10325", 67)],
    )
    def test_gc_data_loading(self, fuel_name: str, num_compounds: int) -> None:
        """Test that the GC data is correctly loaded for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.gc_data.shape[0] == num_compounds
        assert fuel.data.shape[0] == num_compounds

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("decane", 1), ("posf10325", 67)],
    )
    def test_Y_0_loading(self, fuel_name: str, num_compounds: int) -> None:
        """Test that the initial mass fractions (Y_0) are correctly loaded for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.Y_0 is not None
        assert fuel.Y_0.magnitude.shape[0] == num_compounds
        assert fuel.Y_0.units == ""
        assert pytest.approx(fuel.Y_0.magnitude.sum(), abs=1e-8) == 1.0

    def test_pelephysics_keys(self) -> None:
        """Test that the PelePhysics keys are correctly loaded for each compound in the fuel mixture."""
        fuel = Fuel("posf10325")
        assert fuel.pelephysics_keys is not None
        assert len(fuel.pelephysics_keys) == fuel.num_compounds
        assert all(isinstance(key, str) for key in fuel.pelephysics_keys)


class TestRDKitProperties:
    """Unit tests for the RDKit properties of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name",
        [("decane"), ("posf10325")],
    )
    def test_rdkit_mols(self, fuel_name: str) -> None:
        """Test that the RDKit molecule objects are correctly created for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert len(fuel.rdkit_mols) == fuel.num_compounds
        assert all(mol is not None for mol in fuel.rdkit_mols)

    @pytest.mark.parametrize(
        "fuel_name, expected_formulas",
        [("decane", ["C10H22"]), ("heptane-decane", ["C7H16", "C10H22"])],
    )
    def test_formulas(self, fuel_name: str, expected_formulas: list[str]) -> None:
        """Test that the chemical formulas are correctly generated for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.formulas == expected_formulas

    @pytest.mark.parametrize(
        "fuel_name, expected_inchi",
        [
            ("decane", ["InChI=1S/C10H22/c1-3-5-7-9-10-8-6-4-2/h3-10H2,1-2H3"]),
            (
                "heptane-decane",
                [
                    "InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3",
                    "InChI=1S/C10H22/c1-3-5-7-9-10-8-6-4-2/h3-10H2,1-2H3",
                ],
            ),
        ],
    )
    def test_inchi(self, fuel_name: str, expected_inchi: list[str]) -> None:
        """Test that the InChI strings are correctly generated for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.inchi == expected_inchi

    @pytest.mark.parametrize(
        "fuel_name, expected_nC, expected_nH",
        [("decane", [10], [22]), ("heptane-decane", [7, 10], [16, 22])],
    )
    def test_nC_and_nH(
        self, fuel_name: str, expected_nC: list[int], expected_nH: list[int]
    ) -> None:
        """Test the number of carbon atoms for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.nC == expected_nC
        assert fuel.nH == expected_nH

    @pytest.mark.parametrize(
        "fuel_name, expected_MW",
        [("decane", [142.286]), ("heptane-decane", [100.205, 142.286])],
    )
    def test_MW(self, fuel_name: str, expected_MW: list[float]) -> None:
        """Test the molecular weights of the compounds in the fuel mixture and valid conversion to g/mol."""
        fuel = Fuel(fuel_name)
        assert fuel.MW.units == "kg/mol"
        MW = fuel.MW.to("g/mol").magnitude
        assert MW == pytest.approx(expected_MW)

    @pytest.mark.parametrize(
        "fuel_name, expected_hc_type",
        [("decane", ["n-alkane"]), ("posf11498", ["iso-alkane"] * 11 + ["alkene"] * 2)],
    )
    def test_hc_type(self, fuel_name: str, expected_hc_type: list[str]) -> None:
        """Test the hydrocarbon types for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.hc_type == expected_hc_type

    @pytest.mark.parametrize(
        "fuel_name, expected_fam",
        [("decane", [0]), ("posf11498", [0] * 11 + [3] * 2)],
    )
    def test_fam(self, fuel_name: str, expected_fam: list[int]) -> None:
        """Test the family (number of carbon and hydrogen atoms) for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert np.array_equal(fuel.fam, expected_fam)

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("decane", 1), ("posf10325", 67)],
    )
    def test_num_atoms(self, fuel_name: str, num_compounds: int) -> None:
        """Test that property vectors have one entry per compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert len(fuel.get_property(Property.TC)) == num_compounds
        assert len(fuel.gcm_properties["gani"]["tc"]) == num_compounds


class TestMemberFunctions:
    """Test the member functions of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name, expected_mean_MW",
        [("decane", 142.286), ("posf10325", 158.975)],
    )
    def test_mean_molecular_weight(
        self, fuel_name: str, expected_mean_MW: float
    ) -> None:
        """Test the mean_molecular_weight member function of the Fuel class."""
        fuel = Fuel(fuel_name)
        MW = fuel.mean_molecular_weight(fuel.Y_0)
        assert pytest.approx(MW.to("g/mol").magnitude, abs=1e-2) == expected_mean_MW

    def test_mass2Y_single_compound(self) -> None:
        """Test mass2Y for a fuel with a single compound."""
        fuel = Fuel("decane")
        mass = Units.Quantity(np.array([2.5]), "kg")
        Yi = fuel.mass2Y(mass)
        assert Yi.units == "dimensionless"
        assert Yi.magnitude == pytest.approx([1.0])

    def test_mass2Y_multi_compound(self) -> None:
        """Test mass2Y for a fuel with multiple compounds."""
        fuel = Fuel("heptane-decane")
        mass = Units.Quantity(np.array([1.0, 3.0]), "kg")
        Yi = fuel.mass2Y(mass)
        assert Yi.units == "dimensionless"
        assert Yi.magnitude == pytest.approx([0.25, 0.75])
        assert Yi.magnitude.sum() == pytest.approx(1.0)

    def test_mass2Y_zero_mass(self) -> None:
        """Test mass2Y returns zeros when total mass is zero."""
        fuel = Fuel("heptane-decane")
        mass = Units.Quantity(np.array([0.0, 0.0]), "kg")
        Yi = fuel.mass2Y(mass)
        assert Yi.magnitude == pytest.approx([0.0, 0.0])

    def test_mass2X_single_compound(self) -> None:
        """Test mass2X for a fuel with a single compound."""
        fuel = Fuel("decane")
        mass = Units.Quantity(np.array([2.5]), "kg")
        Xi = fuel.mass2X(mass)
        assert Xi.units == "dimensionless"
        assert Xi.magnitude == pytest.approx([1.0])

    def test_mass2X_multi_compound(self) -> None:
        """Test mass2X for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane-decane")
        mass = Units.Quantity(np.array([1.0, 1.0]), "kg")
        Xi = fuel.mass2X(mass)
        MW = fuel.MW.to("kg/mol").magnitude
        num_mole = mass.magnitude / MW
        expected_Xi = num_mole / num_mole.sum()
        assert Xi.units == "dimensionless"
        assert Xi.magnitude == pytest.approx(expected_Xi)
        assert Xi.magnitude.sum() == pytest.approx(1.0)

    def test_mass2X_zero_mass(self) -> None:
        """Test mass2X returns zeros when total moles are zero."""
        fuel = Fuel("heptane-decane")
        mass = Units.Quantity(np.array([0.0, 0.0]), "kg")
        Xi = fuel.mass2X(mass)
        assert Xi.magnitude == pytest.approx([0.0, 0.0])

    def test_X2Y_single_compound(self) -> None:
        """Test X2Y for a fuel with a single compound."""
        fuel = Fuel("decane")
        Xi = Units.Quantity(np.array([1.0]), "dimensionless")
        Yi = fuel.X2Y(Xi)
        assert Yi.units == "dimensionless"
        assert Yi.magnitude == pytest.approx([1.0])

    def test_X2Y_multi_compound(self) -> None:
        """Test X2Y for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane-decane")
        Xi = Units.Quantity(np.array([0.5, 0.5]), "dimensionless")
        Yi = fuel.X2Y(Xi)
        MW = fuel.MW.to("kg/mol").magnitude
        mass = MW * Xi.magnitude
        expected_Yi = mass / mass.sum()
        assert Yi.units == "dimensionless"
        assert Yi.magnitude == pytest.approx(expected_Yi)
        assert Yi.magnitude.sum() == pytest.approx(1.0)

    def test_X2Y_zero_moles(self) -> None:
        """Test X2Y returns zeros when total mass is zero."""
        fuel = Fuel("heptane-decane")
        Xi = Units.Quantity(np.array([0.0, 0.0]), "dimensionless")
        Yi = fuel.X2Y(Xi)
        assert Yi.magnitude == pytest.approx([0.0, 0.0])

    def test_Y2X_single_compound(self) -> None:
        """Test Y2X for a fuel with a single compound."""
        fuel = Fuel("decane")
        Yi = Units.Quantity(np.array([1.0]), "dimensionless")
        Xi = fuel.Y2X(Yi)
        assert Xi.units == "dimensionless"
        assert Xi.magnitude == pytest.approx([1.0])

    def test_Y2X_multi_compound(self) -> None:
        """Test Y2X for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane-decane")
        Yi = Units.Quantity(np.array([0.25, 0.75]), "dimensionless")
        Xi = fuel.Y2X(Yi)
        Mbar = fuel.mean_molecular_weight(Yi)
        MW = fuel.MW.to("kg/mol")
        expected_Xi = (Mbar * Yi / MW).magnitude
        assert Xi.units == "dimensionless"
        assert Xi.magnitude == pytest.approx(expected_Xi)
        assert Xi.magnitude.sum() == pytest.approx(1.0)

    def test_Y2X_zero_mass_fractions(self) -> None:
        """Test Y2X returns zeros when total mass fraction is zero."""
        fuel = Fuel("heptane-decane")
        Yi = Units.Quantity(np.array([0.0, 0.0]), "dimensionless")
        Xi = fuel.Y2X(Yi)
        assert Xi.magnitude == pytest.approx([0.0, 0.0])

    def test_X2Y_Y2X_roundtrip(self) -> None:
        """Test that converting mole fractions to mass fractions and back is consistent."""
        fuel = Fuel("heptane-decane")
        Xi = Units.Quantity(np.array([0.3, 0.7]), "dimensionless")
        Yi = fuel.X2Y(Xi)
        Xi_roundtrip = fuel.Y2X(Yi)
        assert Xi_roundtrip.magnitude == pytest.approx(Xi.magnitude)
