"""Test suite for the Fuel class in the fuellib package."""

import pytest

from fuellib import Fuel


class TestRDKitProperties:
    """Unit tests for the RDKit properties of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name, expected_nC, expected_nH",
        [("decane", [10], [22]), ("heptane-decane", [7, 10], [16, 22])],
    )
    def test_nC_and_nH(
        self, fuel_name: str, expected_nC: list[int], expected_nH: list[int]
    ):
        """Test the number of carbon atoms for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.nC == expected_nC
        assert fuel.nH == expected_nH

    @pytest.mark.parametrize(
        "fuel_name, expected_MW",
        [("decane", [0.142286]), ("heptane-decane", [0.100205, 0.142286])],
    )
    def test_MW(self, fuel_name: str, expected_MW: list[float]):
        """Test the molecular weights of the compounds in the fuel mixture."""
        fuel = Fuel(fuel_name)
        MW = fuel.MW.to("kg/mol").magnitude
        assert MW == pytest.approx(expected_MW)

    @pytest.mark.parametrize(
        "fuel_name, expected_formulas",
        [("decane", ["C10H22"]), ("heptane-decane", ["C7H16", "C10H22"])],
    )
    def test_formulas(self, fuel_name: str, expected_formulas: list[str]):
        fuel = Fuel(fuel_name)
        assert fuel.formulas == expected_formulas
