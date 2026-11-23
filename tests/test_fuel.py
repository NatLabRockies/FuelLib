"""Test suite for the Fuel class in the fuellib package."""

from pathlib import Path

import pandas as pd
import pytest

from fuellib import Fuel, Units
from fuellib.fuel import DEFAULT_DATA_DIR

import numpy as np


class TestParsing:
    """Unit tests for the parsing functions of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("decane", 1), ("posf10325", 67)],
    )
    def test_gcxgc_data_loading(self, fuel_name: str, num_compounds: int) -> None:
        """Test that the GCxGC data is correctly loaded for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.gc_data.shape[0] == num_compounds

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("decane", 1), ("posf10325", 67)],
    )
    def test_y_0_loading(self, fuel_name: str, num_compounds: int) -> None:
        """Test that the initial mass fractions (Y_0) are correctly loaded for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.y0 is not None
        assert fuel.y0.magnitude.shape[0] == num_compounds
        assert fuel.y0.units == ""
        assert pytest.approx(fuel.y0.magnitude.sum(), abs=1e-8) == 1.0

    def test_pelephysics_keys(self) -> None:
        """Test that the PelePhysics keys are correctly loaded for each compound in the fuel mixture."""
        fuel = Fuel("posf10325")
        assert fuel.comps.pelephysics_keys is not None
        assert len(fuel.comps.pelephysics_keys) == fuel.num_compounds
        assert all(isinstance(key, str) for key in fuel.comps.pelephysics_keys)

    @pytest.mark.parametrize(
        "fuel_name",
        ["posf4658", "posf10264", "posf10289", "posf10325", "posf11498"],
    )
    def test_metadata_loading(self, fuel_name: str) -> None:
        """Test that the fuel metadata is correctly loaded for the POSF fuels."""
        fuel = Fuel(fuel_name)
        assert fuel.metadata is not None
        for key in ("name", "category", "source", "reference", "description"):
            assert fuel.metadata[key]
        assert "POSF" in fuel.metadata["name"]

    def test_custom_ref_compounds_extend_and_override_bundled_data(
        self, tmp_path: Path
    ) -> None:
        """Custom rows override bundled rows by InChI without dropping bundled rows."""
        (tmp_path / "gc_data").mkdir()
        pd.read_csv(DEFAULT_DATA_DIR / "gc_data" / "heptane_decane.csv").to_csv(
            tmp_path / "gc_data" / "heptane_decane.csv", index=False
        )

        custom_ref = pd.read_csv(DEFAULT_DATA_DIR / "ref_compounds.csv")
        custom_ref = custom_ref.loc[
            custom_ref["Reference Compound"].eq("n-decane")
        ].copy()
        custom_ref.loc[:, "SMILES"] = "C(C)CCCCCCCC"
        custom_ref.loc[:, "DCN_Value"] = 999.0
        custom_ref.to_csv(tmp_path / "ref_compounds.csv", index=False)

        custom_gani = pd.read_csv(DEFAULT_DATA_DIR / "ref_gani.csv")
        custom_gani = custom_gani.loc[
            custom_gani["Reference Compound"].eq("n-decane")
        ].copy()
        custom_gani.loc[:, "CH3"] = 99
        custom_gani.to_csv(tmp_path / "ref_gani.csv", index=False)

        custom_nannoolal = pd.read_csv(DEFAULT_DATA_DIR / "ref_nannoolal.csv")
        custom_nannoolal = custom_nannoolal.loc[
            custom_nannoolal["Reference Compound"].eq("n-decane")
        ].copy()
        custom_nannoolal.loc[:, "CH3-(ne)"] = 99
        custom_nannoolal.to_csv(tmp_path / "ref_nannoolal.csv", index=False)

        fuel = Fuel("heptane_decane", data_dir=tmp_path)

        assert fuel.comps.names == ["n-heptane", "n-decane"]
        assert fuel._ref_comps["Reference Compound"].eq("n-decane").sum() == 1
        assert fuel._ref_gani["Reference Compound"].eq("n-decane").sum() == 1
        assert fuel._ref_nannoolal["Reference Compound"].eq("n-decane").sum() == 1
        assert fuel.components_data.loc[1, "DCN_Value"] == 999.0
        assert fuel._gani_decomp.loc[1, "CH3"] == 99
        assert fuel._nannoolal_decomp.loc[1, "CH3-(ne)"] == 99


class TestRDKitProperties:
    """Unit tests for the RDKit properties of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name",
        [("decane"), ("posf10325")],
    )
    def test_rdkit_mols(self, fuel_name: str) -> None:
        """Test that the RDKit molecule objects are correctly created for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert len(fuel.comps.rdkit_mols) == fuel.num_compounds
        assert all(mol is not None for mol in fuel.comps.rdkit_mols)

    @pytest.mark.parametrize(
        "fuel_name, expected_num_c, expected_num_h",
        [("decane", [10], [22]), ("heptane_decane", [7, 10], [16, 22])],
    )
    def test_num_c_and_num_h(
        self, fuel_name: str, expected_num_c: list[int], expected_num_h: list[int]
    ) -> None:
        """Test the number of carbon atoms for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert isinstance(fuel.comps.num_c, np.ndarray)
        assert isinstance(fuel.comps.num_h, np.ndarray)
        assert fuel.comps.num_c.tolist() == expected_num_c
        assert fuel.comps.num_h.tolist() == expected_num_h

    @pytest.mark.parametrize(
        "fuel_name, expected_mw",
        [("decane", [142.286]), ("heptane_decane", [100.205, 142.286])],
    )
    def test_mw(self, fuel_name: str, expected_mw: list[float]) -> None:
        """Test the molecular weights of the compounds in the fuel mixture and valid conversion to g/mol."""
        fuel = Fuel(fuel_name)
        assert fuel.comps.mw.units == "kg/mol"
        mw = fuel.comps.mw.to("g/mol").magnitude
        assert mw == pytest.approx(expected_mw)

    @pytest.mark.parametrize(
        "fuel_name, expected_hc_type",
        [("decane", ["n-alkane"]), ("posf11498", ["iso-alkane"] * 11 + ["alkene"] * 2)],
    )
    def test_hc_type(self, fuel_name: str, expected_hc_type: list[str]) -> None:
        """Test the hydrocarbon types for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert fuel.comps.hc_type == expected_hc_type

    @pytest.mark.parametrize(
        "fuel_name, expected_fam",
        [("decane", [0]), ("posf11498", [0] * 11 + [3] * 2)],
    )
    def test_fam(self, fuel_name: str, expected_fam: list[int]) -> None:
        """Test the family (number of carbon and hydrogen atoms) for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert np.array_equal(fuel.comps.fam, expected_fam)

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("decane", 1), ("posf10325", 67)],
    )
    def test_num_atoms(self, fuel_name: str, num_compounds: int) -> None:
        """Test the number of carbon and hydrogen atoms for each compound in the fuel mixture."""
        fuel = Fuel(fuel_name)
        assert len(fuel.comps.tc) == num_compounds


class TestMemberFunctions:
    """Test the member functions of the Fuel class."""

    @pytest.mark.parametrize(
        "fuel_name, expected_mean_mw",
        [("decane", 142.286), ("posf10325", 158.975)],
    )
    def test_mean_molecular_weight(
        self, fuel_name: str, expected_mean_mw: float
    ) -> None:
        """Test the mean_molecular_weight member function of the Fuel class."""
        fuel = Fuel(fuel_name)
        mw = fuel.mix.mw(fuel.y0)
        assert pytest.approx(mw.to("g/mol").magnitude, abs=1e-2) == expected_mean_mw

    def test_mass2y_single_compound(self) -> None:
        """Test mass2y for a fuel with a single compound."""
        fuel = Fuel("decane")
        mass = Units.Quantity(np.array([2.5]), "kg")
        yi = fuel.comps.mass2y(mass)
        assert yi.units == "dimensionless"
        assert yi.magnitude == pytest.approx([1.0])

    def test_mass2y_multi_compound(self) -> None:
        """Test mass2y for a fuel with multiple compounds."""
        fuel = Fuel("heptane_decane")
        mass = Units.Quantity(np.array([1.0, 3.0]), "kg")
        yi = fuel.comps.mass2y(mass)
        assert yi.units == "dimensionless"
        assert yi.magnitude == pytest.approx([0.25, 0.75])
        assert yi.magnitude.sum() == pytest.approx(1.0)

    def test_mass2y_zero_mass(self) -> None:
        """Test mass2y returns zeros when total mass is zero."""
        fuel = Fuel("heptane_decane")
        mass = Units.Quantity(np.array([0.0, 0.0]), "kg")
        yi = fuel.comps.mass2y(mass)
        assert yi.magnitude == pytest.approx([0.0, 0.0])

    def test_mass2x_single_compound(self) -> None:
        """Test mass2x for a fuel with a single compound."""
        fuel = Fuel("decane")
        mass = Units.Quantity(np.array([2.5]), "kg")
        xi = fuel.comps.mass2x(mass)
        assert xi.units == "dimensionless"
        assert xi.magnitude == pytest.approx([1.0])

    def test_mass2x_multi_compound(self) -> None:
        """Test mass2x for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane_decane")
        mass = Units.Quantity(np.array([1.0, 1.0]), "kg")
        xi = fuel.comps.mass2x(mass)
        mw = fuel.comps.mw.to("kg/mol").magnitude
        num_mole = mass.magnitude / mw
        expected_xi = num_mole / num_mole.sum()
        assert xi.units == "dimensionless"
        assert xi.magnitude == pytest.approx(expected_xi)
        assert xi.magnitude.sum() == pytest.approx(1.0)

    def test_mass2x_zero_mass(self) -> None:
        """Test mass2x returns zeros when total moles are zero."""
        fuel = Fuel("heptane_decane")
        mass = Units.Quantity(np.array([0.0, 0.0]), "kg")
        xi = fuel.comps.mass2x(mass)
        assert xi.magnitude == pytest.approx([0.0, 0.0])

    def test_x2y_single_compound(self) -> None:
        """Test x2y for a fuel with a single compound."""
        fuel = Fuel("decane")
        xi = Units.Quantity(np.array([1.0]), "dimensionless")
        yi = fuel.comps.x2y(xi)
        assert yi.units == "dimensionless"
        assert yi.magnitude == pytest.approx([1.0])

    def test_x2y_multi_compound(self) -> None:
        """Test x2y for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane_decane")
        xi = Units.Quantity(np.array([0.5, 0.5]), "dimensionless")
        yi = fuel.comps.x2y(xi)
        mw = fuel.comps.mw.to("kg/mol").magnitude
        mass = mw * xi.magnitude
        expected_yi = mass / mass.sum()
        assert yi.units == "dimensionless"
        assert yi.magnitude == pytest.approx(expected_yi)
        assert yi.magnitude.sum() == pytest.approx(1.0)

    def test_x2y_zero_moles(self) -> None:
        """Test x2y returns zeros when total mass is zero."""
        fuel = Fuel("heptane_decane")
        xi = Units.Quantity(np.array([0.0, 0.0]), "dimensionless")
        yi = fuel.comps.x2y(xi)
        assert yi.magnitude == pytest.approx([0.0, 0.0])

    def test_y2x_single_compound(self) -> None:
        """Test y2x for a fuel with a single compound."""
        fuel = Fuel("decane")
        yi = Units.Quantity(np.array([1.0]), "dimensionless")
        xi = fuel.comps.y2x(yi)
        assert xi.units == "dimensionless"
        assert xi.magnitude == pytest.approx([1.0])

    def test_y2x_multi_compound(self) -> None:
        """Test y2x for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane_decane")
        yi = Units.Quantity(np.array([0.25, 0.75]), "dimensionless")
        xi = fuel.comps.y2x(yi)
        mw_mix = fuel.mix.mw(yi)
        mw = fuel.comps.mw.to("kg/mol")
        expected_xi = (mw_mix * yi / mw).magnitude
        assert xi.units == "dimensionless"
        assert xi.magnitude == pytest.approx(expected_xi)
        assert xi.magnitude.sum() == pytest.approx(1.0)

    def test_y2x_zero_mass_fractions(self) -> None:
        """Test y2x returns zeros when total mass fraction is zero."""
        fuel = Fuel("heptane_decane")
        yi = Units.Quantity(np.array([0.0, 0.0]), "dimensionless")
        xi = fuel.comps.y2x(yi)
        assert xi.magnitude == pytest.approx([0.0, 0.0])

    def test_x2y_y2x_roundtrip(self) -> None:
        """Test that converting mole fractions to mass fractions and back is consistent."""
        fuel = Fuel("heptane_decane")
        xi = Units.Quantity(np.array([0.3, 0.7]), "dimensionless")
        yi = fuel.comps.x2y(xi)
        xi_roundtrip = fuel.comps.y2x(yi)
        assert xi_roundtrip.magnitude == pytest.approx(xi.magnitude)

    def test_y2v_single_compound(self) -> None:
        """Test y2v for a fuel with a single compound."""
        fuel = Fuel("decane")
        yi = Units.Quantity(np.array([1.0]), "dimensionless")
        vi = fuel.comps.y2v(yi, Units.Quantity(298.15, "K"))
        assert vi.units == "dimensionless"
        assert vi.magnitude == pytest.approx([1.0])

    def test_y2v_multi_compound(self) -> None:
        """Test y2v for a fuel with multiple compounds against a manual calculation."""
        fuel = Fuel("heptane_decane")
        temp = Units.Quantity(298.15, "K")
        yi = Units.Quantity(np.array([0.25, 0.75]), "dimensionless")
        vi = fuel.comps.y2v(yi, temp)
        rho = fuel.comps.rho(temp).m_as("kg/m^3")
        vol = yi.magnitude / rho
        assert vi.units == "dimensionless"
        assert vi.magnitude == pytest.approx(vol / vol.sum())
        assert vi.magnitude.sum() == pytest.approx(1.0)

    def test_y2v_zero_mass_fractions(self) -> None:
        """Test y2v returns zeros when total mass fraction is zero."""
        fuel = Fuel("heptane_decane")
        yi = Units.Quantity(np.array([0.0, 0.0]), "dimensionless")
        vi = fuel.comps.y2v(yi, Units.Quantity(298.15, "K"))
        assert vi.magnitude == pytest.approx([0.0, 0.0])

    @pytest.mark.parametrize(
        "fuel_name, expected",
        [("decane", 0.0), ("heptane_decane", 0.0), ("toluene", 100.0)],
    )
    def test_mixture_aromatics_content_pure(
        self, fuel_name: str, expected: float
    ) -> None:
        """Test aromatics content for fuels with no or only aromatic compounds."""
        fuel = Fuel(fuel_name)
        aro = fuel.mix.aromatics(fuel.y0)
        assert aro.units == "percent"
        assert aro.magnitude == pytest.approx(expected)

    def test_mixture_aromatics_content_mixture(self) -> None:
        """Test aromatics content of a jet fuel against a manual volume-fraction sum."""
        fuel = Fuel("posf10325")
        aro = fuel.mix.aromatics(fuel.y0).m_as("percent")
        rho = fuel.comps.rho(Units.Quantity(298.15, "K")).m_as("kg/m^3")
        vol = fuel.y0.magnitude / rho
        is_aromatic = np.array(fuel.comps.hc_type) == "aromatic"
        expected = 100.0 * vol[is_aromatic].sum() / vol.sum()
        assert 0.0 < aro < 100.0
        assert aro == pytest.approx(expected)

    @pytest.mark.parametrize("fuel_name", ["decane", "heptane_decane", "toluene"])
    def test_mixture_naphthalenes_content_none(self, fuel_name: str) -> None:
        """Test naphthalenes content is zero for fuels without naphthalenes."""
        fuel = Fuel(fuel_name)
        nap = fuel.mix.naphthalenes(fuel.y0)
        assert nap.units == "percent"
        assert nap.magnitude == pytest.approx(0.0)

    def test_mixture_naphthalenes_content_mixture(self) -> None:
        """Test naphthalenes content of a jet fuel against a manual volume-fraction sum.

        The naphthalenes are taken as the compounds in the reference ``diaromatic``
        family.
        """
        fuel = Fuel("posf10325")
        nap = fuel.mix.naphthalenes(fuel.y0)
        aro = fuel.mix.aromatics(fuel.y0)
        rho = fuel.comps.rho(Units.Quantity(298.15, "K")).m_as("kg/m^3")
        vol = fuel.y0.magnitude / rho
        is_diaromatic = (fuel.components_data["Family"] == "diaromatic").to_numpy()
        assert nap.units == "percent"
        assert nap.magnitude > 0.0
        assert nap.magnitude == pytest.approx(
            100.0 * vol[is_diaromatic].sum() / vol.sum()
        )
        assert nap < aro
