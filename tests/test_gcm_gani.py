"""Test cases for the Group Contribution Method (GCM) module in FuelLib."""

import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

import fuellib as fl
from fuellib.fuel import Fuel
from fuellib.gcm import GCMRegistry, gani_gcm
from fuellib.gcm.gani import (
    Cp_B,
    Cp_C,
    Cp_stp,
    Gf,
    Hf,
    Hv_stp,
    MW,
    Pc,
    Tb,
    Tc,
    Tm,
    Vc,
    Vm_stp,
    alibakhshi_phi,
    omega,
    rd_A,
    rd_B,
    rd_D,
)
from fuellib.utils import Units


class GCMCoreTestCase(unittest.TestCase):
    """Test cases for the GCM base class and GCMRegistry."""

    @classmethod
    def setUpClass(cls):
        cls.fuel = fl.Fuel("heptane")

    def test_gani_is_registered(self):
        """The built-in Gani GCM should be present in the registry."""
        self.assertIn("gani", GCMRegistry.list_methods())

    def test_get_gcm_returns_registered_gcm(self):
        """get_gcm should retrieve the registered GCM, case-insensitively."""
        gcm = GCMRegistry.get_gcm("gani")
        self.assertIs(gcm, gani_gcm)
        # Case-insensitive lookup
        self.assertIs(GCMRegistry.get_gcm("GANI"), gani_gcm)

    def test_get_gcm_raises_for_unknown_name(self):
        """get_gcm should raise ValueError for an unregistered GCM name."""
        with self.assertRaises(ValueError):
            GCMRegistry.get_gcm("not_a_real_gcm")

    def test_register_new_gcm_and_property(self):
        """register should add a new GCM and register_property should add to it."""

        def dummy_property(fuel):
            return Units.Quantity(np.ones(fuel.num_compounds), "")

        dummy_property.__name__ = "DummyProp"

        gcm = GCMRegistry.register("dummy_test_gcm", property_fns=[dummy_property])
        self.addCleanup(GCMRegistry.methods.remove, gcm)

        self.assertEqual(gcm.name, "dummy_test_gcm")
        self.assertIn("dummyprop", gcm.list_property_fns())

        # register_property should also add new properties after creation
        def another_property(fuel):
            return Units.Quantity(np.zeros(fuel.num_compounds), "")

        another_property.__name__ = "AnotherProp"
        registered = gcm.register_property(another_property)
        self.assertIs(registered, another_property)
        self.assertIn("anotherprop", gcm.list_property_fns())

    def test_get_property_raises_for_unknown_property(self):
        """get_property should raise ValueError for an unimplemented property."""
        with self.assertRaises(ValueError):
            gani_gcm.get_property("not_a_real_property")

    def test_predict_returns_property_result(self):
        """predict should return a Quantity with the expected units and length."""
        result = gani_gcm.predict("Tc", self.fuel)
        self.assertEqual(result.units, Units.Quantity(1.0, "K").units)
        self.assertEqual(len(result.magnitude), self.fuel.num_compounds)

    def test_predict_all_returns_expected_structure(self):
        """predict_all should return every registered property under the GCM name."""
        results = gani_gcm.predict_all(self.fuel)
        self.assertIn("gani", results)
        gani_results = results["gani"]
        self.assertIn("tc", gani_results)
        for prop_name in gani_gcm.list_property_fns():
            self.assertIn(prop_name, gani_results)

    def test_list_property_fns_is_sorted(self):
        """list_property_fns should return property names in sorted order."""
        fns = gani_gcm.list_property_fns()
        self.assertEqual(fns, sorted(fns))


class GaniPropertyTestCase(unittest.TestCase):
    """Test cases for the individual Gani-Constantinou property functions."""

    @classmethod
    def setUpClass(cls):
        cls.fuel = fl.Fuel("heptane")

    def _assert_valid_quantity(self, value, units):
        self.assertEqual(value.units, Units.Quantity(1.0, units).units)
        arr = np.asarray(value.magnitude)
        self.assertEqual(arr.shape[0], self.fuel.num_compounds)
        self.assertTrue(np.all(np.isfinite(arr)))

    def test_tc(self):
        """Tc should return finite critical temperatures in K."""
        self._assert_valid_quantity(Tc(self.fuel), "K")

    def test_pc(self):
        """Pc should return finite critical pressures in bar."""
        self._assert_valid_quantity(Pc(self.fuel), "bar")

    def test_vc(self):
        """Vc should return finite critical volumes in m^3/kmol."""
        self._assert_valid_quantity(Vc(self.fuel), "m^3/kmol")

    def test_tb(self):
        """Tb should return finite normal boiling temperatures in K."""
        self._assert_valid_quantity(Tb(self.fuel), "K")

    def test_tm(self):
        """Tm should return finite melting temperatures in K."""
        self._assert_valid_quantity(Tm(self.fuel), "K")

    def test_hf(self):
        """Hf should return finite enthalpies of formation in kJ/mol."""
        self._assert_valid_quantity(Hf(self.fuel), "kJ/mol")

    def test_gf(self):
        """Gf should return finite Gibbs free energies of formation in kJ/mol."""
        self._assert_valid_quantity(Gf(self.fuel), "kJ/mol")

    def test_hv_stp(self):
        """Hv_stp should return finite enthalpies of vaporization in kJ/mol."""
        self._assert_valid_quantity(Hv_stp(self.fuel), "kJ/mol")

    def test_omega(self):
        """omega should return finite dimensionless acentric factors."""
        self._assert_valid_quantity(omega(self.fuel), "")

    def test_vm_stp(self):
        """Vm_stp should return finite standard molar volumes in m^3/kmol."""
        self._assert_valid_quantity(Vm_stp(self.fuel), "m^3/kmol")

    def test_cp_stp(self):
        """Cp_stp should return finite standard molar heat capacities."""
        self._assert_valid_quantity(Cp_stp(self.fuel), "J/(mol*K)")

    def test_cp_b(self):
        """Cp_B should return finite heat capacity B correction terms."""
        self._assert_valid_quantity(Cp_B(self.fuel), "J/(mol*K)")

    def test_cp_c(self):
        """Cp_C should return finite heat capacity C correction terms."""
        self._assert_valid_quantity(Cp_C(self.fuel), "J/(mol*K)")

    def test_rd_a(self):
        """rd_A should return finite dimensionless Ruzicka-Domalski A coefficients."""
        self._assert_valid_quantity(rd_A(self.fuel), "")

    def test_rd_b(self):
        """rd_B should return finite Ruzicka-Domalski B coefficients in K^-1."""
        self._assert_valid_quantity(rd_B(self.fuel), "K^-1")

    def test_rd_d(self):
        """rd_D should return finite Ruzicka-Domalski D coefficients in K^-2."""
        self._assert_valid_quantity(rd_D(self.fuel), "K^-2")

    def test_alibakhshi_phi(self):
        """alibakhshi_phi should return finite phi parameters in K."""
        self._assert_valid_quantity(alibakhshi_phi(self.fuel), "K")

    def test_mw(self):
        """MW should return finite molecular weights in g/mol."""
        self._assert_valid_quantity(MW(self.fuel), "g/mol")


class GaniHelperFunctionTestCase(unittest.TestCase):
    """Test cases for the internal helper functions in the gani module."""

    def test_get_row_raises_for_unknown_property(self):
        """_get_row should raise KeyError for a property missing from the table."""
        from fuellib.gcm.gani import _get_row

        with pytest.raises(KeyError):
            _get_row("not_a_real_property")

    def test_check_compatible_dims_raises_on_mismatch(self):
        """_check_compatible_dims should raise ValueError on shape mismatch."""
        from fuellib.gcm.gani import _check_compatible_dims

        cj = np.array([1.0, 2.0, 3.0])
        ij = np.zeros((2, 4))
        with pytest.raises(ValueError):
            _check_compatible_dims(cj, ij)

    def test_check_compatible_dims_passes_on_match(self):
        """_check_compatible_dims should not raise when shapes are compatible."""
        from fuellib.gcm.gani import _check_compatible_dims

        cj = np.array([1.0, 2.0, 3.0])
        ij = np.zeros((2, 3))
        # Should not raise
        _check_compatible_dims(cj, ij)

    def test_get_decomp_matches_gani_decomp_columns(self):
        """_get_decomp should return an array shaped by compounds and table columns."""
        from fuellib.gcm.gani import TABLE, _get_decomp

        fuel = fl.Fuel("heptane")
        decomp = _get_decomp(fuel)
        self.assertIsInstance(decomp, np.ndarray)
        self.assertEqual(decomp.shape, (fuel.num_compounds, len(TABLE.columns)))

    def test_get_decomp_resorts_columns_to_match_table(self):
        """_get_decomp should reorder decomp columns to match TABLE.columns order."""
        from fuellib.gcm.gani import TABLE, _get_decomp

        cols = list(TABLE.columns)
        values = {col: idx for idx, col in enumerate(cols)}
        shuffled_cols = cols[::-1]
        decomp_df = pd.DataFrame(
            [[values[col] for col in shuffled_cols]],
            columns=shuffled_cols,
            index=["compoundA"],
        )
        fake_fuel = cast(Fuel, SimpleNamespace(gani_decomp=lambda: decomp_df))

        result = _get_decomp(fake_fuel)

        expected = np.array([[values[col] for col in cols]])
        np.testing.assert_array_equal(result, expected)

    def test_get_decomp_allows_partial_group_coverage(self):
        """_get_decomp should fill missing group columns with 0, not require all groups."""
        from fuellib.gcm.gani import TABLE, _get_decomp

        cols = list(TABLE.columns)
        subset_cols = cols[:3]
        decomp_df = pd.DataFrame(
            [[5.0, 6.0, 7.0]], columns=subset_cols, index=["compoundB"]
        )
        fake_fuel = cast(Fuel, SimpleNamespace(gani_decomp=lambda: decomp_df))

        result = _get_decomp(fake_fuel)

        expected = np.zeros((1, len(cols)))
        expected[0, :3] = [5.0, 6.0, 7.0]
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
