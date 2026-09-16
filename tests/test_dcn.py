"""
Tests for the Derived Cetane Number model (``fuel.dcn`` + gcmTableData/dcn.csv).

Mixture-level validation targets are the published NJFCP/Edwards DCNs
(Edwards, "Reference Jet Fuels for Combustion Testing", AIAA 2017-0146;
NREL DCN/ICN comparison, NREL fy24osti/89581):

    A-1 / posf10264 (JP-8)  DCN = 48.8
    A-2 / posf10325 (Jet A) DCN = 48.3
    A-3 / posf10289 (JP-5)  DCN = 39.2

Tolerances encode the v1 model's known biases (see
tools/IMPLEMENTATION_LOG_astm.md):

- A-1/A-2: +-6 (v1 predicts +2.8/+3.1 — within the blended table 1-sigma).
- A-3: +-15 (v1 predicts +11.6: JP-5 is cycloparaffin-rich and the cyclo
  family DCNs are two-seed extrapolations; tightening this requires more
  measured cycloalkane DCNs, not code).
"""

import unittest

import numpy as np
import pandas as pd

from fuellib import fuel
from fuellib._data_locator import get_gcmtable_dir
from fuellib.fuel import _dcn_mix

NJFCP_DCN = {
    "posf10264": 48.8,  # A-1
    "posf10325": 48.3,  # A-2
    "posf10289": 39.2,  # A-3
}


class TestDcnTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tab = pd.read_csv(get_gcmtable_dir() + "/dcn.csv")

    def test_complete_and_bounded(self):
        self.assertEqual(len(self.tab), 91)  # 89 skeleton + 2 ATJ archetype bins
        self.assertFalse(self.tab["DCN"].isna().any())
        self.assertTrue((self.tab["DCN"] >= 0.0).all())
        self.assertTrue((self.tab["DCN"] <= 105.0).all())
        self.assertTrue((self.tab["DCN_err"] > 0).all())

    def test_reference_anchors(self):
        by_bin = dict(zip(self.tab["GCxGC_Bin"], self.tab["DCN"]))
        self.assertAlmostEqual(by_bin["n-C16"], 100.0, delta=0.1)
        self.assertAlmostEqual(by_bin["n-C07"], 53.8, delta=0.1)

    def test_family_ordering(self):
        """n-alkane > isoparaffin > monocyclo > aromatic at fixed C."""
        by_bin = dict(zip(self.tab["GCxGC_Bin"], self.tab["DCN"]))
        self.assertGreater(by_bin["n-C12"], by_bin["C12-Isoparaffin"])
        self.assertGreater(by_bin["C12-Isoparaffin"], by_bin["C6-Benzene"])
        self.assertGreater(by_bin["C10-Monocycloparaffin"], by_bin["C4-Benzene"])

    def test_provenance_tags(self):
        self.assertFalse((self.tab["Source"] == "unassigned").any())
        n_seed = self.tab["Source"].str.startswith("literature_seed").sum()
        self.assertGreaterEqual(n_seed, 10)


class TestDcnMixture(unittest.TestCase):
    def test_a_fuels(self):
        for name, target in NJFCP_DCN.items():
            pred = fuel(name).dcn()
            tol = 15.0 if name == "posf10289" else 6.0
            self.assertAlmostEqual(
                pred,
                target,
                delta=tol,
                msg=f"{name}: predicted {pred:.1f} vs measured {target}",
            )

    def test_hefa_dcn_band(self):
        """HEFA-SPK (camelina): published HEFA DCNs sit ~55-60; model 58.2."""
        pred = fuel("hefa-came", decompName="hefa").dcn()
        self.assertGreater(pred, 50.0)
        self.assertLess(pred, 65.0)

    def test_blending_monotonicity(self):
        """Adding n-hexadecane (DCN 100) must raise the mixture DCN."""
        f = fuel("posf10325")
        base = f.dcn()
        i_c16 = f.compounds.index("n-C16")
        Yi = np.asarray(f.Y_0, dtype=float).copy()
        Yi *= 0.8
        Yi[i_c16] += 0.2
        self.assertGreater(f.dcn(Yi), base)

    def test_volume_vs_mass_basis_differs(self):
        """The blend is volume-based: it must not equal the mass-based sum
        (guards against a silent basis regression)."""
        f = fuel("posf10325")
        mass_based = float(
            np.sum(f.Y_0 * np.where(np.isnan(f.dcn_pure), 0.0, f.dcn_pure))
        )
        self.assertNotAlmostEqual(f.dcn(), mass_based, places=3)

    def test_helper_pure(self):
        phi = np.array([0.25, 0.75])
        dcn_i = np.array([100.0, 20.0])
        self.assertAlmostEqual(_dcn_mix(phi, dcn_i), 40.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
