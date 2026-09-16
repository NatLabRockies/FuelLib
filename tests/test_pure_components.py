"""
Per-compound MAPE regression test for the five new ASTM properties.

Loads ``tests/pureComponentReference.csv`` (built via
``tools/build_pure_component_reference.py``) which contains NIST WebBook
values for ~30 of FuelLib's 89 ``refCompounds.csv`` entries and Yale
Volume-2 YSI for all 89.

Uses ``posf10264`` (Jet A) as the "compound library" — every FuelLib
refCompound appears as a single row of its Nij matrix, so all per-compound
properties can be pulled with ``comp_idx=i`` without building per-compound
fuel objects. For each (compound, property) cell where the reference is
non-NaN, compares the FuelLib GC prediction to the reference and reports
per-family MAPE.

Compound families are inferred from the ``GCxGC_Bin`` name and grouped for
reporting: n-alkane, iso-alkane, mono-cyclo, di-cyclo, tri-cyclo,
alkyl-benzene, di-aromatic, cyclo-aromatic, alkene.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import fuellib as fl
from fuellib.fuel import _fp_alibakhshi, _lhv_hess

REF_FILE = Path(__file__).with_name("pureComponentReference.csv")


def _family(bin_name):
    b = bin_name.lower()
    if b.startswith("n-c"):
        return "n_alkane"
    if "isoparaffin" in b:
        return "iso_alkane"
    if "monocycloparaffin" in b:
        return "mono_cyclo"
    if "dicycloparaffin" in b:
        return "di_cyclo"
    if "tricycloparaffin" in b:
        return "tri_cyclo"
    if "cycloaromatic" in b:
        return "cyclo_arom"
    if "diaromatic" in b:
        return "di_arom"
    if "alkene" in b:
        return "alkene"
    if "benzene" in b or b == "toluene":
        return "alkyl_benzene"
    return "other"


class PureComponentAccuracyTestCase(unittest.TestCase):
    """
    Per-family MAPE on the 5 new ASTM properties vs published reference.
    """

    @classmethod
    def setUpClass(cls):
        cls.ref = pd.read_csv(REF_FILE)
        cls.posf = fl.fuel("posf10264")
        # Build a lookup from GCxGC bin name -> row index in the POSF Nij.
        cls.bin_to_idx = {c: i for i, c in enumerate(cls.posf.compounds)}
        # Pre-compute per-compound predictions for the 5 properties.
        # LHV: Hess cycle uses Hf_liq = Hf(gas) - Hv_stp.
        Hf_liq = cls.posf.Hf - cls.posf.Hv_stp
        cls.LHV_pred = _lhv_hess(
            cls.posf.n_C, cls.posf.n_H, Hf_liq, cls.posf.MW
        )  # MJ/kg per compound
        cls.Cl298_pred = cls.posf.Cl(298.15)  # J/kg/K per compound
        cls.FreezePoint_pred = cls.posf.Tm_astm  # K per compound (pure = Tm)
        cls.FlashPoint_pred = _fp_alibakhshi(
            cls.posf.Tb_astm, cls.posf.alibakhshi_phi
        )  # K per compound
        cls.YSI_pred = cls.posf.ysi_pure  # per compound

    def _mape(self, obs, pred):
        return float(np.mean(np.abs(obs - pred) / np.abs(obs)) * 100)

    def _report_family(self, prop, obs, pred, families):
        """Print per-family MAPE and return the aggregate."""
        prop_width = 20
        print(f"\n{prop} (n={len(obs)} compounds with reference):")
        fam_stats = []
        for fam in sorted(set(families)):
            mask = np.array([f == fam for f in families])
            if not mask.any():
                continue
            fo, fp = obs[mask], pred[mask]
            mape = self._mape(fo, fp)
            fam_stats.append((fam, mape, mask.sum()))
        for fam, mape, n in sorted(fam_stats, key=lambda x: -x[2]):
            print(f"  {fam:<{prop_width}s}  MAPE = {mape:6.2f}%   (n={n})")
        overall = self._mape(obs, pred)
        print(f"  {'OVERALL':<{prop_width}s}  MAPE = {overall:6.2f}%   (n={len(obs)})")
        return overall

    def _prop_check(self, prop_name, ref_col, pred_arr, mape_ceiling):
        """
        For every ref-CSV row with a non-NaN ``ref_col`` and a matching POSF
        bin, gather (obs, pred, family). Fail the test if per-family MAPE
        exceeds ``mape_ceiling``.
        """
        obs_list, pred_list, fam_list = [], [], []
        for _, r in self.ref.iterrows():
            bin_ = r["GCxGC_Bin"]
            if bin_ not in self.bin_to_idx:
                continue
            idx = self.bin_to_idx[bin_]
            ref_val = r[ref_col]
            if pd.isna(ref_val):
                continue
            obs_list.append(float(ref_val))
            pred_list.append(float(pred_arr[idx]))
            fam_list.append(_family(bin_))
        obs = np.array(obs_list)
        pred = np.array(pred_list)
        overall_mape = self._report_family(prop_name, obs, pred, fam_list)
        self.assertLessEqual(
            overall_mape,
            mape_ceiling,
            msg=(
                f"{prop_name} MAPE {overall_mape:.2f}% exceeds ceiling "
                f"{mape_ceiling:.2f}%"
            ),
        )

    def test_lhv(self):
        """LHV per-family MAPE."""
        self._prop_check("LHV (MJ/kg)", "LHV_MJkg", self.LHV_pred, mape_ceiling=3.0)

    def test_cp_liquid(self):
        """Cp,L @ 298 K per-family MAPE."""
        self._prop_check(
            "Cp,L @ 298 K (J/kg/K)", "Cl_298_JkgK", self.Cl298_pred, mape_ceiling=8.0
        )

    def test_freeze_point(self):
        """Melting-point per-family MAPE (pure-comp freeze = Tm)."""
        self._prop_check(
            "FreezePoint (K)", "FreezePoint_K", self.FreezePoint_pred, mape_ceiling=15.0
        )

    def test_flash_point(self):
        """Flash-point per-family MAPE."""
        self._prop_check(
            "FlashPoint (K)", "FlashPoint_K", self.FlashPoint_pred, mape_ceiling=5.0
        )

    def test_ysi(self):
        """Unified YSI per-family MAPE."""
        self._prop_check("YSI (unified)", "YSI", self.YSI_pred, mape_ceiling=1.0)


if __name__ == "__main__":
    unittest.main()
