"""
Tests for the family-resolved fusion thermodynamics (ASTM-2) feeding
``fuel.freeze_point``.

Context (docs/IMPLEMENTATION_LOG_astm_improvements.md, ASTM-2): Walden's
rule (dSfus = 56.5 J/mol/K for everything) was replaced by per-family
linear correlations (gcmTableData/fusion_families.csv), and the eq-21
mixing-entropy scaling default moved alpha 0.25 -> 1.0 (classical ideal
SLE; 0.25 was only ever valid in tandem with Walden's too-small dSfus).

Interim tolerances: mixture freeze points are now biased LOW by the CG
melting-point error (pure-compound freeze == CG Tm by construction, and
CG Tm runs ~6-26 K low vs NIST). The Tb/Tm experimental anchoring
(ASTM-3) removes that bias; the mixture assertions here are widened
accordingly and must be tightened when ASTM-3 lands.
"""

import unittest

import numpy as np

from fuellib import fuel

# NIST fusion enthalpies at Tm (kJ/mol) — well-established values.
NIST_DHFUS_KJMOL = {
    "n-C07": 14.0,
    "n-C10": 28.7,
    "n-C12": 36.8,
    "n-C16": 53.4,
}

# Experimental freeze points (K): POSF refs from tutorials/astmProperties.py.
# posf11498 (C-1 ATJ) is deliberately absent: its tutorial reference (240 K =
# -33 C) cannot be an ATJ freeze point (would not even qualify as jet fuel;
# real C-1 freezes far below the -47 C spec limit) — part of the reference
# label-scramble documented in docs/IMPLEMENTATION_LOG_astm_improvements.md.
# It gets a spec-limit assertion instead.
FREEZE_REFS_K = {
    "posf10264": 226.0,
    "posf10325": 226.0,
    "posf10289": 219.0,
}


class TestFusionConstants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = fuel("posf10325")

    def test_nalkane_dhfus_vs_nist(self):
        """dHfus within 20% of NIST (residual gap = CG Tm error, ASTM-3)."""
        for comp, ref in NIST_DHFUS_KJMOL.items():
            i = self.f.compounds.index(comp)
            pred = self.f.dHfus[i] / 1e3
            self.assertAlmostEqual(
                pred,
                ref,
                delta=0.20 * ref,
                msg=f"{comp}: dHfus {pred:.1f} vs NIST {ref} kJ/mol",
            )

    def test_walden_no_longer_global(self):
        """n-C12 dSfus must reflect measured ~140 J/mol/K, not Walden 56.5."""
        i = self.f.compounds.index("n-C12")
        self.assertGreater(self.f.dSfus[i], 120.0)
        self.assertLess(self.f.dSfus[i], 155.0)

    def test_family_ordering_and_floor(self):
        """Branched/ring families below n-alkane at same C; global floor."""
        by = {c: i for i, c in enumerate(self.f.compounds)}
        self.assertGreater(
            self.f.dSfus[by["n-C12"]], self.f.dSfus[by["C12-Isoparaffin"]]
        )
        self.assertGreater(
            self.f.dSfus[by["n-C10"]], self.f.dSfus[by["C10-Monocycloparaffin"]]
        )
        self.assertTrue(np.all(self.f.dSfus >= 20.0))

    def test_families_classified(self):
        """Every posf10325 compound must have matched a family (no Walden)."""
        unknown = [
            c for c, fam in zip(self.f.compounds, self.f.bin_family) if fam == "unknown"
        ]
        self.assertEqual(unknown, [])


class TestFreezePoint(unittest.TestCase):
    def test_pure_compound_freeze_equals_Tm(self):
        """Model identity: a pure compound freezes at its own Tm (alpha-free)."""
        f = fuel("heptane")
        i = int(np.argmax(f.Y_0))
        self.assertAlmostEqual(f.freeze_point(), float(f.Tm_astm[i]), delta=0.2)

    def test_posf_mixtures(self):
        """Post-ASTM-3 (Tb/Tm anchoring): freeze within [-10, +5] K of the
        Edwards references (measured: -7.6/-3.1/-0.1/-4.7 K)."""
        for name, ref in FREEZE_REFS_K.items():
            fz = fuel(name).freeze_point()
            self.assertGreater(fz, ref - 10.0, msg=f"{name}: {fz:.1f} K")
            self.assertLess(fz, ref + 5.0, msg=f"{name}: {fz:.1f} K")

    def test_atj_freeze_below_spec_limit(self):
        """C-1 ATJ must freeze below the Jet-A spec limit (226.15 K); no
        trustworthy point reference exists (see FREEZE_REFS_K note)."""
        self.assertLess(fuel("posf11498").freeze_point(), 226.15)

    def test_pure_nalkane_freeze_vs_nist(self):
        """Anchored Tm makes pure n-alkane fuels freeze at NIST values."""
        for name, ref in [("heptane", 182.6), ("decane", 243.5), ("dodecane", 263.6)]:
            fz = fuel(name).freeze_point()
            self.assertAlmostEqual(fz, ref, delta=1.0, msg=f"{name}: {fz:.1f} K")

    def test_alpha_walden_pairing_documented(self):
        """The historic (Walden-paired) alpha=0.25 remains callable and gives
        a HIGHER freeze point than alpha=1 (less depression)."""
        f = fuel("posf10325")
        self.assertGreater(f.freeze_point(alpha=0.25), f.freeze_point(alpha=1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
