"""Test hydrocarbon identification: nC, nH, and hc_type determination."""

import re

import pandas as pd
import pytest

import fuellib as fl
from fuellib.database.database import DEFAULT_DIR, NUM_C_COL

#: Expected `Fuel.hc_type` for each database family.
FAMILY_TO_HC_TYPE = {
    "n-alkane": "n-alkane",
    "iso-alkane": "iso-alkane",
    "alkene": "alkene",
    "monocyclic": "cyclo-alkane",
    "dicyclic": "cyclo-alkane",
    "tricyclic": "cyclo-alkane",
    "alkylbenzene": "aromatic",
    "cycloaromatic": "aromatic",
    "diaromatic": "aromatic",
}


def get_available_fuels():
    """Discover available fuels from the database gcData directory."""
    return sorted(f.stem for f in (DEFAULT_DIR / "gcData").glob("*.csv"))


def extract_c_h_from_formula(formula):
    """Extract carbon and hydrogen counts from formula string (e.g., C7H8 -> (7, 8))."""
    if not formula or pd.isna(formula):
        return None, None

    formula = str(formula).strip()
    # Match CnHm pattern
    match = re.match(r"C(\d+)H(\d+)", formula)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class TestHCIdentification:
    """Test suite for hydrocarbon type, carbon, and hydrogen identification."""

    @pytest.mark.parametrize("fuel_name", get_available_fuels())
    def test_hc_identification(self, fuel_name):
        """Comprehensive test for HC identification: nC, nH, hc_type, and compound classification."""
        # Load fuel
        fuel = fl.Fuel(fuel_name)

        print(f"\n{'=' * 60}")
        print(f"Fuel: {fuel_name}")
        print(f"{'=' * 60}")
        print(f"Compounds: {fuel.num_compounds}")

        # === Test 1: nC matches reference formula ===
        mismatches = []
        formulas = (
            fuel.formulas if fuel.formulas is not None else [None] * fuel.num_compounds
        )
        for compound, formula, nc_calc in zip(fuel.compounds, formulas, fuel.nC):
            if not formula or pd.isna(formula):
                continue

            nc_ref, _ = extract_c_h_from_formula(formula)

            if nc_ref is not None:
                tolerance = 2.0 if "Cycloaromatic" in compound else 0.1

                if abs(nc_calc - nc_ref) > tolerance:
                    mismatches.append(
                        f"{compound}: calculated nC={nc_calc:.1f}, expected nC={nc_ref}"
                    )

        assert not mismatches, "Carbon count mismatches:\n" + "\n".join(mismatches)
        print("✓ nC from decomp matches reference formula")

        # === Test 2: nH matches reference formula ===
        mismatches = []
        formulas = (
            fuel.formulas if fuel.formulas is not None else [None] * fuel.num_compounds
        )
        for compound, formula, nh_calc in zip(fuel.compounds, formulas, fuel.nH):
            if not formula or pd.isna(formula):
                continue

            _, nh_ref = extract_c_h_from_formula(formula)

            if nh_ref is not None:
                tolerance = 2.0 if "Cycloaromatic" in compound else 0.1

                if abs(nh_calc - nh_ref) > tolerance:
                    mismatches.append(
                        f"{compound}: calculated nH={nh_calc:.1f}, expected nH={nh_ref}"
                    )

        assert not mismatches, "Hydrogen count mismatches:\n" + "\n".join(mismatches)
        print("✓ nH from decomp matches reference formula")

        # === Test 3: hc_type is consistent ===
        valid_types = {"n-alkane", "iso-alkane", "cyclo-alkane", "alkene", "aromatic"}

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if hc_type not in valid_types:
                mismatches.append(
                    f"{compound}: invalid hc_type='{hc_type}' "
                    f"(must be one of {valid_types})"
                )

        assert not mismatches, "Invalid hydrocarbon types:\n" + "\n".join(mismatches)
        print("✓ hc_type from decomp is consistent")

        # === Test 4: hc_type matches the database family ===
        mismatches = [
            f"{compound}: hc_type='{hc_type}', family='{family}'"
            for compound, hc_type, family in zip(
                fuel.compounds, fuel.hc_type, fuel.families, strict=True
            )
            if FAMILY_TO_HC_TYPE[family] != hc_type
        ]
        assert not mismatches, "Hydrocarbon type mismatches:\n" + "\n".join(mismatches)
        print("✓ hc_type matches database families")

        # === Test 5: nC matches the database ===
        mismatches = [
            f"{compound}: nC={nc}, database Num_C={num_c}"
            for compound, nc, num_c in zip(
                fuel.compounds, fuel.nC, fuel.data[NUM_C_COL], strict=True
            )
            if nc != num_c
        ]
        assert not mismatches, "Carbon count mismatches:\n" + "\n".join(mismatches)
        print("✓ nC matches database Num_C")
