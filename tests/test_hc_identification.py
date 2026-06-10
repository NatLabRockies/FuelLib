"""Test hydrocarbon identification: nC, nH, and hc_type determination."""

import re
from pathlib import Path
import pytest
import pandas as pd
import fuellib as fl


def get_available_fuels():
    """Discover available fuels from fuelData/gcData directory."""
    gcdata_dir = fl.get_fueldata_gc_dir()
    fuels = sorted(
        [f.name.replace("_init.csv", "") for f in Path(gcdata_dir).glob("*_init.csv")]
    )
    return fuels


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


def infer_hc_type_from_formula(formula, compound_name=""):
    """Infer expected hydrocarbon type from formula."""
    if not formula or pd.isna(formula):
        return None

    nc, nh = extract_c_h_from_formula(formula)
    if nc is None:
        return None

    # Check for aromatic patterns (diaromatics, cycloaromatics, benzenes)
    if "Diaromatic" in compound_name or "aromatic" in compound_name.lower():
        return "aromatic"
    if "Benzene" in compound_name or "Toluene" in compound_name:
        return "aromatic"

    # Check for alkene patterns
    if "Alkene" in compound_name:
        return "alkene"

    # Check for cycloalkane patterns
    if "Cycloparaffin" in compound_name or "Cycloaromatic" in compound_name:
        # These have aromatic rings attached, classified as aromatic
        if "Cycloaromatic" in compound_name:
            return "aromatic"
        return "cyclo-alkane"

    # For alkanes, check saturation
    # Alkanes: CnH(2n+2)
    if nh == 2 * nc + 2:
        # Check if it's iso or n based on name
        if "Isoparaffin" in compound_name or "iso" in compound_name.lower():
            return "iso-alkane"
        elif "n-C" in compound_name or compound_name.startswith("n-"):
            return "n-alkane"
        else:
            # Default branched alkanes to iso-alkane
            return "iso-alkane"

    # Monocycloalkanes: CnH(2n)
    if "Monocyclo" in compound_name and nh == 2 * nc:
        return "cyclo-alkane"

    # Dicycloalkanes: CnH(2n-2)
    if "Dicyclo" in compound_name and nh == 2 * nc - 2:
        return "cyclo-alkane"

    # Tricycloalkanes: CnH(2n-4)
    if "Tricyclo" in compound_name and nh == 2 * nc - 4:
        return "cyclo-alkane"

    return None


class TestHCIdentification:
    """Test suite for hydrocarbon type, carbon, and hydrogen identification."""

    @pytest.mark.parametrize("fuel_name", get_available_fuels())
    def test_hc_identification(self, fuel_name):
        """Comprehensive test for HC identification: nC, nH, hc_type, and compound classification."""
        # Load fuel
        fuel = fl.fuel(fuel_name)

        print(f"\n{'='*60}")
        print(f"Fuel: {fuel_name}")
        print(f"{'='*60}")
        print(f"Compounds: {fuel.num_compounds}")

        # === Test 1: nC matches reference formula ===
        mismatches = []
        for compound, formula, nc_calc in zip(fuel.compounds, fuel.formulas, fuel.nC):
            if not formula or pd.isna(formula):
                continue

            nc_ref, _ = extract_c_h_from_formula(formula)

            if nc_ref is not None:
                tolerance = 2.0 if "Cycloaromatic" in compound else 0.1

                if abs(nc_calc - nc_ref) > tolerance:
                    mismatches.append(
                        f"{compound}: calculated nC={nc_calc:.1f}, expected nC={nc_ref}"
                    )

        assert not mismatches, f"Carbon count mismatches:\n" + "\n".join(mismatches)
        print(f"✓ nC from decomp matches reference formula")

        # === Test 2: nH matches reference formula ===
        mismatches = []
        for compound, formula, nh_calc in zip(fuel.compounds, fuel.formulas, fuel.nH):
            if not formula or pd.isna(formula):
                continue

            _, nh_ref = extract_c_h_from_formula(formula)

            if nh_ref is not None:
                tolerance = 2.0 if "Cycloaromatic" in compound else 0.1

                if abs(nh_calc - nh_ref) > tolerance:
                    mismatches.append(
                        f"{compound}: calculated nH={nh_calc:.1f}, expected nH={nh_ref}"
                    )

        assert not mismatches, f"Hydrogen count mismatches:\n" + "\n".join(mismatches)
        print(f"✓ nH from decomp matches reference formula")

        # === Test 3: hc_type is consistent ===
        valid_types = {"n-alkane", "iso-alkane", "cyclo-alkane", "alkene", "aromatic"}

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if hc_type not in valid_types:
                mismatches.append(
                    f"{compound}: invalid hc_type='{hc_type}' "
                    f"(must be one of {valid_types})"
                )

        assert not mismatches, f"Invalid hydrocarbon types:\n" + "\n".join(mismatches)
        print(f"✓ hc_type from decomp is consistent")

        # === Test 4: hc_type matches formula-derived expectations ===
        mismatches = []
        for compound, formula, hc_type_calc in zip(
            fuel.compounds, fuel.formulas, fuel.hc_type
        ):
            if not formula or pd.isna(formula):
                continue

            hc_type_expected = infer_hc_type_from_formula(formula, compound)

            if hc_type_expected is not None:
                if hc_type_calc != hc_type_expected:
                    mismatches.append(
                        f"{compound}: calculated hc_type='{hc_type_calc}', "
                        f"expected hc_type='{hc_type_expected}' "
                        f"(formula: {formula})"
                    )

        assert not mismatches, f"Hydrocarbon type mismatches:\n" + "\n".join(mismatches)
        print(f"✓ hc_type from decomp matches formula-derived expectations")

        # === Test 5: Aromatic compounds identified ===
        aromatic_names = {
            compound
            for compound, formula in zip(fuel.compounds, fuel.formulas)
            if formula
            and (
                "Benzene" in compound
                or "aromatic" in compound.lower()
                or "naphthalene" in formula.lower()
            )
        }

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if compound in aromatic_names:
                if hc_type != "aromatic":
                    mismatches.append(
                        f"{compound}: should be aromatic but got '{hc_type}'"
                    )

        assert (
            not mismatches
        ), f"Aromatic compounds not correctly identified:\n" + "\n".join(mismatches)
        aromatic_count = len(aromatic_names) if aromatic_names else 0
        print(f"✓ aromatic compounds identified correctly ({aromatic_count} found)")

        # === Test 6: n-alkane compounds identified ===
        nalkane_names = {
            compound for compound in fuel.compounds if compound.startswith("n-C")
        }

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if compound in nalkane_names:
                if hc_type != "n-alkane":
                    mismatches.append(
                        f"{compound}: should be n-alkane but got '{hc_type}'"
                    )

        assert (
            not mismatches
        ), f"n-alkane compounds not correctly identified:\n" + "\n".join(mismatches)
        nalkane_count = len(nalkane_names) if nalkane_names else 0
        print(f"✓ n-alkane compounds identified correctly ({nalkane_count} found)")

        # === Test 7: Cycloalkane compounds identified ===
        cyclo_names = {
            compound
            for compound in fuel.compounds
            if "cycloparaffin" in compound.lower()
            and "aromatic" not in compound.lower()
        }

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if compound in cyclo_names:
                if hc_type != "cyclo-alkane":
                    mismatches.append(
                        f"{compound}: should be cyclo-alkane but got '{hc_type}'"
                    )

        assert (
            not mismatches
        ), f"Cycloalkane compounds not correctly identified:\n" + "\n".join(mismatches)
        cyclo_count = len(cyclo_names) if cyclo_names else 0
        print(f"✓ cyclo-alkane compounds identified correctly ({cyclo_count} found)")

        # === Test 7b: iso-alkane compounds identified ===
        isoalkane_names = {
            compound for compound in fuel.compounds if "Isoparaffin" in compound
        }

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if compound in isoalkane_names:
                if hc_type != "iso-alkane":
                    mismatches.append(
                        f"{compound}: should be iso-alkane but got '{hc_type}'"
                    )

        assert (
            not mismatches
        ), f"iso-alkane compounds not correctly identified:\n" + "\n".join(mismatches)
        isoalkane_count = len(isoalkane_names) if isoalkane_names else 0
        print(f"✓ iso-alkane compounds identified correctly ({isoalkane_count} found)")

        # === Test 8: Alkene compounds identified ===
        alkene_names = {compound for compound in fuel.compounds if "Alkene" in compound}

        mismatches = []
        for compound, hc_type in zip(fuel.compounds, fuel.hc_type):
            if compound in alkene_names:
                if hc_type != "alkene":
                    mismatches.append(
                        f"{compound}: should be alkene but got '{hc_type}'"
                    )

        assert (
            not mismatches
        ), f"Alkene compounds not correctly identified:\n" + "\n".join(mismatches)
        alkene_count = len(alkene_names) if alkene_names else 0
        print(f"✓ alkene compounds identified correctly ({alkene_count} found)")
