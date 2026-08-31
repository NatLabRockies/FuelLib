"""
Build fuelData/unifacDecomposition/<name>.csv for every fuel in fuelData/gcData/.

Run after editing the SMILES table or adding fuels.

Usage:
    python tools/build_unifac_decompositions.py

The script resolves each compound's name (Reference Compound column, falling back
to Compound column for single-component fuels) to a SMILES string via NAME_TO_SMILES.
Tricyclic compounds whose PelePhysics Key column already holds a SMILES string are
used directly. Every resulting decomposition is cross-checked against the molecular
formula derived from the SMILES; mismatches abort the build.
"""

import csv
import os
import sys

# Ensure the FuelLib repo root is on sys.path so we can import paths.py.
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import pandas as pd
from rdkit import Chem

import paths
from decompose_unifac import decompose, verify_formula, UnsupportedGroupError

# --- Compound name → SMILES map ---
#
# Names match the "Reference Compound" column (or "Compound" column for single-
# component fuels). Whitespace is trimmed and case is preserved as written in the
# gcData CSVs. Where the gcData typo'd a name ("2-ethly tetralin", "2-pentyldacalin")
# the typo is preserved as the key so the lookup succeeds.
NAME_TO_SMILES = {
    # --- n-paraffins (linear alkanes) ---
    "n-heptane": "CCCCCCC",
    "n-octane": "CCCCCCCC",
    "n-nonane": "CCCCCCCCC",
    "n-decane": "CCCCCCCCCC",
    "n-undecane": "CCCCCCCCCCC",
    "n-dodecane": "CCCCCCCCCCCC",
    "n-tridecane": "CCCCCCCCCCCCC",
    "n-tetradecane": "CCCCCCCCCCCCCC",
    "n-pentadecane": "CCCCCCCCCCCCCCC",
    "n-hexadecane (cetane)": "CCCCCCCCCCCCCCCC",
    "n-heptadecane": "CCCCCCCCCCCCCCCCC",
    "n-octadecane": "CCCCCCCCCCCCCCCCCC",
    # Single-component fuels use a different naming convention.
    "NC7H16": "CCCCCCC",
    "NC10H22": "CCCCCCCCCC",
    "NC12H26": "CCCCCCCCCCCC",
    # --- iso-paraffins (2-methyl-alkanes) ---
    "2-methyl hexane": "CC(C)CCCC",
    "2-methyl heptane": "CC(C)CCCCC",
    "2-methyl octane": "CC(C)CCCCCC",
    "2-methyl nonane": "CC(C)CCCCCCC",
    "2-methyl decane": "CC(C)CCCCCCCC",
    "2-methyl undecane": "CC(C)CCCCCCCCC",
    "2-methyl dodecane": "CC(C)CCCCCCCCCC",
    "2-methyl tridecane": "CC(C)CCCCCCCCCCC",
    "2-methyl tetradecane": "CC(C)CCCCCCCCCCCC",
    "2-methyl pentadecane": "CC(C)CCCCCCCCCCCCC",
    "2-methyl hexadecane": "CC(C)CCCCCCCCCCCCCC",
    "2-methyl heptadecane": "CC(C)CCCCCCCCCCCCCCC",
    "2-methyl octadecane": "CC(C)CCCCCCCCCCCCCCCC",
    "2-methyl nonadecane": "CC(C)CCCCCCCCCCCCCCCCC",
    "2-methyltricosane": "CC(C)CCCCCCCCCCCCCCCCCCCCC",  # C24
    # --- alpha-olefins (terminal alkenes) ---
    "1-dodecene": "C=CCCCCCCCCCC",
    "1-hexadecene": "C=CCCCCCCCCCCCCCC",
    # --- alkyl benzenes (C6H5-R) ---
    "toluene": "Cc1ccccc1",
    "ethyl benzene": "CCc1ccccc1",
    "propyl benzene": "CCCc1ccccc1",
    "butyl benzene": "CCCCc1ccccc1",
    "pentyl benzene": "CCCCCc1ccccc1",
    "hexyl benzene": "CCCCCCc1ccccc1",
    "heptyl benzene": "CCCCCCCc1ccccc1",
    "octyl benzene": "CCCCCCCCc1ccccc1",
    "nonyl benzene": "CCCCCCCCCc1ccccc1",
    # --- naphthalenes ---
    "naphthalene": "c1ccc2ccccc2c1",
    "1-methyl naphthalene": "Cc1cccc2ccccc12",
    "1-ethyl naphthalene": "CCc1cccc2ccccc12",
    "1-propyl naphthalene": "CCCc1cccc2ccccc12",
    # --- alicyclic-aromatic fused (indane, tetralins) ---
    "indane": "C1Cc2ccccc2C1",
    "tetralin": "C1CCc2ccccc2C1",
    "2-methyl tetralin": "CC1CCc2ccccc2C1",
    "2-ethly tetralin": "CCC1CCc2ccccc2C1",  # gcData typo: "ethly"
    "2-propyl tetralin": "CCCC1CCc2ccccc2C1",
    "2-butyl tetralin": "CCCCC1CCc2ccccc2C1",
    # --- alkyl cyclohexanes ---
    "methyl cyclohexane": "CC1CCCCC1",
    "ethyl cyclohexane": "CCC1CCCCC1",
    "propyl cyclohexane": "CCCC1CCCCC1",
    "butyl cyclohexane": "CCCCC1CCCCC1",
    "pentyl cyclohexane": "CCCCCC1CCCCC1",
    "hexyl cyclohexane": "CCCCCCC1CCCCC1",
    "heptyl cyclohexane": "CCCCCCCC1CCCCC1",
    "octyl cyclohexane": "CCCCCCCCC1CCCCC1",
    "nonyl cyclohexane": "CCCCCCCCCC1CCCCC1",
    "decyl cyclohexane": "CCCCCCCCCCC1CCCCC1",
    "undecyl cyclohexane": "CCCCCCCCCCCC1CCCCC1",
    # --- bicyclic naphthenes (decalin family + hydrindane + octahydropentalene) ---
    "Octahydropentalene": "C1CCC2CCCC12",  # cis/trans-bicyclo[3.3.0]octane
    "Hydrindane": "C1CCC2CCCC2C1",  # cis/trans-bicyclo[4.3.0]nonane
    "Decalin": "C1CCC2CCCCC2C1",  # cis/trans-bicyclo[4.4.0]decane
    "2-methyldecalin": "CC1CCC2CCCCC2C1",
    "2-ethyldecalin": "CCC1CCC2CCCCC2C1",
    "2-propyldecalin": "CCCC1CCC2CCCCC2C1",
    "2-butyldecalin": "CCCCC1CCC2CCCCC2C1",
    "2-pentyldacalin": "CCCCCC1CCC2CCCCC2C1",  # gcData typo: "dacalin"
    # --- tricyclic naphthenes (SMILES taken directly from PelePhysics Key column) ---
    "C1CC2C(C1)C1CCCC21": "C1CC2C(C1)C1CCCC21",
    "C1CC2CC3CCCC3C2C1": "C1CC2CC3CCCC3C2C1",
    "C1CC2CC3CCCC3CC2C1": "C1CC2CC3CCCC3CC2C1",
}


def get_compound_name(row):
    """
    Pick the most informative compound identifier from a gcData row.

    :param row: One row of the gcData CSV as a dict.
    :type row: dict
    :return: The lookup name (Reference Compound if present, else Compound).
    :rtype: str
    """
    name = row.get("Reference Compound") or row.get("Compound") or ""
    return name.strip()


def load_subgroup_columns():
    """
    Return ordered subgroup metadata (113 entries) used to build CSV headers.

    Column headers in ``fuelData/unifacDecomposition/<name>.csv`` are the integer
    ``Subgroup_No`` values (always unique), not the human-readable
    ``Subgroup_Name`` (two subgroups share the name ``CHO``, which would collide
    on a ``pd.read_csv`` roundtrip).

    :return: Tuple ``(subgroup_numbers, subgroup_names)`` where ``subgroup_numbers``
             is a list of strings (header labels) and ``subgroup_names`` is the
             matching list of human-readable names in the same order.
    :rtype: tuple[list[str], list[str]]
    """
    df = pd.read_csv(paths.UNIFAC_SUBGROUP_FILE)
    numbers = [str(int(n)) for n in df["Subgroup_No"].tolist()]
    names = df["Subgroup_Name"].tolist()
    return numbers, names


def build_one_fuel(fuel_name, gc_path, out_path, subgroup_numbers, subgroup_names):
    """
    Build one fuelData/unifacDecomposition/<name>.csv from the gcData rows.

    :param fuel_name: Fuel identifier (e.g., 'heptane', 'posf10325').
    :type fuel_name: str
    :param gc_path: Path to fuelData/gcData/<name>_init.csv.
    :type gc_path: str
    :param out_path: Path to fuelData/unifacDecomposition/<name>.csv to write.
    :type out_path: str
    :param subgroup_numbers: Ordered list of 113 ``Subgroup_No`` strings, used as
                              column headers (always unique).
    :type subgroup_numbers: list[str]
    :param subgroup_names: Ordered list of 113 subgroup names parallel to
                            ``subgroup_numbers`` — used to map decompose()
                            output (keyed by name) into the right column.
    :type subgroup_names: list[str]
    :return: Number of compounds decomposed.
    :rtype: int
    :raises KeyError: If a compound name is not in NAME_TO_SMILES.
    :raises AssertionError: If formula balance fails for any compound.
    """
    with open(gc_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    # Map from subgroup name → column header (subgroup number as string).
    # When two subgroups share the same name (e.g., "CHO"), only the first
    # occurrence's column is used by the decomposer; the second's column stays
    # at 0. The classifier in decompose_unifac.py emits only the first-occurrence
    # names by construction (it never generates "CHO" for ether-like groups).
    name_to_header = {}
    for header, name in zip(subgroup_numbers, subgroup_names):
        name_to_header.setdefault(name, header)

    out_rows = []
    for r in rows:
        name = get_compound_name(r)
        if name not in NAME_TO_SMILES:
            raise KeyError(
                f"{fuel_name}: no SMILES for compound {name!r}. "
                "Add it to NAME_TO_SMILES in this script."
            )
        smiles = NAME_TO_SMILES[name]
        counts = decompose(smiles)
        ok, msg = verify_formula(smiles, counts)
        assert ok, f"{fuel_name}/{name}: formula balance failed — {msg}"
        compound_col = r.get("Compound", name).strip()
        row_out = {header: 0 for header in subgroup_numbers}
        for sg_name, n in counts.items():
            header = name_to_header[sg_name]
            row_out[header] = n
        row_out["Compound"] = compound_col
        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows, columns=["Compound"] + subgroup_numbers)
    out_df.to_csv(out_path, index=False)
    return len(out_rows)


def main():
    """
    Build all 13 fuelData/unifacDecomposition/<name>.csv files.

    :return: None.
    :rtype: NoneType
    """
    subgroup_numbers, subgroup_names = load_subgroup_columns()
    assert len(subgroup_numbers) == 113, "Expected 113 UNIFAC subgroups."

    gc_dir = paths.FUELDATA_GC_DIR
    out_dir = paths.FUELDATA_UNIFAC_DIR
    os.makedirs(out_dir, exist_ok=True)

    fuel_files = sorted(f for f in os.listdir(gc_dir) if f.endswith("_init.csv"))
    summary = []
    for fname in fuel_files:
        fuel_name = fname.replace("_init.csv", "")
        gc_path = os.path.join(gc_dir, fname)
        out_path = os.path.join(out_dir, f"{fuel_name}.csv")
        try:
            n = build_one_fuel(
                fuel_name, gc_path, out_path, subgroup_numbers, subgroup_names
            )
            print(f"  ✓ {fuel_name:<18} {n:>3} compounds → {out_path}")
            summary.append((fuel_name, n, "ok"))
        except (KeyError, UnsupportedGroupError, AssertionError) as e:
            print(f"  ✗ {fuel_name:<18} FAILED: {e}")
            summary.append((fuel_name, 0, f"failed: {e}"))

    n_ok = sum(1 for _, _, s in summary if s == "ok")
    print(f"\n{n_ok}/{len(summary)} fuels built successfully.")
    if n_ok < len(summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
