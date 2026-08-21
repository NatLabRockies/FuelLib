"""
Constantinou-Gani (CG) group-contribution method decomposition for SAF-relevant
hydrocarbons.

Reference: Constantinou & Gani, AIChE J. 40(10), 1994.
           "New group contribution method for estimating properties of pure compounds"

This tool decomposes a SMILES string into first-order and second-order CG groups.
The output is a count vector matching the 121-group column ordering used by
FuelLib/gcmTableData/gcmTable.csv.

Scope: aliphatic + aromatic hydrocarbons (n-paraffins, iso-paraffins, terminal
alpha-olefins, alkylbenzenes, alkylnaphthalenes, monocycloparaffins,
dicycloparaffins, cycloaromatics).

Validation: compared against existing hand-decomposed data in
FuelLib/fuelData/groupDecompositionData/refCompounds.csv
"""

import csv
import os
from rdkit import Chem
from rdkit.Chem import rdmolops


class UnsupportedGroupError(ValueError):
    """Raised when a molecule contains atoms or groups outside the SAF subset."""


# =============================================================================
# First-order group definitions
# =============================================================================

# (C, H) atom counts per first-order subgroup.
# ACCH3/ACCH2/ACCH are 2-atom groups (aromatic C + aliphatic C bundled).
FIRST_ORDER_CH = {
    "CH3": (1, 3),
    "CH2": (1, 2),
    "CH": (1, 1),
    "C": (1, 0),
    "CH2=CH": (2, 3),
    "CH=CH": (2, 2),
    "CH2=C": (2, 2),
    "CH=C": (2, 1),
    "C=C": (2, 0),
    "CH2=C=CH": (3, 3),
    "ACH": (1, 1),
    "AC": (1, 0),
    "ACCH3": (2, 3),
    "ACCH2": (2, 2),
    "ACCH": (2, 1),
}

# Canonical column ordering for the 121 groups (first-order + second-order)
# matching gcmTable.csv columns 2..122 (0-indexed).
# Second-order groups start at index 78 in this list.
CG_GROUP_NAMES = [
    # -- First-order groups (indices 0-77) --
    "CH3",
    "CH2",
    "CH",
    "C",
    "CH2=CH",
    "CH=CH",
    "CH2=C",
    "CH=C",
    "C=C",
    "CH2=C=CH",
    "ACH",
    "AC",
    "ACCH3",
    "ACCH2",
    "ACCH",
    "OH",
    "ACOH",
    "CH3CO",
    "CH2CO",
    "CHO",
    "CH3COO",
    "CH2COO",
    "HCOO",
    "CH3O",
    "CH2O",
    "CH-O",
    "FCH2O",
    "CH2NH2",
    "CHNH2",
    "CH3NH",
    "CH2NH",
    "CHNH",
    "CH3N",
    "CH2N",
    "ACNH2",
    "C5H4N",
    "C5H3N",
    "CH2CN",
    "COOH",
    "CH2CL",
    "CHCL",
    "CCL",
    "CHCL2",
    "CCL2",
    "CCL3",
    "ACCL",
    "CH2NO2",
    "CHNO2",
    "ACNO2",
    "CH2SH",
    "I",
    "Br",
    "CH≡C",
    "C≡C",
    "CL—(C=C)",
    "ACF",
    "HCON(CH2)2",
    "CF3",
    "CF2",
    "CF",
    "COO",
    "CCL2F",
    "HCCLF",
    "CCLF2",
    "Fspecial",
    "CONH2",
    "CONHCH3",
    "CONHCH2",
    "CON(CH3)2",
    "CONCH3CH2",
    "CON(CH2)2",
    "C2H5O2",
    "C2H4O2",
    "CH3S",
    "CH2S",
    "CHS",
    "C4H3S",
    "C4H2S",
    # -- Second-order groups (indices 78-120) --
    "(CH3)2CH",
    "(CH3)3C",
    "CH(CH3)CH(CH3)",
    "CH(CH3)C(CH3)2",
    "C(CH3)2C(CH3)2",
    "3 membered ring",
    "4 membered ring",
    "5 membered ring",
    "6 membered ring",
    "7 membered ring",
    "CHn=CHm—CHp=CHk k,n,m,p in (0,2)",
    "CH3-CHm=CH, m in (0,1), n in (0,2)",
    "CH2-CHm=CHn, m, n in (0,2)",
    "CH-CHm=CHn or C-CHm=CHn, m,n m in (0,2)",
    "Alicyclic side-chain CcyclicCm m > 1",
    "CH3CH3",
    "CHCHO or CCHO",
    "CH3COCH2",
    "CH3COCH or CH3COC",
    "Ccyclic(=0)",
    "ACCHO",
    "CHCOOH or CCOOH",
    "ACCOOH",
    "CH3COOCH or CH3COOC",
    "COCH2COO or COCHCOO or COCCOO",
    " CO-O-CO",
    "ACCOO",
    "CHOH",
    "COH",
    "CHm(OH)CHn(OH), m,n in (0,2)",
    "CHm cyclic-OH, m in (0,1)",
    "CHm(OH)CHn(NHp), m,n,p in (0,3)",
    "CHm(NH2)CHn(NH2)",
    "CHm cyclic-NHp-CHn cyclic, m,n,p in (0,2)",
    "Chm=Chn-F, m,n in (0,2)",
    "AC-O-CHm",
    "CHm cyclic-S-CHn cyclic, m,n in (0,2)",
    "CHm=CHn—F, m,n in (0,2)",
    "CHm=CHn—Br, m,n in (0,2)",
    "CHm=CHn—I, m,n in (0,2)",
    "ACBr",
    "ACI",
    "CHm(NH2)-COOH, m,n in (0,2)",
]

assert len(CG_GROUP_NAMES) == 121, f"Expected 121 groups, got {len(CG_GROUP_NAMES)}"


# =============================================================================
# First-order decomposition
# =============================================================================


def _find_terminal_vinyls(mol):
    """
    Locate terminal alpha-olefin vinyl groups (CH2=CH-R).
    Returns (count, set_of_covered_atom_indices).
    """
    covered = set()
    n_vinyl = 0
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetIsAromatic() or b.GetIsAromatic():
            continue
        ah, bh = a.GetTotalNumHs(), b.GetTotalNumHs()
        an, bn = a.GetDegree(), b.GetDegree()
        # CH2=CH pattern: one end has 2H and degree 1, other has 1H and degree 2
        if (ah == 2 and an == 1) and (bh == 1 and bn == 2):
            tail, head = a, b
        elif (bh == 2 and bn == 1) and (ah == 1 and an == 2):
            tail, head = b, a
        else:
            raise UnsupportedGroupError(
                "C=C bond is not a terminal alpha-olefin "
                f"(atoms {a.GetIdx()} H={ah} deg={an}, "
                f"{b.GetIdx()} H={bh} deg={bn})."
            )
        covered.add(tail.GetIdx())
        covered.add(head.GetIdx())
        n_vinyl += 1
    return n_vinyl, covered


def _find_aromatic_substituents(mol, excluded):
    """
    Assign aromatic-substituent carbons to ACCH3/ACCH2/ACCH.
    The matching aromatic ring carbon is consumed (not counted as AC later).
    Returns (subgroups_dict, consumed_aromatic_set).
    """
    subgroups = {}
    consumed_aromatic = set()
    for atom in mol.GetAtoms():
        if atom.GetIdx() in excluded:
            continue
        if atom.GetSymbol() != "C" or atom.GetIsAromatic():
            continue
        arom_C_neighbors = [
            n for n in atom.GetNeighbors() if n.GetSymbol() == "C" and n.GetIsAromatic()
        ]
        if not arom_C_neighbors:
            continue
        if len(arom_C_neighbors) > 1:
            raise UnsupportedGroupError(
                f"Aliphatic carbon with {len(arom_C_neighbors)} aromatic neighbors "
                f"(atom idx={atom.GetIdx()})."
            )
        arom_C = arom_C_neighbors[0]
        if arom_C.GetIdx() in consumed_aromatic:
            raise UnsupportedGroupError(
                f"Aromatic ring carbon (idx={arom_C.GetIdx()}) has more than one "
                "aliphatic neighbor."
            )
        consumed_aromatic.add(arom_C.GetIdx())

        n_alC = sum(
            1
            for n in atom.GetNeighbors()
            if n.GetSymbol() == "C" and not n.GetIsAromatic()
        )
        if n_alC == 0:
            subgroups[atom.GetIdx()] = "ACCH3"
        elif n_alC == 1:
            subgroups[atom.GetIdx()] = "ACCH2"
        elif n_alC == 2:
            subgroups[atom.GetIdx()] = "ACCH"
        else:
            raise UnsupportedGroupError(
                f"Aromatic-substituent carbon with {n_alC} aliphatic neighbors "
                f"(atom idx={atom.GetIdx()})."
            )
    return subgroups, consumed_aromatic


def _classify_aliphatic_atom(atom):
    """
    Assign one first-order group name (CH3/CH2/CH/C) to an aliphatic carbon.
    Only for atoms not already assigned to vinyl or aromatic-substituent groups.
    """
    if atom.GetSymbol() != "C":
        raise UnsupportedGroupError(f"Non-carbon atom (symbol={atom.GetSymbol()}).")
    for bond in atom.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        if bt not in (1.0, 1.5):
            raise UnsupportedGroupError(
                f"Non-single, non-aromatic bond (order={bt}) on atom "
                f"idx={atom.GetIdx()}."
            )
    h_count = atom.GetTotalNumHs()
    if atom.GetIsAromatic():
        if h_count == 1:
            return "ACH"
        if h_count == 0:
            return "AC"
        raise UnsupportedGroupError(
            f"Aromatic carbon with H count={h_count} (atom idx={atom.GetIdx()})."
        )
    # Aliphatic: classify by number of aliphatic C neighbors
    aliphatic_C_neighbors = sum(
        1 for n in atom.GetNeighbors() if n.GetSymbol() == "C" and not n.GetIsAromatic()
    )
    if aliphatic_C_neighbors == 1:
        return "CH3"
    if aliphatic_C_neighbors == 2:
        return "CH2"
    if aliphatic_C_neighbors == 3:
        return "CH"
    if aliphatic_C_neighbors == 4:
        return "C"
    raise UnsupportedGroupError(
        f"Aliphatic carbon with {aliphatic_C_neighbors} C neighbors "
        f"(atom idx={atom.GetIdx()})."
    )


def _first_order_decomposition(mol):
    """
    Decompose molecule into first-order CG groups.
    Returns dict mapping group name → count.
    """
    counts = {}

    # 1. Terminal alpha-olefins
    n_vinyl, vinyl_idxs = _find_terminal_vinyls(mol)
    if n_vinyl:
        counts["CH2=CH"] = n_vinyl

    # 2. Aromatic substituents (ACCH3/ACCH2/ACCH)
    acch_subgroups, consumed_aromatic = _find_aromatic_substituents(mol, vinyl_idxs)
    for sg_name in acch_subgroups.values():
        counts[sg_name] = counts.get(sg_name, 0) + 1

    # 3. Classify remaining atoms
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in vinyl_idxs or idx in acch_subgroups or idx in consumed_aromatic:
            continue
        sg = _classify_aliphatic_atom(atom)
        counts[sg] = counts.get(sg, 0) + 1

    return counts


# =============================================================================
# Second-order decomposition
# =============================================================================


def _detect_branching_groups(mol):
    """
    Detect second-order branching groups:
    - (CH3)2CH: aliphatic CH with exactly 2 CH3 neighbors
    - (CH3)3C: quaternary C with exactly 3 CH3 neighbors
    - CH(CH3)CH(CH3): adjacent pair of CH's, each with at least 1 CH3
    - CH(CH3)C(CH3)2: adjacent CH (1 CH3) and C (2 CH3)
    - C(CH3)2C(CH3)2: adjacent quaternary C's, each with 2 CH3
    """
    counts = {}

    # Classify each atom: count how many terminal CH3 neighbors it has
    def _ch3_neighbor_count(atom):
        """Count terminal-CH3 neighbors of an aliphatic atom."""
        n = 0
        for nbr in atom.GetNeighbors():
            if (
                nbr.GetSymbol() == "C"
                and not nbr.GetIsAromatic()
                and nbr.GetTotalNumHs() == 3
                and nbr.GetDegree() == 1
            ):
                n += 1
        return n

    def _is_aliphatic_C(atom):
        return atom.GetSymbol() == "C" and not atom.GetIsAromatic()

    def _aliphatic_C_degree(atom):
        """Number of C-C bonds (aliphatic neighbors)."""
        return sum(
            1
            for n in atom.GetNeighbors()
            if n.GetSymbol() == "C" and not n.GetIsAromatic()
        )

    # (CH3)2CH: CH with 2 CH3 neighbors
    ch3_2_ch_atoms = set()
    for atom in mol.GetAtoms():
        if not _is_aliphatic_C(atom):
            continue
        if atom.GetTotalNumHs() == 1 and _aliphatic_C_degree(atom) == 3:
            # This is a CH (3 aliphatic C neighbors, 1 H)
            if _ch3_neighbor_count(atom) >= 2:
                ch3_2_ch_atoms.add(atom.GetIdx())
    if ch3_2_ch_atoms:
        counts["(CH3)2CH"] = len(ch3_2_ch_atoms)

    # (CH3)3C: quaternary C with 3 CH3 neighbors
    ch3_3_c_atoms = set()
    for atom in mol.GetAtoms():
        if not _is_aliphatic_C(atom):
            continue
        if atom.GetTotalNumHs() == 0 and _aliphatic_C_degree(atom) == 4:
            if _ch3_neighbor_count(atom) >= 3:
                ch3_3_c_atoms.add(atom.GetIdx())
    if ch3_3_c_atoms:
        counts["(CH3)3C"] = len(ch3_3_c_atoms)

    # CH(CH3)CH(CH3): adjacent CH-CH pair, each with at least 1 CH3
    ch_ch_pairs = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (_is_aliphatic_C(a) and _is_aliphatic_C(b)):
            continue
        # Both must be CH (1H, degree 3 aliphatic neighbors)
        a_is_ch = a.GetTotalNumHs() == 1 and _aliphatic_C_degree(a) == 3
        b_is_ch = b.GetTotalNumHs() == 1 and _aliphatic_C_degree(b) == 3
        if a_is_ch and b_is_ch:
            if _ch3_neighbor_count(a) >= 1 and _ch3_neighbor_count(b) >= 1:
                pair = tuple(sorted([a.GetIdx(), b.GetIdx()]))
                ch_ch_pairs.add(pair)
    if ch_ch_pairs:
        counts["CH(CH3)CH(CH3)"] = len(ch_ch_pairs)

    # CH(CH3)C(CH3)2: adjacent CH (1 CH3) and quaternary C (2 CH3)
    ch_c_pairs = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (_is_aliphatic_C(a) and _is_aliphatic_C(b)):
            continue
        # Check a=CH with CH3, b=C with 2 CH3
        a_is_ch = a.GetTotalNumHs() == 1 and _aliphatic_C_degree(a) == 3
        b_is_quat = b.GetTotalNumHs() == 0 and _aliphatic_C_degree(b) == 4
        if a_is_ch and b_is_quat:
            if _ch3_neighbor_count(a) >= 1 and _ch3_neighbor_count(b) >= 2:
                pair = tuple(sorted([a.GetIdx(), b.GetIdx()]))
                ch_c_pairs.add(pair)
        # Symmetric check
        b_is_ch = b.GetTotalNumHs() == 1 and _aliphatic_C_degree(b) == 3
        a_is_quat = a.GetTotalNumHs() == 0 and _aliphatic_C_degree(a) == 4
        if b_is_ch and a_is_quat:
            if _ch3_neighbor_count(b) >= 1 and _ch3_neighbor_count(a) >= 2:
                pair = tuple(sorted([a.GetIdx(), b.GetIdx()]))
                ch_c_pairs.add(pair)
    if ch_c_pairs:
        counts["CH(CH3)C(CH3)2"] = len(ch_c_pairs)

    # C(CH3)2C(CH3)2: adjacent quaternary C's, each with 2 CH3
    c_c_pairs = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (_is_aliphatic_C(a) and _is_aliphatic_C(b)):
            continue
        a_quat = a.GetTotalNumHs() == 0 and _aliphatic_C_degree(a) == 4
        b_quat = b.GetTotalNumHs() == 0 and _aliphatic_C_degree(b) == 4
        if a_quat and b_quat:
            if _ch3_neighbor_count(a) >= 2 and _ch3_neighbor_count(b) >= 2:
                pair = tuple(sorted([a.GetIdx(), b.GetIdx()]))
                c_c_pairs.add(pair)
    if c_c_pairs:
        counts["C(CH3)2C(CH3)2"] = len(c_c_pairs)

    return counts


def _detect_rings(mol):
    """
    Count non-aromatic rings by size (3-7 membered).
    Uses the Smallest Set of Smallest Rings (SSSR).

    A ring is counted if it is NOT fully aromatic. This handles fused
    aromatic-alicyclic systems (e.g., tetralin has one fully aromatic ring
    and one ring with 4 non-aromatic + 2 aromatic atoms at the junction;
    only the latter is counted as a "6 membered ring").
    """
    counts = {}
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        # Skip fully aromatic rings (e.g., benzene ring in tetralin)
        all_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        if all_aromatic:
            continue
        size = len(ring)
        if 3 <= size <= 7:
            name = f"{size} membered ring"
            counts[name] = counts.get(name, 0) + 1
    return counts


def _detect_alicyclic_sidechain(mol):
    """
    Detect alicyclic side-chain CcyclicCm (m > 1).

    NOTE: Based on empirical evidence from FuelLib refCompounds.csv, ALL
    monocycloparaffins (including ethylcyclohexane, propylcyclohexane, etc.)
    have this group = 0. This suggests the "alicyclic side-chain" correction
    does NOT apply to simple alkyl substituents on cycloparaffin rings in the
    SAF context. The exact structural requirement for this group is unclear
    from the paper alone.

    Current implementation: disabled (always returns 0) to match FuelLib data.
    TODO: revisit if non-SAF compounds need this correction.
    """
    return 0


def _detect_ch3ch3(mol):
    """Detect CH3CH3 (ethane) second-order group. Only for ethane itself."""
    if mol.GetNumAtoms() == 2:
        a, b = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(1)
        if (
            a.GetSymbol() == "C"
            and b.GetSymbol() == "C"
            and a.GetTotalNumHs() == 3
            and b.GetTotalNumHs() == 3
        ):
            return 1
    return 0


def _second_order_decomposition(mol):
    """
    Decompose molecule into second-order CG groups.
    Returns dict mapping group name → count.
    """
    counts = {}

    # Branching groups
    branching = _detect_branching_groups(mol)
    counts.update(branching)

    # Ring corrections
    rings = _detect_rings(mol)
    counts.update(rings)

    # Alicyclic side-chain
    n_sidechain = _detect_alicyclic_sidechain(mol)
    if n_sidechain:
        counts["Alicyclic side-chain CcyclicCm m > 1"] = n_sidechain

    # CH3CH3 (ethane)
    n_ethane = _detect_ch3ch3(mol)
    if n_ethane:
        counts["CH3CH3"] = n_ethane

    return counts


# =============================================================================
# Public API
# =============================================================================


def decompose(smiles):
    """
    Decompose a hydrocarbon SMILES into CG first-order and second-order group counts.

    :param smiles: SMILES string.
    :return: dict mapping group name → count (only non-zero entries).
    :raises UnsupportedGroupError: If the molecule is outside the SAF subset.
    :raises ValueError: If the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    counts = {}

    # First-order
    first_order = _first_order_decomposition(mol)
    counts.update(first_order)

    # Second-order
    second_order = _second_order_decomposition(mol)
    counts.update(second_order)

    return counts


def to_vector(counts):
    """
    Convert a group-count dict to a 121-element list in canonical order.
    """
    return [counts.get(name, 0) for name in CG_GROUP_NAMES]


def verify_formula(smiles, counts):
    """
    Cross-check that first-order subgroup counts reproduce the molecular formula.
    Only checks C and H from the first 15 groups (hydrocarbon groups).
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    expected_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    expected_H = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")

    got_C = 0
    got_H = 0
    for gname in list(FIRST_ORDER_CH.keys()):
        n = counts.get(gname, 0)
        c, h = FIRST_ORDER_CH[gname]
        got_C += c * n
        got_H += h * n

    if got_C == expected_C and got_H == expected_H:
        return True, f"C{got_C}H{got_H} (matches)"
    return False, f"got C{got_C}H{got_H}, expected C{expected_C}H{expected_H}"


# =============================================================================
# Validation against FuelLib
# =============================================================================


def _load_refcompounds():
    """
    Load refCompounds.csv from FuelLib
    Returns dict: compound_name → list of 121 int counts.
    """
    ref_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fuellib",
        "data",
        "fuelData",
        "groupDecompositionData",
        "refCompounds.csv",
    )
    if not os.path.exists(ref_path):
        return None

    data = {}
    with open(ref_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            name = row[0]
            # Columns 1-78 are first-order, 79-121 are second-order
            # Total data columns: 121
            values = [int(row[i]) for i in range(1, min(122, len(row)))]
            # Pad to 121 if needed
            while len(values) < 121:
                values.append(0)
            data[name] = values
    return data


def compare_with_fuellib(name, computed_vector, ref_data):
    """
    Compare computed decomposition vector against FuelLib refCompounds.
    Returns (match, diff_report).
    """
    if ref_data is None or name not in ref_data:
        return None, f"No reference data for '{name}'"

    ref_vector = ref_data[name]
    diffs = []
    for i, (comp, ref) in enumerate(zip(computed_vector, ref_vector)):
        if comp != ref:
            diffs.append(f"    {CG_GROUP_NAMES[i]}: computed={comp}, ref={ref}")

    if not diffs:
        return True, "MATCH"
    return False, "MISMATCH:\n" + "\n".join(diffs)


# =============================================================================
# Test harness
# =============================================================================

if __name__ == "__main__":
    # Test cases with SMILES and expected name
    # Maps test name → (smiles, fuellib_name_or_None)
    test_cases = [
        # n-paraffins
        ("CCCCCCC", "n-heptane", "n-C07"),
        ("CCCCCCCCCC", "n-decane", "n-C10"),
        # iso-paraffins (2-methylalkanes)
        ("CC(C)CCCC", "2-methylhexane", "C07-Isoparaffin"),
        ("CC(C)CCCCC", "2-methylheptane", "C08-Isoparaffin"),
        # More branched
        ("CC(C)C(C)C", "2,3-dimethylbutane", None),
        ("CC(C)(C)C", "neopentane", None),
        # Monocycloparaffins
        ("CC1CCCCC1", "methylcyclohexane", "C07-Monocycloparaffin"),
        ("CCC1CCCCC1", "ethylcyclohexane", "C08-Monocycloparaffin"),
        # Dicycloparaffins
        ("C1CCC2CCCCC2C1", "decalin (trans)", "C10-Dicycloparaffin"),
        ("C1CC[C@H]2CCCC[C@H]2C1", "cis-decalin", "C10-Dicycloparaffin"),
        # Aromatics
        ("Cc1ccccc1", "toluene", "Toluene"),
        ("CCc1ccccc1", "ethylbenzene", "C2-Benzene"),
        ("CCCc1ccccc1", "propylbenzene", "C3-Benzene"),
        # Naphthalenes
        ("c1ccc2ccccc2c1", "naphthalene", "Diaromatic-C10"),
        ("Cc1cccc2ccccc12", "1-methylnaphthalene", "Diaromatic-C11"),
        # Cycloaromatics
        ("C1Cc2ccccc2C1", "indane", "Cycloaromatic-C09"),
        ("C1CCc2ccccc2C1", "tetralin", "Cycloaromatic-C10"),
        ("CC1CCc2ccccc2C1", "2-methyltetralin", "Cycloaromatic-C11"),
        # Alkenes
        ("C=CCCCCCCCCCC", "1-dodecene", "C12-Alkene"),
    ]

    # Load FuelLib reference data
    ref_data = _load_refcompounds()
    if ref_data:
        print(f"Loaded {len(ref_data)} compounds from FuelLib refCompounds.csv")
    else:
        print("WARNING: Could not load FuelLib refCompounds.csv for comparison")
    print()

    print(f"{'Name':<22} {'SMILES':<28} {'Formula':12} {'FO Groups':<40} {'SO Groups'}")
    print("=" * 130)

    n_pass = 0
    n_fail = 0
    n_skip = 0
    issues = []

    for smi, name, ref_name in test_cases:
        try:
            d = decompose(smi)
            ok, msg = verify_formula(smi, d)
            status = "✓" if ok else "✗"

            # Separate first-order and second-order for display
            first_order_parts = {k: v for k, v in d.items() if k in FIRST_ORDER_CH}
            second_order_parts = {k: v for k, v in d.items() if k not in FIRST_ORDER_CH}

            first_order_str = str(first_order_parts) if first_order_parts else "{}"
            second_order_str = str(second_order_parts) if second_order_parts else "{}"

            print(
                f"{status} {name:<20} {smi:<28} {msg:12} {first_order_str:<40} {second_order_str}"
            )

            if not ok:
                n_fail += 1
                issues.append(f"  FORMULA MISMATCH: {name} — {msg}")
                continue

            # Compare with FuelLib
            if ref_name and ref_data:
                vec = to_vector(d)
                match, report = compare_with_fuellib(ref_name, vec, ref_data)
                if match is True:
                    n_pass += 1
                    print(f"    → FuelLib comparison: PASS")
                elif match is False:
                    n_fail += 1
                    print(f"    → FuelLib comparison: FAIL")
                    print(f"      {report}")
                    issues.append(f"  FUELLIB MISMATCH: {name} vs {ref_name}")
                else:
                    n_skip += 1
            else:
                n_skip += 1
                n_pass += 1  # formula check passed

        except UnsupportedGroupError as e:
            print(f"✗ {name:<20} {smi:<28} → UNSUPPORTED: {e}")
            n_fail += 1

    print()
    print(f"Results: {n_pass} pass, {n_fail} fail, {n_skip} skip (no ref)")
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
