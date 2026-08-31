"""
UNIFAC subgroup decomposition for SAF-relevant hydrocarbons.

helper for generating ``fuelData/unifacDecomposition/<name>.csv``
files. Requires RDKit (install with ``pip install rdkit``).

Scope: aliphatic + aromatic hydrocarbons whose subgroups lie in UNIFAC main groups
1 (CH2), 2 (C=C, terminal alpha-olefins only), 3 (ACH/AC), 4 (ACCH2). These cover
all SAF jet-fuel surrogate molecules: n-paraffins, iso-paraffins, terminal
alpha-olefins, alkylbenzenes, alkylnaphthalenes, monocycloparaffins,
dicycloparaffins (decalin family), and aromatic-alicyclic fused systems (indane,
tetralin family).

UNIFAC subgroup conventions (Magnussen 1981; same as the existing FuelLib GCM
decomposition files):
- ``ACCH3``, ``ACCH2``, ``ACCH`` are **2-atom subgroups** containing one
  aromatic ring carbon AND its directly-bonded aliphatic substituent carbon
  together. Their R/Q values equal AC + (CH3/CH2/CH).
- ``AC`` is **only** used for aromatic ring carbons that have no H AND no
  aliphatic substituent (i.e., ring-fusion junctions like the 4a,8a positions
  of naphthalene).
- ``ACH`` is each aromatic ring carbon with one H.

A compound containing atoms or groups outside this subset raises
``UnsupportedGroupError`` so the build script flags it instead of silently
producing wrong counts.
"""

import os
import sys

try:
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "RDKit is required for the UNIFAC decomposition tool. "
        "Install with: pip install rdkit"
    ) from e


class UnsupportedGroupError(ValueError):
    """Raised when a molecule contains atoms or groups outside the SAF subset."""


# Subgroup number per Magnussen 1981 / UNIFAC 2.0 ``unifac_subgroups.csv``.
# Names match the ``Subgroup_Name`` column verbatim.
SUBGROUP_NUMBERS = {
    "CH3": 1,
    "CH2": 2,
    "CH": 3,
    "C": 4,
    "CH2=CH": 5,  # terminal alpha-olefin vinyl group
    "ACH": 9,
    "AC": 10,
    "ACCH3": 11,
    "ACCH2": 12,
    "ACCH": 13,
}


# (C, H) atom counts per subgroup, encoding the convention above.
SUBGROUP_CH = {
    "CH3": (1, 3),
    "CH2": (1, 2),
    "CH": (1, 1),
    "C": (1, 0),
    "CH2=CH": (2, 3),  # =CH2 + =CH-, 2 carbons total
    "ACH": (1, 1),
    "AC": (1, 0),
    "ACCH3": (2, 3),  # AC + CH3 bundled
    "ACCH2": (2, 2),  # AC + CH2 bundled
    "ACCH": (2, 1),  # AC + CH bundled
}


def _find_terminal_vinyls(mol):
    """
    Locate terminal alpha-olefin vinyl groups (CH2=CH-R) in a molecule.

    :param mol: RDKit molecule (implicit-H form).
    :type mol: rdkit.Chem.rdchem.Mol
    :return: Tuple ``(vinyl_count, covered_atom_idxs)`` — number of CH2=CH groups
             found and the set of atom indices they cover.
    :rtype: tuple[int, set[int]]
    :raises UnsupportedGroupError: If a C=C bond is internal, di-substituted, or
                                    otherwise not a clean terminal vinyl.
    """
    covered = set()
    n_vinyl = 0
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetIsAromatic() or b.GetIsAromatic():
            continue  # aromatic ring bond, not olefinic
        ah, bh = a.GetTotalNumHs(), b.GetTotalNumHs()
        an, bn = a.GetDegree(), b.GetDegree()
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
    Assign each aliphatic-carbon aromatic substituent its ACCH3/ACCH2/ACCH subgroup.

    The matching aromatic ring carbon is marked as "consumed" so it is not
    later counted as an AC.

    :param mol: RDKit molecule (implicit-H form).
    :type mol: rdkit.Chem.rdchem.Mol
    :param excluded: Atom indices already assigned to other subgroups (e.g.,
                     vinyl atoms). Atoms in this set are skipped.
    :type excluded: set[int]
    :return: Tuple ``(subgroups, consumed_aromatic)`` where ``subgroups`` is a
             dict mapping substituent atom index → subgroup name, and
             ``consumed_aromatic`` is the set of aromatic ring atom indices
             that have been bundled into an ACCH_x.
    :rtype: tuple[dict[int, str], set[int]]
    :raises UnsupportedGroupError: If an aliphatic carbon bonds to more than one
                                    aromatic carbon, or if the substituent has
                                    too many aliphatic neighbors (quaternary
                                    aromatic substituent is outside the SAF subset).
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
                f"(atom idx={atom.GetIdx()}). Bridge carbon between aromatic rings "
                "is not in the SAF subset."
            )
        arom_C = arom_C_neighbors[0]
        if arom_C.GetIdx() in consumed_aromatic:
            raise UnsupportedGroupError(
                f"Aromatic ring carbon (idx={arom_C.GetIdx()}) has more than one "
                "aliphatic neighbor — not in the SAF subset."
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
                f"(atom idx={atom.GetIdx()}). Quaternary aromatic substituent is "
                "not in the SAF subset."
            )
    return subgroups, consumed_aromatic


def _classify_atom(atom):
    """
    Assign one UNIFAC subgroup name to a single atom whose subgroup is one of
    ACH, AC, CH3, CH2, CH, C. Aromatic-substituent carbons (ACCH3/2/H) and vinyl
    atoms are handled separately by :func:`_find_aromatic_substituents` and
    :func:`_find_terminal_vinyls`; this function must not be called on them.

    :param atom: RDKit atom (must be carbon).
    :type atom: rdkit.Chem.rdchem.Atom
    :return: Subgroup name.
    :rtype: str
    :raises UnsupportedGroupError: If the atom or its bonds cannot be mapped.
    """
    if atom.GetSymbol() != "C":
        raise UnsupportedGroupError(
            f"Non-carbon atom encountered (symbol={atom.GetSymbol()}). "
            "Only hydrocarbons are supported."
        )

    for bond in atom.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        if bt not in (1.0, 1.5):
            raise UnsupportedGroupError(
                f"Non-single, non-aromatic bond (order={bt}) on atom "
                f"idx={atom.GetIdx()}. Internal alkenes and alkynes are not in "
                "the SAF subset."
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


def decompose(smiles):
    """
    Decompose a hydrocarbon SMILES into UNIFAC subgroup counts.

    :param smiles: SMILES string of the molecule.
    :type smiles: str
    :return: Mapping from subgroup name to integer count
             (e.g., ``{"CH3": 2, "CH2": 5}``).
    :rtype: dict[str, int]
    :raises UnsupportedGroupError: If the molecule contains atoms or groups
                                    outside the SAF subset (see module docstring).
    :raises ValueError: If the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    counts = {}

    # 1. Peel off terminal alpha-olefins as CH2=CH groups.
    n_vinyl, vinyl_idxs = _find_terminal_vinyls(mol)
    if n_vinyl:
        counts["CH2=CH"] = n_vinyl

    # 2. Assign aromatic-substituent carbons to ACCH3/ACCH2/ACCH and consume
    #    their aromatic ring partners.
    accH_subgroups, consumed_aromatic = _find_aromatic_substituents(mol, vinyl_idxs)
    for sg_name in accH_subgroups.values():
        counts[sg_name] = counts.get(sg_name, 0) + 1

    # 3. Classify remaining atoms (ACH, AC, CH3/CH2/CH/C).
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in vinyl_idxs or idx in accH_subgroups or idx in consumed_aromatic:
            continue
        sg = _classify_atom(atom)
        counts[sg] = counts.get(sg, 0) + 1

    return counts


def verify_formula(smiles, subgroup_counts):
    """
    Cross-check that subgroup counts reproduce the molecular formula (C, H only).

    :param smiles: SMILES string of the molecule.
    :type smiles: str
    :param subgroup_counts: Mapping from subgroup name to count
                             (as returned by :func:`decompose`).
    :type subgroup_counts: dict[str, int]
    :return: Tuple ``(ok, message)`` where ``ok`` is ``True`` iff the subgroup
             counts match the molecular formula derived from the SMILES.
    :rtype: tuple[bool, str]
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    expected_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    expected_H = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")

    got_C = sum(SUBGROUP_CH[sg][0] * n for sg, n in subgroup_counts.items())
    got_H = sum(SUBGROUP_CH[sg][1] * n for sg, n in subgroup_counts.items())

    if got_C == expected_C and got_H == expected_H:
        return True, f"C{got_C}H{got_H} (matches)"
    return False, f"got C{got_C}H{got_H}, expected C{expected_C}H{expected_H}"


if __name__ == "__main__":
    test_cases = [
        ("CCCCCCC", "n-heptane"),
        ("CCCCCCCCCC", "n-decane"),
        ("CC(C)CCCC", "2-methylhexane"),
        ("Cc1ccccc1", "toluene"),
        ("CCc1ccccc1", "ethylbenzene"),
        ("CCCCCc1ccccc1", "pentylbenzene"),
        ("CC1CCCCC1", "methylcyclohexane"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("Cc1cccc2ccccc12", "1-methylnaphthalene"),
        ("C1CCc2ccccc2C1", "tetralin"),
        ("C1Cc2ccccc2C1", "indane"),
        ("CC1CCc2ccccc2C1", "2-methyltetralin"),
        ("C1CCC2CCCCC2C1", "decalin"),
        ("C1CC[C@H]2CCCC[C@H]2C1", "cis-decalin"), 
        ("CC1CCC2CCCCC2C1", "2-methyldecalin"),
        ("C=CCCCCCCCCCC", "1-dodecene"),
        ("C=CCCCCCCCCCCCCCC", "1-hexadecene"),
    ]
    for smi, name in test_cases:
        try:
            d = decompose(smi)
            ok, msg = verify_formula(smi, d)
            status = "✓" if ok else "✗"
            print(f"{status} {name:<22} {smi:<25} → {d}  [{msg}]")
        except UnsupportedGroupError as e:
            print(f"✗ {name:<22} {smi:<25} → UNSUPPORTED: {e}")
