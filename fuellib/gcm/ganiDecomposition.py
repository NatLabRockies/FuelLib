"""RDKit-based Constantinou-Gani hydrocarbon group decomposition."""

from rdkit import Chem
from rdkit.Chem import Mol


class UnsupportedGroupError(ValueError):
    """Raised when a molecule lies outside the supported hydrocarbon subset."""


def _find_terminal_vinyls(mol: Mol) -> tuple[int, set[int]]:
    covered: set[int] = set()
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        if first.GetIsAromatic() or second.GetIsAromatic():
            continue
        first_is_tail = first.GetTotalNumHs() == 2 and first.GetDegree() == 1
        second_is_head = second.GetTotalNumHs() == 1 and second.GetDegree() == 2
        second_is_tail = second.GetTotalNumHs() == 2 and second.GetDegree() == 1
        first_is_head = first.GetTotalNumHs() == 1 and first.GetDegree() == 2
        if not ((first_is_tail and second_is_head) or (second_is_tail and first_is_head)):
            raise UnsupportedGroupError("C=C bond is not a terminal alpha-olefin.")
        covered.update((first.GetIdx(), second.GetIdx()))
        count += 1
    return count, covered


def _find_aromatic_substituents(
    mol: Mol, excluded: set[int]
) -> tuple[dict[int, str], set[int]]:
    groups: dict[int, str] = {}
    consumed_aromatic: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetIdx() in excluded or atom.GetSymbol() != "C" or atom.GetIsAromatic():
            continue
        aromatic_neighbors = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() == "C" and neighbor.GetIsAromatic()
        ]
        if not aromatic_neighbors:
            continue
        if len(aromatic_neighbors) != 1:
            raise UnsupportedGroupError("Aliphatic carbon has multiple aromatic neighbors.")
        aromatic_atom = aromatic_neighbors[0]
        if aromatic_atom.GetIdx() in consumed_aromatic:
            raise UnsupportedGroupError("Aromatic carbon has multiple aliphatic neighbors.")
        consumed_aromatic.add(aromatic_atom.GetIdx())
        aliphatic_neighbors = sum(
            neighbor.GetSymbol() == "C" and not neighbor.GetIsAromatic()
            for neighbor in atom.GetNeighbors()
        )
        try:
            groups[atom.GetIdx()] = {0: "ACCH3", 1: "ACCH2", 2: "ACCH"}[aliphatic_neighbors]
        except KeyError as error:
            raise UnsupportedGroupError("Unsupported aromatic substituent.") from error
    return groups, consumed_aromatic


def _classify_aliphatic_atom(atom) -> str:
    if atom.GetSymbol() != "C":
        raise UnsupportedGroupError(f"Non-carbon atom: {atom.GetSymbol()}.")
    if any(bond.GetBondTypeAsDouble() not in (1.0, 1.5) for bond in atom.GetBonds()):
        raise UnsupportedGroupError("Unsupported non-single, non-aromatic bond.")
    hydrogens = atom.GetTotalNumHs()
    if atom.GetIsAromatic():
        if hydrogens == 1:
            return "ACH"
        if hydrogens == 0:
            return "AC"
        raise UnsupportedGroupError("Unsupported aromatic carbon hydrogen count.")
    neighbors = sum(
        neighbor.GetSymbol() == "C" and not neighbor.GetIsAromatic()
        for neighbor in atom.GetNeighbors()
    )
    try:
        return {1: "CH3", 2: "CH2", 3: "CH", 4: "C"}[neighbors]
    except KeyError as error:
        raise UnsupportedGroupError("Unsupported aliphatic carbon degree.") from error


def _first_order_decomposition(mol: Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    vinyl_count, vinyl_atoms = _find_terminal_vinyls(mol)
    if vinyl_count:
        counts["CH2=CH"] = vinyl_count
    substituents, consumed_aromatic = _find_aromatic_substituents(mol, vinyl_atoms)
    for group in substituents.values():
        counts[group] = counts.get(group, 0) + 1
    for atom in mol.GetAtoms():
        if atom.GetIdx() in vinyl_atoms | set(substituents) | consumed_aromatic:
            continue
        group = _classify_aliphatic_atom(atom)
        counts[group] = counts.get(group, 0) + 1
    return counts


def _is_aliphatic_carbon(atom) -> bool:
    return atom.GetSymbol() == "C" and not atom.GetIsAromatic()


def _aliphatic_carbon_degree(atom) -> int:
    return sum(_is_aliphatic_carbon(neighbor) for neighbor in atom.GetNeighbors())


def _methyl_neighbors(atom) -> int:
    return sum(
        _is_aliphatic_carbon(neighbor)
        and neighbor.GetTotalNumHs() == 3
        and neighbor.GetDegree() == 1
        for neighbor in atom.GetNeighbors()
    )


def _detect_branching_groups(mol: Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    is_ch = lambda atom: (
        _is_aliphatic_carbon(atom)
        and atom.GetTotalNumHs() == 1
        and _aliphatic_carbon_degree(atom) == 3
    )
    is_quaternary = lambda atom: (
        _is_aliphatic_carbon(atom)
        and atom.GetTotalNumHs() == 0
        and _aliphatic_carbon_degree(atom) == 4
    )
    two_methyl_ch = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if is_ch(atom) and _methyl_neighbors(atom) >= 2
    }
    if two_methyl_ch:
        counts["(CH3)2CH"] = len(two_methyl_ch)
    three_methyl_c = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if is_quaternary(atom) and _methyl_neighbors(atom) >= 3
    }
    if three_methyl_c:
        counts["(CH3)3C"] = len(three_methyl_c)

    ch_ch_pairs: set[tuple[int, int]] = set()
    ch_c_pairs: set[tuple[int, int]] = set()
    c_c_pairs: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (_is_aliphatic_carbon(first) and _is_aliphatic_carbon(second)):
            continue
        pair = tuple(sorted((first.GetIdx(), second.GetIdx())))
        if is_ch(first) and is_ch(second) and _methyl_neighbors(first) and _methyl_neighbors(second):
            ch_ch_pairs.add(pair)
        if (
            (is_ch(first) and is_quaternary(second) and _methyl_neighbors(first) >= 1 and _methyl_neighbors(second) >= 2)
            or (is_ch(second) and is_quaternary(first) and _methyl_neighbors(second) >= 1 and _methyl_neighbors(first) >= 2)
        ):
            ch_c_pairs.add(pair)
        if is_quaternary(first) and is_quaternary(second) and _methyl_neighbors(first) >= 2 and _methyl_neighbors(second) >= 2:
            c_c_pairs.add(pair)
    if ch_ch_pairs:
        counts["CH(CH3)CH(CH3)"] = len(ch_ch_pairs)
    if ch_c_pairs:
        counts["CH(CH3)C(CH3)2"] = len(ch_c_pairs)
    if c_c_pairs:
        counts["C(CH3)2C(CH3)2"] = len(c_c_pairs)
    return counts


def _detect_rings(mol: Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring):
            continue
        ring_size = len(ring)
        if 3 <= ring_size <= 7:
            group = f"{ring_size} membered ring"
            counts[group] = counts.get(group, 0) + 1
    return counts


def _detect_ch3ch3(mol: Mol) -> int:
    if mol.GetNumAtoms() != 2:
        return 0
    first, second = mol.GetAtoms()
    return int(
        first.GetSymbol() == "C"
        and second.GetSymbol() == "C"
        and first.GetTotalNumHs() == 3
        and second.GetTotalNumHs() == 3
    )


def _second_order_decomposition(mol: Mol) -> dict[str, int]:
    counts = _detect_branching_groups(mol)
    counts.update(_detect_rings(mol))
    if _detect_ch3ch3(mol):
        counts["CH3CH3"] = 1
    return counts


def decompose_gani(smiles: str) -> dict[str, int]:
    """Decompose a supported hydrocarbon SMILES string into Gani group counts.

    The returned mapping contains only groups with non-zero counts. The group names
    match the columns in FuelLib's ``refGani.csv`` reference table.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    counts = _first_order_decomposition(mol)
    counts.update(_second_order_decomposition(mol))
    return counts


__all__ = ["UnsupportedGroupError", "decompose_gani"]
