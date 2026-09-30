"""RDKit Mol-related utilities and functions for the FuelLib package."""

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol


# Instantiation functions
def from_smiles(smiles: str) -> Mol:
    """Instantiate an RDKit Mol object from a SMILES string.

    Args:
        smiles: SMILES string representing the molecule.

    Returns:
        RDKit Mol object instantiated from the SMILES string.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    mol: Mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol = Chem.AddHs(mol)
    return mol


def smiles(mol: Mol) -> str:
    """Get the SMILES string representation of `mol`.

    Args:
        mol: RDKit Mol object.

    Returns:
        SMILES string representation of the molecule.
    """
    mol = Chem.RemoveAllHs(mol)
    return Chem.MolToSmiles(mol)


def from_inchi(inchi: str) -> Mol:
    """Instantiate an RDKit Mol object from an InChI string.

    Args:
        inchi: InChI string representing the molecule.

    Returns:
        RDKit Mol object instantiated from the InChI string.

    Raises:
        ValueError: If the InChI string is invalid.
    """
    mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"Invalid InChI string: {inchi}")
    mol = Chem.AddHs(mol)
    return mol


def inchi(mol: Mol) -> str:
    """Get the InChI string representation of `mol`.

    Args:
        mol: RDKit Mol object.

    Returns:
        InChI string representing the molecule.
    """
    return Chem.inchi.MolBlockToInchi(Chem.rdmolfiles.MolToMolBlock(mol))


def hill_formula(mol: Mol) -> str:
    """Get the Hill formula representation of `mol`.

    Args:
        mol: RDKit Mol object.

    Returns:
        Hill formula string representing the molecule.
    """
    atom_counts_dict = atom_counts(mol)
    if "C" not in atom_counts_dict:
        return "".join(
            f"{atom}{count}" if count > 1 else atom
            for atom, count in sorted(atom_counts_dict.items())
        )
    nC = atom_counts_dict.pop("C")
    formula_parts = [f"C{nC}" if nC > 1 else "C"]
    nH = atom_counts_dict.pop("H", 0)
    if nH > 0:
        formula_parts.append(f"H{nH}" if nH > 1 else "H")
    for atom, count in sorted(atom_counts_dict.items()):
        formula_parts.append(f"{atom}{count}" if count > 1 else atom)
    return "".join(formula_parts)


# Structural analysis functions
def atom_counts(mol: Mol) -> dict[str, int]:
    """Count the number of each type of atom in `mol`.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dictionary mapping atom symbols to their counts.
    """
    mol = Chem.AddHs(mol)
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def has_aromatic(mol: Mol) -> bool:
    """Check if `mol` contains any aromatic atoms.

    Args:
        mol: RDKit Mol object.

    Returns:
        True if the molecule contains any aromatic atoms, False otherwise.
    """
    return any(atom.GetIsAromatic() for atom in mol.GetAtoms())


def has_ring(mol: Mol) -> bool:
    """Check if `mol` contains any ring structures.

    Args:
        mol: RDKit Mol object.

    Returns:
        True if the molecule contains any ring structures, False otherwise.
    """
    return mol.GetRingInfo().NumRings() > 0


def has_double_bond(mol: Mol) -> bool:
    """Check if `mol` contains any double bonds.

    Args:
        mol: RDKit Mol object.

    Returns:
        True if the molecule contains any double bonds, False otherwise.
    """
    return any(
        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in mol.GetBonds()
    )


def has_branch(mol: Mol) -> bool:
    """Check if `mol` contains any branches.

    A branch is identified by an atom bonded to more than two neighbors
    (degree > 2) after removing explicit hydrogens. If such an atom is not
    part of a ring, it is a branch point. If it is part of a ring, it is
    only considered a branch point if at least one of its neighbors lies
    outside the ring (i.e., a substituent hanging off the ring), since
    ring atoms themselves commonly have degree > 2 without representing a
    branch (i.e., fused-ring systems).

    Args:
        mol: RDKit Mol object.

    Returns:
        True if the molecule contains any branches, False otherwise.
    """
    mol = Chem.RemoveAllHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 2:
            if not atom.IsInRing():
                return True
            for neighbor in atom.GetNeighbors():
                if not neighbor.IsInRing():
                    return True
    return False


# Molecular property calculations
def molecular_weight(mol: Mol, *, exact: bool = False) -> float:
    """Calculate the molecular weight of `mol`.

    Args:
        mol: RDKit Mol object.
        exact: Whether to calculate the monoisotopic molecular weight, i.e.
            the mass of the molecule computed using the most abundant
            isotope of each element rather than each element's standard
            atomic weight (which is averaged over isotopic abundance).
            Defaults to `False`, which uses standard atomic weights.

    Returns:
        Molecular weight of the molecule in atomic mass units (amu).
    """
    if exact:
        return Descriptors.ExactMolWt(mol)  # ty: ignore[unresolved-attribute]
    return Descriptors.MolWt(mol)  # ty: ignore[unresolved-attribute]


__all__ = [
    "atom_counts",
    "from_inchi",
    "from_smiles",
    "has_aromatic",
    "has_branch",
    "has_double_bond",
    "has_ring",
    "inchi",
    "molecular_weight",
    "smiles",
]
