"""Nannoolal group contribution parameters."""

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
from rdkit import Chem

from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
nannoolal_gcm = GCMRegistry.register("nannoolal", property_fns=[])


def _get_row(property_name: str) -> types.Array1D:
    """Get property row from GCM table.

    Args:
        property_name: Name of the property to retrieve from the GCM table.

    Returns:
        The property row as a 1D numpy array.

    Raises:
        KeyError: If the property is not found in the GCM table.
    """
    if property_name not in TABLE.columns:
        msg = f"Property '{property_name}' not found in GCM table."
        raise KeyError(msg)
    return TABLE[property_name].to_numpy(dtype=float)


def _get_decomp(fuel: "Fuel") -> types.Array2D:
    """Organize the decomposition matrix for the fuel and convert it to a numpy array.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The decomposition matrix as a 2D numpy array.
    """
    decomp = fuel.nannoolalDecomp.reindex(columns=TABLE.index, fill_value=0)
    return decomp.to_numpy()


def _check_compatible_dims(
    xj: Annotated[types.Array1D, "GCM property vector"],
    Aij: Annotated[types.Array2D, "Group decomposition matrix"],
) -> None:
    """Check if the shape of the GCM properties and group decomposition matrix match.

    Args:
        xj: The GCM property vector.
        Aij: The group decomposition matrix.

    Raises:
        ValueError: If the dimensions of the GCM contribution vector and the group
            decomposition matrix are incompatible.
    """
    if Aij.shape[1] != xj.shape[0]:
        msg = (
            f"Incompatible dimensions: GCM contribution vector has length {xj.shape[0]}"
            f", but group decomposition matrix has {Aij.shape[1]} columns."
        )
        raise ValueError(msg)


# Connected to N, O, F or Cl (aliphatic or aromatic N/O)
E = "$(*~[#7,#8,F,Cl])"

# (priority, Nannoolal group name, SMARTS); lower priority number = matched first
order1_groups = [
    (5, ">C=C=C<", "[C]=[C]=[C]"),
    (6, ">C=C-C=C< (ring)", "[C]=@[C]-[C]=@[C]"),
    (7, ">C=C-C=C< (chain)", "[C]=!@[C]-[C]=!@[C]"),
    (8, "-C#C-C#C-", "[C]#[C]-[C]#[C]"),
    (56, "HC#C-", "[CH1]#[C]"),
    (57, "H2C(c)=C<", "[CH2]=[C]"),
    (58, "-(e)C(c)=C(c)<", f"[C;{E}]=!@[C]"),
    (59, ">C(c)=C(c)<(C(a))", "[C;$(*~c)]=!@[C]"),
    (60, ">C(r)=C(r)<", "[C]=@[C]"),
    (61, "-C#C-", "[CH0]#[CH0]"),
    (62, ">C(c)=C(c)<", "[C;!H2]=!@[C;!H2]"),
    (103, "CH3-(e)", f"[CH3;{E}]"),
    (104, "CH3-(a)", "[CH3;$(*~a)]"),
    (105, "CH3-(ne)", f"[CH3;!{E}]"),
    (106, "=C(a)H-", "[cH1]"),
    (107, ">C(r)<(Ca)", "[CX4;R;$(*~c)]"),
    (108, ">C(c)<(e)", f"[CX4;!R;!H3;{E}]"),
    (109, ">C(c)<(a)", "[CX4;!R;!H3;$(*~c)]"),
    (110, ">C(r)<(e,c)", "[CX4;R;$(*!@[#7,#8]),$(*~[F,Cl])]"),
    (111, ">C(r)<(e,r)", "[CX4;R;$(*@[#7,#8])]"),
    (112, "-C(c)H2-", "[CH2X4;!R]"),
    (113, "-C(r)H2-", "[CH2X4;R]"),
    (114, "=C(a)<(e)", f"[cH0;{E}]"),
    (115, "(a)=C(a)<2(a)", "[cH0;$(*(:a)(:a):a)]"),
    (116, "C(a)=C(a)<C2(a) (bridge)", "[cH0;$(*(:a)(:a)!:a)]"),
    (117, "=C(a)<(ne)", f"[cH0;!{E}]"),
    (118, ">C(r)H-", "[CH1X4;R]"),
    (119, ">C(c)H-", "[CH1X4;!R]"),
    (120, ">C(r)<", "[CH0X4;R]"),
    (121, ">C(c)<", "[CH0X4;!R]"),
]
# Carbon whose heavy-atom neighbors are all carbon
CC = "!$(*~[!#6;!#1])"

# (Nannoolal group name, SMARTS); additive, so every unique match is counted
order2_groups = [
    ("((C=)(C)C-CC3)", f"[#6X3;D3;{CC};$(*=,:[#6])]-[CX4;D4;{CC}]"),
    ("C2C-CC2", f"[CX4;D3;{CC}]-[CX4;D3;{CC}]"),
    ("C3C-CC2", f"[CX4;D4;{CC}]-[CX4;D3;{CC}]"),
    ("C3C-CC3", f"[CX4;D4;{CC}]-[CX4;D4;{CC}]"),
]

# Ring position distance between two substituents on a benzene ring
PAIR_GROUPS = {1: "Ortho-pair(s)", 2: "Meta-pair(s)", 3: "Para-pair(s)"}


def decompose(smiles: str) -> pd.DataFrame:
    """Decompose a molecule into first and second order groups based on SMARTS patterns.

    Args:
        smiles: The SMILES string of the molecule to decompose.

    Returns:
        A single-row DataFrame indexed by the SMILES string, with a column for every
        group in the GCM table containing the number of occurrences of that group.

    Raises:
        ValueError: If the SMILES string cannot be parsed or if not all atoms are
            matched to first order groups.
        KeyError: If a matched group is not a column of the GCM table.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Error parsing SMILES: {smiles}")

    atom_count = mol.GetNumAtoms()
    matched = set()
    group1_matches = dict()

    for _, group, smarts in sorted(order1_groups):
        query = Chem.MolFromSmarts(smarts)
        if not query:
            msg = f"Error parsing SMARTS pattern: {smarts}"
            raise ValueError(msg)

        matches = mol.GetSubstructMatches(query)
        for match in matches:
            if any(atom_id in matched for atom_id in match):
                continue

            matched.update(match)
            group1_matches.setdefault(group, []).append(set(match))

    if len(matched) != atom_count:
        msg = (
            "Not all atoms were matched to first order groups."
            f"Matched {len(matched)} out of {atom_count} atoms."
        )
        raise ValueError(msg)

    group2_counts: dict[str, int] = dict()

    for group, smarts in order2_groups:
        query = Chem.MolFromSmarts(smarts)
        if not query:
            msg = f"Error parsing SMARTS pattern: {smarts}"
            raise ValueError(msg)

        if n_matches := len(mol.GetSubstructMatches(query)):
            group2_counts[group] = n_matches

    n_hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    if n_hydrogens == 0:
        group2_counts["No hydrogen"] = 1
    elif n_hydrogens == 1:
        group2_counts["One hydrogen"] = 1

    # Count each non-aromatic ring once (rings may share atoms in fused systems)
    ring_info = mol.GetRingInfo()
    for atom_ring, bond_ring in zip(
        ring_info.AtomRings(), ring_info.BondRings(), strict=True
    ):
        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_ring):
            continue
        if len(atom_ring) in (3, 4):
            group2_counts["3/4 ring"] = group2_counts.get("3/4 ring", 0) + 1
        elif len(atom_ring) == 5:
            group2_counts["5 ring"] = group2_counts.get("5 ring", 0) + 1

    # Substituent pairs on aromatic six-membered rings (substituents are attached
    # via non-ring bonds). Counted once, and only if a single pair type exists.
    pair_groups = set()
    for atom_ring in ring_info.AtomRings():
        if len(atom_ring) != 6 or not all(
            mol.GetAtomWithIdx(i).GetIsAromatic() for i in atom_ring
        ):
            continue
        substituted = [
            pos
            for pos, atom_id in enumerate(atom_ring)
            if any(not b.IsInRing() for b in mol.GetAtomWithIdx(atom_id).GetBonds())
        ]
        for pos_i, pos_j in combinations(substituted, 2):
            pair_groups.add(PAIR_GROUPS[min(pos_j - pos_i, 6 - (pos_j - pos_i))])
    if len(pair_groups) == 1:
        group2_counts[pair_groups.pop()] = 1

    invalid = (group1_matches.keys() | group2_counts.keys()) - set(TABLE.index)
    if invalid:
        msg = f"Groups not found in GCM table: {sorted(invalid)}"
        raise KeyError(msg)

    df = pd.DataFrame(0, index=[smiles], columns=TABLE.index)
    for group, matches in group1_matches.items():
        df.loc[smiles, group] += len(matches)
    for group, count in group2_counts.items():
        df.loc[smiles, group] += count

    return df


@nannoolal_gcm.register_property
def dBv(fuel: "Fuel") -> types.Quantity1D:
    """Get the saturated liquid viscosity curve slope for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The dimensionless group contribution sums for the saturated liquid
        viscosity curve slope (dBv) of the fuel's components.
        Converting these sums into dBv requires Nannoolal Eq. (7).
    """
    nij = _get_decomp(fuel)
    dBvj = _get_row("dBv")
    _check_compatible_dims(dBvj, nij)
    # Only absent groups are zeroed; NaN propagates for used, unsupported groups
    dBvi = np.where(nij != 0, nij * dBvj, 0.0).sum(axis=1)
    return Units.Quantity(dBvi, "")


@nannoolal_gcm.register_property
def Tv(fuel: "Fuel") -> types.Quantity1D:
    """Get the saturated liquid viscosity reference temperature for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The dimensionless group contribution sums for the saturated liquid
        viscosity reference temperature (Tv) of the fuel's components.
        Converting these sums into Tv in K requires Nannoolal Eq. (8).
    """
    nij = _get_decomp(fuel)
    Tvj = _get_row("Tv")
    _check_compatible_dims(Tvj, nij)
    Tvi = np.where(nij != 0, nij * Tvj, 0.0).sum(axis=1)
    return Units.Quantity(Tvi, "")
