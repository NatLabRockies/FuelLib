"""Nannoolal viscosity-group decomposition derived from Gani RDKit groups."""

import math
from collections.abc import Mapping

from .ganiDecomposition import decompose_gani

_SOURCE_CARBON_GROUPS = (
    "CH3",
    "CH2",
    "CH",
    "C",
    "ACH",
    "AC",
    "ACCH3",
    "ACCH2",
    "ACCH",
)
_SOURCE_ALKENE_GROUPS = ("CH2=CH", "CH=CH", "CH2=C", "CH=C", "C=C")
_SOURCE_RING_GROUPS = tuple(f"{size} membered ring" for size in range(3, 8))
_SOURCE_GROUPS = frozenset(
    _SOURCE_CARBON_GROUPS + _SOURCE_ALKENE_GROUPS + _SOURCE_RING_GROUPS
)


def map_gani_to_nannoolal_groups(
    gani_groups: Mapping[str, int | float],
) -> tuple[dict[int, int], int]:
    """Map first-order Gani counts into Nannoolal viscosity group IDs.

    The mapping follows the hydrocarbon assumptions in Nannoolal et al. (2009):
    saturated polycycles are pairwise fused, and cycloaromatic and fused-aromatic
    structures must match the explicitly supported topologies.

    Args:
        gani_groups: Non-negative, integral first-order Gani and ring-group counts.

    Returns:
        Nannoolal group-ID counts and the corresponding non-hydrogen atom count.

    Raises:
        ValueError: If the counts are invalid or cannot represent a supported
            Nannoolal hydrocarbon topology.
    """
    for name, count in gani_groups.items():
        if not math.isfinite(count) or count < 0 or count != math.floor(count):
            raise ValueError(
                f"Group {name!r} must have a finite non-negative integer count."
            )
        if count and name not in _SOURCE_GROUPS:
            raise ValueError(
                f"Unsupported Gani group for Nannoolal viscosity: {name!r}"
            )

    counts = {name: int(gani_groups.get(name, 0)) for name in _SOURCE_GROUPS}
    mapped: dict[int, int] = {}

    def add(group_id: int, count: int) -> None:
        if count:
            mapped[group_id] = mapped.get(group_id, 0) + count

    ring_count = sum(counts[name] for name in _SOURCE_RING_GROUPS)
    aromatic_atoms = sum(counts[name] for name in _SOURCE_CARBON_GROUPS[4:])
    double_bond_count = sum(counts[name] for name in _SOURCE_ALKENE_GROUPS)
    attached_aromatic = counts["ACCH3"] + counts["ACCH2"] + counts["ACCH"]

    if double_bond_count and (ring_count or aromatic_atoms or double_bond_count > 1):
        raise ValueError(
            "Gani groups cannot resolve conjugated, cyclic, or "
            "aromatic-adjacent double bonds."
        )
    if aromatic_atoms not in (0, 6, 10):
        raise ValueError(
            "Nannoolal mapping requires a six- or ten-carbon aromatic skeleton."
        )
    if aromatic_atoms == 10 and (ring_count or counts["AC"] != 2):
        raise ValueError("Fused-aromatic mapping requires two fused aromatic carbons.")
    if (
        aromatic_atoms
        and ring_count
        and (ring_count != 1 or counts["ACCH3"] or counts["ACCH"])
    ):
        raise ValueError(
            "Cycloaromatic mapping requires one unsubstituted fused saturated ring."
        )

    ring_atoms = sum(size * counts[f"{size} membered ring"] for size in range(3, 8))
    if ring_count:
        ring_atoms -= 2 if aromatic_atoms else 2 * (ring_count - 1)
        ring_ch2 = ring_atoms - counts["CH"] - counts["C"]
        if aromatic_atoms:
            ring_ch2 -= counts["ACCH2"]
        chain_ch2 = counts["CH2"] - ring_ch2
        if min(ring_ch2, chain_ch2) < 0:
            raise ValueError("Inconsistent saturated-ring atom and Gani group counts.")
        add(9, ring_ch2)
        add(10, counts["CH"])
        add(11, counts["C"])
    else:
        chain_ch2 = counts["CH2"]
        add(5, counts["CH"])
        add(6, counts["C"])

    add(1, counts["CH3"])
    add(3, counts["ACCH3"])
    add(4, chain_ch2)
    add(8 if not ring_count else 14, counts["ACCH2"] + counts["ACCH"])
    add(15, counts["ACH"])
    add(18 if aromatic_atoms == 10 else 16, counts["AC"])
    add(16, attached_aromatic)
    add(61, counts["CH2=CH"] + counts["CH2=C"])
    add(58, counts["CH=CH"] + counts["CH=C"] + counts["C=C"])
    add(125, counts["3 membered ring"] + counts["4 membered ring"])
    add(126, counts["5 membered ring"])

    atom_count = sum(counts[name] for name in _SOURCE_CARBON_GROUPS)
    atom_count += attached_aromatic + 2 * double_bond_count
    mapped_atoms = sum(
        (2 if group_id in (58, 61) else 1) * count
        for group_id, count in mapped.items()
        if group_id < 125
    )
    if atom_count <= 0 or mapped_atoms != atom_count:
        raise ValueError(
            "Nannoolal mapping does not conserve the hydrocarbon carbon count."
        )
    return mapped, atom_count


def _groups_from_smiles(smiles: str) -> tuple[dict[int, int], int]:
    gani_groups = decompose_gani(smiles)
    source_groups = {
        name: count for name, count in gani_groups.items() if name in _SOURCE_GROUPS
    }
    return map_gani_to_nannoolal_groups(source_groups)


def decompose_nannoolal(smiles: str) -> dict[str, int]:
    """Decompose a supported hydrocarbon SMILES string into Nannoolal group counts.

    Args:
        smiles: SMILES string for the hydrocarbon.

    Returns:
        Mapping of Nannoolal group IDs, represented as strings, to non-zero counts.

    """
    groups, _atom_count = _groups_from_smiles(smiles)
    return {str(group_id): count for group_id, count in groups.items()}


__all__ = ["decompose_nannoolal", "map_gani_to_nannoolal_groups"]
