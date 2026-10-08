"""Nannoolal liquid-viscosity equations for hydrocarbon group decompositions."""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._data_locator import get_gcmtable_dir


def cg_to_nannoolal_groups(groups):
    """Map CG hydrocarbon groups, assuming pairwise fused saturated rings."""
    carbon_groups = ("CH3", "CH2", "CH", "C", "ACH", "AC", "ACCH3", "ACCH2", "ACCH")
    alkene_groups = ("CH2=CH", "CH=CH", "CH2=C", "CH=C", "C=C")
    ring_groups = tuple(f"{size} membered ring" for size in range(3, 8))
    supported = set(carbon_groups + alkene_groups + ring_groups)
    for name, count in groups.items():
        if not np.isfinite(count) or count < 0 or count != np.floor(count):
            raise ValueError(
                f"Group {name!r} must have a finite non-negative integer count."
            )
        if count and name not in supported:
            raise ValueError(f"Unsupported CG group for Nannoolal viscosity: {name!r}")
    counts = {name: int(groups.get(name, 0)) for name in supported}
    mapped = {}

    def add(group, count):
        if count:
            mapped[group] = mapped.get(group, 0) + count

    n_rings = sum(counts[name] for name in ring_groups)
    aromatic_atoms = sum(counts[name] for name in carbon_groups[4:])
    n_double_bonds = sum(counts[name] for name in alkene_groups)
    attached = counts["ACCH3"] + counts["ACCH2"] + counts["ACCH"]
    if n_double_bonds and (n_rings or aromatic_atoms or n_double_bonds > 1):
        raise ValueError(
            "CG groups cannot resolve conjugated, cyclic, or aromatic-adjacent double bonds."
        )
    if aromatic_atoms not in (0, 6, 10):
        raise ValueError(
            "Nannoolal mapping requires a six- or ten-carbon aromatic skeleton."
        )
    if aromatic_atoms == 10 and (n_rings or counts["AC"] != 2):
        raise ValueError(
            "Nannoolal fused-aromatic mapping requires two fused aromatic carbons."
        )
    if (
        aromatic_atoms
        and n_rings
        and (n_rings != 1 or counts["ACCH3"] or counts["ACCH"])
    ):
        raise ValueError(
            "Nannoolal cycloaromatic mapping requires one unsubstituted fused saturated ring."
        )

    ring_atoms = sum(size * counts[f"{size} membered ring"] for size in range(3, 8))
    if n_rings:
        ring_atoms -= 2 if aromatic_atoms else 2 * (n_rings - 1)
        ring_ch2 = ring_atoms - counts["CH"] - counts["C"]
        if aromatic_atoms:
            ring_ch2 -= counts["ACCH2"]
        chain_ch2 = counts["CH2"] - ring_ch2
        if min(ring_ch2, chain_ch2) < 0:
            raise ValueError("Inconsistent saturated-ring atom and group counts.")
        add(9, ring_ch2)
        add(10, counts["CH"])
        add(11, counts["C"])
    else:
        chain_ch2 = counts["CH2"]
        add(5, counts["CH"])
        add(6, counts["C"])

    add(1, counts["CH3"])
    add(4, chain_ch2)
    add(15, counts["ACH"])
    add(18 if aromatic_atoms == 10 else 16, counts["AC"])
    add(16, attached)
    add(3, counts["ACCH3"])
    add(14 if n_rings else 8, counts["ACCH2"] + counts["ACCH"])
    add(61, counts["CH2=CH"] + counts["CH2=C"])
    add(58, counts["CH=CH"] + counts["CH=C"] + counts["C=C"])
    add(125, counts["3 membered ring"] + counts["4 membered ring"])
    add(126, counts["5 membered ring"])
    n_atoms = (
        sum(counts[name] for name in carbon_groups) + attached + 2 * n_double_bonds
    )
    mapped_atoms = sum(
        (2 if group in (58, 61) else 1) * count
        for group, count in mapped.items()
        if group < 125
    )
    if n_atoms <= 0 or mapped_atoms != n_atoms:
        raise ValueError(
            "Nannoolal mapping does not conserve the hydrocarbon carbon count."
        )
    return mapped, n_atoms


@lru_cache(maxsize=1)
def _contributions():
    """Read the Nannoolal group contributions for dBv and Tv."""
    table = Path(get_gcmtable_dir()) / "nannoolal_viscosity.csv"
    with table.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["Group"]): (float(row["dBv"]), float(row["Tv"]))
            for row in csv.DictReader(stream)
        }


def nannoolal_parameters(groups, n_atoms, Tb):
    """Return dBv and Tv (K) using Nannoolal (2009), equations 7 and 8."""
    if (
        not np.isfinite([n_atoms, Tb]).all()
        or min(n_atoms, Tb) <= 0
        or n_atoms != np.floor(n_atoms)
    ):
        raise ValueError(
            "Nannoolal atom count and boiling point must be positive and finite."
        )
    table = _contributions()
    missing = set(groups) - set(table)
    if missing:
        raise ValueError(f"Unsupported Nannoolal group IDs: {sorted(missing)}")
    counts = np.asarray(list(groups.values()), dtype=float)
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Nannoolal group counts must be finite non-negative integers.")
    sum_dbv = sum(count * table[group][0] for group, count in groups.items())
    sum_tv = sum(count * table[group][1] for group, count in groups.items())
    if sum_tv <= 0:
        raise ValueError("Nannoolal Tv group sum must be positive.")
    dbv = sum_dbv / (n_atoms**-2.5635 + 0.0685) + 3.7777
    tv = 21.8444 * np.sqrt(Tb) + sum_tv**0.9315 / (n_atoms**0.6577 + 4.9259) - 231.1361
    return dbv, tv


def nannoolal_dynamic_viscosity(T, dbv, tv):
    """Return Nannoolal dynamic viscosity in Pa*s for temperature T in K."""
    temperature = np.asarray(T, dtype=float)
    if not np.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError("Nannoolal temperature must be positive and finite.")
    denominator = temperature - tv / 16.0
    if (denominator <= 1e-12).any():
        raise ValueError("Nannoolal temperature must be above the Tv/16 singularity.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        viscosity = 1.3e-3 * np.exp(-dbv * (temperature - tv) / denominator)
    if not np.isfinite(viscosity).all() or (viscosity <= 0).any():
        raise ValueError("Nannoolal produced a non-physical dynamic viscosity.")
    return viscosity
