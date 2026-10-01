"""Boehm (2022) group contribution parameters."""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..rdk import mol
from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
boehm_gcm = GCMRegistry.register("boehm", property_fns=[])


def _identify_families(fuel: "Fuel") -> list[str]:
    """Identify the families of a fuel's components.

    =============== ==========================================
    Family          Description
    =============== ==========================================
    n-alkane        Straight-chain alkanes
    iso-alkane      Branched-chain alkanes
    alkene          Unsaturated hydrocarbons with double bonds
    monocyclic      Single-ring hydrocarbons
    dicyclic        Fused two-ring hydrocarbons
    tricyclic       Fused three-ring hydrocarbons
    alkylbenzene    Alkyl-substituted benzene derivatives
    cycloaromatic   Fused cyclic-aromatic hydrocarbons
    diaromatic      Fused two-ring aromatic hydrocarbons
    =============== ==========================================

    Args:
        fuel: The fuel object to identify the family for.

    Returns:
        The family names as a list of strings.

    Raises:
        ValueError: If the family cannot be identified.
    """
    families = []
    for m in fuel.rdkit_mols:
        aromatic = mol.has_aromatic(m)
        fused = mol.has_fused_rings(m)
        fused_two = mol.has_fused_rings(m, number_of_rings=2)
        fused_three = mol.has_fused_rings(m, number_of_rings=3)
        n_aromatic_rings = mol.count_aromatic_rings(m)

        if fused_two and n_aromatic_rings >= 2:
            family = "diaromatic"
        elif fused_two and n_aromatic_rings == 1:
            family = "cycloaromatic"
        elif aromatic and not fused:
            family = "alkylbenzene"
        elif fused_three and not aromatic:
            family = "tricyclic"
        elif fused_two and not aromatic:
            family = "dicyclic"
        elif fused:
            msg = f"Cannot identify family for molecule: {mol.smiles(m)}"
            raise ValueError(msg)
        elif mol.has_ring(m):
            family = "monocyclic"
        elif mol.has_double_bond(m):
            family = "alkene"
        elif mol.has_branch(m):
            family = "iso-alkane"
        else:
            family = "n-alkane"
        families.append(family)
    return families


#: Walden-rule fusion entropy fallback (J/(mol*K)) for unclassified compounds.
WALDEN_RULE_DS_FUS = 56.5


@boehm_gcm.register_property
def dS_fus(fuel: "Fuel") -> types.Quantity1D:
    """Return the fusion entropy (ΔS_fus) for each component in the fuel.

    Compounds whose family is not present in the family correlation table
    fall back to the Walden-rule value of 56.5 J/(mol*K).

    Args:
        fuel: Fuel object.

    Returns:
        The predicted fusion entropy (ΔS_fus) values in J/(mol*K).
    """
    families = _identify_families(fuel)
    result = []
    for i in range(fuel.num_compounds):
        family = families[i]
        nC = fuel.nC[i]
        if family in TABLE.index:
            row = TABLE.loc[family]
            dsFus_i = float(row["dSfus_A"]) + float(row["dSfus_B"]) * (
                nC - int(row["C_ref"])
            )
            result.append(max(dsFus_i, 20.0))
        else:
            result.append(WALDEN_RULE_DS_FUS)
    return Units.Quantity(result, "J/(mol*K)")
