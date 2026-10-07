"""Boehm (2022) group contribution parameters."""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
boehm_gcm = GCMRegistry.register("boehm", property_fns=[])


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
    result = []
    for family, nC in zip(fuel.families, fuel.nC, strict=True):
        if family in TABLE.index:
            row = TABLE.loc[family]
            dsFus_i = float(row["dSfus_A"]) + float(row["dSfus_B"]) * (
                nC - int(row["C_ref"])
            )
            result.append(max(dsFus_i, 20.0))
        else:
            result.append(WALDEN_RULE_DS_FUS)
    return Units.Quantity(result, "J/(mol*K)")
