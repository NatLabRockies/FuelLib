"""Nannoolal group-contribution parameters for liquid viscosity."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils import Units, types
from .core import GCMRegistry
from .nannoolalDecomposition import _groups_from_smiles

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), index_col="Group")
nannoolal_gcm = GCMRegistry.register("nannoolal", property_fns=[])


def _parameters_for_smiles(smiles: str, boiling_point: float) -> tuple[float, float]:
    """Calculate Nannoolal dBv and Tv parameters for one supported SMILES string.

    Returns:
        Dimensionless dBv and Tv in K.

    Raises:
        ValueError: If the boiling point or SMILES decomposition is unsupported.
    """
    groups, atom_count = _groups_from_smiles(smiles)
    if not np.isfinite(boiling_point) or boiling_point <= 0:
        raise ValueError("Nannoolal boiling point must be positive and finite.")

    contribution_rows = TABLE.reindex(groups)
    if contribution_rows.isna().any(axis=None):
        missing = sorted(set(groups).difference(TABLE.index))
        raise ValueError(f"Unsupported Nannoolal group IDs: {missing}")

    counts = np.asarray(list(groups.values()), dtype=float)
    dbv_sum = float(np.dot(counts, contribution_rows["dBv"].to_numpy()))
    tv_sum = float(np.dot(counts, contribution_rows["Tv"].to_numpy()))
    if tv_sum <= 0:
        raise ValueError("Nannoolal Tv group sum must be positive.")

    dbv = dbv_sum / (atom_count**-2.5635 + 0.0685) + 3.7777
    tv = (
        21.8444 * np.sqrt(boiling_point)
        + tv_sum**0.9315 / (atom_count**0.6577 + 4.9259)
        - 231.1361
    )
    return float(dbv), float(tv)


def parameters(
    fuel: "Fuel", *, strict: bool = True
) -> tuple[types.Array1D, types.Array1D, list[str | None]]:
    """Calculate Nannoolal viscosity parameters for every component in a fuel.

    Args:
        fuel: Fuel whose components will be mapped from their SMILES strings.
        strict: Raise for the first unsupported component when true. Otherwise,
            return NaN parameters and the error message for unsupported components.

    Returns:
        Arrays of dBv and Tv values, and one optional error message per component.

    Raises:
        ValueError: If ``strict`` is true and a component cannot be mapped.
    """
    boiling_points = fuel.Tb.to("K").magnitude
    dbv = np.full(fuel.num_compounds, np.nan)
    tv = np.full(fuel.num_compounds, np.nan)
    errors: list[str | None] = [None] * fuel.num_compounds
    for index, (compound, smiles, boiling_point) in enumerate(
        zip(fuel.compounds, fuel.smiles, boiling_points)
    ):
        try:
            dbv[index], tv[index] = _parameters_for_smiles(smiles, boiling_point)
        except ValueError as error:
            message = f"Nannoolal viscosity for {compound!r}: {error}"
            if strict:
                raise ValueError(message) from error
            errors[index] = message
    return dbv, tv, errors


def dynamic_viscosity(
    temperature: float | types.Array1D, dbv: types.Array1D, tv: types.Array1D
) -> types.Array1D:
    """Evaluate the Nannoolal dynamic-viscosity equation in Pa*s.

    Args:
        temperature: Liquid temperature in K.
        dbv: Dimensionless Nannoolal dBv parameters.
        tv: Nannoolal reference temperatures in K.

    Returns:
        Dynamic viscosities in Pa*s.

    Raises:
        ValueError: If the inputs are non-physical or evaluate at the Tv/16
            singularity.
    """
    temperature_array = np.asarray(temperature, dtype=float)
    if not np.isfinite(temperature_array).all() or (temperature_array <= 0).any():
        raise ValueError("Nannoolal temperature must be positive and finite.")
    denominator = temperature_array - tv / 16.0
    if (denominator <= 1e-12).any():
        raise ValueError("Nannoolal temperature must be above the Tv/16 singularity.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        viscosity = 1.3e-3 * np.exp(-dbv * (temperature_array - tv) / denominator)
    if not np.isfinite(viscosity).all() or (viscosity <= 0).any():
        raise ValueError("Nannoolal produced a non-physical dynamic viscosity.")
    return viscosity


@nannoolal_gcm.register_property
def dBv(fuel: "Fuel") -> types.Quantity1D:
    """Predict the Nannoolal dBv viscosity parameter for each component.

    Args:
        fuel: Fuel whose components will be mapped to Nannoolal groups.

    Returns:
        Dimensionless dBv parameter values.
    """
    dbv, _tv, _errors = parameters(fuel)
    return Units.Quantity(dbv, "")


@nannoolal_gcm.register_property
def Tv(fuel: "Fuel") -> types.Quantity1D:
    """Predict the Nannoolal reference temperature for each component.

    Args:
        fuel: Fuel whose components will be mapped to Nannoolal groups.

    Returns:
        Nannoolal reference temperatures in K.
    """
    _dbv, tv, _errors = parameters(fuel)
    return Units.Quantity(tv, "K")


__all__ = ["TABLE", "Tv", "dBv", "dynamic_viscosity", "nannoolal_gcm", "parameters"]
