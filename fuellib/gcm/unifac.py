"""UNIFAC 2.0 group contribution method."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
from rdkit import Chem

from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

# Subgroup names as columns (duplicate "CHO" becomes "CHO.1", matching refUnifac.csv)
_RAW = pd.read_csv(Path(__file__).with_suffix(".csv"), header=1, index_col=0)
TABLE = _RAW.loc[["R", "Q"]].astype(float)
MAIN_GROUP = _RAW.loc["Main Group No."].astype(int)
# Interaction parameters a_mn [K] indexed by main group numbers (row m, column n)
AMN = pd.read_csv(Path(__file__).with_name("unifac_amn.csv"), index_col=0)
AMN.columns = AMN.columns.astype(int)
unifac_gcm = GCMRegistry.register("UNIFAC", property_fns=[])


def _get_row(property_name: str) -> types.Array1D:
    """Get property row from GCM table.

    Args:
        property_name: Name of the property to retrieve from the GCM table.

    Returns:
        The property row as a 1D numpy array.

    Raises:
        KeyError: If the property is not found in the GCM table.
    """
    row = TABLE.loc[property_name]
    if row.empty:
        msg = f"Property '{property_name}' not found in GCM table."
        raise KeyError(msg)
    return row.to_numpy().flatten()


def _get_decomp(fuel: "Fuel") -> types.Array2D:
    """Organize the decomposition matrix for the fuel and convert it to a numpy array.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The decomposition matrix as a 2D numpy array.
    """
    decomp = fuel.unifacDecomp.reindex(columns=TABLE.columns, fill_value=0)
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


# Groups organized by priority (highest to lowest)
order1_groups = [
    ("ACCH3", "c-[CH3]"),
    ("ACCH2", "c-[CH2]"),
    ("ACCH", "c-[CH1]"),
    ("ACH", "[cH1]"),
    ("AC", "[cH0]"),
    ("CH2=CH", "[CH2]=[CH1]"),
    ("CH=CH", "[CH1]=[CH1]"),
    ("CH2=C", "[CH2]=[CH0]"),
    ("CH=C", "[CH1]=[CH0]"),
    ("C=C", "[CH0]=[CH0]"),
    ("CH3", "[CH3]"),
    ("CH2", "[CH2]"),
    ("CH", "[CH1]"),
    ("C", "[CH0]"),
]


def decompose(smiles: str) -> pd.DataFrame:
    """Decompose a molecule into first order groups based on SMARTS patterns.

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

    for group, smarts in order1_groups:
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

    invalid = set(group1_matches.keys()) - set(TABLE.columns)
    if invalid:
        msg = f"Groups not found in GCM table columns: {sorted(invalid)}"
        raise KeyError(msg)

    df = pd.DataFrame(0, index=[smiles], columns=TABLE.columns)
    for group, matches in group1_matches.items():
        df.loc[smiles, group] += len(matches)

    return df


@unifac_gcm.register_property
def R(fuel: "Fuel") -> types.Quantity1D:
    """Predict the R parameter for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted R parameters (dimensionless).
    """
    nij = _get_decomp(fuel)
    Rj = _get_row("R")
    _check_compatible_dims(Rj, nij)
    Ri: types.Array1D = np.matmul(nij, Rj)
    return Units.Quantity(Ri, "")


@unifac_gcm.register_property
def Q(fuel: "Fuel") -> types.Quantity1D:
    """Predict the Q parameter for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted Q parameters (dimensionless).
    """
    nij = _get_decomp(fuel)
    Qj = _get_row("Q")
    _check_compatible_dims(Qj, nij)
    Qi: types.Array1D = np.matmul(nij, Qj)
    return Units.Quantity(Qi, "")


def _ln_group_gamma(
    Xk: types.Array2D, Qk: types.Array1D, psi: types.Array2D
) -> types.Array2D:
    """Residual group activity coefficients ln(Gamma_k) for each row of Xk.

    Args:
        Xk: Group mole fractions, one row per mixture (shape (n, K)).
        Qk: Group surface parameters (shape (K,)).
        psi: Group interaction matrix Psi_mn (shape (K, K)).

    Returns:
        ln(Gamma_k) for each row of Xk (shape (n, K)).
    """
    theta = Qk * Xk
    theta /= theta.sum(axis=1, keepdims=True)
    s = theta @ psi  # s_k = sum_m theta_m psi_mk
    return Qk * (1.0 - np.log(s) - (theta / s) @ psi.T)


def parameters(
    fuel: "Fuel",
) -> tuple[types.Array2D, types.Array1D, types.Array1D, types.Array2D]:
    """Collect the UNIFAC parameters of the subgroups present in a fuel.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        Subgroup counts per component (shape (N, K)), subgroup volume parameters
        R_k (shape (K,)), subgroup surface parameters Q_k (shape (K,)), and
        interaction parameters a_mn in K between the subgroups (shape (K, K)).
    """
    nu_all = _get_decomp(fuel)
    used = nu_all.any(axis=0)
    main = MAIN_GROUP.to_numpy()[used]
    return (
        nu_all[:, used],
        _get_row("R")[used],
        _get_row("Q")[used],
        AMN.loc[main, main].to_numpy(),
    )


def ln_gamma_x(
    x: types.Array1D,
    T: float,
    nu: types.Array2D,
    Rk: types.Array1D,
    Qk: types.Array1D,
    a_mn: types.Array2D,
) -> types.Array1D:
    """Compute UNIFAC ln(gamma_i) from mole fractions with plain NumPy arrays.

    Args:
        x: Liquid mole fractions of each component (shape (N,)).
        T: Temperature in K.
        nu: Subgroup counts per component (shape (N, K)).
        Rk: Subgroup volume parameters (shape (K,)).
        Qk: Subgroup surface parameters (shape (K,)).
        a_mn: Subgroup interaction parameters in K (shape (K, K)).

    Returns:
        Natural log of the activity coefficient of each component (shape (N,)).
    """
    # Combinatorial part
    r = nu @ Rk
    q = nu @ Qk
    V = r / (x @ r)
    F = q / (x @ q)
    ln_gamma_C = 1.0 - V + np.log(V) - 5.0 * q * (1.0 - V / F + np.log(V / F))

    # Residual part
    psi = np.exp(-a_mn / T)
    nx = x @ nu
    ln_G_mix = _ln_group_gamma((nx / nx.sum())[None, :], Qk, psi)
    ln_G_pure = _ln_group_gamma(nu / nu.sum(axis=1, keepdims=True), Qk, psi)
    ln_gamma_R = np.sum(nu * (ln_G_mix - ln_G_pure), axis=1)

    return ln_gamma_C + ln_gamma_R


def ln_gamma(
    fuel: "Fuel", T: types.Quantity0D, Yi: types.Quantity1D | None = None
) -> types.Quantity1D:
    """Compute UNIFAC 2.0 activity coefficients ln(gamma_i) of a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.
        T: Temperature.
        Yi: Liquid mass fractions of each component.
            Defaults to `fuel.Y_0` (initial mass fractions).

    Returns:
        Natural log of the activity coefficient of each component (dimensionless).
    """
    Yi = Yi if Yi is not None else fuel.Y_0
    T_K = T.to("K").magnitude
    x = fuel.Y2X(Yi).to("dimensionless").magnitude
    return Units.Quantity(ln_gamma_x(x, T_K, *parameters(fuel)), "")
