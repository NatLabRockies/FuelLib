"""Gani-Constantinou and extended group contribution parameters."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd

from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
gani_gcm = GCMRegistry.register("gani", property_fns=[])


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
    decomp = fuel.gani_decomp().reindex(columns=TABLE.columns, fill_value=0)
    return decomp.to_numpy()


def _check_compatible_dims(
    cj: Annotated[types.Array1D, "GCM property vector"],
    ij: Annotated[types.Array2D, "Group decomposition matrix"],
) -> None:
    """Check if the shape of the GCM properties and group decomposition matrix match.

    Args:
        cj: The GCM contribution vector.
        ij: The group decomposition matrix.

    Raises:
        ValueError: If the dimensions of the GCM contribution vector and the group
            decomposition matrix are incompatible.
    """
    if ij.shape[1] != cj.shape[0]:
        msg = (
            f"Incompatible dimensions: GCM contribution vector has length {cj.shape[0]}"
            f", but group decomposition matrix has {ij.shape[1]} columns."
        )
        raise ValueError(msg)


@gani_gcm.register_property
def Tc(fuel: "Fuel") -> types.Quantity1D:
    """Predict the critical temperature (Tc) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted critical temperatures (Tc) in K.
    """
    nij = _get_decomp(fuel)
    tcj = _get_row("tck")
    _check_compatible_dims(tcj, nij)
    tci: types.Array1D = 181.128 * np.log(np.matmul(nij, tcj))
    return Units.Quantity(tci, "K")


@gani_gcm.register_property
def Pc(fuel: "Fuel") -> types.Quantity1D:
    """Predict the critical pressure (Pc) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted critical pressures (Pc) in bar.
    """
    nij = _get_decomp(fuel)
    pcj = _get_row("pck")
    _check_compatible_dims(pcj, nij)
    pci: types.Array1D = 1.3705 + np.power(np.matmul(nij, pcj) + 0.10022, -2.0)
    return Units.Quantity(pci, "bar")


@gani_gcm.register_property
def Vc(fuel: "Fuel") -> types.Quantity1D:
    """Predict the critical volume (Vc) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted critical volumes (Vc) in m^3/kmol.
    """
    nij = _get_decomp(fuel)
    vcj = _get_row("vck")
    _check_compatible_dims(vcj, nij)
    vci: types.Array1D = -0.00435 + (np.matmul(nij, vcj))
    return Units.Quantity(vci, "m^3/kmol")


@gani_gcm.register_property
def Tb(fuel: "Fuel") -> types.Quantity1D:
    """Predict the normal boiling temperature (Tb) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted normal boiling temperatures (Tb) in K.
    """
    nij = _get_decomp(fuel)
    tbj = _get_row("tbk")
    _check_compatible_dims(tbj, nij)
    tbi: types.Array1D = 204.359 * np.log(np.matmul(nij, tbj))
    return Units.Quantity(tbi, "K")


@gani_gcm.register_property
def Tm(fuel: "Fuel") -> types.Quantity1D:
    """Predict the melting temperature (Tm) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted melting temperatures (Tm) in K.
    """
    nij = _get_decomp(fuel)
    tmj = _get_row("tmk")
    _check_compatible_dims(tmj, nij)
    tmi: types.Array1D = 102.425 * np.log(np.matmul(nij, tmj))
    return Units.Quantity(tmi, "K")


@gani_gcm.register_property
def Hf(fuel: "Fuel") -> types.Quantity1D:
    """Predict the standard enthalpy of formation (Hf) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted standard enthalpies of formation (Hf) in kJ/mol.
    """
    nij = _get_decomp(fuel)
    hfj = _get_row("hfk")
    _check_compatible_dims(hfj, nij)
    hfi: types.Array1D = 10.835 + np.matmul(nij, hfj)
    return Units.Quantity(hfi, "kJ/mol")


@gani_gcm.register_property
def Gf(fuel: "Fuel") -> types.Quantity1D:
    """Predict the Gibbs free energy of formation (Gf) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted standard Gibbs free energies of formation (Gf) in kJ/mol.
    """
    nij = _get_decomp(fuel)
    gfj = _get_row("gfk")
    _check_compatible_dims(gfj, nij)
    gfi: types.Array1D = -14.828 + np.matmul(nij, gfj)
    return Units.Quantity(gfi, "kJ/mol")


@gani_gcm.register_property
def Hv_stp(fuel: "Fuel") -> types.Quantity1D:
    """Predict the enthalpy of vaporization at STP (Hv_stp) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted standard enthalpies of vaporization at standard temperature and
        pressure (Hv_stp) in kJ/mol.
    """
    nij = _get_decomp(fuel)
    hvj = _get_row("hvk")
    _check_compatible_dims(hvj, nij)
    hvi: types.Array1D = 6.829 + (np.matmul(nij, hvj))
    return Units.Quantity(hvi, "kJ/mol")


@gani_gcm.register_property
def omega(fuel: "Fuel") -> types.Quantity1D:
    """Predict the acentric factor (omega) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted acentric factors (omega).
    """
    nij = _get_decomp(fuel)
    wj = _get_row("wk")
    _check_compatible_dims(wj, nij)
    wi: types.Array1D = 0.4085 * np.power(
        np.log(np.matmul(nij, wj) + 1.1507), (1.0 / 0.5050)
    )
    return Units.Quantity(wi, "")


@gani_gcm.register_property
def Vm_stp(fuel: "Fuel") -> types.Quantity1D:
    """Predict the standard molar volume at STP (Vm_stp) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted standard molar volumes at STP (Vm_stp).
    """
    nij = _get_decomp(fuel)
    vmj = _get_row("vmk")
    _check_compatible_dims(vmj, nij)
    vmi: types.Array1D = 0.01211 + np.matmul(nij, vmj)
    return Units.Quantity(vmi, "m^3/kmol")


@gani_gcm.register_property
def Cp_stp(fuel: "Fuel") -> types.Quantity1D:
    """Predict the standard molar heat capacity at STP (Cp_stp) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted standard molar heat capacities at STP (Cp_stp).
    """
    nij = _get_decomp(fuel)
    cpj = _get_row("CpAk")
    _check_compatible_dims(cpj, nij)
    cpi: types.Array1D = np.matmul(nij, cpj) - 19.7779
    return Units.Quantity(cpi, "J/(mol*K)")


@gani_gcm.register_property
def Cp_B(fuel: "Fuel") -> types.Quantity1D:
    """Predict the molar heat capacity B correction (Cp_B) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted molar heat capacity B corrections (Cp_B).
    """
    nij = _get_decomp(fuel)
    cpbj = _get_row("CpBk")
    _check_compatible_dims(cpbj, nij)
    cpbi: types.Array1D = np.matmul(nij, cpbj)
    return Units.Quantity(cpbi, "J/(mol*K)")


@gani_gcm.register_property
def Cp_C(fuel: "Fuel") -> types.Quantity1D:
    """Predict the molar heat capacity C correction (Cp_C) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted molar heat capacity C corrections (Cp_C).
    """
    nij = _get_decomp(fuel)
    cpcj = _get_row("CpCk")
    _check_compatible_dims(cpcj, nij)
    cpci: types.Array1D = np.matmul(nij, cpcj)
    return Units.Quantity(cpci, "J/(mol*K)")


@gani_gcm.register_property
def rd_A(fuel: "Fuel") -> types.Quantity1D:
    """Predict the RD liquid heat-capacity coefficient A (rd_A) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted Ruzicka-Domalski liquid heat-capacity A coefficients (rd_A).
    """
    nij = _get_decomp(fuel)
    rd_aj = _get_row("rd_A")
    _check_compatible_dims(rd_aj, nij)
    rd_ai: types.Array1D = np.matmul(nij, rd_aj)
    return Units.Quantity(rd_ai, "")


@gani_gcm.register_property
def rd_B(fuel: "Fuel") -> types.Quantity1D:
    """Predict the RD liquid heat-capacity coefficient B (rd_B) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted Ruzicka-Domalski liquid heat-capacity B coefficients (rd_B).
    """
    nij = _get_decomp(fuel)
    rd_bj = _get_row("rd_B")
    _check_compatible_dims(rd_bj, nij)
    rd_bi: types.Array1D = np.matmul(nij, rd_bj)
    return Units.Quantity(rd_bi, "K^-1")


@gani_gcm.register_property
def rd_D(fuel: "Fuel") -> types.Quantity1D:
    """Predict the RD liquid heat-capacity coefficient D (rd_D) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted Ruzicka-Domalski liquid heat-capacity D coefficients (rd_D).
    """
    nij = _get_decomp(fuel)
    rd_dj = _get_row("rd_D")
    _check_compatible_dims(rd_dj, nij)
    rd_di: types.Array1D = np.matmul(nij, rd_dj)
    return Units.Quantity(rd_di, "K^-2")


@gani_gcm.register_property
def alibakhshi_phi(fuel: "Fuel") -> types.Quantity1D:
    """Predict the Alibakhshi phi parameter for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted Alibakhshi phi parameter.
    """
    nij = _get_decomp(fuel)
    phi_j = _get_row("alibakhshi_phi")
    _check_compatible_dims(phi_j, nij)
    phi_i: types.Array1D = np.matmul(nij, phi_j)
    return Units.Quantity(phi_i, "K")


@gani_gcm.register_property
def MW(fuel: "Fuel") -> types.Quantity1D:
    """Predict the molecular weight (MW) for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The predicted molecular weights (MW) in g/mol.
    """
    nij = _get_decomp(fuel)
    mw_j = _get_row("MW")
    _check_compatible_dims(mw_j, nij)
    mw_i: types.Array1D = np.matmul(nij, mw_j)
    return Units.Quantity(mw_i, "g/mol")
