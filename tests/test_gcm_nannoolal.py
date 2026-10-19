"""Test the Nannoolal decomposition against the bundled reference data."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fuellib import DEFAULT_DATA_DIR, Fuel, Units
from fuellib.correlate import components, mixture
from fuellib.gcm.nannoolal import TABLE, decompose

REF_COMPOUNDS = pd.read_csv(DEFAULT_DATA_DIR / "refCompounds.csv").set_index(
    "Reference Compound"
)
REF_NANNOOLAL = (
    pd
    .read_csv(DEFAULT_DATA_DIR / "refNannoolal.csv")
    .set_index("Reference Compound")
    .fillna(0)
    .astype(int)
)


def test_reference_names_match_ref_compounds():
    """Every refNannoolal compound should be named as in refCompounds."""
    missing = REF_NANNOOLAL.index.difference(REF_COMPOUNDS.index)
    assert missing.empty, f"Not in refCompounds.csv: {list(missing)}"


def test_reference_groups_match_gcm_table():
    """Every refNannoolal group column should be a row of the GCM table."""
    missing = REF_NANNOOLAL.columns.difference(TABLE.index)
    assert missing.empty, f"Not in nannoolal.csv: {list(missing)}"


@pytest.mark.parametrize(
    "name",
    REF_NANNOOLAL.index.intersection(REF_COMPOUNDS.index),
)
def test_reference_decomposition(name: str):
    """The SMILES decomposition should match the tabulated Nannoolal groups."""
    smiles = REF_COMPOUNDS.loc[name, "SMILES"]
    computed = decompose(smiles).iloc[0]
    expected = REF_NANNOOLAL.loc[name].reindex(TABLE.index, fill_value=0)

    diff = {
        group: (int(computed[group]), int(expected[group]))
        for group in TABLE.index
        if computed[group] != expected[group]
    }
    assert not diff, f"{name} ({smiles}) group (computed, tabulated): {diff}"


@pytest.mark.parametrize("fuel_name, warns", [("heptane", False), ("posf10325", True)])
def test_viscosity(fuel_name: str, warns: bool):
    """Nannoolal viscosities should be finite and positive with correct units."""
    fuel = Fuel(fuel_name)
    T = Units.Quantity(298.15, "K")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mu_i = components.dynamic_viscosity_nannoolal(fuel, T)
    zero_warnings = [w for w in caught if "no published contribution" in str(w.message)]
    assert bool(zero_warnings) == warns

    nu_i = components.kinematic_viscosity_nannoolal(fuel, T)
    for prop, units in [(mu_i, "Pa*s"), (nu_i, "m^2/s")]:
        assert prop.check(units)
        assert prop.magnitude.shape == (fuel.num_compounds,)
        assert np.all(np.isfinite(prop.magnitude))
        assert np.all(prop.magnitude > 0)

    mu = mixture.dynamic_viscosity_nannoolal(fuel, T)
    nu = mixture.kinematic_viscosity_nannoolal(fuel, T)
    for prop, units in [(mu, "Pa*s"), (nu, "m^2/s")]:
        assert prop.check(units)
        assert np.ndim(prop.magnitude) == 0
        assert np.isfinite(prop.magnitude)
        assert prop.magnitude > 0
