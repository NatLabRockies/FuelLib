"""Test the UNIFAC 2.0 decomposition, group parameters, and activity coefficients."""

import numpy as np
import pandas as pd
import pytest

from fuellib import DEFAULT_DATA_DIR, Fuel, Units
from fuellib.gcm.unifac import AMN, MAIN_GROUP, TABLE, Q, R, decompose, ln_gamma

REF_COMPOUNDS = pd.read_csv(DEFAULT_DATA_DIR / "refCompounds.csv").set_index(
    "Reference Compound"
)
REF_UNIFAC = (
    pd
    .read_csv(DEFAULT_DATA_DIR / "refUnifac.csv")
    .set_index("Reference Compound")
    .fillna(0)
    .astype(int)
)
T_REF = Units.Quantity(298.15, "K")


def _ln_gamma_at_x(fuel: Fuel, T, x: np.ndarray) -> np.ndarray:
    """Evaluate ln(gamma) at mole fractions x."""
    Yi = fuel.X2Y(Units.Quantity(x, "dimensionless"))
    return ln_gamma(fuel, T, Yi).magnitude


def test_table_parsing():
    """R and Q should be numeric for all 113 subgroups with valid main groups."""
    assert list(TABLE.index) == ["R", "Q"]
    assert TABLE.shape == (2, 113)
    assert all(dtype == float for dtype in TABLE.dtypes)
    assert "CHO" in TABLE.columns and "CHO.1" in TABLE.columns
    assert MAIN_GROUP.index.equals(TABLE.columns)


def test_interaction_matrix():
    """a_mn should be square, cover every main group, and have a zero diagonal."""
    assert AMN.shape[0] == AMN.shape[1]
    assert list(AMN.index) == list(AMN.columns)
    missing = set(MAIN_GROUP.unique()) - set(AMN.index)
    assert not missing, f"Main groups missing from unifac_amn.csv: {sorted(missing)}"
    assert np.all(np.isfinite(AMN.to_numpy()))
    np.testing.assert_array_equal(np.diag(AMN.to_numpy()), 0.0)


def test_reference_names_match_ref_compounds():
    """Every refUnifac compound should be named as in refCompounds."""
    missing = REF_UNIFAC.index.difference(REF_COMPOUNDS.index)
    assert missing.empty, f"Not in refCompounds.csv: {list(missing)}"


def test_reference_groups_match_gcm_table():
    """Every refUnifac group column should be a column of the GCM table."""
    missing = REF_UNIFAC.columns.difference(TABLE.columns)
    assert missing.empty, f"Not in unifac.csv: {list(missing)}"


@pytest.mark.parametrize("name", REF_UNIFAC.index.intersection(REF_COMPOUNDS.index))
def test_reference_decomposition(name: str):
    """The SMILES decomposition should match the tabulated UNIFAC groups."""
    smiles = REF_COMPOUNDS.loc[name, "SMILES"]
    computed = decompose(smiles).iloc[0]
    expected = REF_UNIFAC.loc[name].reindex(TABLE.columns, fill_value=0)

    diff = {
        group: (int(computed[group]), int(expected[group]))
        for group in TABLE.columns
        if computed[group] != expected[group]
    }
    assert not diff, f"{name} ({smiles}) group (computed, tabulated): {diff}"


def test_r_q_decane():
    """Decane = 2 CH3 + 8 CH2."""
    fuel = Fuel("decane")
    np.testing.assert_allclose(R(fuel).magnitude, [2 * 0.9011 + 8 * 0.6744])
    np.testing.assert_allclose(Q(fuel).magnitude, [2 * 0.848 + 8 * 0.54])


@pytest.mark.parametrize("T_K", [250.0, 298.15, 400.0])
def test_ln_gamma_pure_component_is_zero(T_K: float):
    """A pure component is ideal: ln(gamma) = 0."""
    fuel = Fuel("decane")
    lng = ln_gamma(fuel, Units.Quantity(T_K, "K"))
    assert lng.check("dimensionless")
    np.testing.assert_allclose(lng.magnitude, 0.0, atol=1e-12)


@pytest.mark.parametrize("fuel_name", ["heptane-decane", "posf10325"])
def test_ln_gamma_shape_and_finite(fuel_name: str):
    """ln(gamma) should have one finite value per component."""
    fuel = Fuel(fuel_name)
    lng = ln_gamma(fuel, T_REF)
    assert lng.check("dimensionless")
    assert lng.magnitude.shape == (fuel.num_compounds,)
    assert np.all(np.isfinite(lng.magnitude))


def test_ln_gamma_alkane_mixture_near_ideal():
    """Similar n-alkanes should be nearly ideal."""
    lng = ln_gamma(Fuel("heptane-decane"), T_REF).magnitude
    assert np.all(np.abs(lng) < 0.1)


def test_ln_gamma_default_composition():
    """Omitting Yi should use the fuel's initial mass fractions."""
    fuel = Fuel("posf10325")
    np.testing.assert_allclose(
        ln_gamma(fuel, T_REF).magnitude, ln_gamma(fuel, T_REF, fuel.Y_0).magnitude
    )


def test_ln_gamma_temperature_units():
    """Temperature should be unit-converted."""
    fuel = Fuel("posf10325")
    np.testing.assert_allclose(
        ln_gamma(fuel, T_REF).magnitude,
        ln_gamma(fuel, Units.Quantity(25.0, "degC")).magnitude,
    )


@pytest.mark.parametrize("fuel_name", ["heptane-decane", "posf10325"])
def test_ln_gamma_gibbs_duhem(fuel_name: str):
    """At constant T, sum_i x_i d(ln gamma_i) = 0 for any composition change."""
    fuel = Fuel(fuel_name)
    x0 = fuel.Y2X(fuel.Y_0).magnitude
    rng = np.random.default_rng(0)
    z = rng.standard_normal(x0.size)
    d = x0 * (z - x0 @ z)  # sum(d) = 0 and keeps x0 +/- eps*d positive
    eps = 1e-5

    dlng = (
        _ln_gamma_at_x(fuel, T_REF, x0 + eps * d)
        - _ln_gamma_at_x(fuel, T_REF, x0 - eps * d)
    ) / (2 * eps)

    assert np.max(np.abs(dlng)) > 0  # non-trivial perturbation
    assert abs(x0 @ dlng) < 1e-6 * np.max(np.abs(dlng))
