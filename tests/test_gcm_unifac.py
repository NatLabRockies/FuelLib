"""Test the UNIFAC decomposition and activity coefficient model."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from fuellib import DEFAULT_DATA_DIR, Fuel, Units
from fuellib.gcm import unifac
from fuellib.gcm.unifac import AMN, MAIN_GROUP, TABLE, decompose

REF_COMPOUNDS = pd.read_csv(DEFAULT_DATA_DIR / "refCompounds.csv").set_index(
    "Reference Compound"
)
REF_UNIFAC = (
    pd
    .read_csv(DEFAULT_DATA_DIR / "refUNIFAC.csv")
    .set_index("Reference Compound")
    .fillna(0)
    .astype(int)
)


def _params(*smiles: str):
    """UNIFAC parameters for a mixture of the given SMILES strings."""
    decomp = pd.concat([decompose(s) for s in smiles])
    # parameters() only reads `fuel.unifacDecomp`, so a lightweight stand-in suffices
    return unifac.parameters(cast(Fuel, SimpleNamespace(unifacDecomp=decomp)))


def _ln_gamma_combinatorial_reference(x, nu, Rk, Qk):
    """Textbook Staverman-Guggenheim combinatorial term (z = 10)."""
    r, q = nu @ Rk, nu @ Qk
    phi = r * x / (r @ x)
    theta = q * x / (q @ x)
    ell = 5.0 * (r - q) - (r - 1.0)
    return np.log(phi / x) + 5.0 * q * np.log(theta / phi) + ell - phi / x * (x @ ell)


def _ln_gamma_residual_reference(x, T, nu, Qk, a_mn):
    """Textbook UNIFAC residual term, written with explicit loops."""
    psi = np.exp(-a_mn / T)
    K = len(Qk)

    def ln_Gamma(X):
        theta = Qk * X / np.sum(Qk * X)
        out = np.empty(K)
        for k in range(K):
            s_k = sum(theta[m] * psi[m, k] for m in range(K))
            t = sum(
                theta[m] * psi[k, m] / sum(theta[n] * psi[n, m] for n in range(K))
                for m in range(K)
            )
            out[k] = Qk[k] * (1.0 - np.log(s_k) - t)
        return out

    ln_G_mix = ln_Gamma(x @ nu / np.sum(x @ nu))
    result = np.zeros(len(x))
    for i in range(len(x)):
        mask = nu[i] > 0
        ln_G_pure = np.zeros(K)
        ln_G_pure[mask] = ln_Gamma(nu[i] / nu[i].sum())[mask]
        result[i] = np.sum(nu[i] * (ln_G_mix - ln_G_pure))
    return result


# ---------------------------------------------------------------------------
# Data tables
# ---------------------------------------------------------------------------


def test_table_parameters():
    """R and Q parameters should be finite and non-negative."""
    assert list(TABLE.index) == ["R", "Q"]
    assert np.all(np.isfinite(TABLE.to_numpy()))
    assert np.all(TABLE.to_numpy() >= 0)
    assert TABLE.columns.is_unique
    assert MAIN_GROUP.index.equals(TABLE.columns)


@pytest.mark.parametrize(
    "group, R, Q",
    [
        ("CH3", 0.9011, 0.848),
        ("CH2", 0.6744, 0.540),
        ("CH", 0.4469, 0.228),
        ("C", 0.2195, 0.0),
        ("ACH", 0.5313, 0.400),
        ("AC", 0.3652, 0.120),
        ("ACCH3", 1.2663, 0.968),
        ("ACCH2", 1.0396, 0.660),
    ],
)
def test_hydrocarbon_subgroup_parameters(group: str, R: float, Q: float):
    """Hydrocarbon subgroup R/Q should match the published UNIFAC values."""
    assert TABLE.loc["R", group] == pytest.approx(R)
    assert TABLE.loc["Q", group] == pytest.approx(Q)


def test_interaction_matrix():
    """a_mn should be square, complete, zero on the diagonal, and cover all groups."""
    assert AMN.shape[0] == AMN.shape[1]
    assert list(AMN.index) == list(AMN.columns)
    assert not AMN.isna().to_numpy().any()
    assert np.all(np.diag(AMN.to_numpy()) == 0)
    missing = set(MAIN_GROUP.unique()) - set(AMN.index)
    assert not missing, f"Main groups missing from unifac_amn.csv: {sorted(missing)}"


def test_registered_gcm():
    """UNIFAC should expose R and Q through the GCM registry."""
    assert unifac.unifac_gcm.list_property_fns() == ["q", "r"]


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_reference_names_match_ref_compounds():
    """Every refUNIFAC compound should be named as in refCompounds."""
    missing = REF_UNIFAC.index.difference(REF_COMPOUNDS.index)
    assert missing.empty, f"Not in refCompounds.csv: {list(missing)}"


def test_reference_groups_match_gcm_table():
    """Every refUNIFAC group column should be a column of the GCM table."""
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


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CCCCCCC", {"CH3": 2, "CH2": 5}),
        ("CC(C)C", {"CH3": 3, "CH": 1}),
        ("CC(C)(C)C", {"CH3": 4, "C": 1}),
        ("C1CCCCC1", {"CH2": 6}),
        ("c1ccccc1", {"ACH": 6}),
        ("Cc1ccccc1", {"ACH": 5, "ACCH3": 1}),
        ("CCc1ccccc1", {"ACH": 5, "ACCH2": 1, "CH3": 1}),
        ("CC(C)c1ccccc1", {"ACH": 5, "ACCH": 1, "CH3": 2}),
        ("c1ccc2ccccc2c1", {"ACH": 8, "AC": 2}),
        ("C=CCCC", {"CH2=CH": 1, "CH2": 2, "CH3": 1}),
        ("CC=CC", {"CH=CH": 1, "CH3": 2}),
        ("C=C(C)C", {"CH2=C": 1, "CH3": 2}),
        ("CC=C(C)C", {"CH=C": 1, "CH3": 3}),
        ("CC(C)=C(C)C", {"C=C": 1, "CH3": 4}),
    ],
)
def test_decompose(smiles: str, expected: dict[str, int]):
    """Simple hydrocarbons should decompose into the expected subgroups."""
    df = decompose(smiles)
    assert list(df.index) == [smiles]
    assert list(df.columns) == list(TABLE.columns)
    row = df.iloc[0]
    assert row[row > 0].to_dict() == expected


@pytest.mark.parametrize("smiles", ["not-a-smiles", "C1CC"])
def test_decompose_invalid_smiles(smiles: str):
    """Unparsable SMILES should raise a ValueError."""
    with pytest.raises(ValueError, match="Error parsing SMILES"):
        decompose(smiles)


@pytest.mark.parametrize("smiles", ["CCO", "CCN", "CCCl"])
def test_decompose_unmatched_atoms(smiles: str):
    """Heteroatoms are not covered by the hydrocarbon patterns."""
    with pytest.raises(ValueError, match="Not all atoms were matched"):
        decompose(smiles)


# ---------------------------------------------------------------------------
# Fuel-level R/Q and parameters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def heptane() -> Fuel:
    return Fuel("heptane")


@pytest.fixture(scope="module")
def posf10325() -> Fuel:
    return Fuel("posf10325")


def test_R_Q_heptane(heptane: Fuel):
    """R and Q for n-heptane: 2 CH3 + 5 CH2."""
    R = unifac.unifac_gcm.predict("R", heptane)
    Q = unifac.unifac_gcm.predict("Q", heptane)
    for prop in (R, Q):
        assert prop.check("")
        assert prop.magnitude.shape == (1,)
    assert R.magnitude[0] == pytest.approx(2 * 0.9011 + 5 * 0.6744)
    assert Q.magnitude[0] == pytest.approx(2 * 0.848 + 5 * 0.540)


def test_R_Q_multicomponent(posf10325: Fuel):
    """R and Q should be positive for every component of a real fuel."""
    for name in ("R", "Q"):
        prop = unifac.unifac_gcm.predict(name, posf10325).magnitude
        assert prop.shape == (posf10325.num_compounds,)
        assert np.all(np.isfinite(prop))
        assert np.all(prop > 0)


def test_check_compatible_dims():
    """Mismatched decomposition and contribution sizes should raise."""
    with pytest.raises(ValueError, match="Incompatible dimensions"):
        unifac._check_compatible_dims(np.ones(3), np.ones((2, 4)))


def test_parameters_only_used_groups(posf10325: Fuel):
    """parameters() should drop unused subgroups and keep consistent shapes."""
    nu, Rk, Qk, a_mn = unifac.parameters(posf10325)
    N, K = nu.shape
    assert N == posf10325.num_compounds
    assert Rk.shape == Qk.shape == (K,)
    assert a_mn.shape == (K, K)
    assert np.all(nu.any(axis=0))
    assert np.all(nu.sum(axis=1) > 0)
    full = unifac.unifac_gcm.predict("R", posf10325).magnitude
    np.testing.assert_allclose(nu @ Rk, full)


# ---------------------------------------------------------------------------
# Activity coefficient kernel
# ---------------------------------------------------------------------------

BINARIES = [
    ("CCCCCCC", "Cc1ccccc1"),  # heptane + toluene
    ("C1CCCCC1", "c1ccccc1"),  # cyclohexane + benzene
    ("CCCCCCCCCCCCCCCC", "C=CCCCC"),  # hexadecane + 1-hexene
]


@pytest.mark.parametrize("pair", BINARIES)
@pytest.mark.parametrize("T", [250.0, 300.0, 450.0])
def test_ln_gamma_matches_reference(pair, T):
    """Vectorized kernel should match an independent textbook implementation."""
    nu, Rk, Qk, a_mn = _params(*pair)
    for x1 in (0.05, 0.3, 0.5, 0.9):
        x = np.array([x1, 1.0 - x1])
        expected = _ln_gamma_combinatorial_reference(
            x, nu, Rk, Qk
        ) + _ln_gamma_residual_reference(x, T, nu, Qk, a_mn)
        np.testing.assert_allclose(
            unifac.ln_gamma_x(x, T, nu, Rk, Qk, a_mn), expected, atol=1e-12
        )


def test_ln_gamma_matches_reference_multicomponent(posf10325: Fuel):
    """Kernel should match the textbook implementation for a real fuel."""
    nu, Rk, Qk, a_mn = unifac.parameters(posf10325)
    rng = np.random.default_rng(0)
    x = rng.dirichlet(np.ones(nu.shape[0]))
    T = 350.0
    expected = _ln_gamma_combinatorial_reference(
        x, nu, Rk, Qk
    ) + _ln_gamma_residual_reference(x, T, nu, Qk, a_mn)
    np.testing.assert_allclose(
        unifac.ln_gamma_x(x, T, nu, Rk, Qk, a_mn), expected, atol=1e-10
    )


@pytest.mark.parametrize("pair", BINARIES)
def test_pure_component_limit(pair):
    """ln(gamma_i) -> 0 as x_i -> 1."""
    params = _params(*pair)
    for i in range(2):
        x = np.full(2, 1e-12)
        x[i] = 1.0 - 1e-12
        ln_g = unifac.ln_gamma_x(x, 300.0, *params)
        assert ln_g[i] == pytest.approx(0.0, abs=1e-9)


def test_same_main_group_is_athermal():
    """Alkane mixtures have only combinatorial nonideality (a_mn = 0)."""
    nu, Rk, Qk, a_mn = _params("CCCCC", "CCCCCCCCCCCCCCCC")
    assert np.all(a_mn == 0)
    x = np.array([0.4, 0.6])
    expected = _ln_gamma_combinatorial_reference(x, nu, Rk, Qk)
    for T in (250.0, 500.0):
        np.testing.assert_allclose(
            unifac.ln_gamma_x(x, T, nu, Rk, Qk, a_mn), expected, atol=1e-12
        )


def test_high_temperature_limit():
    """As T -> infinity, Psi -> 1 and the residual part vanishes."""
    nu, Rk, Qk, a_mn = _params("CCCCCCC", "Cc1ccccc1")
    x = np.array([0.3, 0.7])
    np.testing.assert_allclose(
        unifac.ln_gamma_x(x, 1e15, nu, Rk, Qk, a_mn),
        _ln_gamma_combinatorial_reference(x, nu, Rk, Qk),
        atol=1e-10,
    )


@pytest.mark.parametrize("pair", BINARIES)
def test_gibbs_duhem(pair):
    """sum_i x_i d ln(gamma_i) = 0 at constant T and P."""
    params = _params(*pair)
    eps = 1e-6
    for x1 in (0.1, 0.5, 0.8):
        x = np.array([x1, 1.0 - x1])
        v = np.array([1.0, -1.0])
        d = (
            unifac.ln_gamma_x(x + eps * v, 320.0, *params)
            - unifac.ln_gamma_x(x - eps * v, 320.0, *params)
        ) / (2 * eps)
        assert x @ d == pytest.approx(0.0, abs=1e-7)


def test_gibbs_duhem_multicomponent(posf10325: Fuel):
    """Gibbs-Duhem should hold along an arbitrary direction for a real fuel."""
    params = unifac.parameters(posf10325)
    N = params[0].shape[0]
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.full(N, 10.0))
    v = rng.standard_normal(N)
    v -= v.mean()
    v *= 0.5 * x.min() / np.abs(v).max()
    eps = 1e-5
    d = (
        unifac.ln_gamma_x(x + eps * v, 320.0, *params)
        - unifac.ln_gamma_x(x - eps * v, 320.0, *params)
    ) / (2 * eps)
    assert abs(x @ d) <= 1e-6 * np.abs(x * d).sum()


def test_aromatic_alkane_positive_deviation():
    """Toluene in heptane shows positive deviation from Raoult's law."""
    params = _params("CCCCCCC", "Cc1ccccc1")
    x = np.array([0.5, 0.5])
    ln_g = unifac.ln_gamma_x(x, 300.0, *params)
    assert np.all(ln_g > 0)
    # Infinite dilution activity coefficient exceeds the finite-concentration one.
    ln_g_inf = unifac.ln_gamma_x(np.array([1 - 1e-10, 1e-10]), 300.0, *params)
    assert ln_g_inf[1] > ln_g[1]


# ---------------------------------------------------------------------------
# Fuel-level ln_gamma
# ---------------------------------------------------------------------------


def test_ln_gamma_single_component(heptane: Fuel):
    """A pure fuel is ideal: ln(gamma) = 0."""
    ln_g = unifac.ln_gamma(heptane, Units.Quantity(300.0, "K"))
    assert ln_g.check("")
    np.testing.assert_allclose(ln_g.magnitude, 0.0, atol=1e-12)


def test_ln_gamma_fuel(posf10325: Fuel):
    """ln(gamma) for a real fuel should be finite, dimensionless and consistent."""
    T = Units.Quantity(300.0, "K")
    ln_g = unifac.ln_gamma(posf10325, T)
    assert ln_g.check("")
    assert ln_g.magnitude.shape == (posf10325.num_compounds,)
    assert np.all(np.isfinite(ln_g.magnitude))

    # Default Yi is the initial composition
    np.testing.assert_allclose(
        unifac.ln_gamma(posf10325, T, posf10325.Y_0).magnitude, ln_g.magnitude
    )

    # Temperature units are converted
    np.testing.assert_allclose(
        unifac.ln_gamma(posf10325, Units.Quantity(26.85, "degC")).magnitude,
        ln_g.magnitude,
        rtol=1e-10,
    )

    # Matches the kernel called with mole fractions
    x = posf10325.Y2X(posf10325.Y_0).to("dimensionless").magnitude
    np.testing.assert_allclose(
        ln_g.magnitude,
        unifac.ln_gamma_x(x, 300.0, *unifac.parameters(posf10325)),
    )


def test_ln_gamma_fuel_custom_composition(posf10325: Fuel):
    """Changing composition should change ln(gamma)."""
    T = Units.Quantity(300.0, "K")
    N = posf10325.num_compounds
    Y = Units.Quantity(np.full(N, 1.0 / N), "")
    assert not np.allclose(
        unifac.ln_gamma(posf10325, T, Y).magnitude,
        unifac.ln_gamma(posf10325, T).magnitude,
    )
