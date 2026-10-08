"""Reference equations and integration tests for selectable viscosity models."""

import numpy as np
import pytest
from get_pred_and_data import get_pred_and_data

import fuellib as fl
from fuellib._viscosity import (
    cg_to_nannoolal_groups,
    nannoolal_dynamic_viscosity,
    nannoolal_parameters,
)


def test_nannoolal_toluene_reference_equations():
    """Verify equations 7 and 8 independently of the packaged table loader."""
    groups = {15: 5, 16: 1, 3: 1}
    boiling_point = 383.75
    temperature = 298.15
    sum_dbv = 5 * -0.002783983 + 0.04594031 - 0.01106602
    sum_tv = 5 * 113.9028 - 26.61952 + 80.96984
    expected_dbv = sum_dbv / (7**-2.5635 + 0.0685) + 3.7777
    expected_tv = (
        21.8444 * np.sqrt(boiling_point)
        + sum_tv**0.9315 / (7**0.6577 + 4.9259)
        - 231.1361
    )
    dbv, tv = nannoolal_parameters(groups, 7, boiling_point)
    assert dbv == pytest.approx(expected_dbv, rel=1e-13)
    assert tv == pytest.approx(expected_tv, rel=1e-13)
    expected_mpas = 1.3 * np.exp(
        -expected_dbv * (temperature - expected_tv) / (temperature - expected_tv / 16)
    )
    assert nannoolal_dynamic_viscosity(temperature, dbv, tv) == pytest.approx(
        expected_mpas * 1e-3, rel=1e-13
    )
    assert nannoolal_dynamic_viscosity(tv, dbv, tv) == pytest.approx(1.3e-3)


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ({"CH3": 2, "CH2": 10}, {1: 2, 4: 10}),
        ({"CH3": 7, "CH2": 2, "CH": 1, "C": 2}, {1: 7, 4: 2, 5: 1, 6: 2}),
        (
            {"CH3": 1, "CH2": 8, "CH": 1, "6 membered ring": 1},
            {1: 1, 4: 3, 9: 5, 10: 1},
        ),
        ({"CH2": 8, "CH": 2, "6 membered ring": 2}, {9: 8, 10: 2}),
        (
            {"CH2": 6, "CH": 4, "5 membered ring": 2, "4 membered ring": 1},
            {9: 6, 10: 4, 125: 1, 126: 2},
        ),
        ({"CH3": 1, "CH2": 2, "ACCH2": 1, "ACH": 5}, {1: 1, 4: 2, 8: 1, 15: 5, 16: 1}),
        ({"ACH": 8, "AC": 2}, {15: 8, 18: 2}),
        (
            {"CH2": 1, "ACH": 4, "ACCH2": 2, "5 membered ring": 1},
            {9: 1, 14: 2, 15: 4, 16: 2, 126: 1},
        ),
        ({"CH3": 1, "CH2": 9, "CH2=CH": 1}, {1: 1, 4: 9, 61: 1}),
        ({"CH3": 2, "CH2": 3, "CH=CH": 1}, {1: 2, 4: 3, 58: 1}),
        ({"CH3": 2, "CH2": 4, "CH2=C": 1}, {1: 2, 4: 4, 61: 1}),
        ({"CH3": 3, "CH2": 4, "CH=C": 1}, {1: 3, 4: 4, 58: 1}),
        ({"CH3": 4, "C=C": 1}, {1: 4, 58: 1}),
    ],
)
def test_cg_mapping(groups, expected):
    """Check hydrocarbon mappings and carbon conservation."""
    mapped, n_atoms = cg_to_nannoolal_groups(groups)
    assert mapped == expected
    assert (
        sum(
            (2 if group in (58, 61) else 1) * count
            for group, count in mapped.items()
            if group < 125
        )
        == n_atoms
    )


@pytest.mark.parametrize(
    "fuel_name",
    [
        "heptane",
        "decane",
        "dodecane",
        "posf10264",
        "posf10289",
        "posf10325",
        "posf11498",
    ],
)
@pytest.mark.parametrize("temperature", [233.15, 298.15, 373.15])
def test_selectable_component_viscosities(fuel_name, temperature):
    """Preserve Dutt results while supporting indexed and all-component Nannoolal."""
    fuel = fl.fuel(fuel_name)
    assert np.array_equal(
        fuel.viscosity_kinematic(temperature),
        fuel.viscosity_kinematic(temperature, model="Dutt"),
    )
    assert np.array_equal(
        fuel.viscosity_dynamic(temperature),
        fuel.viscosity_dynamic(temperature, model="Dutt"),
    )
    dynamic = fuel.viscosity_dynamic(temperature, model="Nannoolal")
    kinematic = fuel.viscosity_kinematic(temperature, model="nannoolal")
    assert dynamic.shape == kinematic.shape == (fuel.num_compounds,)
    assert np.isfinite(dynamic).all() and (dynamic > 0).all()
    np.testing.assert_allclose(
        dynamic, kinematic * fuel.density(temperature), rtol=1e-13
    )
    for index in range(fuel.num_compounds):
        assert fuel.viscosity_dynamic(
            temperature, index, model="NANNOOLAL"
        ) == pytest.approx(dynamic[index])
        assert fuel.viscosity_kinematic(
            temperature, index, model="Nannoolal"
        ) == pytest.approx(kinematic[index])


@pytest.mark.parametrize("correlation", ["Kendall-Monroe", "Arrhenius"])
def test_selected_model_reaches_mixture_rules(correlation):
    """Use mole-fraction kinematic mixing followed by FuelLib mixture density."""
    fuel = fl.fuel("heptane-decane")
    temperature = 298.15
    component = fuel.viscosity_kinematic(temperature, model="Nannoolal")
    fractions = fuel.Y2X(fuel.Y_0)
    expected = (
        (fractions @ np.cbrt(component)) ** 3
        if correlation == "Kendall-Monroe"
        else np.exp(fractions @ np.log(component))
    )
    mixture = fuel.mixture_kinematic_viscosity(
        fuel.Y_0, temperature, correlation, model="Nannoolal"
    )
    assert mixture == pytest.approx(expected)
    dynamic = fuel.mixture_dynamic_viscosity(
        fuel.Y_0, temperature, correlation, model="Nannoolal"
    )
    assert dynamic == pytest.approx(
        mixture * fuel.mixture_density(fuel.Y_0, temperature)
    )


@pytest.mark.parametrize("model", ["Dutt", "Nannoolal"])
@pytest.mark.parametrize("correlation", ["Kendall-Monroe", "Arrhenius"])
def test_pure_component_mixture_limit(model, correlation):
    """Both mixing rules recover the pure-component viscosity."""
    fuel = fl.fuel("decane")
    temperature = 298.15
    assert fuel.mixture_kinematic_viscosity(
        fuel.Y_0, temperature, correlation, model=model
    ) == pytest.approx(fuel.viscosity_kinematic(temperature, 0, model=model))
    assert fuel.mixture_dynamic_viscosity(
        fuel.Y_0, temperature, correlation, model=model
    ) == pytest.approx(fuel.viscosity_dynamic(temperature, 0, model=model))


@pytest.mark.parametrize(
    "groups",
    [
        {},
        {"CH3": -1},
        {"CH2": 0.5},
        {"CH2": np.nan},
        {"CH2": np.inf},
        {"CH3": 1, "OH": 1},
        {"CH2": 4, "6 membered ring": 1},
        {"ACH": 7},
        {"ACH": 8, "AC": 1, "ACCH3": 1},
        {"CH2=CH": 2},
        {"CH2=C=CH": 1},
        {"CH2=CH": 1, "5 membered ring": 1},
        {"CH2=CH": 1, "ACH": 5, "AC": 1},
    ],
)
def test_invalid_or_unsupported_mapping(groups):
    """Do not invent contributions for incomplete or unsupported structures."""
    with pytest.raises(ValueError):
        cg_to_nannoolal_groups(groups)


@pytest.mark.parametrize(
    "temperature", [0, -1, np.nan, np.inf, 300 / 16, 300 / 16 + 1e-10]
)
def test_invalid_nannoolal_temperature(temperature):
    """Reject invalid temperatures, singularities, and numerical overflow."""
    with pytest.raises(ValueError):
        nannoolal_dynamic_viscosity(temperature, 4.0, 300.0)


@pytest.mark.parametrize(
    ("groups", "n_atoms", "boiling_point"),
    [
        ({999: 1}, 7, 383.75),
        ({1: -1}, 7, 383.75),
        ({1: 0.5}, 7, 383.75),
        ({1: np.nan}, 7, 383.75),
        ({}, 7, 383.75),
        ({1: 1}, 0.5, 383.75),
        ({1: 1}, 0, 383.75),
        ({1: 1}, 7, np.inf),
        ({1: 1}, 7, 0),
    ],
)
def test_invalid_nannoolal_parameters(groups, n_atoms, boiling_point):
    """Reject invalid parameters before calculating a temperature curve."""
    with pytest.raises(ValueError):
        nannoolal_parameters(groups, n_atoms, boiling_point)


@pytest.mark.parametrize("model", ["hybrid", "unknown", None])
def test_unknown_model_rejected(model):
    """Every viscosity entry point validates the requested model."""
    fuel = fl.fuel("decane")
    for method in (fuel.viscosity_dynamic, fuel.viscosity_kinematic):
        with pytest.raises(ValueError, match="Unknown viscosity model"):
            method(298.15, model=model)
    for method in (fuel.mixture_dynamic_viscosity, fuel.mixture_kinematic_viscosity):
        with pytest.raises(ValueError, match="Unknown viscosity model"):
            method(fuel.Y_0, 298.15, model=model)


def test_mapping_is_lazy_and_errors_identify_component():
    """Unsupported rows must not affect Dutt or an indexed supported row."""
    fuel = fl.fuel("heptane-decane")
    fuel.Nij = fuel.Nij.astype(float)
    fuel.Nij[1, 0] += 0.5
    assert not fuel._nannoolal_cache
    assert np.isfinite(fuel.viscosity_dynamic(298.15)).all()
    assert fuel.viscosity_dynamic(298.15, 0, model="Nannoolal") > 0
    assert set(fuel._nannoolal_cache) == {0}
    with pytest.raises(ValueError, match=fuel.compounds[1]):
        fuel.viscosity_dynamic(298.15, model="Nannoolal")


def test_nannoolal_dynamic_is_independent_of_density(monkeypatch):
    """Only conversion to kinematic viscosity requires a density prediction."""
    fuel = fl.fuel("decane")
    monkeypatch.setattr(fuel, "density", lambda temperature, comp_idx=None: np.nan)
    assert fuel.viscosity_dynamic(298.15, model="Nannoolal") > 0
    with pytest.raises(ValueError, match="liquid density"):
        fuel.viscosity_kinematic(298.15, model="Nannoolal")


@pytest.mark.parametrize(
    ("fuel_name", "expected_mape"),
    [
        ("heptane", 24.121001),
        ("decane", 10.809488),
        ("dodecane", 3.674697),
        ("posf10264", 8.113293),
        ("posf10289", 14.699858),
        ("posf10325", 8.645421),
    ],
)
def test_nannoolal_packaged_reference_accuracy(fuel_name, expected_mape):
    """Record accuracy against measurements without replacing the Dutt baselines."""
    temperatures, observed, predicted = get_pred_and_data(
        fuel_name, "Viscosity", viscosity_model="Nannoolal"
    )
    _, _, dutt = get_pred_and_data(fuel_name, "Viscosity")
    relative_errors = (predicted - observed) / observed
    mape = np.mean(np.abs(relative_errors)) * 100
    bias = np.mean(relative_errors) * 100
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    dutt_mape = np.mean(np.abs(dutt - observed) / observed) * 100
    print(
        f"\n{fuel_name}: n={len(temperatures)}, Dutt MAPE={dutt_mape:.4f}%, "
        f"Nannoolal MAPE={mape:.4f}%, bias={bias:.4f}%, RMSE={rmse:.6f} mm^2/s"
    )
    assert mape == pytest.approx(expected_mape, abs=1e-5)
