"""Tests for the Greenfield et al. (1998) ASTM D86 distillation model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuellib import DEFAULT_DATA_DIR, Fuel, Units
from fuellib.correlate.distillation import (
    D86Result,
    _heat_transfer_coefficient,
    distillation_curve_greenfield,
)
from fuellib.gcm import gani, nannoolal, unifac

# Ford synfuel (Greenfield et al., 1998, Table 2): mass %, and critical properties
# from Table 3 (Tc in K, Pc in bar, Vc in cm^3/mol, omega)
SYNFUEL = {
    "isopentane": ("CC(C)CC", 17.03, 460.4, 33.9, 306.0, 0.227),
    "cyclohexane": ("C1CCCCC1", 24.23, 553.5, 40.7, 308.0, 0.212),
    "toluene": ("CC1=CC=CC=C1", 17.35, 591.8, 41.0, 316.0, 0.263),
    "ethylbenzene": ("CCC1=CC=CC=C1", 12.71, 617.2, 36.0, 374.0, 0.302),
    "isooctane": ("CC(C)CC(C)(C)C", 19.11, 544.0, 25.7, 468.0, 0.303),
    "naphthalene": ("C1=CC=C2C=CC=CC2=C1", 1.08, 748.4, 40.5, 413.0, 0.302),
    "n-decane": ("CCCCCCCCCC", 8.49, 617.7, 21.2, 603.0, 0.489),
}
SYNFUEL_D86 = {  # recovered volume fraction: measured D86 temperature in K
    0.05: 324,
    0.1: 333,
    0.2: 343,
    0.3: 354,
    0.4: 363,
    0.5: 370,
    0.6: 375,
    0.7: 381,
    0.8: 391,
    0.9: 412,
    0.95: 428,
}


@pytest.fixture(scope="module")
def synfuel(tmp_path_factory: pytest.TempPathFactory) -> Fuel:
    """Ford synfuel with the critical properties used by Greenfield et al."""
    data_dir: Path = tmp_path_factory.mktemp("synfuel")
    (data_dir / "gcData").mkdir()
    columns = pd.read_csv(DEFAULT_DATA_DIR / "refCompounds.csv", nrows=0).columns
    rows = []
    for name, (smiles, _, Tc, Pc, Vc, omega) in SYNFUEL.items():
        row: dict[str, object] = dict.fromkeys(columns, np.nan)
        row.update({
            "Family": "test",
            "Carbon Number": 0,
            "Reference Compound": name,
            "SMILES": smiles,
            "Tc_Value": Tc,
            "Tc_Units": "K",
            "Pc_Value": Pc,
            "Pc_Units": "bar",
            "Vc_Value": Vc,
            "Vc_Units": "cm^3/mol",
            "Omega_Value": omega,
            "Omega_Units": "",
        })
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(
        data_dir / "refCompounds.csv", index=False
    )
    for module, file_name in [
        (gani, "refGani.csv"),
        (nannoolal, "refNannoolal.csv"),
        (unifac, "refUnifac.csv"),
    ]:
        decomp = pd.concat([
            module.decompose(smiles).set_axis([name])
            for name, (smiles, *_) in SYNFUEL.items()
        ])
        decomp.rename_axis("Reference Compound").reset_index().to_csv(
            data_dir / file_name, index=False
        )
    pd.DataFrame({
        "Reference Compound": list(SYNFUEL),
        "Weight %": [values[1] for values in SYNFUEL.values()],
    }).to_csv(data_dir / "gcData" / "synfuel.csv", index=False)
    return Fuel("synfuel", fuelDataDir=data_dir)


@pytest.fixture(scope="module")
def synfuel_d86(synfuel: Fuel) -> D86Result:
    """D86 simulation of Ford synfuel with the full model."""
    return distillation_curve_greenfield(synfuel)


def test_heat_transfer_coefficient():
    """Natural convection from the D86 neck gives a few W/(m^2*K)."""
    assert _heat_transfer_coefficient(298.15, 298.15, 0.13) == 0.0
    h = _heat_transfer_coefficient(350.0, 298.15, 0.13)
    assert 4.0 < h < 8.0


def test_synfuel_matches_greenfield_table_2(synfuel_d86: D86Result):
    """The full model reproduces the measured synfuel D86 curve."""
    fractions = np.array(list(SYNFUEL_D86))
    measured = np.array(list(SYNFUEL_D86.values()))
    predicted = synfuel_d86.temperature_at(fractions).to("K").magnitude
    np.testing.assert_allclose(predicted, measured, atol=6.0)
    assert np.sqrt(np.mean((predicted - measured) ** 2)) < 3.0


def test_synfuel_result_histories(synfuel_d86: D86Result):
    """Time histories are consistent with the D86 procedure."""
    r = synfuel_d86
    vol = r.volume_recovered.to("dimensionless").magnitude
    assert np.all(np.diff(r.time.to("s").magnitude) > 0)
    assert np.all(np.diff(vol) >= 0)
    assert 0.95 < vol[-1] < 1.0
    # The vapor (thermometer) never exceeds the boiling liquid
    assert np.all(r.T_vapor.magnitude <= r.T_liquid.magnitude + 1e-9)
    assert np.all(r.T_condenser.magnitude <= r.T_liquid.magnitude + 1e-9)
    X = r.liquid_mole_fractions.magnitude
    np.testing.assert_allclose(X.sum(axis=1), 1.0)
    # The residue is mostly the heaviest compounds
    assert X[-1, -2:].sum() > 0.9


def test_cstr_lowers_initial_temperatures(synfuel: Fuel, synfuel_d86: D86Result):
    """Without the air-filled CSTR the early D86 temperatures are higher."""
    partial = distillation_curve_greenfield(synfuel, air_cstr=False)
    assert partial.temperature_at(0.05) > synfuel_d86.temperature_at(
        0.05
    ) + Units.Quantity(5.0, "K")


def test_pure_compound_boils_at_constant_temperature():
    """A pure compound distills at its (correlated) normal boiling point."""
    fuel = Fuel("decane")
    r = distillation_curve_greenfield(fuel)
    psat = fuel.psat(r.temperature_at(0.5)).to("Pa").magnitude[0]
    assert psat == pytest.approx(101325.0, rel=1e-3)
    T = r.temperature_at(np.array([0.4, 0.5, 0.9])).to("K").magnitude
    np.testing.assert_allclose(T, T[1], atol=0.1)


def test_custom_composition_shifts_curve():
    """A lighter charge of the same compounds distills at lower temperatures."""
    fuel = Fuel("heptane-decane")
    Yi = Units.Quantity([0.95, 0.05], "dimensionless")
    base = distillation_curve_greenfield(fuel)
    light = distillation_curve_greenfield(fuel, Yi)
    assert light.temperature_at(0.5) < base.temperature_at(0.5)
