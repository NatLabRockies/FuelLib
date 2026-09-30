from pathlib import Path
import pytest
import os
import unittest

import numpy as np
import pandas as pd

from fuellib.utils import Units
from fuellib import Fuel, correlate

# Locate the tests baseline directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions")

BOLD = "\033[1;1m"
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[1;34m"
STOP = "\033[0m"


class BaselineCrossCheckTestCase(unittest.TestCase):
    """Verify the legacy per-fuel baselines match the new mixture baseline.

    TEMPORARY: This test only exists to prove that refactoring the API from
    `Fuel.mixture_*` methods (used by `CompTestCase` / the legacy per-fuel
    `<fuel>.csv` baseline files) to `fuellib.correlate.mixture.*` functions
    (used by `MixtureTestCase` / `mixture_baseline.csv`) did not change any
    predicted values. It should be deleted once this PR is merged.
    """

    data_dir = Path(__file__).parent / "baselinePredictions"
    mixture_baseline = pd.read_csv(data_dir / "mixture_baseline.csv")

    fuel_names = [
        "heptane",
        "decane",
        "dodecane",
        "posf10264",
        "posf10325",
        "posf10289",
    ]
    prop_names = [
        "Density",
        "Viscosity",
        "VaporPressure",
        "SurfaceTension",
        "ThermalConductivity",
    ]

    def test_legacy_and_mixture_baselines_match(self):
        """Old and new baseline predictions must agree for shared fuels/properties."""
        total_checks = 0
        mismatches = []

        for fuel_name in self.fuel_names:
            legacy_file = self.data_dir / f"{fuel_name}.csv"
            df_legacy = pd.read_csv(legacy_file)

            t_vals = df_legacy.Temperature.iloc[1:].to_numpy(dtype=float)
            t_units = df_legacy.Temperature.iloc[0]
            legacy_temps = Units.Quantity(t_vals, t_units).to("K").magnitude

            fuel_mixture = self.mixture_baseline[
                self.mixture_baseline["Fuel"] == fuel_name
            ]

            for prop in self.prop_names:
                legacy_vals = df_legacy[prop].iloc[1:].to_numpy(dtype=float)
                legacy_units = df_legacy[prop].iloc[0]

                prop_rows = fuel_mixture[fuel_mixture["Property"] == prop]
                new_temps_k = np.array([
                    Units.Quantity(row.Temp, row.Temp_Units).to("K").magnitude
                    for row in prop_rows.itertuples()
                ])

                for temp_k, legacy_val in zip(legacy_temps, legacy_vals):
                    if np.isnan(legacy_val):
                        continue
                    total_checks += 1

                    match_idx = np.flatnonzero(
                        np.isclose(new_temps_k, temp_k, atol=1e-6)
                    )
                    if match_idx.size == 0:
                        mismatches.append(
                            f"{fuel_name}/{prop} @ {temp_k:.4f} K: no matching "
                            "temperature found in mixture_baseline.csv"
                        )
                        continue

                    match_row = prop_rows.iloc[match_idx[0]]
                    legacy_q = Units.Quantity(legacy_val, legacy_units)
                    new_q = Units.Quantity(
                        match_row.Baseline_Value, match_row.Property_Units
                    )

                    if not np.isclose(
                        legacy_q.to(new_q.units).magnitude,
                        new_q.magnitude,
                        rtol=1e-6,
                    ):
                        mismatches.append(
                            f"{fuel_name}/{prop} @ {temp_k:.4f} K: "
                            f"legacy={legacy_q}, new={new_q}"
                        )

        print(
            f"\n{total_checks - len(mismatches)}/{total_checks} baseline "
            "values match between old and new format"
        )
        self.assertFalse(
            mismatches,
            msg=(
                "Baseline mismatches found between legacy and mixture "
                "baselines:\n" + "\n".join(mismatches)
            ),
        )


class MixtureTestCase(unittest.TestCase):
    """Test class for verifying the accuracy of mixture predictions."""

    data_dir = Path(__file__).parent / "baselinePredictions"
    base_file = data_dir / "mixture_baseline.csv"
    base_data = pd.read_csv(base_file)

    method_map = {
        "density": correlate.mixture.density,
        "viscosity": correlate.mixture.kinematic_viscosity_dutt,
        "vaporpressure": correlate.mixture.saturated_vapor_pressure,
        "dynamicviscosity": correlate.mixture.dynamic_viscosity_dutt,
        "surfacetension": correlate.mixture.surface_tension,
        "thermalconductivity": correlate.mixture.thermal_conductivity_latini,
        "cp": correlate.components.molar_specific_heat,
    }
    prop_width = max(len(prop) for prop in method_map.keys())

    def test_mixture_accuracy(self) -> None:
        """Test the accuracy of mixture predictions."""
        # Implement the test logic here
        passed_checks = 0
        total_checks = 0

        for fuel_name in self.base_data["Fuel"].unique():
            fuel = Fuel(fuel_name)
            fuel_data = self.base_data[self.base_data["Fuel"] == fuel_name]

            print(
                f"\n\n{BLUE}{fuel_name.capitalize()} Accuracy Regression Check via MAPE:{STOP}"
            )
            for prop_name in fuel_data["Property"].unique():
                with self.subTest(fuel=fuel_name, property=prop_name):
                    total_checks += 1

                    temps = []
                    temp_units = []
                    base_mapes = []
                    pred_mapes = []
                    prop_data = fuel_data[fuel_data["Property"] == prop_name]
                    method = self.method_map.get(
                        prop_name.replace(" ", "").strip().lower(), None
                    )
                    if method is None:
                        msg = f"No method found for property '{prop_name}'."
                        raise ValueError(msg)

                    for row in prop_data.itertuples():
                        T = Units.Quantity(row.Temp, row.Temp_Units)
                        pred = method(fuel=fuel, T=T).to(row.Property_Units)
                        pred_val = pred.magnitude
                        if isinstance(pred_val, np.ndarray):
                            if pred_val.size != 1:
                                msg = f"Expected a single value for property '{prop_name}' of fuel '{fuel_name}', but got an array of size {pred_val.size}."
                                raise ValueError(msg)
                            pred_val = pred_val.item()
                        # Fetch the baseline prediction and recreate the known property value from baseline value + error
                        # (This method skips an extra lookup)
                        base_val = row.Baseline_Value  # Baseline prediction value
                        prop_val = base_val + row.Baseline_Error  # Known value

                        temps.append(row.Temp)
                        temp_units.append(row.Temp_Units)
                        base_mapes.append(
                            abs(prop_val - base_val) / abs(prop_val) * 100
                        )
                        pred_mapes.append(
                            abs(prop_val - pred_val) / abs(prop_val) * 100
                        )

                    # Analyze the collected mapes for this property
                    base_mapes = np.array(base_mapes)
                    pred_mapes = np.array(pred_mapes)

                    regression_ok = np.allclose(pred_mapes, base_mapes) or np.all(
                        pred_mapes <= base_mapes
                    )
                    if regression_ok:
                        passed_checks += 1
                        print(
                            f"\n  {GREEN}✓ {prop_name:<{self.prop_width}}{STOP}"
                            f"\n    Baseline   = {np.mean(base_mapes):8.4f}%"
                            f"\n    New        = {np.mean(pred_mapes):8.4f}%"
                            f"\n    Difference = {np.mean(pred_mapes) - np.mean(base_mapes):8.4f}%"
                            "\n"
                        )

                    else:
                        print()
                        print(f"  {RED}✗ {prop_name:<{self.prop_width}}{STOP}")
                        header = (
                            f"    {'Temperature'.center(14)}  {'Baseline'.center(10)}  "
                            f"{'New'.center(10)}  {'Difference'.center(10)}"
                        )
                        print(f"    {'-' * (len(header) - 4)}")
                        print(header)
                        print(f"    {'-' * (len(header) - 4)}")
                        for temp, temp_unit, base_mape, pred_mape in zip(
                            temps, temp_units, base_mapes, pred_mapes
                        ):
                            diff = pred_mape - base_mape
                            temp_str = f"{temp:.2f} {temp_unit}"
                            print(
                                f"    {temp_str:>14}  {base_mape:>9.4f}%  "
                                f"{pred_mape:>9.4f}%  {diff:>9.4f}%"
                            )

                    self.assertTrue(
                        regression_ok,
                        msg=(
                            f"{fuel_name} / {prop_name}: MAPE regressed from "
                            f"{np.mean(base_mapes):.4f}% (baseline) to "
                            f"{np.mean(pred_mapes):.4f}%."
                        ),
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")


class TestFuelMWAccuracy:
    """Test class for verifying the accuracy of fuel molecular weight predictions."""

    @pytest.mark.parametrize(
        "fuel_name, expected_mw",
        [
            ("heptane", 0.10020),
            ("posf10325", 0.15897),
        ],
    )
    def test_fuel_mw(self, fuel_name: str, expected_mw: float) -> None:
        """Test that the mean molecular weight of the fuel roughly matches the expected value."""
        fuel = Fuel(fuel_name)
        mw = fuel.mean_molecular_weight(fuel.Y_0).magnitude  # kg/mol expected
        assert np.isclose(mw, expected_mw, atol=1e-4), (
            f"{fuel_name}: expected {expected_mw}, got {mw}"
        )


if __name__ == "__main__":
    unittest.main()
