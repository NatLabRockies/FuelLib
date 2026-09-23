import os
import unittest

import numpy as np
import pandas as pd
from get_pred_and_data import get_pred_and_data

from fuellib.units import PintUnits

# Locate the tests baseline directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions")

BOLD = "\033[1;1m"
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[1;34m"
STOP = "\033[0m"


class CompTestCase(unittest.TestCase):
    """Test that prediction accuracy is preserved across PRs."""

    def test_accuracy(self):
        """Compare MAPE of PR vs. stored baseline"""

        # Fuels to test
        fuel_names = [
            "heptane",
            "decane",
            "dodecane",
            "posf10264",
            "posf10325",
            "posf10289",
        ]

        # Properties to test
        prop_names = [
            "Density",
            "Viscosity",
            "VaporPressure",
            "SurfaceTension",
            "ThermalConductivity",
        ]
        prop_width = max(len(p) for p in prop_names)

        total_checks = 0
        passed_checks = 0

        print(f"\n\n{BLUE}Accuracy Regression Check via MAPE:{STOP}")

        for fuel_name in fuel_names:
            baseline_file = os.path.join(TESTS_BASELINE_DIR, f"{fuel_name}.csv")
            df_base = pd.read_csv(baseline_file)
            print(f"\n{BOLD}{fuel_name}:{STOP}\n")
            t_vals = df_base.Temperature.iloc[1:].to_numpy(dtype=float)
            t_units = df_base.Temperature.iloc[0]
            base_temps = PintUnits.Quantity(t_vals, t_units).to("K")

            for prop in prop_names:
                with self.subTest(fuel=fuel_name, prop=prop):
                    total_checks += 1

                    base_props = PintUnits.Quantity(
                        df_base[prop].iloc[1:].to_numpy(dtype=float),
                        df_base[prop].iloc[0],
                    )
                    valid_idxs = ~np.isnan(base_props)
                    valid_temps = base_temps[valid_idxs]
                    base_props = base_props[valid_idxs]

                    # Current model predictions and experimental reference data
                    T, data, pred = get_pred_and_data(fuel_name, prop)

                    self.assertTrue(
                        np.allclose(T, valid_temps),
                        msg=(
                            f"{fuel_name} / {prop}: baseline temperatures do not match "
                            "current data temperatures."
                        ),
                    )
                    mape_base = (
                        np.mean(np.abs(data - base_props) / np.abs(data)) * 100
                    )
                    mape = np.mean(np.abs(data - pred) / np.abs(data)) * 100

                    # Regression check: MAPE must not exceed Baseline.
                    # np.isclose handles tiny floating-point noise when values
                    # are numerically equal but differ at machine precision.
                    regression_ok = (mape <= mape_base) or np.isclose(mape, mape_base)

                    if regression_ok:
                        passed_checks += 1
                        print(
                            f"  {GREEN}"
                            f"✓ {prop:<{prop_width}}"
                            f"{STOP}"
                            f"\n    Baseline   = {mape_base.magnitude:8.4f}%"
                            f"\n    New        = {mape.magnitude:8.4f}%"
                            f"\n    Difference = {mape.magnitude - mape_base.magnitude:8.4f}%"
                            "\n"
                        )
                    else:
                        print(
                            f"  {RED}"
                            f"✗ {prop:<{prop_width}}"
                            f"{STOP}"
                            f"\n    Baseline   = {mape_base.magnitude:8.4f}%"
                            f"\n    New        = {mape.magnitude:8.4f}%"
                            f"\n    Difference = {mape.magnitude - mape_base.magnitude:8.4f}%"
                            "\n"
                        )

                    self.assertTrue(
                        regression_ok,
                        msg=(
                            f"{fuel_name} / {prop}: MAPE regressed from "
                            f"{mape_base.magnitude:.4f}% (baseline) to "
                            f"{mape.magnitude:.4f}%."
                        ),
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")


if __name__ == "__main__":
    unittest.main()
