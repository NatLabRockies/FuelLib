import os
import unittest

import numpy as np
import pandas as pd
from get_pred_and_data import get_pred_and_data

from fuellib.units import PintUnits

# Locate the tests baseline directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions")

BOLD = "\033[1;1m"  # ANSI escape code for bold text
RED = "\033[31m"  # ANSI escape code for red text
GREEN = "\033[32m"  # ANSI escape code for green text
BLUE = "\033[1;34m"  # ANSI escape code for blue text
STOP = "\033[0m"  # ANSI escape code to reset text color


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

            t_vals = df_base.Temperature.iloc[1:].to_numpy(dtype=float)
            t_units = df_base.Temperature.iloc[0]
            base_temps = PintUnits.Quantity(t_vals, t_units).to("K")

            print(f"\n{BOLD}{fuel_name}:{STOP}\n")
            for prop in prop_names:
                with self.subTest(fuel=fuel_name, prop=prop):
                    total_checks += 1

                    prop_vals = df_base[prop].iloc[1:].to_numpy(dtype=float)
                    prop_units = df_base[prop].iloc[0]
                    base_props = PintUnits.Quantity(prop_vals, prop_units)

                    # Current model predictions and experimental reference data
                    data_temps, data_props, pred_props = get_pred_and_data(
                        fuel_name, prop
                    )

                    # Align indices where baseline properties are not NaN.
                    valid_idxs = ~np.isnan(base_props)
                    valid_temps = base_temps[valid_idxs]
                    valid_props = base_props[valid_idxs]

                    self.assertTrue(
                        np.allclose(
                            data_temps,
                            valid_temps,
                            atol=1e-8,  # Account for fp-error
                        )
                    )

                    mape_base = (
                        np.mean(np.abs(data_props - valid_props) / np.abs(data_props))
                        * 100
                    )
                    mape_pred = (
                        np.mean(np.abs(data_props - pred_props) / np.abs(pred_props))
                        * 100
                    )
                    regression_ok = (
                        (mape_pred <= mape_base)
                        or np.isclose(
                            mape_pred,
                            mape_base,
                            atol=5e-1,  # Accept 0.5% difference as negligible (i.e., rounding).
                        )
                    )

                    if regression_ok:
                        passed_checks += 1
                        print(
                            f"  {GREEN}"
                            f"✓ {prop:<{prop_width}}"
                            f"{STOP}"
                            f"\n    Baseline   = {mape_base.magnitude:8.4f}%"
                            f"\n    New        = {mape_pred.magnitude:8.4f}%"
                            f"\n    Difference = {mape_pred.magnitude - mape_base.magnitude:8.4f}%"
                            "\n"
                        )
                    else:
                        print(
                            f"  {RED}"
                            f"✗ {prop:<{prop_width}}  "
                            f"{STOP}"
                            f"\n    Baseline   = {mape_base.magnitude:8.4f}%"
                            f"\n    New        = {mape_pred.magnitude:8.4f}%"
                            f"\n    Difference = {mape_pred.magnitude - mape_base.magnitude:8.4f}%"
                            "\n"
                        )

                    self.assertTrue(
                        regression_ok,
                        msg=(
                            f"{fuel_name} / {prop}: MAPE regressed from "
                            f"{mape_base.magnitude:.4f}% (baseline) to {mape_pred.magnitude:.4f}%."
                        ),
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")


if __name__ == "__main__":
    unittest.main()
