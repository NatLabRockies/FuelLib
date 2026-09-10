import os
import unittest

import astropy.units as u
import numpy as np
import pandas as pd
from get_pred_and_data import get_pred_and_data

from fuellib.units import convert_temperature

# Locate the tests baseline directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions")


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

        print("\n\nAccuracy Regression Check via MAPE:")

        for fuel_name in fuel_names:
            baseline_file = os.path.join(TESTS_BASELINE_DIR, f"{fuel_name}.csv")
            df_base = pd.read_csv(baseline_file)

            # Extract unit from column
            base_temp_unit = str(df_base["Temperature"].iloc[0])
            base_temp_data = df_base.Temperature.iloc[1:].to_numpy(dtype=float)
            base_temp_data = convert_temperature(
                u.Quantity(base_temp_data, base_temp_unit), "K"
            )

            print(f"\n{fuel_name}:")
            for prop in prop_names:
                with self.subTest(fuel=fuel_name, prop=prop):
                    total_checks += 1

                    # Current model predictions and experimental reference data
                    data_temps, data_props, pred_props = get_pred_and_data(
                        fuel_name,
                        prop_name=prop,  # ty: ignore[invalid-argument-type]
                    )

                    # Baseline: align stored baseline predictions to the same
                    # temperature points, then compare against the same reference data.
                    base_prop_unit = str(df_base[prop].iloc[0])
                    base_prop_data = df_base[prop].iloc[1:].to_numpy(dtype=float)

                    valid_idxs = ~np.isnan(base_prop_data)
                    base_props = u.Quantity(base_prop_data[valid_idxs], base_prop_unit)
                    base_temps = base_temp_data[valid_idxs]

                    self.assertTrue(
                        np.allclose(
                            data_temps, base_temps, atol=1e-8
                        ),  # Account for fp-error
                        msg=(
                            f"{fuel_name} / {prop}: baseline temperatures do not match "
                            "current data temperatures."
                        ),
                    )

                    mape_base = (
                        np.mean(np.abs(data_props - base_props) / np.abs(data_props))
                        * 100
                    )
                    mape_new = (
                        np.mean(np.abs(data_props - pred_props) / np.abs(data_props))
                        * 100
                    )

                    regression_ok = (mape_new <= mape_base) or np.isclose(
                        mape_new,
                        mape_base,
                        rtol=1e-8,  # Account for fp-error
                    )

                    if regression_ok:
                        passed_checks += 1
                        print(
                            "  "
                            f"✓ {prop:<{prop_width}}  "
                            f"New={mape_new:8.4f}%  "
                            f"Baseline={mape_base:8.4f}%"
                        )
                    else:
                        print(
                            "  "
                            f"✗ {prop:<{prop_width}}  "
                            f"New={mape_new:8.4f}% exceeds "
                            f"Baseline={mape_base:8.4f}%"
                        )

                    self.assertTrue(
                        regression_ok,
                        msg=(
                            f"{fuel_name} / {prop}: MAPE regressed from "
                            f"{mape_base:.4f}% (baseline) to {mape_new:.4f}%."
                        ),
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")


if __name__ == "__main__":
    unittest.main()
