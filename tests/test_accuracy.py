import os
import sys
import numpy as np
import pandas as pd
from get_pred_and_data import get_pred_and_data
import unittest

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)
from paths import *


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
            df_base = pd.read_csv(baseline_file, skiprows=[1])
            print(f"\n{fuel_name}:")

            for prop in prop_names:
                with self.subTest(fuel=fuel_name, prop=prop):
                    total_checks += 1

                    # Current model predictions and experimental reference data
                    T, data, pred = get_pred_and_data(fuel_name, prop)

                    # Baseline: align stored baseline predictions to the same
                    # temperature points, then compare against the same reference data.
                    df_base_prop = df_base[["Temperature", prop]].dropna()
                    pred_base = (
                        df_base_prop.set_index("Temperature")
                        .reindex(T)[prop]
                        .to_numpy()
                    )
                    mape_base = np.mean(np.abs(data - pred_base) / np.abs(data)) * 100
                    mape = np.mean(np.abs(data - pred) / np.abs(data)) * 100

                    # Regression check: MAPE must not exceed Baseline.
                    # np.isclose handles tiny floating-point noise when values
                    # are numerically equal but differ at machine precision.
                    regression_ok = (mape <= mape_base) or np.isclose(mape, mape_base)

                    if regression_ok:
                        passed_checks += 1
                        print(
                            "  "
                            f"✓ {prop:<{prop_width}}  "
                            f"New={mape:8.4f}%  "
                            f"Baseline={mape_base:8.4f}%"
                        )
                    else:
                        print(
                            "  "
                            f"✗ {prop:<{prop_width}}  "
                            f"New={mape:8.4f}% exceeds "
                            f"Baseline={mape_base:8.4f}%"
                        )

                    self.assertTrue(
                        regression_ok,
                        msg=(
                            f"{fuel_name} / {prop}: MAPE regressed from "
                            f"{mape_base:.4f}% (baseline) to {mape:.4f}%."
                        ),
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")


if __name__ == "__main__":
    unittest.main()
