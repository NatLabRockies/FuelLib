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
import FuelLib as fl

# Maximum MAPE (%) allowed for each property, regardless of baseline.
# These act as absolute sanity checks to catch catastrophic regressions.
ABS_MAPE_THRESHOLDS = {
    "Density": 5.0,
    "Viscosity": 35.0,
    "VaporPressure": 65.0,
    "SurfaceTension": 15.0,
    "ThermalConductivity": 15.0,
}


class CompTestCase(unittest.TestCase):
    """Test that prediction accuracy is preserved across PRs."""

    def test_accuracy(self):
        """MAPE must not regress beyond the stored baseline for any fuel/property."""

        # Maximum fractional increase in MAPE relative to baseline that is tolerated.
        max_error_diff = 1e-6

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

        total_checks = 0
        passed_checks = 0

        print("\nAccuracy Regression Check:")

        for fuel_name in fuel_names:
            baseline_file = os.path.join(TESTS_BASELINE_DIR, f"{fuel_name}.csv")
            df_base = pd.read_csv(baseline_file, skiprows=[1])
            print(f"\n{fuel_name}:")

            for prop in prop_names:
                with self.subTest(fuel=fuel_name, prop=prop):
                    total_checks += 1

                    # Current model predictions and experimental reference data
                    T, data, pred = get_pred_and_data(fuel_name, prop)

                    # Baseline MAPE: align stored baseline predictions to the same
                    # temperature points, then compare against the same reference data.
                    df_base_prop = df_base[["Temperature", prop]].dropna()
                    pred_base = (
                        df_base_prop.set_index("Temperature")
                        .reindex(T)[prop]
                        .to_numpy()
                    )
                    mape_base = np.mean(np.abs(data - pred_base) / np.abs(data)) * 100
                    mape = np.mean(np.abs(data - pred) / np.abs(data)) * 100
                    abs_limit = ABS_MAPE_THRESHOLDS[prop]

                    # 1. Regression check: MAPE must not exceed the baseline MAPE.
                    mape_limit = mape_base * (1 + max_error_diff)
                    regression_ok = mape <= mape_limit
                    absolute_ok = mape < abs_limit

                    if regression_ok and absolute_ok:
                        passed_checks += 1
                        print(
                            "  "
                            f"✓ {prop}: MAPE={mape:.4f}%, "
                            f"baseline={mape_base:.4f}%, "
                            f"regression_limit={mape_limit:.4f}%, "
                            f"absolute_limit={abs_limit:.4f}%"
                        )
                    elif not regression_ok:
                        print(
                            "  "
                            f"✗ {prop}: MAPE={mape:.4f}% exceeds "
                            f"regression_limit={mape_limit:.4f}% "
                            f"(baseline={mape_base:.4f}%)"
                        )
                    else:
                        print(
                            "  "
                            f"✗ {prop}: MAPE={mape:.4f}% exceeds "
                            f"absolute_limit={abs_limit:.4f}%"
                        )

                    self.assertTrue(
                        regression_ok,
                        msg=(
                            f"{fuel_name} / {prop}: MAPE regressed from "
                            f"{mape_base:.4f}% (baseline) to {mape:.4f}%."
                        ),
                    )
                    self.assertTrue(
                        absolute_ok,
                        msg=(
                            f"{fuel_name} / {prop}: MAPE of {mape:.4f}% exceeds "
                            f"absolute threshold of {abs_limit}%."
                        ),
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")


if __name__ == "__main__":
    unittest.main()
