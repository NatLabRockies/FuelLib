from pathlib import Path
import pytest
import os
import unittest

import numpy as np
import pandas as pd

try:
    import jax.numpy as jnp
    from fuellib.simulate.inverse import mixing

    HAS_INVERSE = True
except ImportError:
    HAS_INVERSE = False
from fuellib.utils import Units
from fuellib import Fuel, correlate
from baselinePredictions.generate_baseline import method_map

# Locate the tests baseline directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions")

BOLD = "\033[1;1m"
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[1;34m"
STOP = "\033[0m"


class MixtureTestCase(unittest.TestCase):
    """Test class for verifying the accuracy of mixture predictions."""

    data_dir = Path(__file__).parent / "baselinePredictions"
    base_file = data_dir / "mixture_baseline.csv"
    base_data = pd.read_csv(base_file, float_precision="round_trip")

    prop_width = max(len(prop) for prop in method_map.keys())

    def test_mixture_accuracy(self) -> None:
        """Test the accuracy of mixture predictions."""
        # Implement the test logic here
        passed_checks = 0
        total_checks = 0

        for fuel_name in self.base_data["Fuel"].unique():
            fuel = Fuel(fuel_name, useRefProperties=False)
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
                    method = method_map.get(
                        prop_name.replace(" ", "").strip().lower(), None
                    )
                    if method is None:
                        msg = f"No method found for property '{prop_name}'."
                        raise ValueError(msg)

                    for row in prop_data.itertuples():
                        # Temperature-independent properties (e.g., freeze point) have
                        # no Temp (parsed as NaN), so do not pass T to the method.
                        kwargs = {}
                        if not pd.isna(row.Temp):
                            kwargs["T"] = Units.Quantity(row.Temp, row.Temp_Units)
                        pred = method(fuel=fuel, **kwargs).to(row.Property_Units)
                        pred_val = pred.magnitude
                        if isinstance(pred_val, np.ndarray):
                            if pred_val.size != 1:
                                msg = f"Expected a single value for property '{prop_name}' of fuel '{fuel_name}', but got an array of size {pred_val.size}."
                                raise ValueError(msg)
                            pred_val = pred_val.item()
                        # Fetch the baseline prediction and recreate the known property value from baseline value - error
                        # (Baseline_Error = prediction - known; this skips an extra lookup)
                        base_val = row.Baseline_Value  # Baseline prediction value
                        prop_val = base_val - row.Baseline_Error  # Known value

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


@pytest.mark.skipif(not HAS_INVERSE, reason="requires the 'inverse' extra")
class TestInverseConsistency:
    """Test class for verifying inverse accuracy against the main codebase."""

    @pytest.mark.parametrize(
        "fuel_name",
        [
            "heptane",
            "dodecane",
            "posf10264",
            "posf10289",
            "posf10325",
        ],
    )
    def test_mixture_density(self, fuel_name: str) -> None:
        """Test that the inverse density predictions are consistent with the main codebase."""
        fuel = Fuel(fuel_name)
        T = Units.Quantity(298.15, "K")

        Yi = jnp.asarray(fuel.Y_0.m_as(""))
        rho_i = jnp.asarray(fuel.density(T).m_as("kg/m^3"))
        rho_inv = mixing.density(Yi, rho_i)
        rho_base = fuel.mixture_density(fuel.Y_0, T).m_as("kg/m^3")

        assert np.isclose(rho_inv, rho_base), (
            f"{fuel_name}: inverse density {rho_inv} does not match base density {rho_base}"
        )

    @pytest.mark.parametrize(
        "fuel_name",
        [
            "heptane",
            "dodecane",
            "posf10264",
            "posf10289",
            "posf10325",
        ],
    )
    def test_mixture_viscosity(self, fuel_name: str) -> None:
        """Test that the inverse viscosity predictions are consistent with the main codebase."""
        fuel = Fuel(fuel_name)
        T = Units.Quantity(298.15, "K")

        Yi = jnp.asarray(fuel.Y_0.m_as(""))
        mu_i = jnp.asarray(fuel.viscosity_kinematic(T).m_as("m^2/s"))
        mw_i = jnp.asarray(fuel.MW.m_as("kg/mol"))
        mu_inv = mixing.viscosity(Yi, mu_i, mw_i)
        mu_base = fuel.mixture_kinematic_viscosity(fuel.Y_0, T).m_as("m^2/s")

        assert np.isclose(mu_inv, mu_base), (
            f"{fuel_name}: inverse viscosity {mu_inv} does not match base viscosity {mu_base}"
        )

    @pytest.mark.parametrize(
        "fuel_name",
        [
            "heptane",
            "dodecane",
            "posf10264",
            "posf10289",
            "posf10325",
        ],
    )
    def test_mixture_aromatics(self, fuel_name: str) -> None:
        """Test that the inverse aromatics predictions are consistent with the main codebase."""
        fuel = Fuel(fuel_name)
        T = Units.Quantity(298.15, "K")

        Yi = jnp.asarray(fuel.Y_0.m_as(""))
        rho_i = jnp.asarray(fuel.density(T).m_as("kg/m^3"))
        is_aromatic = jnp.asarray(np.array(fuel.hc_type) == "aromatic", dtype=float)
        aro_inv = mixing.aromatics_content(Yi, rho_i, is_aromatic)
        aro_base = fuel.mixture_aromatics_content(fuel.Y_0).m_as("percent")

        assert np.isclose(aro_inv, aro_base), (
            f"{fuel_name}: inverse aromatics {aro_inv} does not match base aromatics {aro_base}"
        )

    @pytest.mark.parametrize(
        "fuel_name",
        [
            "heptane",
            "dodecane",
            "posf10264",
            "posf10289",
            "posf10325",
        ],
    )
    def test_mixture_heat_of_combustion(self, fuel_name: str) -> None:
        """Test that the inverse heat of combustion predictions are consistent with the main codebase."""
        fuel = Fuel(fuel_name)

        Yi = jnp.asarray(fuel.Y_0.m_as(""))
        Lhv_i = jnp.asarray(
            correlate.components.lower_heating_value(fuel).m_as("MJ/kg")
        )
        h_inv = mixing.heat_of_combustion(Yi, Lhv_i)
        h_base = fuel.mixture_heat_of_combustion(fuel.Y_0).m_as("MJ/kg")

        assert np.isclose(h_inv, h_base), (
            f"{fuel_name}: inverse heat of combustion {h_inv} does not match base heat of combustion {h_base}"
        )

    @pytest.mark.parametrize(
        "fuel_name",
        [
            "heptane",
            "dodecane",
            "posf10264",
            "posf10289",
            "posf10325",
        ],
    )
    def test_mixture_flash_point(self, fuel_name: str) -> None:
        """Test that the inverse flash point predictions are consistent with the main codebase."""
        fuel = Fuel(fuel_name)
        T = Units.Quantity(298.15, "K")

        Yi = jnp.asarray(fuel.Y_0.m_as(""))
        Tfp_i = jnp.asarray(fuel.flash_point().m_as("celsius"))
        Tc_i = jnp.asarray(fuel.Tc.m_as("K"))
        Pc_i = jnp.asarray(fuel.Pc.m_as("Pa"))
        omega_i = jnp.asarray(fuel.omega.m_as(""))
        MW_i = jnp.asarray(fuel.MW.m_as("kg/mol"))
        Tfp_inv = mixing.flash_point(Yi, Tfp_i, Tc_i, Pc_i, omega_i, MW_i)
        Tfp_base = fuel.mixture_flash_point(fuel.Y_0).m_as("celsius")

        assert np.isclose(Tfp_inv, Tfp_base), (
            f"{fuel_name}: inverse flash point {Tfp_inv} does not match base flash point {Tfp_base}"
        )

    @pytest.mark.parametrize(
        "fuel_name",
        [
            "heptane",
            "dodecane",
            "posf10264",
            "posf10289",
            "posf10325",
        ],
    )
    def test_mixture_freeze_point(self, fuel_name: str) -> None:
        """Test that the inverse freeze point predictions are consistent with the main codebase."""
        fuel = Fuel(fuel_name)
        T = Units.Quantity(298.15, "K")

        Yi = jnp.asarray(fuel.Y_0.m_as(""))
        Tm_i = jnp.asarray(fuel.Tm.m_as("kelvin"))
        dS_fus_i = jnp.asarray(fuel.dS_fus.m_as("J/(mol*K)"))
        Cp_i = jnp.asarray(
            correlate.components.molar_specific_heat_capacity_ruzicka(
                fuel, Units.Quantity(298.15, "K")
            ).m_as("J/(mol*K)")
        )
        MW_i = jnp.asarray(fuel.MW.m_as("kg/mol"))
        Tfp_inv = mixing.freeze_point(Yi, Tm_i, dS_fus_i, Cp_i, MW_i)
        Tfp_base = fuel.mixture_freeze_point(fuel.Y_0).m_as("celsius")

        assert np.isclose(Tfp_inv, Tfp_base, atol=5e-3), (
            f"{fuel_name}: inverse freeze point {Tfp_inv} does not match base freeze point {Tfp_base}"
        )  # A small tolerance (0.005) due to root-finding technique


if __name__ == "__main__":
    unittest.main()
