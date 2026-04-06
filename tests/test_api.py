import inspect
import os
import sys
import unittest
import numpy as np

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)
from paths import *
import FuelLib as fl


def _normalize_signature(sig):
    """Normalize path-like defaults so signatures are stable across machines."""

    parts = []
    for name, param in sig.parameters.items():
        text = str(param)
        if (
            name == "path"
            and param.default is not inspect.Parameter.empty
            and isinstance(param.default, str)
            and param.default.endswith("exportData")
        ):
            text = "path='<EXPORTDATA_PATH>'"
        parts.append(text)
    return f"({', '.join(parts)})"


def _public_module_functions(module):
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    }


def _public_class_methods(cls):
    return {
        name: obj
        for name, obj in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


class ApiContractTestCase(unittest.TestCase):
    def test_fuellib_module_api(self):
        print("\nFuelLib Module API:")
        expected = {
            "C2K": "(T)",
            "K2C": "(T)",
            "mixing_rule": "(var_n, X, pseudo_prop='arithmetic')",
            "droplet_volume": "(r)",
            "droplet_mass": "(fuel, r, Yi, T)",
        }

        actual = _public_module_functions(fl)
        self.assertEqual(
            set(actual.keys()),
            set(expected.keys()),
            msg=(
                "FuelLib module public function list changed. "
                f"Expected: {sorted(expected.keys())}; Found: {sorted(actual.keys())}"
            ),
        )

        for name in sorted(expected.keys()):
            actual_sig = _normalize_signature(inspect.signature(actual[name]))
            self.assertEqual(
                actual_sig,
                expected[name],
                msg=f"FuelLib module function signature changed: {name}",
            )
            print(f"  ✓ {name}{actual_sig}")

    def test_fuellib_class_api(self):
        print("\nFuelLib.fuel Class API:")
        expected = {
            "Cl": "(self, T, comp_idx=None)",
            "Cp": "(self, T, comp_idx=None)",
            "X2Y": "(self, Xi)",
            "Y2X": "(self, Yi)",
            "density": "(self, T, comp_idx=None)",
            "diffusion_coeff": "(self, p, T, sigma_gas=3.62e-10, epsilonByKB_gas=97.0, MW_gas=0.02897, correlation='Tee')",
            "latent_heat_vaporization": "(self, T, comp_idx=None)",
            "mass2X": "(self, mass)",
            "mass2Y": "(self, mass)",
            "mean_molecular_weight": "(self, Yi)",
            "mixture_density": "(self, Yi, T)",
            "mixture_dynamic_viscosity": "(self, Yi, T, correlation='Kendall-Monroe')",
            "mixture_kinematic_viscosity": "(self, Yi, T, correlation='Kendall-Monroe')",
            "mixture_surface_tension": "(self, Yi, T, correlation='Brock-Bird')",
            "mixture_thermal_conductivity": "(self, Yi, T)",
            "mixture_vapor_pressure": "(self, Yi, T, correlation='Lee-Kesler')",
            "mixture_vapor_pressure_antoine_coeffs": "(self, Yi, Tvals=None, units='mks', correlation='Lee-Kesler')",
            "molar_liquid_vol": "(self, T, comp_idx=None)",
            "psat": "(self, T, comp_idx=None, correlation='Lee-Kesler')",
            "psat_antoine_coeffs": "(self, Tvals=None, units='mks', correlation='Lee-Kesler')",
            "surface_tension": "(self, T, comp_idx=None, correlation='Brock-Bird')",
            "thermal_conductivity": "(self, T, comp_idx=None)",
            "viscosity_dynamic": "(self, T, comp_idx=None)",
            "viscosity_kinematic": "(self, T, comp_idx=None)",
        }

        actual = _public_class_methods(fl.fuel)
        self.assertEqual(
            set(actual.keys()),
            set(expected.keys()),
            msg=(
                "FuelLib.fuel public method list changed. "
                f"Expected: {sorted(expected.keys())}; Found: {sorted(actual.keys())}"
            ),
        )

        for name in sorted(expected.keys()):
            actual_sig = _normalize_signature(inspect.signature(actual[name]))
            self.assertEqual(
                actual_sig,
                expected[name],
                msg=f"FuelLib.fuel method signature changed: {name}",
            )
            print(f"  ✓ fuel.{name}{actual_sig}")


class FuelLibFunctionEvalTestCase(unittest.TestCase):
    """Smoke test all current FuelLib public functions for single and multicomponent fuels."""

    @classmethod
    def setUpClass(cls):
        cls.fuels = {
            "decane": fl.fuel("decane"),
            "posf10325": fl.fuel("posf10325"),
        }
        cls.T = 320.0
        cls.p = 101325.0

    def _assert_finite_and_positive(self, value):
        arr = np.asarray(value)
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertTrue(np.all(arr > 0.0))

    def test_all_fuel_methods_for_single_and_multicomponent(self):
        print("\n")  # Add newline to separate from unittest verbose output

        for fuel_name, fuel in self.fuels.items():
            print(f"\n{fuel_name.upper()}:")

            with self.subTest(fuel=fuel_name):
                Yi = fuel.Y_0.copy()
                self.assertAlmostEqual(np.sum(Yi), 1.0)

                # Utility functions (run per fuel for consistent CI grouping)
                print("  Utility Functions:")
                self.assertAlmostEqual(fl.C2K(25.0), 298.15)
                print("    ✓ C2K")
                self.assertAlmostEqual(fl.K2C(298.15), 25.0)
                print("    ✓ K2C")
                self.assertAlmostEqual(
                    fl.droplet_volume(1e-4), 4.0 / 3.0 * np.pi * (1e-4) ** 3
                )
                print("    ✓ droplet_volume")

                Xi = fuel.Y2X(Yi)
                self._assert_finite_and_positive(
                    fl.mixing_rule(fuel.Tc, Xi, pseudo_prop="arithmetic")
                )
                print("    ✓ mixing_rule (arithmetic)")
                self._assert_finite_and_positive(
                    fl.mixing_rule(fuel.Tc, Xi, pseudo_prop="geometric")
                )
                print("    ✓ mixing_rule (geometric)")

                # Composition conversion methods
                print("  Composition Conversions:")
                methods_to_test = [
                    ("mean_molecular_weight", lambda: fuel.mean_molecular_weight(Yi)),
                ]
                for method_name, method_call in methods_to_test:
                    try:
                        result = method_call()
                        self._assert_finite_and_positive(result)
                        print(f"    ✓ {method_name}")
                    except AssertionError as e:
                        print(f"    ✗ {method_name}: {e}")
                        raise

                Yi_back = fuel.X2Y(Xi)
                self.assertTrue(np.allclose(Yi, Yi_back, rtol=1e-10, atol=1e-12))
                print("    ✓ Y2X/X2Y roundtrip")

                mass = Yi * 1.0e-6
                self.assertTrue(
                    np.allclose(fuel.mass2Y(mass), Yi, rtol=1e-10, atol=1e-12)
                )
                print("    ✓ mass2Y")

                Xi_from_mass = fuel.mass2X(mass)
                self.assertTrue(np.allclose(np.sum(Xi_from_mass), 1.0))
                print("    ✓ mass2X")

                # Component properties (all components and explicit component index)
                print("  Component Properties (both all and indexed):")
                component_methods = [
                    "density",
                    "viscosity_kinematic",
                    "viscosity_dynamic",
                    "Cp",
                    "Cl",
                    "psat",
                    "molar_liquid_vol",
                    "latent_heat_vaporization",
                    "surface_tension",
                    "thermal_conductivity",
                ]
                for method_name in component_methods:
                    method = getattr(fuel, method_name)
                    self._assert_finite_and_positive(method(self.T))
                    self._assert_finite_and_positive(method(self.T, comp_idx=0))
                    print(f"    ✓ {method_name}")

                # Alternate correlation branches
                print("  Alternate Correlations:")
                alt_methods = [
                    (
                        "psat (Ambrose-Walton)",
                        lambda: fuel.psat(self.T, correlation="Ambrose-Walton"),
                    ),
                    (
                        "surface_tension (Pitzer)",
                        lambda: fuel.surface_tension(self.T, correlation="Pitzer"),
                    ),
                    (
                        "diffusion_coeff (Tee)",
                        lambda: fuel.diffusion_coeff(self.p, self.T, correlation="Tee"),
                    ),
                    (
                        "diffusion_coeff (Wilke)",
                        lambda: fuel.diffusion_coeff(
                            self.p, self.T, correlation="Wilke"
                        ),
                    ),
                ]
                for method_name, method_call in alt_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Antoine coefficient fits (individual compounds)
                A, B, C, D = fuel.psat_antoine_coeffs(
                    Tvals=np.array([300.0, 340.0]),
                    units="atm",
                    correlation="Lee-Kesler",
                )
                self.assertEqual(len(A), fuel.num_compounds)
                self.assertEqual(len(B), fuel.num_compounds)
                self.assertEqual(len(C), fuel.num_compounds)
                self.assertEqual(len(D), fuel.num_compounds)
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive(A)
                self._assert_finite_and_positive(B)
                self._assert_finite_and_positive(D)
                self.assertTrue(np.all(np.isfinite(C)))
                print("    ✓ psat_antoine_coeffs (individual)")

                # Antoine coefficient fits (mixture)
                A_mix, B_mix, C_mix, D_mix = fuel.mixture_vapor_pressure_antoine_coeffs(
                    Yi,
                    Tvals=np.array([300.0, 340.0]),
                    units="bar",
                    correlation="Lee-Kesler",
                )
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive([A_mix, B_mix, D_mix])
                self.assertTrue(np.isfinite(C_mix))
                print("    ✓ mixture_vapor_pressure_antoine_coeffs")

                # Mixture properties
                print("  Mixture Properties:")
                mixture_methods = [
                    ("mixture_density", lambda: fuel.mixture_density(Yi, self.T)),
                    (
                        "mixture_kinematic_viscosity (Kendall-Monroe)",
                        lambda: fuel.mixture_kinematic_viscosity(
                            Yi, self.T, correlation="Kendall-Monroe"
                        ),
                    ),
                    (
                        "mixture_kinematic_viscosity (Arrhenius)",
                        lambda: fuel.mixture_kinematic_viscosity(
                            Yi, self.T, correlation="Arrhenius"
                        ),
                    ),
                    (
                        "mixture_dynamic_viscosity",
                        lambda: fuel.mixture_dynamic_viscosity(Yi, self.T),
                    ),
                    (
                        "mixture_vapor_pressure (Lee-Kesler)",
                        lambda: fuel.mixture_vapor_pressure(
                            Yi, self.T, correlation="Lee-Kesler"
                        ),
                    ),
                    (
                        "mixture_vapor_pressure (Ambrose-Walton)",
                        lambda: fuel.mixture_vapor_pressure(
                            Yi, self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "mixture_surface_tension (Brock-Bird)",
                        lambda: fuel.mixture_surface_tension(
                            Yi, self.T, correlation="Brock-Bird"
                        ),
                    ),
                    (
                        "mixture_surface_tension (Pitzer)",
                        lambda: fuel.mixture_surface_tension(
                            Yi, self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "mixture_thermal_conductivity",
                        lambda: fuel.mixture_thermal_conductivity(Yi, self.T),
                    ),
                ]
                for method_name, method_call in mixture_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Droplet helpers
                print("  Droplet Properties:")
                m = fl.droplet_mass(fuel, 2.0e-5, Yi, self.T)
                self.assertEqual(m.shape, fuel.MW.shape)
                self.assertTrue(np.all(m >= 0.0))
                self.assertTrue(
                    np.allclose(fl.droplet_mass(fuel, 0.0, Yi, self.T), 0.0)
                )
                print("    ✓ droplet_mass")


if __name__ == "__main__":
    unittest.main()
