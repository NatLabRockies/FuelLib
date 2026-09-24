import unittest
from functools import partial
from pathlib import Path

import numpy as np

import fuellib as fl
from fuellib.utils import Units


class FuelLibFunctionEvalTestCase(unittest.TestCase):
    """Smoke test all current FuelLib public functions for single and multicomponent fuels."""

    @classmethod
    def setUpClass(cls):
        cls.fuels = {
            "decane": fl.Fuel("decane"),
            "posf10325": fl.Fuel("posf10325"),
        }
        cls.T = Units.Quantity(320.0, "K")
        cls.p = Units.Quantity(101325.0, "Pa")

    def _assert_finite_and_positive(self, value):
        arr = np.asarray(value.magnitude if hasattr(value, "magnitude") else value)
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
                self.assertAlmostEqual(fl.convert.C2K(25.0), 298.15)
                print("    ✓ convert.C2K")
                self.assertAlmostEqual(fl.convert.K2C(298.15), 25.0)
                print("    ✓ convert.K2C")
                self.assertAlmostEqual(
                    fl.utility.droplet_volume(1e-4), 4.0 / 3.0 * np.pi * (1e-4) ** 3
                )
                print("    ✓ utility.droplet_volume")

                Xi = fuel.Y2X(Yi)
                self._assert_finite_and_positive(
                    fl.utility.mixing_rule(fuel.Tc, Xi, pseudo_prop="arithmetic")
                )
                print("    ✓ utility.mixing_rule (arithmetic)")
                self._assert_finite_and_positive(
                    fl.utility.mixing_rule(fuel.Tc, Xi, pseudo_prop="geometric")
                )
                print("    ✓ utility.mixing_rule (geometric)")

                # Composition conversion methods
                print("  Composition Conversions:")
                methods_to_test = [
                    (
                        "mean_molecular_weight",
                        lambda fuel=fuel, Yi=Yi: fuel.mean_molecular_weight(Yi),
                    ),
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

                mass = Units.Quantity(Yi * 1.0e-6, "kg")
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
                        lambda fuel=fuel: fuel.psat(
                            self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "surface_tension (Pitzer)",
                        lambda fuel=fuel: fuel.surface_tension(
                            self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "diffusion_coeff (Tee)",
                        lambda fuel=fuel: fuel.diffusion_coeff(
                            self.p, self.T, correlation="Tee"
                        ),
                    ),
                    (
                        "diffusion_coeff (Wilke)",
                        lambda fuel=fuel: fuel.diffusion_coeff(
                            self.p, self.T, correlation="Wilke"
                        ),
                    ),
                ]
                for method_name, method_call in alt_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Antoine coefficient fits (individual compounds)
                A, B, C, D = fuel.psat_antoine_coeffs(
                    Tvals=Units.Quantity(np.array([300.0, 340.0]), "K"),
                    units="mks",
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
                    Tvals=Units.Quantity(np.array([300.0, 340.0]), "K"),
                    units="cgs",
                    correlation="Lee-Kesler",
                )
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive([A_mix, B_mix, D_mix])
                self.assertTrue(np.isfinite(C_mix))
                print("    ✓ mixture_vapor_pressure_antoine_coeffs")

                # Mixture properties
                print("  Mixture Properties:")
                mixture_methods = [
                    (
                        "mixture_density",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_density(Yi, self.T),
                    ),
                    (
                        "mixture_kinematic_viscosity (Kendall-Monroe)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_kinematic_viscosity(
                            Yi, self.T, correlation="Kendall-Monroe"
                        ),
                    ),
                    (
                        "mixture_kinematic_viscosity (Arrhenius)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_kinematic_viscosity(
                            Yi, self.T, correlation="Arrhenius"
                        ),
                    ),
                    (
                        "mixture_dynamic_viscosity",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_dynamic_viscosity(
                            Yi, self.T
                        ),
                    ),
                    (
                        "mixture_vapor_pressure (Lee-Kesler)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_vapor_pressure(
                            Yi, self.T, correlation="Lee-Kesler"
                        ),
                    ),
                    (
                        "mixture_vapor_pressure (Ambrose-Walton)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_vapor_pressure(
                            Yi, self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "mixture_surface_tension (Brock-Bird)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_surface_tension(
                            Yi, self.T, correlation="Brock-Bird"
                        ),
                    ),
                    (
                        "mixture_surface_tension (Pitzer)",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_surface_tension(
                            Yi, self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "mixture_thermal_conductivity",
                        lambda fuel=fuel, Yi=Yi: fuel.mixture_thermal_conductivity(
                            Yi, self.T
                        ),
                    ),
                ]
                for method_name, method_call in mixture_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Droplet helpers
                print("  Droplet Properties:")
                m = fl.utility.droplet_mass(fuel, 2.0e-5, Yi, self.T)
                self.assertEqual(m.shape, fuel.MW.shape)
                self.assertTrue(np.all(m >= 0.0))
                self.assertTrue(
                    np.allclose(fl.utility.droplet_mass(fuel, 0.0, Yi, self.T), 0.0)
                )
                print("    ✓ utility.droplet_mass")


class FuelLibAPIContractTestCase(unittest.TestCase):
    """Check that the FuelLib API persists across versions.

    This test uses partial functions to check that FuelLib API methods can be called
    using explicitly defined arguments. Because we are using partial functions, using
    `F2` to rename variables will not affect the test calls--thus, the API persistence
    can be verified independently. Additionally, using `__getattribute__` ensures that
    the method names are accessed by expected strings, reinforcing the API check.
    """

    def test__fuel_class_attribute_persistence(self):
        """Test that the Fuel class attributes persist across versions."""
        fuel = fl.Fuel("heptane")
        attributes = [
            "fuelDataDir",
            "fuelDataGcDir",
            "fuelDataDecompDir",
            "fuelDataPropsDir",
            "name",
            "compounds",
            "formulas",
            "Y_0",
            "Nij",
            "num_compounds",
            "num_groups",
            "MW",
            "Tc",
            "Pc",
            "Vc",
            "Tb",
            "Tm",
            "Hf",
            "Gf",
            "Hv_stp",
            "Lv_stp",
            "Cp_stp",
            "Vm_stp",
            "omega",
            "sigma",
            "epsilonByKB",
            "hc_type",
            "fam",
            "nC",
            "nH",
            "pelephysics_keys",
        ]
        for attr in attributes:
            self.assertTrue(
                hasattr(fuel, attr), f"Fuel class is missing attribute '{attr}'"
            )

    def test__fuel_module_api_call_persistence(self):
        """Test that the FuelLib.fuel module API persists across versions."""
        fuelDataDir = Path(__file__).parent.parent / "fuellib/data/fuelData"
        fuel_init = partial(
            fl.Fuel, name="heptane", decompName="heptane", fuelDataDir=str(fuelDataDir)
        )
        fuel_init()  # Check that the Fuel object can be instantiated.

        fuel = fl.Fuel(name="heptane")

        # Define some common variables for the tests.
        Yi = fuel.Y_0
        mass = fl.Units.Quantity(fuel.MW.magnitude * 100.0, "kg")
        temp = fl.Units.Quantity(298.15, "K")
        temps = fl.Units.Quantity([298.15, 300.0, 310.0], "K")
        press = fl.Units.Quantity(101325.0, "Pa")
        # List of FuelLib.fuel methods to check for API persistence.
        to_check = [
            partial(fuel.__getattribute__("mean_molecular_weight"), Yi=Yi),
            partial(fuel.__getattribute__("mass2X"), mass=mass),
            partial(fuel.__getattribute__("mass2Y"), mass=mass),
            partial(fuel.__getattribute__("density"), T=temp, comp_idx=0),
            partial(fuel.__getattribute__("viscosity_kinematic"), T=temp, comp_idx=0),
            partial(fuel.__getattribute__("viscosity_dynamic"), T=temp, comp_idx=0),
            partial(fuel.__getattribute__("Cl"), T=temp, comp_idx=0),
            partial(fuel.__getattribute__("Cp"), T=temp, comp_idx=0),
            partial(
                fuel.__getattribute__("psat"),
                T=temp,
                comp_idx=0,
                correlation="Lee-Kesler",
            ),
            partial(
                fuel.__getattribute__("psat_antoine_coeffs"),
                Tvals=temps,
                units="mks",
                correlation="Lee-Kesler",
            ),
            partial(fuel.__getattribute__("molar_liquid_vol"), T=temp, comp_idx=0),
            partial(
                fuel.__getattribute__("latent_heat_vaporization"), T=temp, comp_idx=0
            ),
            partial(
                fuel.__getattribute__("diffusion_coeff"),
                p=press,
                T=temp,
                correlation="Tee",
            ),
            partial(
                fuel.__getattribute__("surface_tension"),
                T=temp,
                comp_idx=0,
                correlation="Brock-Bird",
            ),
            partial(fuel.__getattribute__("thermal_conductivity"), T=temp, comp_idx=0),
            partial(fuel.__getattribute__("mixture_density"), Yi=Yi, T=temp),
            partial(
                fuel.__getattribute__("mixture_kinematic_viscosity"),
                Yi=Yi,
                T=temp,
                correlation="Kendall-Monroe",
            ),
            partial(
                fuel.__getattribute__("mixture_dynamic_viscosity"),
                Yi=Yi,
                T=temp,
                correlation="Kendall-Monroe",
            ),
            partial(
                fuel.__getattribute__("mixture_vapor_pressure"),
                Yi=Yi,
                T=temp,
                correlation="Lee-Kesler",
            ),
            partial(
                fuel.__getattribute__("mixture_vapor_pressure_antoine_coeffs"),
                Yi=Yi,
                Tvals=temps,
                units="cgs",
                correlation="Lee-Kesler",
            ),
            partial(
                fuel.__getattribute__("mixture_surface_tension"),
                Yi=Yi,
                T=temp,
                correlation="Brock-Bird",
            ),
            partial(
                fuel.__getattribute__("mixture_thermal_conductivity"),
                Yi=Yi,
                T=temp,
            ),
        ]

        for func in to_check:
            module = func.func.__module__ if hasattr(func.func, "__module__") else ""
            name = func.func.__name__ if hasattr(func.func, "__name__") else ""
            with self.subTest(module=module, name=name):
                try:
                    func()
                except Exception as e:  # noqa: BLE001
                    self.fail(f"Method failed with expected call. Exception: {e}")

    def test__convert_module_api_call_persistence(self):
        """Check that the FuelLib.convert module API persists across versions."""
        to_check = [
            partial(fl.convert.C2K, T=20.0),
            partial(fl.convert.K2C, T=293.15),
            partial(fl.convert.C2F, T=20.0),
            partial(fl.convert.F2C, T=68.0),
            partial(fl.convert.F2K, T=68.0),
            partial(fl.convert.K2F, T=293.15),
            partial(
                fl.convert.epsilon_to_characteristic_temperature, epsilon_j_per_mol=1.0
            ),
        ]
        for func in to_check:
            module = func.func.__module__ if hasattr(func.func, "__module__") else ""
            name = func.func.__name__ if hasattr(func.func, "__name__") else ""
            with self.subTest(module=module, name=name):
                try:
                    func()
                except Exception as e:  # noqa: BLE001
                    self.fail(f"Method failed with expected call. Exception: {e}")


if __name__ == "__main__":
    unittest.main()
