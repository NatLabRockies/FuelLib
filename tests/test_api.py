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
                yi = fuel.y0.copy()
                self.assertAlmostEqual(np.sum(yi), 1.0)

                # Utility functions (run per fuel for consistent CI grouping)
                print("  Utility Functions:")
                droplet_radius = Units.Quantity(1e-4, "m")
                self.assertAlmostEqual(
                    fl.utility.droplet_volume(droplet_radius).to("m^3").magnitude,
                    4.0 / 3.0 * np.pi * (1e-4) ** 3,
                )
                print("    ✓ utility.droplet_volume")

                xi = fuel.comps.y2x(yi)
                self._assert_finite_and_positive(
                    fl.utility.mixing_rule(fuel.comps.tc, xi, pseudo_prop="arithmetic")
                )
                print("    ✓ utility.mixing_rule (arithmetic)")
                self._assert_finite_and_positive(
                    fl.utility.mixing_rule(fuel.comps.tc, xi, pseudo_prop="geometric")
                )
                print("    ✓ utility.mixing_rule (geometric)")
                self._assert_finite_and_positive(
                    fl.correlate.mixture.arithmetic(xi, fuel.comps.tc)
                )
                print("    ✓ correlate.mixture.arithmetic")
                self._assert_finite_and_positive(
                    fl.correlate.mixture.geometric(xi, fuel.comps.tc)
                )
                print("    ✓ correlate.mixture.geometric")
                component_nu = fuel.comps.nu(self.T)
                self._assert_finite_and_positive(
                    fl.correlate.mixture.arrhenius(xi, component_nu)
                )
                print("    ✓ correlate.mixture.arrhenius")
                self._assert_finite_and_positive(
                    fl.correlate.mixture.kendall_monroe(xi, component_nu)
                )
                print("    ✓ correlate.mixture.kendall_monroe")

                # Composition conversion methods
                print("  Composition Conversions:")
                methods_to_test = [
                    (
                        "mix.mw",
                        lambda fuel=fuel, yi=yi: fuel.mix.mw(yi),
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

                yi_back = fuel.comps.x2y(xi)
                self.assertTrue(np.allclose(yi, yi_back, rtol=1e-10, atol=1e-12))
                print("    ✓ y2x/x2y roundtrip")

                mass = Units.Quantity(yi.magnitude * 1.0e-6, "kg")
                self.assertTrue(
                    np.allclose(fuel.comps.mass2y(mass), yi, rtol=1e-10, atol=1e-12)
                )
                print("    ✓ mass2y")

                xi_from_mass = fuel.comps.mass2x(mass)
                self.assertTrue(np.allclose(np.sum(xi_from_mass), 1.0))
                print("    ✓ mass2x")

                # Component properties (all components and explicit component index)
                print("  Component Properties (both all and indexed):")
                component_methods = [
                    ("rho", {}),
                    ("nu", {}),
                    ("mu", {}),
                    ("cp", {"basis": "molar"}),
                    ("cp", {"basis": "mass"}),
                    ("pvap", {}),
                    ("vm", {}),
                    ("hv", {"basis": "molar"}),
                    ("hv", {"basis": "mass"}),
                    ("sigma", {}),
                    ("k", {}),
                ]
                for method_name, kwargs in component_methods:
                    method = getattr(fuel.comps, method_name)
                    self._assert_finite_and_positive(method(self.T, **kwargs))
                    self._assert_finite_and_positive(
                        method(self.T, comp_idx=0, **kwargs)
                    )
                    label = f"{method_name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
                    print(f"    ✓ {label}")

                # Alternate correlation branches
                print("  Alternate Correlations:")
                alt_methods = [
                    (
                        "pvap (Ambrose-Walton)",
                        lambda fuel=fuel: fuel.comps.pvap(
                            self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "sigma (Pitzer)",
                        lambda fuel=fuel: fuel.comps.sigma(
                            self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "d (Tee)",
                        lambda fuel=fuel: fuel.comps.d(
                            self.p, self.T, correlation="Tee"
                        ),
                    ),
                    (
                        "d (Wilke)",
                        lambda fuel=fuel: fuel.comps.d(
                            self.p, self.T, correlation="Wilke"
                        ),
                    ),
                ]
                for method_name, method_call in alt_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Antoine coefficient fits (individual compounds)
                a, b, c, d = fuel.comps.pvap_antoine_coeffs(
                    temp_vals=Units.Quantity(np.array([300.0, 340.0]), "K"),
                    units="mks",
                    correlation="Lee-Kesler",
                )
                self.assertEqual(len(a), fuel.num_compounds)
                self.assertEqual(len(b), fuel.num_compounds)
                self.assertEqual(len(c), fuel.num_compounds)
                self.assertEqual(len(d), fuel.num_compounds)
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive(a)
                self._assert_finite_and_positive(b)
                self._assert_finite_and_positive(d)
                self.assertTrue(np.all(np.isfinite(c)))
                print("    ✓ pvap_antoine_coeffs (individual)")

                # Antoine coefficient fits (mixture)
                a_mix, b_mix, c_mix, d_mix = fuel.mix.pvap_antoine_coeffs(
                    yi,
                    temp_vals=Units.Quantity(np.array([300.0, 340.0]), "K"),
                    units="cgs",
                    correlation="Lee-Kesler",
                )
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive([a_mix, b_mix, d_mix])
                self.assertTrue(np.isfinite(c_mix))
                print("    ✓ mix.pvap_antoine_coeffs")

                # Mixture properties
                print("  Mixture Properties:")
                mixture_methods = [
                    (
                        "mix.rho",
                        lambda fuel=fuel, yi=yi: fuel.mix.rho(yi, self.T),
                    ),
                    (
                        "mix.nu (Kendall-Monroe)",
                        lambda fuel=fuel, yi=yi: fuel.mix.nu(
                            yi, self.T, mixing_rule="Kendall-Monroe"
                        ),
                    ),
                    (
                        "mix.nu (Arrhenius)",
                        lambda fuel=fuel, yi=yi: fuel.mix.nu(
                            yi, self.T, mixing_rule="Arrhenius"
                        ),
                    ),
                    (
                        "mix.mu",
                        lambda fuel=fuel, yi=yi: fuel.mix.mu(yi, self.T),
                    ),
                    (
                        "mix.pvap (Lee-Kesler)",
                        lambda fuel=fuel, yi=yi: fuel.mix.pvap(
                            yi, self.T, correlation="Lee-Kesler"
                        ),
                    ),
                    (
                        "mix.pvap (Ambrose-Walton)",
                        lambda fuel=fuel, yi=yi: fuel.mix.pvap(
                            yi, self.T, correlation="Ambrose-Walton"
                        ),
                    ),
                    (
                        "mix.sigma (Brock-Bird)",
                        lambda fuel=fuel, yi=yi: fuel.mix.sigma(
                            yi, self.T, correlation="Brock-Bird"
                        ),
                    ),
                    (
                        "mix.sigma (Pitzer)",
                        lambda fuel=fuel, yi=yi: fuel.mix.sigma(
                            yi, self.T, correlation="Pitzer"
                        ),
                    ),
                    (
                        "mix.k",
                        lambda fuel=fuel, yi=yi: fuel.mix.k(yi, self.T),
                    ),
                ]
                for method_name, method_call in mixture_methods:
                    self._assert_finite_and_positive(method_call())
                    print(f"    ✓ {method_name}")

                # Droplet helpers
                print("  Droplet Properties:")
                r = Units.Quantity(2.0e-5, "m")
                m = fl.utility.droplet_mass(fuel, r, yi, self.T)
                self.assertEqual(m.shape, fuel.comps.mw.shape)
                self.assertTrue(np.all(m >= 0.0))
                self.assertTrue(
                    np.allclose(
                        fl.utility.droplet_mass(
                            fuel, Units.Quantity(0.0, "m"), yi, self.T
                        ),
                        0.0,
                    )
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
            "name",
            "data_dir",
            "gc_data",
            "components_data",
            "prop_data",
            "y0",
            "x0",
            "num_compounds",
            "comps",
            "mix",
            "metadata",
        ]
        for attr in attributes:
            self.assertTrue(
                hasattr(fuel, attr), f"Fuel class is missing attribute '{attr}'"
            )

        comps_attributes = [
            "names",
            "smiles",
            "pelephysics_keys",
            "mw",
            "tc",
            "pc",
            "vc",
            "tb",
            "tm",
            "hf_stp",
            "gf_stp",
            "hv_stp",
            "vm_stp",
            "omega",
            "lj_sigma",
            "lj_eps_by_kb",
            "hc_type",
            "fam",
            "num_c",
            "num_h",
        ]
        for attr in comps_attributes:
            self.assertTrue(
                hasattr(fuel.comps, attr),
                f"Fuel.comps is missing attribute '{attr}'",
            )

    def test__fuel_module_api_call_persistence(self):
        """Test that the FuelLib.fuel module API persists across versions."""
        fuel_data_dir = Path(__file__).parent.parent / "fuellib/data"
        fuel_init = partial(fl.Fuel, name="heptane", data_dir=str(fuel_data_dir))
        fuel_init()  # Check that the Fuel object can be instantiated.

        fuel = fl.Fuel(name="heptane")

        # Define some common variables for the tests.
        yi = fuel.y0
        mass = fl.Units.Quantity(fuel.comps.mw.magnitude * 100.0, "kg")
        temp = fl.Units.Quantity(298.15, "K")
        temp_vals = fl.Units.Quantity([298.15, 300.0, 310.0], "K")
        press = fl.Units.Quantity(101325.0, "Pa")
        # List of FuelLib.fuel methods to check for API persistence.
        to_check = [
            partial(fuel.mix.mw, yi=yi),
            partial(fuel.comps.mass2x, mass=mass),
            partial(fuel.comps.mass2y, mass=mass),
            partial(fuel.comps.rho, temp=temp, comp_idx=0),
            partial(fuel.comps.nu, temp=temp, comp_idx=0),
            partial(fuel.comps.mu, temp=temp, comp_idx=0),
            partial(fuel.comps.cp, temp=temp, comp_idx=0, basis="mass"),
            partial(fuel.comps.cp, temp=temp, comp_idx=0, basis="molar"),
            partial(
                fuel.comps.pvap,
                temp=temp,
                comp_idx=0,
                correlation="Lee-Kesler",
            ),
            partial(
                fuel.comps.pvap_antoine_coeffs,
                temp_vals=temp_vals,
                units="mks",
                correlation="Lee-Kesler",
            ),
            partial(fuel.comps.vm, temp=temp, comp_idx=0),
            partial(fuel.comps.hv, temp=temp, comp_idx=0, basis="mass"),
            partial(fuel.comps.hv, temp=temp, comp_idx=0, basis="molar"),
            partial(
                fuel.comps.d,
                p=press,
                temp=temp,
                correlation="Tee",
            ),
            partial(
                fuel.comps.sigma,
                temp=temp,
                comp_idx=0,
                correlation="Brock-Bird",
            ),
            partial(fuel.comps.k, temp=temp, comp_idx=0),
            partial(fuel.mix.rho, yi=yi, temp=temp),
            partial(
                fuel.mix.nu,
                yi=yi,
                temp=temp,
                mixing_rule="Kendall-Monroe",
            ),
            partial(
                fuel.mix.mu,
                yi=yi,
                temp=temp,
                mixing_rule="Kendall-Monroe",
            ),
            partial(
                fuel.mix.pvap,
                yi=yi,
                temp=temp,
                correlation="Lee-Kesler",
            ),
            partial(
                fuel.mix.pvap_antoine_coeffs,
                yi=yi,
                temp_vals=temp_vals,
                units="cgs",
                correlation="Lee-Kesler",
            ),
            partial(
                fuel.mix.sigma,
                yi=yi,
                temp=temp,
                correlation="Brock-Bird",
            ),
            partial(
                fuel.mix.k,
                yi=yi,
                temp=temp,
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
