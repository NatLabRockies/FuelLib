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


class FuelLibFunctionTestCase(unittest.TestCase):
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

    def test_utility_functions(self):
        self.assertAlmostEqual(fl.C2K(25.0), 298.15)
        self.assertAlmostEqual(fl.K2C(298.15), 25.0)
        self.assertAlmostEqual(fl.droplet_volume(1e-4), 4.0 / 3.0 * np.pi * (1e-4) ** 3)

        var_n = np.array([1.0, 2.0])
        X = np.array([0.25, 0.75])
        mix_arith = fl.mixing_rule(var_n, X, pseudo_prop="arithmetic")
        mix_geom = fl.mixing_rule(var_n, X, pseudo_prop="geometric")
        self._assert_finite_and_positive(mix_arith)
        self._assert_finite_and_positive(mix_geom)

    def test_all_fuel_methods_for_single_and_multicomponent(self):
        for fuel_name, fuel in self.fuels.items():
            with self.subTest(fuel=fuel_name):
                Yi = fuel.Y_0.copy()
                self.assertAlmostEqual(np.sum(Yi), 1.0)

                # Composition conversion methods
                Mbar = fuel.mean_molecular_weight(Yi)
                self._assert_finite_and_positive(Mbar)

                Xi = fuel.Y2X(Yi)
                Yi_back = fuel.X2Y(Xi)
                self.assertTrue(np.allclose(Yi, Yi_back, rtol=1e-10, atol=1e-12))

                mass = Yi * 1.0e-6
                self.assertTrue(
                    np.allclose(fuel.mass2Y(mass), Yi, rtol=1e-10, atol=1e-12)
                )
                Xi_from_mass = fuel.mass2X(mass)
                self.assertTrue(np.allclose(np.sum(Xi_from_mass), 1.0))

                # Zero-input branches
                zeros = np.zeros_like(Yi)
                self.assertEqual(fuel.mean_molecular_weight(zeros), 0.0)
                self.assertTrue(np.allclose(fuel.mass2Y(zeros), zeros))
                self.assertTrue(np.allclose(fuel.mass2X(zeros), zeros))
                self.assertTrue(np.allclose(fuel.Y2X(zeros), zeros))
                self.assertTrue(np.allclose(fuel.X2Y(zeros), zeros))

                # Component properties (all components and explicit component index)
                for method in [
                    fuel.density,
                    fuel.viscosity_kinematic,
                    fuel.viscosity_dynamic,
                    fuel.Cp,
                    fuel.Cl,
                    fuel.psat,
                    fuel.molar_liquid_vol,
                    fuel.latent_heat_vaporization,
                    fuel.surface_tension,
                    fuel.thermal_conductivity,
                ]:
                    self._assert_finite_and_positive(method(self.T))
                    self._assert_finite_and_positive(method(self.T, comp_idx=0))

                # Alternate correlation branches
                self._assert_finite_and_positive(
                    fuel.psat(self.T, correlation="Ambrose-Walton")
                )
                self._assert_finite_and_positive(
                    fuel.surface_tension(self.T, correlation="Pitzer")
                )
                self._assert_finite_and_positive(
                    fuel.diffusion_coeff(self.p, self.T, correlation="Tee")
                )
                self._assert_finite_and_positive(
                    fuel.diffusion_coeff(self.p, self.T, correlation="Wilke")
                )

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

                # Mixture properties
                self._assert_finite_and_positive(fuel.mixture_density(Yi, self.T))
                self._assert_finite_and_positive(
                    fuel.mixture_kinematic_viscosity(
                        Yi, self.T, correlation="Kendall-Monroe"
                    )
                )
                self._assert_finite_and_positive(
                    fuel.mixture_kinematic_viscosity(
                        Yi, self.T, correlation="Arrhenius"
                    )
                )
                self._assert_finite_and_positive(
                    fuel.mixture_dynamic_viscosity(Yi, self.T)
                )
                self._assert_finite_and_positive(
                    fuel.mixture_vapor_pressure(Yi, self.T, correlation="Lee-Kesler")
                )
                self._assert_finite_and_positive(
                    fuel.mixture_vapor_pressure(
                        Yi, self.T, correlation="Ambrose-Walton"
                    )
                )

                A_mix, B_mix, C_mix, D_mix = fuel.mixture_vapor_pressure_antoine_coeffs(
                    Yi,
                    Tvals=np.array([300.0, 340.0]),
                    units="bar",
                    correlation="Lee-Kesler",
                )
                # A, B, D must be positive; C can be negative (it's a temperature offset)
                self._assert_finite_and_positive([A_mix, B_mix, D_mix])
                self.assertTrue(np.isfinite(C_mix))

                self._assert_finite_and_positive(
                    fuel.mixture_surface_tension(Yi, self.T, correlation="Brock-Bird")
                )
                self._assert_finite_and_positive(
                    fuel.mixture_surface_tension(Yi, self.T, correlation="Pitzer")
                )
                self._assert_finite_and_positive(
                    fuel.mixture_thermal_conductivity(Yi, self.T)
                )

                # Droplet helpers
                m = fl.droplet_mass(fuel, 2.0e-5, Yi, self.T)
                self.assertEqual(m.shape, fuel.MW.shape)
                self.assertTrue(np.all(m >= 0.0))
                self.assertTrue(
                    np.allclose(fl.droplet_mass(fuel, 0.0, Yi, self.T), 0.0)
                )


if __name__ == "__main__":
    unittest.main()
