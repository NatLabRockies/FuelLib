import inspect
import os
import sys
import unittest

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
            and param.default is not inspect._empty
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


if __name__ == "__main__":
    unittest.main()
