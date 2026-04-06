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
import Export4Converge as e4c
import Export4Pele as e4p


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

    def test_export4converge_api(self):
        print("\nExport4Converge API:")
        expected_module = {
            "export_converge": "(fuel, path='<EXPORTDATA_PATH>', units='mks', temp_min=0, temp_max=1000, temp_step=10, export_mix=False)",
            "main": "()",
            "validate_fuel_files": "(fuel_name, fuel_data_dir)",
        }
        expected_converter = {
            "create_data_dict": "(self, T, T_crit, mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity)"
        }

        actual_module = _public_module_functions(e4c)
        self.assertEqual(
            set(actual_module.keys()),
            set(expected_module.keys()),
            msg=(
                "Export4Converge public function list changed. "
                f"Expected: {sorted(expected_module.keys())}; Found: {sorted(actual_module.keys())}"
            ),
        )

        for name in sorted(expected_module.keys()):
            actual_sig = _normalize_signature(inspect.signature(actual_module[name]))
            self.assertEqual(
                actual_sig,
                expected_module[name],
                msg=f"Export4Converge function signature changed: {name}",
            )
            print(f"  ✓ {name}{actual_sig}")

        actual_converter = _public_class_methods(e4c.UnitConverter)
        self.assertEqual(
            set(actual_converter.keys()),
            set(expected_converter.keys()),
            msg=(
                "Export4Converge.UnitConverter public method list changed. "
                f"Expected: {sorted(expected_converter.keys())}; Found: {sorted(actual_converter.keys())}"
            ),
        )

        for name in sorted(expected_converter.keys()):
            actual_sig = _normalize_signature(inspect.signature(actual_converter[name]))
            self.assertEqual(
                actual_sig,
                expected_converter[name],
                msg=f"Export4Converge.UnitConverter method signature changed: {name}",
            )
            print(f"  ✓ UnitConverter.{name}{actual_sig}")

    def test_export4pele_api(self):
        print("\nExport4Pele API:")
        expected_module = {
            "create_individual_compounds_dataframe": "(fuel, compound_names, converter)",
            "create_mixture_dataframe": "(fuel, export_mix_name, converter)",
            "export_pele": "(fuel, path='<EXPORTDATA_PATH>', units='mks', dep_fuel_names=None, use_pp_keys=True, export_mix=False, export_mix_name=None, liq_prop_model='gcm', psat_antoine=True)",
            "get_filename": "(fuel_name, liq_prop_model, export_mix, path)",
            "get_git_info": "()",
            "main": "()",
            "vec_to_str": "(vec)",
        }

        actual_module = _public_module_functions(e4p)
        self.assertEqual(
            set(actual_module.keys()),
            set(expected_module.keys()),
            msg=(
                "Export4Pele public function list changed. "
                f"Expected: {sorted(expected_module.keys())}; Found: {sorted(actual_module.keys())}"
            ),
        )

        for name in sorted(expected_module.keys()):
            actual_sig = _normalize_signature(inspect.signature(actual_module[name]))
            self.assertEqual(
                actual_sig,
                expected_module[name],
                msg=f"Export4Pele function signature changed: {name}",
            )
            print(f"  ✓ {name}{actual_sig}")


if __name__ == "__main__":
    unittest.main()
