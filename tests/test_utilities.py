"""Tests for FuelLib utility functions and CLI commands."""

import subprocess
import unittest

import fuellib as fl


class TestUtilityFunctions(unittest.TestCase):
    """Test standalone utility functions."""

    def test_epsilon_to_characteristic_temperature(self):
        """Test epsilon conversion from J/mol to Kelvin."""
        # Test with a known value
        # epsilon = 1000 J/mol
        # epsilon_molecule = 1000 / 6.02214076e23 J
        # T* = epsilon_molecule / k_B = (1000 / 6.02214076e23) / 1.380649e-23 K
        result = fl.convert.epsilon_to_characteristic_temperature(1000.0)
        expected = (1000.0 / fl.constants.N_A) / fl.constants.k_B
        self.assertAlmostEqual(result, expected, places=10)

    def test_epsilon_to_characteristic_temperature_zero(self):
        """Test epsilon conversion with zero input."""
        result = fl.convert.epsilon_to_characteristic_temperature(0.0)
        self.assertEqual(result, 0.0)

    def test_epsilon_to_characteristic_temperature_negative(self):
        """Test epsilon conversion with negative input."""
        result = fl.convert.epsilon_to_characteristic_temperature(-1000.0)
        expected = (-1000.0 / fl.constants.N_A) / fl.constants.k_B
        self.assertAlmostEqual(result, expected, places=10)

    def test_C2K(self):
        """Test Celsius to Kelvin conversion."""
        # 0°C = 273.15 K
        self.assertAlmostEqual(fl.convert.C2K(0.0), 273.15, places=10)
        # 25°C = 298.15 K
        self.assertAlmostEqual(fl.convert.C2K(25.0), 298.15, places=10)
        # 100°C = 373.15 K
        self.assertAlmostEqual(fl.convert.C2K(100.0), 373.15, places=10)

    def test_K2C(self):
        """Test Kelvin to Celsius conversion."""
        # 273.15 K = 0°C
        self.assertAlmostEqual(fl.convert.K2C(273.15), 0.0, places=10)
        # 298.15 K = 25°C
        self.assertAlmostEqual(fl.convert.K2C(298.15), 25.0, places=10)
        # 373.15 K = 100°C
        self.assertAlmostEqual(fl.convert.K2C(373.15), 100.0, places=10)

    def test_C2K_K2C_roundtrip(self):
        """Test roundtrip conversion between Celsius and Kelvin."""
        original_c = 42.5
        kelvin = fl.convert.C2K(original_c)
        result_c = fl.convert.K2C(kelvin)
        self.assertAlmostEqual(result_c, original_c, places=10)


class TestUtilityCLI(unittest.TestCase):
    """Test command-line interface utilities."""

    def _run_cli_command(self, command, *args):
        """Run a CLI command and return stdout."""
        try:
            result = subprocess.run(
                [command] + list(args),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            self.skipTest(f"Command '{command}' not found in PATH")

    def test_fl_eps2K_basic(self):
        """Test fl-eps2K CLI command."""
        returncode, stdout, stderr = self._run_cli_command("fl-eps2K", "1000.0")
        self.assertEqual(returncode, 0, f"stderr: {stderr}")
        self.assertIn("Characteristic temperature", stdout)
        self.assertIn("K", stdout)

    def test_fl_eps2K_invalid_input(self):
        """Test fl-eps2K CLI with invalid input."""
        returncode, stdout, stderr = self._run_cli_command("fl-eps2K", "not_a_number")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Error", stdout)

    def test_fl_eps2K_no_args(self):
        """Test fl-eps2K CLI with no arguments."""
        returncode, stdout, stderr = self._run_cli_command("fl-eps2K")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Usage", stdout)

    def test_fl_C2K_basic(self):
        """Test fl-C2K CLI command."""
        returncode, stdout, stderr = self._run_cli_command("fl-C2K", "25.0")
        self.assertEqual(returncode, 0, f"stderr: {stderr}")
        self.assertIn("298.15", stdout)
        self.assertIn("K", stdout)

    def test_fl_C2K_invalid_input(self):
        """Test fl-C2K CLI with invalid input."""
        returncode, stdout, stderr = self._run_cli_command("fl-C2K", "not_a_number")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Error", stdout)

    def test_fl_C2K_no_args(self):
        """Test fl-C2K CLI with no arguments."""
        returncode, stdout, stderr = self._run_cli_command("fl-C2K")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Usage", stdout)

    def test_fl_K2C_basic(self):
        """Test fl-K2C CLI command."""
        returncode, stdout, stderr = self._run_cli_command("fl-K2C", "298.15")
        self.assertEqual(returncode, 0, f"stderr: {stderr}")
        self.assertIn("25.00", stdout)
        self.assertIn("C", stdout)

    def test_fl_K2C_invalid_input(self):
        """Test fl-K2C CLI with invalid input."""
        returncode, stdout, stderr = self._run_cli_command("fl-K2C", "not_a_number")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Error", stdout)

    def test_fl_K2C_no_args(self):
        """Test fl-K2C CLI with no arguments."""
        returncode, stdout, stderr = self._run_cli_command("fl-K2C")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Usage", stdout)


if __name__ == "__main__":
    unittest.main()
