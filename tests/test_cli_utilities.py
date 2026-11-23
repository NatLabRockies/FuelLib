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
        # T* = epsilon_molecule / K_B = (1000 / 6.02214076e23) / 1.380649e-23 K
        result = fl.utility.epsilon_to_characteristic_temperature(1000.0)
        expected = fl.Units.Quantity(1000.0, "J/mol") / (
            fl.constants.N_A * fl.constants.K_B
        )
        self.assertAlmostEqual(result.magnitude, expected.magnitude, places=10)

    def test_epsilon_to_characteristic_temperature_zero(self):
        """Test epsilon conversion with zero input."""
        result = fl.utility.epsilon_to_characteristic_temperature(0.0)
        self.assertEqual(result.magnitude, 0.0)

    def test_epsilon_to_characteristic_temperature_negative(self):
        """Test epsilon conversion with negative input."""
        result = fl.utility.epsilon_to_characteristic_temperature(-1000.0)
        expected = fl.Units.Quantity(-1000.0, "J/mol") / (
            fl.constants.N_A * fl.constants.K_B
        )
        self.assertAlmostEqual(result.magnitude, expected.magnitude, places=10)


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
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            self.skipTest(f"Command '{command}' not found in PATH")

    def test_fl_eps2k_basic(self):
        """Test fl convert eps2k CLI command."""
        returncode, stdout, stderr = self._run_cli_command(
            "fl", "convert", "eps2k", "1000.0"
        )
        self.assertEqual(returncode, 0, f"stderr: {stderr}")
        self.assertIn("Characteristic temperature", stdout)
        self.assertIn("K", stdout)

    def test_fl_eps2k_invalid_input(self):
        """Test fl convert eps2k CLI with invalid input."""
        returncode, _stdout, stderr = self._run_cli_command(
            "fl", "convert", "eps2k", "not_a_number"
        )
        self.assertNotEqual(returncode, 0)
        self.assertIn("error", stderr)

    def test_fl_eps2k_no_args(self):
        """Test fl convert eps2k CLI with no arguments."""
        returncode, _stdout, stderr = self._run_cli_command("fl", "convert", "eps2k")
        self.assertNotEqual(returncode, 0)
        self.assertIn("usage", stderr)

    def test_fl_c2k_basic(self):
        """Test fl convert c2k CLI command."""
        returncode, stdout, stderr = self._run_cli_command(
            "fl", "convert", "c2k", "25.0"
        )
        self.assertEqual(returncode, 0, f"stderr: {stderr}")
        self.assertIn("298.15", stdout)
        self.assertIn("K", stdout)

    def test_fl_c2k_invalid_input(self):
        """Test fl convert c2k CLI with invalid input."""
        returncode, _stdout, stderr = self._run_cli_command(
            "fl", "convert", "c2k", "not_a_number"
        )
        self.assertNotEqual(returncode, 0)
        self.assertIn("error", stderr)

    def test_fl_c2k_no_args(self):
        """Test fl convert c2k CLI with no arguments."""
        returncode, _stdout, stderr = self._run_cli_command("fl", "convert", "c2k")
        self.assertNotEqual(returncode, 0)
        self.assertIn("usage", stderr)

    def test_fl_k2c_basic(self):
        """Test fl convert k2c CLI command."""
        returncode, stdout, stderr = self._run_cli_command(
            "fl", "convert", "k2c", "298.15"
        )
        self.assertEqual(returncode, 0, f"stderr: {stderr}")
        self.assertIn("25.00", stdout)
        self.assertIn("C", stdout)

    def test_fl_k2c_invalid_input(self):
        """Test fl convert k2c CLI with invalid input."""
        returncode, _stdout, stderr = self._run_cli_command(
            "fl", "convert", "k2c", "not_a_number"
        )
        self.assertNotEqual(returncode, 0)
        self.assertIn("error", stderr)

    def test_fl_k2c_no_args(self):
        """Test fl convert k2c CLI with no arguments."""
        returncode, _stdout, stderr = self._run_cli_command("fl", "convert", "k2c")
        self.assertNotEqual(returncode, 0)
        self.assertIn("usage", stderr)


if __name__ == "__main__":
    unittest.main()
