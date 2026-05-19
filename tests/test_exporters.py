"""Test exporters for FuelLib.

This test module verifies that the export CLI commands work correctly.
"""

import subprocess
import sys
import tempfile
import shutil
import os
import fuellib as fl


def run_export_command(cmd):
    """Run an export command and verify it succeeds.

    :param cmd: Command to run as a list of strings.
    :type cmd: list
    :raises RuntimeError: If the command fails.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Export command failed: {' '.join(cmd)}")
        print(f"✓ {' '.join(cmd)}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Export command timed out: {' '.join(cmd)}")


def test_pele_individual_component():
    """Test fl-export-pele individual component export."""
    run_export_command(["fl-export-pele", "-f", "posf10264"])


def test_pele_mixture_gcm():
    """Test fl-export-pele mixture export with GCM model."""
    run_export_command(["fl-export-pele", "-f", "posf10264", "-m", "true"])


def test_pele_mixture_mp():
    """Test fl-export-pele mixture export with MP model."""
    run_export_command(["fl-export-pele", "-f", "posf10264", "-m", "true", "-l", "mp"])


def test_pele_mixture_cgs():
    """Test fl-export-pele mixture export with CGS units."""
    run_export_command(["fl-export-pele", "-f", "posf10264", "-m", "true", "-u", "cgs"])


def test_pele_deposit_species():
    """Test fl-export-pele single deposit species."""
    run_export_command(["fl-export-pele", "-f", "posf10264", "-dep", "POSF10264"])


def test_converge_individual_component():
    """Test fl-export-converge individual component export."""
    run_export_command(["fl-export-converge", "-f", "posf10264"])


def test_converge_mixture():
    """Test fl-export-converge mixture export."""
    run_export_command(
        [
            "fl-export-converge",
            "-f",
            "posf10264",
            "-m",
            "true",
            "-t",
            "280",
            "-T",
            "400",
            "-s",
            "10",
        ]
    )


def test_pele_custom_fuel_data_dir():
    """Test fl-export-pele with custom fuel data directory (not embedded data)."""
    # Create a temporary directory and copy fuelData to it
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the embedded fuelData to a temp location
        embedded_fueldata = fl.get_fueldata_dir()
        custom_fueldata = os.path.join(tmpdir, "fuelData")
        shutil.copytree(embedded_fueldata, custom_fueldata)

        # Export from custom location
        run_export_command(
            ["fl-export-pele", "-f", "posf10264", "-dir", custom_fueldata]
        )


def test_converge_custom_fuel_data_dir():
    """Test fl-export-converge with custom fuel data directory (not embedded data)."""
    # Create a temporary directory and copy fuelData to it
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the embedded fuelData to a temp location
        embedded_fueldata = fl.get_fueldata_dir()
        custom_fueldata = os.path.join(tmpdir, "fuelData")
        shutil.copytree(embedded_fueldata, custom_fueldata)

        # Export from custom location
        run_export_command(
            [
                "fl-export-converge",
                "-f",
                "posf10264",
                "-dir",
                custom_fueldata,
                "-m",
                "true",
                "-t",
                "280",
                "-T",
                "400",
                "-s",
                "10",
            ]
        )


if __name__ == "__main__":
    print("Running exporter tests...\n")

    tests = [
        (
            "fl-export-pele - individual component export",
            test_pele_individual_component,
        ),
        ("fl-export-pele - mixture export with GCM model", test_pele_mixture_gcm),
        ("fl-export-pele - mixture export with MP model", test_pele_mixture_mp),
        ("fl-export-pele - mixture export with CGS units", test_pele_mixture_cgs),
        ("fl-export-pele - single deposit species", test_pele_deposit_species),
        (
            "fl-export-pele - custom fuel data directory",
            test_pele_custom_fuel_data_dir,
        ),
        (
            "fl-export-converge - individual component export",
            test_converge_individual_component,
        ),
        ("fl-export-converge - mixture export", test_converge_mixture),
        (
            "fl-export-converge - custom fuel data directory",
            test_converge_custom_fuel_data_dir,
        ),
    ]

    failed = []
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            failed.append(test_name)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        print("Failed tests:")
        for test_name in failed:
            print(f"  - {test_name}")
        sys.exit(1)
