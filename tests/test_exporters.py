"""Test exporters for FuelLib.

This test module verifies that the export CLI commands work correctly.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import fuellib as fl


def run_export_command(cmd, output_dir=None):
    """Run an export command and verify it succeeds.

    :param cmd: Command to run as a list of strings.
    :type cmd: list
    :param output_dir: Optional output directory for the export. If provided, adds -o flag.
    :type output_dir: str, optional
    :raises RuntimeError: If the command fails.
    """
    # Add output directory flag if provided
    if output_dir is not None:
        cmd = cmd + ["-o", output_dir]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
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
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(["fl-export-pele", "-f", "posf10264"], output_dir=tmpdir)


def test_pele_default_deposition_species():
    """Test that deposition species default to the emitted fuel species."""
    for options, expected_species in [
        ([], "n-C07 n-C10"),
        (["-pp"], "NC7H16 NC10H22"),
    ]:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_export_command(
                ["fl-export-pele", "-f", "heptane-decane", *options],
                output_dir=tmpdir,
            )
            output_file = os.path.join(tmpdir, "sprayPropsGCM_heptane-decane.inp")
            with open(output_file) as output:
                lines = output.readlines()

            fuel_species = next(
                line for line in lines if line.startswith("particles.fuel_species =")
            )
            dep_fuel_species = next(
                line
                for line in lines
                if line.startswith("particles.dep_fuel_species =")
            )

            assert fuel_species.rstrip().endswith(expected_species)
            assert dep_fuel_species.rstrip().endswith(expected_species)


def test_pele_mixture_gcm():
    """Test fl-export-pele mixture export with GCM model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(
            ["fl-export-pele", "-f", "posf10264", "-m"], output_dir=tmpdir
        )


def test_pele_mixture_mp():
    """Test fl-export-pele mixture export with MP model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(
            ["fl-export-pele", "-f", "posf10264", "-m", "-l", "mp", "-psat"],
            output_dir=tmpdir,
        )


def test_pele_mixture_cgs():
    """Test fl-export-pele mixture export with CGS units."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(
            ["fl-export-pele", "-f", "posf10264", "-m", "-u", "cgs"],
            output_dir=tmpdir,
        )


def test_pele_deposit_species():
    """Test fl-export-pele single deposit species."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(
            ["fl-export-pele", "-f", "posf10264", "-dep", "POSF10264"],
            output_dir=tmpdir,
        )


def test_converge_individual_component():
    """Test fl-export-converge individual component export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(["fl-export-converge", "-f", "posf10264"], output_dir=tmpdir)


def test_converge_mixture():
    """Test fl-export-converge mixture export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_export_command(
            [
                "fl-export-converge",
                "-f",
                "posf10264",
                "-m",
                "-t",
                "280",
                "-T",
                "400",
                "-s",
                "10",
            ],
            output_dir=tmpdir,
        )


def test_pele_custom_fuel_data_dir():
    """Test fl-export-pele with custom fuel data directory (not embedded data)."""
    # Create temporary directories for fuel data and output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the embedded fuelData to a temp location
        embedded_fueldata = fl.get_fueldata_dir()
        custom_fueldata = os.path.join(tmpdir, "fuelData")
        shutil.copytree(embedded_fueldata, custom_fueldata)

        # Export from custom location to temp output directory
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)
        run_export_command(
            ["fl-export-pele", "-f", "posf10264", "-dir", custom_fueldata],
            output_dir=output_dir,
        )


def test_converge_custom_fuel_data_dir():
    """Test fl-export-converge with custom fuel data directory (not embedded data)."""
    # Create temporary directories for fuel data and output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the embedded fuelData to a temp location
        embedded_fueldata = fl.get_fueldata_dir()
        custom_fueldata = os.path.join(tmpdir, "fuelData")
        shutil.copytree(embedded_fueldata, custom_fueldata)

        # Export from custom location to temp output directory
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)
        run_export_command(
            [
                "fl-export-converge",
                "-f",
                "posf10264",
                "-dir",
                custom_fueldata,
                "-m",
                "-t",
                "280",
                "-T",
                "400",
                "-s",
                "10",
            ],
            output_dir=output_dir,
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
        except Exception as e:  # noqa: BLE001
            print(f"✗ {test_name}: {e}")
            failed.append(test_name)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        print("Failed tests:")
        for test_name in failed:
            print(f"  - {test_name}")
        sys.exit(1)
