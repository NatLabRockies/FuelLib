"""
Export Pele-formatted critical properties and initial mass fraction data.

This script is designed to be run from the command line and creates a file named
``sprayPropsGCM_<fuel_name>.csv`` or ``sprayPropsMP_<fuel_name>.csv`` in the specified
output directory. The file contains properties for each compound in the fuel, formatted
for use with Pele.

Usage:
    `fl-export-pele -f <fuel_name>

For detailed options, see:
    `fl-export-pele -h`
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from typing import Literal
from urllib import error, request

import numpy as np
import pandas as pd
import pint
from scipy import stats

import fuellib
from fuellib import PintUnits as Units
from fuellib.utility import mixing_rule

# Disable warnings from pandas about Pint units
warnings.filterwarnings("ignore", message="The unit of the quantity is stripped*")
# Disable warnings from numpy about invalid value in power
warnings.filterwarnings("ignore", message="invalid value encountered in power")


# Helper functions for finding repository host
def _git_info():
    """Get git commit hash and remote URL for FuelLib (with fallbacks)."""
    # Get the directory where FuelLib is installed
    fuellib_dir = os.path.dirname(os.path.dirname(os.path.abspath(fuellib.__file__)))

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "-C", fuellib_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode("utf-8")
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        # Fall back to package version
        try:
            git_commit = fuellib.__version__
        except AttributeError:
            git_commit = "N/A"

    try:
        git_remote = (
            subprocess.check_output(
                ["git", "-C", fuellib_dir, "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode("utf-8")
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        # Try to get repository URL from PyPI metadata
        git_remote = _get_pypi_repo_url()

    return git_commit, git_remote


def _get_pypi_repo_url():
    """Get the repository URL from PyPI package metadata."""
    try:
        version = fuellib.__version__
        pypi_api_url = f"https://pypi.org/pypi/fuellib/{version}/json"

        with request.urlopen(pypi_api_url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

            # Try to get repository URL from project URLs
            if "info" in data and "project_urls" in data["info"]:
                project_urls = data["info"]["project_urls"]
                if project_urls:
                    # Look for common repository URL keys
                    for key in ["Repository", "Homepage", "Source Code", "Code"]:
                        if key in project_urls:
                            return project_urls[key]

            # Fallback to home page
            if (
                "info" in data
                and "home_page" in data["info"]
                and data["info"]["home_page"]
            ):
                return data["info"]["home_page"]
    except (KeyError, TypeError, error.URLError):
        pass

    # Final fallback: PyPI package URL
    try:
        version = fuellib.__version__
        return f"https://pypi.org/project/fuellib/{version}/"
    except AttributeError:
        return "https://pypi.org/project/fuellib/"


# Helper for generating filename
def _filename(
    fuel_name: str, liq_prop_model: Literal["gcm", "mp"], export_mix: bool, path: str
) -> str:
    """Generate a filename for the exported data based on fuel name, liquid property model, and export mix flag."""
    prefix = f"sprayProps{liq_prop_model.upper()}"
    if export_mix:
        prefix += "_mixture"
    return os.path.join(path, f"{prefix}_{fuel_name}.inp")


# ANSI codes
BOLD = "\033[1m"  # ANSI code for bold text
RED = "\033[91m"  # ANSI code for red text
GREEN = "\033[92m"  # ANSI code for green text
YELLOW = "\033[93m"  # ANSI code for yellow text
BLUE = "\033[94m"  # ANSI code for blue text
RESET = "\033[0m"  # ANSI code to reset text color

# Get default data directory
FUELDATA_DIR = fuellib.get_fueldata_dir()

# Preferred unit labels
UNITS_LABELS = {
    "kelvin": "K",
    "gram/centimeter/second": "Poise",
    "gram/second**2": "dyne/cm",
    "centimeter**2/second**2": "erg/g",
    "gram/centimeter/second**2": "dyne/cm^2",
    "gram/centimeter**3": "g/cm^3",
    "centimeter**2/kelvin/second**2": "erg/g/K",
    "centimeter*gram/kelvin/second**3": "erg/cm/s/K",
    "gram/mole": "g/mol",
    "kilogram/meter/second": "Pa*s",
    "kilogram/second**2": "N/m",
    "meter**2/second**2": "J/kg",
    "kilogram/meter/second**2": "Pa",
    "kilogram/meter**3": "kg/m^3",
    "meter**2/kelvin/second**2": "J/kg/K",
    "kilogram*meter/kelvin/second**3": "W/m/K",
    "kilogram/mole": "kg/mol",
    "centimeter**3/mole": "cm^3/mol",
    "meter**3/mole": "m^3/mol",
    "dimensionless": "-",
}

# Exported properties by liquid property model
GCM_PROPS = [
    "family",
    "molar_weight",
    "crit_temp",
    "crit_press",
    "crit_vol",
    "boil_temp",
    "omega",
    "molar_vol",
    "cp_a",
    "cp_b",
    "cp_c",
    "cp",
    "latent",
]

MP_PROPS = [
    "molar_weight",
    "crit_temp",
    "boil_temp",
    "latent",
    "cp",
    "rho",
    "psat",
]


def _get_label(quantity: pint.Quantity):
    unit = str(quantity.units).replace(" ", "")
    return UNITS_LABELS.get(unit.lower(), unit)


def _quantity_cells(quantity: pint.Quantity) -> list[pint.Quantity]:
    """Split an array-valued pint Quantity into a list of scalar Quantities."""
    return [
        Units.Quantity(v, quantity.units) for v in np.atleast_1d(quantity.magnitude)
    ]


# Mandatory argument for fuel name
parser = argparse.ArgumentParser(
    description="Export Pele-formatted critical properties and initial mass fraction data."
)
parser.add_argument(
    "-f",
    "--fuel_name",
    required=True,
    metavar="NAME",
    help="Name of the fuel (mandatory).",
)
parser.add_argument(
    "-dir",
    "--fuel_data_dir",
    default=FUELDATA_DIR,
    metavar="PATH",
    help="Directory where fuel data files are located (optional, default: FuelLib/fuelData).",
)
parser.add_argument(
    "-decomp",
    "--fuel_decomp_name",
    default=None,
    metavar="NAME",
    help="Name of the decomposition file (optional). If not provided, defaults to fuel_name.",
)
parser.add_argument(
    "-u",
    "--units",
    type=str,
    choices=["mks", "cgs"],
    default="mks",
    metavar="{mks,cgs}",
    help="Units for critical properties (optional, default: mks).",
)
parser.add_argument(
    "-dep",
    "--dep_fuel_names",
    type=list[str],
    nargs="+",  # Accepts one or more values
    default=None,
    metavar="NAME",
    help="Space-separated list or single fuel that each compound deposits to (optional, default: fuel.compounds).",
)
parser.add_argument(
    "-pp",
    "--use_pp_keys",
    type=lambda x: str(x).lower() in ["true", "1"],
    default=True,
    metavar="{true,false}",
    help="Use PelePhysics keys for each compound (optional, default: true).",
)
parser.add_argument(
    "-o",
    "--export_dir",
    default=os.getcwd(),
    metavar="PATH",
    help="Directory to export the properties (optional, default: current working directory).",
)
parser.add_argument(
    "-m",
    "--export_mix",
    type=lambda x: str(x).lower() in ["true", "1"],
    default=False,
    metavar="{true,false}",
    help="Export mixture properties of the fuel (optional, default: false).",
)
parser.add_argument(
    "-mn",
    "--export_mix_name",
    default=None,
    metavar="NAME",
    help="Name the mixture if different than fuel_name (optional, default: fuel_name).",
)
parser.add_argument(
    "-l",
    "--liq_prop_model",
    type=str,
    choices=["gcm", "mp"],
    default="gcm",
    metavar="{gcm,mp}",
    help="Model for liquid properties (optional, default: gcm).",
)
parser.add_argument(
    "-psat",
    "--psat_antoine",
    type=lambda x: str(x).lower() in ["true", "1"],
    default=True,
    metavar="{true,false}",
    help="Use Antoine coefficients for vapor pressure in MP model (optional, default: true).",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Enable verbose console output (optional, default: false).",
)

# Parse arguments
args = parser.parse_args()
fuel_name = args.fuel_name
fuel_dir = args.fuel_data_dir
fuel_decomp_name = args.fuel_decomp_name or fuellib.get_metadata_decomp_name(
    fuel_name, fuel_dir
)
units = args.units
dep_fuel_names = args.dep_fuel_names
use_pp_keys = args.use_pp_keys
export_dir = args.export_dir
export_mix = args.export_mix
export_mix_name = args.export_mix_name
liq_prop_model = args.liq_prop_model
psat_antoine = args.psat_antoine

# Set up logging
logger = logging.getLogger(__name__)
# Log file handler
file_handler = logging.FileHandler(os.path.join(export_dir, "fl-export-pele.log"))
file_handler.setLevel(logging.INFO)
# Console output handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO if args.verbose else logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s \n",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler, console_handler],
)

# Check for valid Fuel
fuel = fuellib.Fuel(fuel_name, fuelDataDir=fuel_dir)
if not len(fuel.compounds) == len(fuel.Y_0) == fuel.num_compounds:
    msg = f"{fuel_name} does not have valid compounds or initial mole fractions."
    logger.error(msg)
    sys.exit(1)

os.makedirs(export_dir, exist_ok=True)

msg = (
    f"{BOLD}{BLUE}"
    f"Exporting critical properties for {fuel_name}:"
    f"{RESET}"
    f"\n  Decomposition name        : {fuel_decomp_name}"
    f"\n  Units                     : {units}"
    f"\n  Liquid property model     : {liq_prop_model}"
    f"\n  Antoine coefficients      : {psat_antoine if liq_prop_model == 'mp' else 'N/A'}"
    f"\n  Export mixture properties : {export_mix}"
    f"\n  Export directory          : {export_dir}"
    f"\n  Fuel data directory       : {fuel_dir}"
)
logger.info(msg)

if use_pp_keys and fuel.pelephysics_keys is not None:
    msg = (
        f"{BOLD}{GREEN}"
        "Using PelePhysics keys from GCxGC data. Please ensure consistency with PelePhysics mechanism."
        f"{RESET}"
    )
    logger.info(msg)
    compound_names = fuel.pelephysics_keys
elif use_pp_keys and fuel.pelephysics_keys is None:
    msg = (
        f"{BOLD}{YELLOW}"
        "PelePhysics keys not found in GCxGC data. Using compound names instead."
        f"{RESET}"
    )
    logger.warning(msg)
    compound_names = fuel.compounds
elif not use_pp_keys and fuel.pelephysics_keys is not None:
    msg = (
        f"{BOLD}{YELLOW}"
        "PelePhysics keys are available but not used. Using compound names instead."
        f"{RESET}"
    )
    logger.info(msg)
    compound_names = fuel.compounds

for c in compound_names:
    if " " in c:
        msg = (
            f"{BOLD}{RED}"
            f"Compound '{c}' contains spaces, which PelePhysics does not accept."
            "Use '-' instead of spaces in compound names."
            f"{RESET}"
        )
        logger.warning(msg)
        sys.exit(1)

if not export_mix:
    msg = f"{BOLD}Calculating GCM properties for individual compounds in {fuel.name}.{RESET}"
    logger.info(msg)

    if dep_fuel_names is None:
        dep_fuel_names = list(compound_names)
    elif len(dep_fuel_names) == 1:
        dep_fuel_names = [dep_fuel_names[0]] * len(compound_names)
    elif len(dep_fuel_names) != len(compound_names):
        msg = (
            f"{BOLD}{RED}"
            "The number of dependent fuel names must be one or match the number of compounds."
            f"{RESET}"
        )
        logger.error(msg)
        sys.exit(1)

    Cp_A = fuel.Cp_stp / fuel.MW
    Cp_B = fuel.Cp_B / fuel.MW
    Cp_C = fuel.Cp_C / fuel.MW

    df = pd.DataFrame(
        {
            "Compound": compound_names,
            "family": fuel.fam,
            "Y_0": fuel.Y_0,
            "molar_weight": _quantity_cells(fuel.MW.to_base_units(units)),
            "crit_temp": _quantity_cells(fuel.Tc.to_base_units(units)),
            "crit_press": _quantity_cells(fuel.Pc.to_base_units(units)),
            "crit_vol": _quantity_cells(fuel.Vc.to_base_units(units)),
            "boil_temp": _quantity_cells(fuel.Tb.to_base_units(units)),
            "omega": _quantity_cells(fuel.omega.to_base_units(units)),
            "molar_vol": _quantity_cells(fuel.Vm_stp.to_base_units(units)),
            "cp_a": _quantity_cells(Cp_A.to_base_units(units)),
            "cp_b": _quantity_cells(Cp_B.to_base_units(units)),
            "cp_c": _quantity_cells(Cp_C.to_base_units(units)),
            "cp": _quantity_cells(Cp_A.to_base_units(units)),
            "latent": _quantity_cells(fuel.Lv_stp.to_base_units(units)),
        }
    )

else:
    msg = f"{BOLD}Calculating mixture GCM properties for {fuel.name} at standard conditions.{RESET}"
    logger.info(msg)

    export_mix_name = export_mix_name or fuel.name
    if "posf" in export_mix_name.lower():
        export_mix_name = export_mix_name.upper()

    X_0 = fuel.Y2X(fuel.Y_0)
    Cp_A = fuel.Cp_stp / fuel.MW
    Cp_B = fuel.Cp_B / fuel.MW
    Cp_C = fuel.Cp_C / fuel.MW

    def _mix(prop: pint.Quantity, X_0: np.ndarray, units: str) -> pint.Quantity:
        """Mix a given property according to the mole fractions then format it to the specified units."""
        val = mixing_rule(prop, X_0).to_base_units(units)
        return val

    df = pd.DataFrame(
        {
            "Compound": [export_mix_name],
            "family": [stats.mode(fuel.fam).mode],
            "Y_0": [1.0],
            "molar_weight": fuel.mean_molecular_weight(fuel.Y_0).to_base_units(units),
            "crit_temp": [_mix(fuel.Tc, X_0, units)],
            "crit_press": [_mix(fuel.Pc, X_0, units)],
            "crit_vol": [_mix(fuel.Vc, X_0, units)],
            "boil_temp": [_mix(fuel.Tb, X_0, units)],
            "omega": [_mix(fuel.omega, X_0, units)],
            "molar_vol": [_mix(fuel.Vm_stp, X_0, units)],
            "cp_a": [_mix(Cp_A, X_0, units)],
            "cp_b": [_mix(Cp_B, X_0, units)],
            "cp_c": [_mix(Cp_C, X_0, units)],
            "cp": [_mix(Cp_A, X_0, units)],
            "latent": [_mix(fuel.Lv_stp, X_0, units)],
        }
    )

    compound_names = df["Compound"].tolist()
    dep_fuel_names = dep_fuel_names or compound_names

if liq_prop_model == "gcm":
    prop_names = GCM_PROPS

else:
    prop_names = MP_PROPS
    if not psat_antoine:
        prop_names.remove("psat")
    # Calculate density at 298.15 K
    ref_T = Units.Quantity(298.15, "K")
    rho = fuel.mixture_density(fuel.Y_0, ref_T) if export_mix else fuel.density(ref_T)
    df["rho"] = rho.to_base_units(units) if export_mix else _quantity_cells(rho)

    if psat_antoine:
        if export_mix:
            psat_A, psat_B, psat_C, psat_D = fuel.mixture_vapor_pressure_antoine_coeffs(
                fuel.Y_0, units=units
            )
        else:
            psat_A, psat_B, psat_C, psat_D = fuel.psat_antoine_coeffs(units=units)

        df["psat_A"] = psat_A
        df["psat_B"] = psat_B
        df["psat_C"] = psat_C
        df["psat_D"] = psat_D


# Get header information
now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
git_commit, git_remote = _git_info()

file_name = _filename(fuel.name, liq_prop_model, export_mix, export_dir)
msg = f"{BOLD}Exporting properties to {file_name}.{RESET}"
logger.info(msg)
with open(file_name, "w") as f:

    def _vec_to_str(vec: list | pd.Series | pd.DataFrame) -> str:
        """Convert a list or array to a string."""
        if isinstance(vec, list):
            return " ".join(str(x) for x in vec)
        return " ".join(str(x) for x in vec.values)

    f.write(
        f"# -----------------------------------------------------------------------------\n"
        f"# Liquid fuel properties for {liq_prop_model.upper()} in Pele\n"
        f"# Fuel: {fuel.name}\n"
        f"# Number of compounds: {len(compound_names)}\n"
        f"# Generated: {now}\n"
        f"# FuelLib remote URL: {git_remote}\n"
        f"# Git commit: {git_commit}\n"
        f"# Units: {units.upper()}\n"
        f"# -----------------------------------------------------------------------------\n\n"
    )
    f.write(f"particles.fuel_species = {_vec_to_str(df['Compound'].tolist())}\n")
    f.write(f"particles.Y_0 = {_vec_to_str(df['Y_0'].tolist())}\n")
    f.write(f"particles.dep_fuel_species = {_vec_to_str(dep_fuel_names)}\n")
    if liq_prop_model == "mp":
        f.write(f"particles.fuel_ref_temp = {ref_T.to('K').magnitude} # K\n")

    for comp in compound_names:
        f.write(f"\n# Properties for {comp} in {units.upper()}\n")
        for prop in prop_names:
            if prop == "family":
                _labels = [
                    "saturated hydrocarbons",
                    "aromatics",
                    "cycloparaffins",
                    "olefins",
                ]
                cell = df.loc[df["Compound"] == comp, prop].values[0]
                f.write(f"particles.{comp}_{prop} = {cell} # {_labels[cell]}\n")
                continue

            if prop == "psat":
                A = df.loc[df["Compound"] == comp, "psat_A"].values[0]
                B = df.loc[df["Compound"] == comp, "psat_B"].values[0]
                C = df.loc[df["Compound"] == comp, "psat_C"].values[0]
                D = df.loc[df["Compound"] == comp, "psat_D"].values[0]
                psat_coeffs = [A, B, C, D]

                _Q = Units.Quantity(1, "Pa").to_base_units(units)
                f.write(
                    f"particles.{comp}_{prop} = {_vec_to_str(psat_coeffs)} # {_get_label(_Q)}\n"
                )
                continue

            cell = df.loc[df["Compound"] == comp, prop].values[0]
            cell_units = _get_label(cell) if isinstance(cell, pint.Quantity) else ""
            cell_val = cell.magnitude if isinstance(cell, pint.Quantity) else cell

            f.write(f"particles.{comp}_{prop} = {cell_val:.6f} # {cell_units}\n")

msg = f"{BOLD}{GREEN}Properties exported to {file_name}.{RESET}"
logger.info(msg)
