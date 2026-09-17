"""
Export Converge-formatted mixture / component properties over a temperature range.

This script is designed to be run from the command line and creates a file named
``mixturePropsGCM_<fuel_name>.csv`` in the specified output directory. The file
contains mixture properties for the fuel, formatted for use with Converge.

Usage:
    `fl-export-converge -f <fuel_name>

For detailed options, see:
    `fl-export-converge -h`
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pint

import fuellib
from fuellib import PintUnits as Units

# Disable warnings from pandas about Pint units
warnings.filterwarnings("ignore", message="The unit of the quantity is stripped*")
# Disable warnings from numpy about invalid value in power
warnings.filterwarnings("ignore", message="invalid value encountered in power")

# ANSI codes
BOLD = "\033[1m"  # ANSI code for bold text
RED = "\033[91m"  # ANSI code for red text
GREEN = "\033[92m"  # ANSI code for green text
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
}


def _get_label(quantity: pint.Quantity):
    unit = str(quantity.units).replace(" ", "")
    return UNITS_LABELS.get(unit.lower(), unit)


# Set up argument parser
parser = argparse.ArgumentParser(
    description="Export mixture fuel properties for Converge simulations."
)
parser.add_argument(
    "-f",
    "--fuel_name",
    type=str,
    required=True,
    metavar="NAME",
    help="Name of the fuel (mandatory).",
)
parser.add_argument(
    "-dir",
    "--fuel_data_dir",
    type=str,
    default=FUELDATA_DIR,
    metavar="PATH",
    help="Directory where fuel data files are located (optional, default: FuelLib/fuelData).",
)
parser.add_argument(
    "-u",
    "--units",
    type=str,
    default="mks",
    choices=["mks", "cgs"],
    metavar="{mks,cgs}",
    help="Unit system for critical properties (optional, default: mks).",
)
parser.add_argument(
    "-t",
    "--temp_min",
    type=float,
    default=0,
    metavar="temp_units",
    help="Minimum temperature for property calculations (optional, default: 0 K).",
)
parser.add_argument(
    "-T",
    "--temp_max",
    type=float,
    default=1000,
    metavar="temp_units",
    help="Maximum temperature for property calculations (optional, default: 1000 K).",
)
parser.add_argument(
    "-s",
    "--temp_step",
    type=float,
    default=10,
    metavar="temp_units",
    help="Step size for temperature (optional, default: 10 K).",
)
parser.add_argument(
    "-tu",
    "--temp_units",
    type=str,
    default="K",
    choices=["K", "kelvin", "°C", "celsius", "°F", "fahrenheit"],
    metavar="UNIT",
    help="Set units for provided temperatures (optional, default: K).",
)
parser.add_argument(
    "-o",
    "--export_dir",
    type=str,
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
# Alternative flag for `-m`/`--export_mix`
parser.add_argument(
    "-M",
    "--export_mix_flag",
    action="store_true",  # Presence of the flag sets export_mix to True
    help="Export mixture properties of the fuel (optional, default: false).",
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
units = args.units
temp_min = args.temp_min
temp_max = args.temp_max
temp_step = args.temp_step
temp_units = args.temp_units
export_dir = args.export_dir
export_mix = args.export_mix or args.export_mix_flag

# Set up logging
logger = logging.getLogger(__name__)
# Log file handler
file_handler = logging.FileHandler(os.path.join(export_dir, "fl-export-converge.log"))
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

# Check for valid arguments
fuel = fuellib.Fuel(fuel_name, fuelDataDir=fuel_dir)
if not len(fuel.compounds) == len(fuel.Y_0) == fuel.num_compounds:
    msg = f"{fuel_name} does not have valid compounds or initial mole fractions."
    logger.error(msg)
    sys.exit(1)

temp_min = Units.Quantity(temp_min, temp_units)
temp_max = Units.Quantity(temp_max, temp_units)
temp_step = Units.Quantity(temp_step, temp_units)

if temp_min.to("K").magnitude < 0.0:
    msg = "Minimum temperature cannot be below absolute zero."
    logger.error(msg)
    sys.exit(1)

if temp_step.to("K").magnitude <= 0.0:
    msg = "Temperature step must be greater than zero."
    logger.error(msg)
    sys.exit(1)

if not temp_max > temp_min:
    msg = "Maximum temperature must be greater than minimum temperature."
    logger.error(msg)
    sys.exit(1)

os.makedirs(export_dir, exist_ok=True)  # Ensure output directory exists

# Log the parsed arguments
msg = (
    f"{BOLD}{BLUE}"
    f"Exporting {'mixture' if export_mix else 'component'} properties for {fuel_name}:"
    f"{RESET}"
    f"\n  Units               : {units}"
    f"\n  Minimum temperature : {temp_min} {temp_units}"
    f"\n  Maximum temperature : {temp_max} {temp_units}"
    f"\n  Temperature step    : {temp_step} {temp_units}"
    f"\n  Fuel data directory : {fuel_dir}"
    f"\n  Export directory    : {export_dir}"
)
logger.info(msg)

# Build the temperature array
_nT = int(((temp_max - temp_min) / temp_step).magnitude) + 1
temps = Units.Quantity(np.linspace(temp_min, temp_max, _nT)).to("K")


# Helper for min/max allowed temp
def _get_allowed_temperature(
    T_array: pint.Quantity, T_boundary: pint.Quantity, mode: Literal["min", "max"]
) -> pint.Quantity:
    """Get the minimum or maximum allowed temperature based on the boundary temperature."""
    idxs = T_array > T_boundary if mode == "max" else T_array < T_boundary
    distances = np.abs(T_array - T_boundary)
    distances[idxs] = Units.Quantity(
        np.inf, temp_units
    )  # Set distances for temperatures outside the allowed boundary to infinity
    return T_array[distances.argmin()]


components = [fuel.name] if export_mix else fuel.compounds
for comp_idx, comp in enumerate(components):
    # Initialize property arrays
    #: Tc = Critical Temperature
    Tc = Units.Quantity(np.zeros_like(temps), "K").to_base_units(units)
    #: mu = Dynamic Viscosity
    mu = Units.Quantity(np.zeros_like(temps), "Pa*s").to_base_units(units)
    #: gamma = Surface Tension
    gamma = Units.Quantity(np.zeros_like(temps), "N/m").to_base_units(units)
    #: Lv = Latent Heat of Vaporization
    Lv = Units.Quantity(np.zeros_like(temps), "J/kg").to_base_units(units)
    # pv = Vapor Pressure
    pv = Units.Quantity(np.zeros_like(temps), "Pa").to_base_units(units)
    #: rho = Density
    rho = Units.Quantity(np.zeros_like(temps), "kg/m^3").to_base_units(units)
    #: Cl = Specific Heat at Constant Pressure
    Cl = Units.Quantity(np.zeros_like(temps), "J/(kg*K)").to_base_units(units)
    #: kappa = Thermal Conductivity
    kappa = Units.Quantity(np.zeros_like(temps), "W/(m*K)").to_base_units(units)
    #: MW = Molecular Weight
    MW = Units.Quantity(np.zeros_like(temps), units="kg/mol").to_base_units(units)

    if export_mix:
        file_name = os.path.join(export_dir, f"mixturePropsGCM_{fuel.name}.csv")
        # Compute X once up front
        X_0 = fuel.Y2X(fuel.Y_0)
        # Compute MW once up front
        _MW = fuel.mean_molecular_weight(fuel.Y_0).to_base_units(units)

        T_freeze = fuellib.utility.mixing_rule(fuel.Tm, X_0).to("K")
        T_crit = fuellib.utility.mixing_rule(fuel.Tc, X_0).to("K")
        msg = (
            f"{BLUE}"
            f"Estimated mixture freezing temperature: {T_freeze:.2f}"
            f"{RESET}"
            f"\nMinimum component freezing temperature: {min(fuel.Tm):.2f}"
            f"\nMaximum component freezing temperature: {max(fuel.Tm):.2f}"
        )
        logger.info(msg)
        msg = (
            f"{BLUE}"
            f"Estimated mixture critical temperature: {T_crit:.2f}"
            f"{RESET}"
            f"\nMinimum component critical temperature: {min(fuel.Tc):.2f}"
            f"\nMaximum component critical temperature: {max(fuel.Tc):.2f}"
        )
        logger.info(msg)

        # Get the minimum allowed temperature based on the estimated mixture freezing temperature
        T_min_allowed = _get_allowed_temperature(temps, T_freeze, mode="min")
        if np.any(temps < T_min_allowed):
            msg = (
                f"{BOLD}{RED}"
                "Warning: Some components have freezing temperatures above the estimated mixture freezing\n"
                f"temperature ({T_freeze:.2f}). Property calculations will be performed at a minimum of {T_min_allowed:.2f}."
                f"{RESET}"
            )
            logger.warning(msg)

        # Get the maximum allowed temperature based on the estimated mixture critical temperature
        T_max_allowed = _get_allowed_temperature(temps, T_crit, mode="max")
        if np.any(temps > T_max_allowed):
            msg = (
                f"{BOLD}{RED}"
                "Warning: Some components have critical temperatures below the estimated mixture critical\n"
                f"temperature ({T_crit:.2f}). Property calculations will be performed at a maximum of {T_max_allowed:.2f}."
                f"{RESET}"
            )
            logger.warning(msg)

        T = temps[(temps >= T_min_allowed) & (temps <= T_max_allowed)]
        for i, temp in enumerate(T):
            Tc[i] = T_crit
            # Compute the property for each temperature
            mu[i] = fuel.mixture_dynamic_viscosity(fuel.Y_0, temp).to_base_units(units)
            gamma[i] = fuel.mixture_surface_tension(fuel.Y_0, temp).to_base_units(units)
            pv[i] = fuel.mixture_vapor_pressure(fuel.Y_0, temp).to_base_units(units)
            rho[i] = fuel.mixture_density(fuel.Y_0, temp).to_base_units(units)
            kappa[i] = fuel.mixture_thermal_conductivity(fuel.Y_0, temp).to_base_units(
                units
            )
            ## Mixing rules for latent heat and specific heat
            _Lv = fuel.latent_heat_vaporization(temp)
            Lv[i] = fuellib.utility.mixing_rule(_Lv, X_0).to_base_units(units)
            _Cl = fuel.Cl(temp)
            Cl[i] = fuellib.utility.mixing_rule(_Cl, X_0).to_base_units(units)
            MW[i] = _MW

    else:
        msg = f"{BOLD}Processing component {comp_idx + 1}/{len(components)}: {comp}{RESET}"
        logger.info(msg)

        file_name = os.path.join(export_dir, f"{comp_idx}_{comp}.csv")
        # Compute MW once up front
        _MW = fuel.MW[comp_idx].to_base_units(units)

        T_freeze = fuel.Tm[comp_idx]
        T_crit = fuel.Tc[comp_idx]

        # Get allowed temperature range
        T_min_allowed = _get_allowed_temperature(temps, T_freeze, mode="min")
        T_max_allowed = _get_allowed_temperature(temps, T_crit, mode="max")
        if np.any(temps < T_min_allowed) or np.any(temps > T_max_allowed):
            msg = (
                f"{RED}"
                f"{comp} property calculations will be performed from {T_min_allowed:.2f} to {T_max_allowed:.2f}."
                f"{RESET}"
            )
            logger.warning(msg)

        T = temps[(temps >= T_min_allowed) & (temps <= T_max_allowed)]
        for i, temp in enumerate(T):
            Tc[i] = T_crit
            # Compute the property for each temperature
            mu[i] = fuel.viscosity_dynamic(temp, comp_idx).to_base_units(units)
            gamma[i] = fuel.surface_tension(temp, comp_idx).to_base_units(units)
            pv[i] = fuel.psat(temp, comp_idx).to_base_units(units)
            rho[i] = fuel.density(temp, comp_idx).to_base_units(units)
            kappa[i] = fuel.thermal_conductivity(temp, comp_idx).to_base_units(units)
            Lv[i] = fuel.latent_heat_vaporization(temp, comp_idx).to_base_units(units)
            Cl[i] = fuel.Cl(temp, comp_idx).to_base_units(units)
            MW[i] = _MW

    drop_after = i + 1  # Index after last data entry
    data = {
        f"Temperature ({_get_label(T)})": T.magnitude,
        f"Critical Temperature ({_get_label(Tc)})": Tc[:drop_after].magnitude,
        f"Viscosity ({_get_label(mu)})": mu[:drop_after].magnitude,
        f"Surface Tension ({_get_label(gamma)})": gamma[:drop_after],
        f"Heat of Vaporization ({_get_label(Lv)})": Lv[:drop_after],
        f"Vapor Pressure ({_get_label(pv)})": pv[:drop_after],
        f"Density ({_get_label(rho)})": rho[:drop_after],
        f"Specific Heat ({_get_label(Cl)})": Cl[:drop_after],
        f"Thermal Conductivity ({_get_label(kappa)})": kappa[:drop_after],
        f"Molecular Weight ({_get_label(MW)})": MW[:drop_after],
    }

    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)

    msg = f"{GREEN}Properties exported to {file_name}.{RESET}"
    logger.info(msg)
