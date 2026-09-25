import argparse
import os

import numpy as np
import pandas as pd

import fuellib as fl

from ..utils import Units

# Default data directory - use fuellib's embedded data
FUELDATA_DIR = fl.get_fueldata_dir()

"""
Script that exports mixture properties over large temperature range for use in
Converge simulations.

This script is designed to be run from the command line and will create
a file named "mixturePropsGCM_<fuel_name>.csv" in the specified directory.
The file contains mixture properties for the fuel, formatted for Converge.

Usage:
    fl-export-converge -f <fuel_name>

For detailed options, run:
    fl-export-converge -h
"""


class UnitConverter:
    """MKS column labels for Converge property files."""

    def __init__(self):
        """Initialize MKS labels."""
        self._set_labels()

    def _set_labels(self):
        """Set unit labels for DataFrame columns."""
        self.labels = {
            "temperature": "Temperature (K)",
            "critical_temp": "Critical Temperature (K)",
            "viscosity": "Viscosity (Pa*s)",
            "surface_tension": "Surface Tension (N/m)",
            "heat_vaporization": "Heat of Vaporization (J/kg)",
            "vapor_pressure": "Vapor Pressure (Pa)",
            "density": "Density (kg/m^3)",
            "specific_heat": "Specific Heat (J/kg/K)",
            "thermal_conductivity": "Thermal Conductivity (W/m/K)",
            "molecular_weight": "Molecular Weight (kg/mol)",
        }

    def create_data_dict(
        self, T, T_crit, mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity
    ):
        """Create an MKS data dictionary for CSV output.

        :param T: Temperature array.
        :type T: np.ndarray
        :param T_crit: Critical temperature.
        :type T_crit: float
        :param mu: Viscosity array.
        :type mu: np.ndarray
        :param surface_tension: Surface tension array.
        :type surface_tension: np.ndarray
        :param Lv: Heat of vaporization array.
        :type Lv: np.ndarray
        :param pv: Vapor pressure array.
        :type pv: np.ndarray
        :param rho: Density array.
        :type rho: np.ndarray
        :param Cl: Specific heat array.
        :type Cl: np.ndarray
        :param thermal_conductivity: Thermal conductivity array.
        :type thermal_conductivity: np.ndarray
        :return: Dictionary with converted properties and labels.
        :rtype: dict
        """
        return {
            self.labels["temperature"]: T.to("K").magnitude,
            self.labels["critical_temp"]: T_crit.to("K").magnitude
            + np.zeros_like(T.magnitude),
            self.labels["viscosity"]: mu.to("Pa*s").magnitude,
            self.labels["surface_tension"]: surface_tension.to("N/m").magnitude,
            self.labels["heat_vaporization"]: Lv.to("J/kg").magnitude,
            self.labels["vapor_pressure"]: pv.to("Pa").magnitude,
            self.labels["density"]: rho.to("kg/m^3").magnitude,
            self.labels["specific_heat"]: Cl.to("J/(kg*K)").magnitude,
            self.labels["thermal_conductivity"]: thermal_conductivity.to(
                "W/(m*K)"
            ).magnitude,
        }


def export_converge(
    fuel,
    path=None,
    temp_min=0,
    temp_max=1000,
    temp_step=10,
    export_mix=False,
):
    """Export mixture fuel properties to csv files for Converge simulations.

    :param fuel: Fuel object containing properties to export.
    :type fuel: fl.Fuel

    :param path: Directory to save the input file.
    :type path: str, optional (default: current working directory)

    :param temp_min: Minimum temperature (K) for the property calculations.
    :type temp_min: float, optional (default: 0)

    :param temp_max: Maximum temperature (K)for the property calculations.
    :type temp_max: float, optional (default: 1000)

    :param temp_step: Step size for temperature (K).
    :type temp_step: int, optional (default: 10)

    :param export_mix: Whether to export individual component or mixture properties.
    :type export_mix: bool, optional (default: False)

    :return: None
    :rtype: None

    :raises ValueError: If input parameters are invalid
    :raises TypeError: If fuel object is not a FuelLib fuel instance
    """
    if path is None:
        path = os.getcwd()

    temp_min = (
        Units.Quantity(temp_min, "K")
        if not hasattr(temp_min, "to")
        else temp_min.to("K")
    )
    temp_max = (
        Units.Quantity(temp_max, "K")
        if not hasattr(temp_max, "to")
        else temp_max.to("K")
    )
    temp_step = (
        Units.Quantity(temp_step, "K")
        if not hasattr(temp_step, "to")
        else temp_step.to("K")
    )

    # Input validation
    if not hasattr(fuel, "compounds") or not hasattr(fuel, "Y_0"):
        raise TypeError("fuel parameter must be a valid FuelLib fuel object")

    if temp_min.magnitude < 0:
        raise ValueError(f"temp_min must be non-negative, got {temp_min}")

    if temp_max <= temp_min:
        raise ValueError(
            f"temp_max ({temp_max}) must be greater than temp_min ({temp_min})"
        )

    if temp_step.magnitude <= 0:
        raise ValueError(f"temp_step must be positive, got {temp_step}")

    # Ensure output directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    if export_mix:
        # Export mixture properties only
        file_name = os.path.join(path, f"mixturePropsGCM_{fuel.name}.csv")
        components = [fuel.name]
    else:
        # Export individual component properties and composition
        path = os.path.join(path, fuel.name)
        components = fuel.compounds

    # Initialize unit converter
    converter = UnitConverter()

    def nearest_temp(x, base=temp_step):
        """Round to nearest multiple of temp_step.

        :param x: Temperature value to round.
        :type x: float
        :param base: Base value for rounding (temp_step).
        :type base: float
        :return: Rounded temperature.
        :rtype: float
        """
        return base * round((x / base).magnitude)

    def nearest_floor(array, value):
        """Find the largest value in the array that is less than or equal to the given value.

        :param array: Array of temperature values.
        :type array: np.ndarray
        :param value: Reference value.
        :type value: float
        :return: Largest array value <= reference value.
        :rtype: float
        :raises ValueError: If no array value is <= reference value.
        """
        if np.any(array <= value):
            return array[array <= value].max()
        else:
            raise ValueError(
                f"No temperature in the array is less than or equal to the critical point {value}. Choose a lower temp_min"
            )

    def nearest_ceil(array, value):
        """Find the smallest value in the array that is greater than or equal to the given value.

        :param array: Array of temperature values.
        :type array: np.ndarray
        :param value: Reference value.
        :type value: float
        :return: Smallest array value >= reference value.
        :rtype: float
        :raises ValueError: If no array value is >= reference value.
        """
        if np.any(array >= value):
            return array[array >= value].min()
        else:
            raise ValueError(
                f"No temperature in the array is greater than or equal the freezing point {value}. Choose a higher temp_max"
            )

    def validate_temperature_range(T_array, T_freeze, T_crit, is_mixture=True):
        """Validate and adjust temperature range based on freezing and critical temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param T_freeze: Freezing temperature.
        :type T_freeze: float
        :param T_crit: Critical temperature.
        :type T_crit: float
        :param is_mixture: Whether this is for mixture properties.
        :type is_mixture: bool
        :return: Tuple of (T_min_allowed, T_max_allowed, adjusted_T_array).
        :rtype: tuple
        """
        T_min_allowed = nearest_temp(T_freeze)
        T_max_allowed = T_crit

        # Handle minimum temperature warnings
        if np.any(T_array < T_min_allowed):
            T_min_allowed = nearest_ceil(T_array, T_min_allowed)
            compound_type = "mixture" if is_mixture else "compound"
            print("!" * 88)
            print(
                "   Warning: Some compounds have freezing temperatures above the estimated"
            )
            print(
                f"   freezing temperature of the {compound_type} ({T_freeze:.2f} K). All properties calculated"
            )
            print(
                f"   below {T_min_allowed} K will be set using a temperature of {T_min_allowed} K."
            )
            print("!" * 88)

        # Handle maximum temperature warnings for mixtures
        if is_mixture:
            T_max_allowed = min(fuel.Tc)
            if np.any(T_array > T_max_allowed):
                T_max_allowed = nearest_floor(T_array, T_max_allowed)
                print("!" * 88)
                print(
                    "   Warning: Some compounds have critical temperatures below the estimated"
                )
                print(
                    f"   critical temperature of the mixture ({T_crit:.2f} K). All properties will be"
                )
                print(f"   calculated up to {T_max_allowed} K.")
                print("!" * 88)

        # Filter temperature array to allowed range
        adjusted_T = T_array[(T_array >= T_min_allowed) & (T_array <= T_max_allowed)]

        return T_min_allowed, T_max_allowed, adjusted_T

    def calculate_mixture_properties(T_array, fuel):
        """Calculate mixture properties for a range of temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.Fuel
        :return: Tuple of property arrays (mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = Units.Quantity(np.zeros_like(T_array.magnitude), "Pa*s")
        surface_tension = Units.Quantity(np.zeros_like(T_array.magnitude), "N/m")
        Lv = Units.Quantity(np.zeros_like(T_array.magnitude), "J/kg")
        pv = Units.Quantity(np.zeros_like(T_array.magnitude), "Pa")
        rho = Units.Quantity(np.zeros_like(T_array.magnitude), "kg/m^3")
        Cl = Units.Quantity(np.zeros_like(T_array.magnitude), "J/(kg*K)")
        thermal_conductivity = Units.Quantity(
            np.zeros_like(T_array.magnitude), "W/(m*K)"
        )

        for k, Temp in enumerate(T_array):
            Y_li = fuel.Y_0
            X_li = fuel.Y2X(Y_li)

            # Standard mixing rules for properties
            rho[k] = fuel.mixture_density(Y_li, Temp)  # kg/m^3
            mu[k] = fuel.mixture_dynamic_viscosity(Y_li, Temp)  # Pa*s
            pv[k] = fuel.mixture_vapor_pressure(Y_li, Temp)  # Pa
            surface_tension[k] = fuel.mixture_surface_tension(Y_li, Temp)  # N/m
            thermal_conductivity[k] = fuel.mixture_thermal_conductivity(Y_li, Temp)

            # Generic mixing rules for latent heat and specific heat
            Lv[k] = fl.utility.mixing_rule(
                fuel.latent_heat_vaporization(Temp), X_li
            )  # J/kg
            Cl[k] = fl.utility.mixing_rule(fuel.Cl(Temp), X_li)  # J/kg/K

        return mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity

    def calculate_component_properties(T_array, fuel, comp_idx):
        """Calculate individual component properties for a range of temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.Fuel
        :param comp_idx: Index of the component.
        :type comp_idx: int
        :return: Tuple of property arrays (mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = Units.Quantity(np.zeros_like(T_array.magnitude), "Pa*s")
        surface_tension = Units.Quantity(np.zeros_like(T_array.magnitude), "N/m")
        Lv = Units.Quantity(np.zeros_like(T_array.magnitude), "J/kg")
        pv = Units.Quantity(np.zeros_like(T_array.magnitude), "Pa")
        rho = Units.Quantity(np.zeros_like(T_array.magnitude), "kg/m^3")
        Cl = Units.Quantity(np.zeros_like(T_array.magnitude), "J/(kg*K)")
        thermal_conductivity = Units.Quantity(
            np.zeros_like(T_array.magnitude), "W/(m*K)"
        )

        for k, Temp in enumerate(T_array):
            rho[k] = fuel.density(Temp, comp_idx=comp_idx)  # kg/m^3
            mu[k] = fuel.viscosity_dynamic(Temp, comp_idx=comp_idx)  # Pa*s
            pv[k] = fuel.psat(Temp, comp_idx=comp_idx)  # Pa
            surface_tension[k] = fuel.surface_tension(Temp, comp_idx=comp_idx)  # N/m
            thermal_conductivity[k] = fuel.thermal_conductivity(Temp, comp_idx=comp_idx)
            Lv[k] = fuel.latent_heat_vaporization(Temp, comp_idx=comp_idx)  # J/kg
            Cl[k] = fuel.Cl(Temp, comp_idx=comp_idx)  # J/kg/K

        return mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity

    def export_properties_to_csv(file_path, data_dict, overwrite=True):
        """Export properties data to CSV file.

        :param file_path: Path to the output CSV file.
        :type file_path: str
        :param data_dict: Dictionary containing property data.
        :type data_dict: dict
        :param overwrite: Whether to overwrite existing file.
        :type overwrite: bool
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Remove existing file if overwrite is True
        if overwrite and os.path.exists(file_path):
            os.remove(file_path)

        # Create and save DataFrame
        df = pd.DataFrame(data_dict)
        df.to_csv(file_path, index=False)

    if export_mix:
        # Vector of evenly spaced temperatures
        nT = int(((temp_max - temp_min) / temp_step).magnitude) + 1
        T = Units.Quantity(np.linspace(temp_min.magnitude, temp_max.magnitude, nT), "K")

        # Estimate freezing point and critical temp of mixture
        T_freeze = fl.utility.mixing_rule(fuel.Tm, fuel.Y2X(fuel.Y_0))
        T_crit = fl.utility.mixing_rule(fuel.Tc, fuel.Y2X(fuel.Y_0))

        print(f"\nEstimated mixture freezing temp: {T_freeze:.2f} K")
        print(f"Min freezing temp min(Tm_i): {min(fuel.Tm):.2f} K")
        print(f"Max freezing temp max(Tm_i): {max(fuel.Tm):.2f} K")
        print(f"Estimated mixture critical temp: {T_crit:.2f} K")
        print(f"Min critical temp min(Tc_i): {min(fuel.Tc):.2f} K")
        print(f"Max critical temp max(Tc_i): {max(fuel.Tc):.2f} K")

        # Validate and adjust temperature range
        T_min_allowed, T_max_allowed, T = validate_temperature_range(
            T, T_freeze, T_crit, is_mixture=True
        )

    for comp_idx, compound in enumerate(components):
        if not export_mix:
            # Get component-specific temperature limits
            T_freeze = fuel.Tm[comp_idx]
            T_crit = fuel.Tc[comp_idx]
            T_min_allowed = nearest_temp(T_freeze)

            # Create temperature array up to critical temperature
            maxtemps = Units.Quantity(
                np.array([
                    (nearest_temp(T_crit) - temp_step).magnitude,
                    nearest_temp(T_crit).magnitude,
                    (nearest_temp(T_crit) + temp_step).magnitude,
                ]),
                "K",
            )
            T_nearest_floor = nearest_floor(maxtemps, T_crit)
            nT = int(((T_nearest_floor - T_min_allowed) / temp_step).magnitude) + 1
            T = Units.Quantity(
                np.linspace(T_min_allowed.magnitude, T_nearest_floor.magnitude, nT),
                "K",
            )
            T = Units.Quantity(np.append(T.magnitude, T_crit.to("K").magnitude), "K")
            T_max_allowed = T_crit
        # Calculate GCM properties for a range of temperatures
        comp_text = "" if export_mix else f"for {compound}"
        print(
            f"\nCalculating properties {comp_text} over {len(T)} temperatures from {T_min_allowed} K to {T_max_allowed} K..."
        )

        if export_mix:
            (
                mu,
                surface_tension,
                Lv,
                pv,
                rho,
                Cl,
                thermal_conductivity,
            ) = calculate_mixture_properties(T, fuel)
        else:
            (
                mu,
                surface_tension,
                Lv,
                pv,
                rho,
                Cl,
                thermal_conductivity,
            ) = calculate_component_properties(T, fuel, comp_idx)

        # Create data dictionary with converted units
        data = converter.create_data_dict(
            T, T_crit, mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity
        )

        # Export the properties to CSV file
        if export_mix:
            print(f"\nWriting mixture properties to {file_name}")
        else:
            file_name = os.path.join(path, f"{comp_idx}_{compound}.csv")
            print(f"\nWriting properties for {compound} to {file_name}")

        export_properties_to_csv(file_name, data)

    if not export_mix:
        # Also export the initial mass fractions
        composition_file = os.path.join(path, f"composition_{fuel.name}.csv")
        print(f"\nWriting mass fractions for {fuel.name} to {composition_file}")
        composition_data = {
            "Index": range(len(fuel.compounds)),
            "Component": fuel.compounds,
            "Mass Fraction": fuel.Y_0,
            "Mole Fraction": fuel.Y2X(fuel.Y_0),
            converter.labels["molecular_weight"]: fuel.MW.to("kg/mol").magnitude,
        }
        export_properties_to_csv(composition_file, composition_data)


def main():
    """Main function to execute the export process.

    :param --fuel_name: Name of the fuel (mandatory).
    :type --fuel_name: str

    :param --fuel_data_dir: Directory where fuel data files are located.
    :type --fuel_data_dir: str, optional (default: FuelLib/fuelData)

    :param --temp_min: Minimum temperature (K) for the property calculations.
    :type --temp_min: float, optional (default: 0 K)

    :param --temp_max: Maximum temperature (K) for the property calculations.
    :type --temp_max: float, optional (default: 1000 K)

    :param --temp_step: Step size for temperature (K).
    :type --temp_step: float, optional (default: 10 K)

    :param --export_dir: Directory to export the properties.
    :type --export_dir: str, optional (default: current working directory)

    :param --export-mix: Export mixture properties instead of component properties.
    :type --export-mix: bool, optional (default: False)

    :raises FileNotFoundError: If required files for the specified fuel are not found.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Export mixture fuel properties for Converge simulations."
    )

    # Mandatory argument for fuel name
    parser.add_argument(
        "-f",
        "--fuel_name",
        required=True,
        metavar="NAME",
        help="Name of the fuel (mandatory).",
    )

    # Optional argument for fuel data directory
    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=FUELDATA_DIR,
        metavar="PATH",
        help="Directory where fuel data files are located (optional, default: FuelLib/fuelData).",
    )

    # Optional argument for minimum temperature
    parser.add_argument(
        "-t",
        "--temp_min",
        type=float,
        default=0,
        metavar="K",
        help="Minimum temperature for property calculations (optional, default: 0).",
    )

    # Optional argument for maximum temperature
    parser.add_argument(
        "-T",
        "--temp_max",
        type=float,
        default=1000,
        metavar="K",
        help="Maximum temperature for property calculations (optional, default: 1000).",
    )

    # Optional argument for temperature step size
    parser.add_argument(
        "-s",
        "--temp_step",
        type=int,
        default=10,
        metavar="K",
        help="Step size for temperature (optional, default: 10).",
    )

    # Optional argument for export directory
    parser.add_argument(
        "-o",
        "--export_dir",
        default=os.getcwd(),
        metavar="PATH",
        help="Directory to export the properties (optional, default: current working directory).",
    )

    # Optional argument for exporting mixture properties
    parser.add_argument(
        "-m",
        "--export-mix",
        "--export_mix",
        dest="export_mix",
        action="store_true",
        help="Export mixture properties instead of component properties.",
    )

    # Parse arguments
    args = parser.parse_args()
    fuel_name = args.fuel_name
    fuel_data_dir = args.fuel_data_dir
    temp_min = args.temp_min
    temp_max = args.temp_max
    temp_step = args.temp_step
    export_dir = args.export_dir
    export_mix = args.export_mix

    # Print the parsed arguments
    print("Preparing to export mixture properties:")
    print(f"    Fuel name: {fuel_name}")
    if export_mix:
        print("    Exporting mixture properties: True")
    print("    Units: mks")
    print(f"    Minimum temperature: {temp_min} K")
    print(f"    Maximum temperature: {temp_max} K")
    print(f"    Temperature step size: {temp_step} K")
    print(f"    Export directory: {export_dir}")
    print(f"    Fuel data directory: {fuel_data_dir}")

    # Get decomposition name from metadata (required)
    # Note: decomp_name not currently used, but kept for API consistency
    _ = fl.get_metadata_decomp_name(fuel_name, fuel_data_dir)

    # Create the fuel object
    fuel = fl.Fuel(fuel_name, fuelDataDir=fuel_data_dir)

    # Export properties for Converge
    export_converge(
        fuel,
        path=export_dir,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_step=temp_step,
        export_mix=export_mix,
    )

    print("\nExport completed successfully!")


if __name__ == "__main__":
    main()
