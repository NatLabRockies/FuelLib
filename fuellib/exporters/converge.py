import os
import numpy as np
import pandas as pd
import argparse
import fuellib as fl

# Default data directory - use fuellib's embedded data
FUELDATA_DIR = fl.get_fueldata_dir()

"""
Script that exports mixture properties over large temperature range for use in
Converge simulations.

This script is designed to be run from the command line and will create
a file named "mixturePropsGCM_<fuel_name>.csv" in the specified directory.
The file contains mixture properties for the fuel, formatted for Converge.

Usage:
    fl-export-converge --fuel_name <fuel_name>

Options:
        --units <units>
        --temp_min <temp_min> (K) 
        --temp_max <temp_max> (K)
        --temp_step <temp_step> (K)
        --export_dir <export_dir>
        --export_mix <export_mix>
"""


class UnitConverter:
    """Unit conversion factors and labels for different unit systems."""

    def __init__(self, units: str):
        """
        Initialize converter for specified unit system.

        :param units: Unit system ('cgs' or 'mks').
        :type units: str
        """
        self.units = units.lower()
        self._set_conversion_factors()
        self._set_labels()

    def _set_conversion_factors(self):
        """Set conversion factors based on unit system."""
        if self.units == "cgs":
            # Convert from MKS to CGS
            self.mw = 1e3  # kg/mol to g/mol
            self.mu = 1e2  # Pa*s to Poise
            self.surface_tension = 1e7  # N/m to dyne/cm
            self.Lv = 1e4  # J/kg to erg/g
            self.P = 1e1  # Pa to dyne/cm^2
            self.rho = 1e3  # kg/m^3 to g/cm^3
            self.Cl = 1e4  # J/kg/K to erg/g/K
            self.thermal_conductivity = 1e5  # W/m/K to erg/cm/s/K
        else:
            # MKS units (no conversion)
            self.mw = 1
            self.mu = 1
            self.surface_tension = 1
            self.Lv = 1
            self.P = 1
            self.rho = 1
            self.Cl = 1
            self.thermal_conductivity = 1

    def _set_labels(self):
        """Set unit labels for DataFrame columns."""
        if self.units == "cgs":
            self.labels = {
                "temperature": "Temperature (K)",
                "critical_temp": "Critical Temperature (K)",
                "viscosity": "Viscosity (Poise)",
                "surface_tension": "Surface Tension (dyne/cm)",
                "heat_vaporization": "Heat of Vaporization (erg/g)",
                "vapor_pressure": "Vapor Pressure (dyne/cm^2)",
                "density": "Density (g/cm^3)",
                "specific_heat": "Specific Heat (erg/g/K)",
                "thermal_conductivity": "Thermal Conductivity (erg/cm/s/K)",
                "molecular_weight": "Molecular Weight (g/mol)",
            }
        else:
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
        """
        Create a data dictionary with converted units and appropriate labels.

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
            self.labels["temperature"]: T,
            self.labels["critical_temp"]: T_crit + np.zeros_like(T),
            self.labels["viscosity"]: mu * self.mu,
            self.labels["surface_tension"]: surface_tension * self.surface_tension,
            self.labels["heat_vaporization"]: Lv * self.Lv,
            self.labels["vapor_pressure"]: pv * self.P,
            self.labels["density"]: rho * self.rho,
            self.labels["specific_heat"]: Cl * self.Cl,
            self.labels["thermal_conductivity"]: thermal_conductivity
            * self.thermal_conductivity,
        }


def export_converge(
    fuel,
    path=None,
    units="mks",
    temp_min=0,
    temp_max=1000,
    temp_step=10,
    export_mix=False,
):
    """
    Export mixture fuel properties to csv files for Converge simulations.

    :param fuel: Fuel object containing properties to export.
    :type fuel: fl.fuel

    :param path: Directory to save the input file.
    :type path: str, optional (default: current directory/exportData)

    :param units: Units for the properties ("mks" for SI, "cgs" for CGS).
    :type units: str, optional (default: "mks")

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
        path = os.path.join(os.getcwd(), "exportData")

    # Input validation
    if not hasattr(fuel, "compounds") or not hasattr(fuel, "Y_0"):
        raise TypeError("fuel parameter must be a valid FuelLib fuel object")

    if units.lower() not in ["mks", "cgs"]:
        raise ValueError(f"Units must be 'mks' or 'cgs', got '{units}'")

    if temp_min < 0:
        raise ValueError(f"temp_min must be non-negative, got {temp_min}")

    if temp_max <= temp_min:
        raise ValueError(
            f"temp_max ({temp_max}) must be greater than temp_min ({temp_min})"
        )

    if temp_step <= 0:
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
    converter = UnitConverter(units)

    def nearest_temp(x, base=temp_step):
        """
        Round to nearest multiple of temp_step.

        :param x: Temperature value to round.
        :type x: float
        :param base: Base value for rounding (temp_step).
        :type base: float
        :return: Rounded temperature.
        :rtype: float
        """
        return base * round(x / base)

    def nearest_floor(array, value):
        """
        Find the largest value in the array that is less than or equal to the given value.

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
        """
        Find the smallest value in the array that is greater than or equal to the given value.

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
        """
        Validate and adjust temperature range based on freezing and critical temperatures.

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
                f"   Warning: Some compounds have freezing temperatures above the estimated"
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
                    f"   Warning: Some compounds have critical temperatures below the estimated"
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
        """
        Calculate mixture properties for a range of temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.fuel
        :return: Tuple of property arrays (mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = np.zeros_like(T_array)
        surface_tension = np.zeros_like(T_array)
        Lv = np.zeros_like(T_array)
        pv = np.zeros_like(T_array)
        rho = np.zeros_like(T_array)
        Cl = np.zeros_like(T_array)
        thermal_conductivity = np.zeros_like(T_array)

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
            Lv[k] = fl.mixing_rule(fuel.latent_heat_vaporization(Temp), X_li)  # J/kg
            Cl[k] = fl.mixing_rule(fuel.Cl(Temp), X_li)  # J/kg/K

        return mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity

    def calculate_component_properties(T_array, fuel, comp_idx):
        """
        Calculate individual component properties for a range of temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.fuel
        :param comp_idx: Index of the component.
        :type comp_idx: int
        :return: Tuple of property arrays (mu, surface_tension, Lv, pv, rho, Cl, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = np.zeros_like(T_array)
        surface_tension = np.zeros_like(T_array)
        Lv = np.zeros_like(T_array)
        pv = np.zeros_like(T_array)
        rho = np.zeros_like(T_array)
        Cl = np.zeros_like(T_array)
        thermal_conductivity = np.zeros_like(T_array)

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
        """
        Export properties data to CSV file.

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
        nT = int((temp_max - temp_min) / temp_step) + 1
        T = np.linspace(temp_min, temp_max, nT)

        # Estimate freezing point and critical temp of mixture
        T_freeze = fl.mixing_rule(fuel.Tm, fuel.Y2X(fuel.Y_0))
        T_crit = fl.mixing_rule(fuel.Tc, fuel.Y2X(fuel.Y_0))

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
            maxtemps = np.array(
                [
                    nearest_temp(T_crit) - temp_step,
                    nearest_temp(T_crit),
                    nearest_temp(T_crit) + temp_step,
                ]
            )
            T_nearest_floor = nearest_floor(maxtemps, T_crit)
            nT = int((T_nearest_floor - T_min_allowed) / temp_step) + 1
            T = np.linspace(T_min_allowed, T_nearest_floor, nT)
            T = np.append(T, T_crit)
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
            converter.labels["molecular_weight"]: fuel.MW * converter.mw,
        }
        export_properties_to_csv(composition_file, composition_data)


def main():
    """
    Main function to execute the export process.

    :param --fuel_name: Name of the fuel (mandatory).
    :type --fuel_name: str

    :param --fuel_data_dir: Directory where fuel data files are located.
    :type --fuel_data_dir: str, optional (default: FuelLib/fuelData)

    :param --units: Units for critical properties. Options are mks or cgs.
    :type --units: str, optional (default: mks)

    :param --temp_min: Minimum temperature (K) for the property calculations.
    :type --temp_min: float, optional (default: 0 K)

    :param --temp_max: Maximum temperature (K) for the property calculations.
    :type --temp_max: float, optional (default: 1000 K)

    :param --temp_step: Step size for temperature (K).
    :type --temp_step: float, optional (default: 10 K)

    :param --export_dir: Directory to export the properties.
    :type --export_dir: str, optional (default: current directory/exportData)

    :param --export_mix: Whether to export individual component or mixture properties.
    :type --export_mix: bool, optional (default: False)

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
        "-D",
        "--fuel_data_dir",
        default=FUELDATA_DIR,
        metavar="PATH",
        help="Directory where fuel data files are located (optional, default: FuelLib/fuelData).",
    )

    # Optional argument for units
    # Default is 'mks', but can be set to 'cgs'
    parser.add_argument(
        "-u",
        "--units",
        default="mks",
        metavar="{mks,cgs}",
        help="Units for critical properties (optional, default: mks).",
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
        default=os.path.join(os.getcwd(), "exportData"),
        metavar="PATH",
        help="Directory to export the properties (optional, default: ./exportData).",
    )

    # Optional argument for exporting mixture properties
    parser.add_argument(
        "-m",
        "--export_mix",
        type=lambda x: str(x).lower() in ["true", "1"],
        default=False,
        metavar="{true,false}",
        help="Export mixture properties of the fuel (optional, default: false).",
    )

    # Parse arguments
    args = parser.parse_args()
    fuel_name = args.fuel_name
    fuel_data_dir = args.fuel_data_dir
    units = args.units.lower()
    temp_min = args.temp_min
    temp_max = args.temp_max
    temp_step = args.temp_step
    export_dir = args.export_dir
    export_mix = args.export_mix

    # Print the parsed arguments
    print(f"Preparing to export mixture properties:")
    print(f"    Fuel name: {fuel_name}")
    if export_mix:
        print(f"    Exporting mixture properties: True")
    print(f"    Units: {units}")
    print(f"    Minimum temperature: {temp_min} K")
    print(f"    Maximum temperature: {temp_max} K")
    print(f"    Temperature step size: {temp_step} K")
    print(f"    Export directory: {export_dir}")
    print(f"    Fuel data directory: {fuel_data_dir}")

    # Check if necessary files exist in the fuelData directory
    print("\nChecking for required files...")
    gcxgc_file = os.path.join(fuel_data_dir, f"gcData/{fuel_name}_init.csv")
    decomp_file = os.path.join(fuel_data_dir, f"groupDecompositionData/{fuel_name}.csv")
    if not os.path.exists(gcxgc_file):
        err = f"GCXGC file for {fuel_name} not found in {fuel_data_dir}/gcData. gcxgc_file = {gcxgc_file}"
        raise FileNotFoundError(err)
    if not os.path.exists(decomp_file):
        err = f"Decomposition file for {fuel_name} not found in {fuel_data_dir}/groupDecompositionData. decomp_file = {decomp_file}"
        raise FileNotFoundError(err)
    print("All required files found.")

    # Create the fuel object
    fuel = fl.fuel(fuel_name, fuelDataDir=fuel_data_dir)

    # Export properties for Converge
    export_converge(
        fuel,
        path=export_dir,
        units=units,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_step=temp_step,
        export_mix=export_mix,
    )

    print("\nExport completed successfully!")


if __name__ == "__main__":
    main()
