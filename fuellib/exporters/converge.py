import os
from pathlib import Path

import numpy as np
import pandas as pd

import fuellib as fl

from ..utils import Units

# Default data directory - use fuellib's embedded data
FUELDATA_DIR = Path(__file__).parent.parent / "data"

"""
Script that exports mixture properties over large temperature range for use in
Converge simulations.

This script is designed to be run from the command line (``fl export converge``)
and will create
a file named "mixturePropsGCM_<fuel_name>.csv" in the specified directory.
The file contains mixture properties for the fuel, formatted for Converge.

Usage:
    fl export converge -f <fuel_name>

For detailed options, run:
    fl export converge -h
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
        self, temp_vals, tc, mu, surface_tension, hv, pv, rho, cp, thermal_conductivity
    ):
        """Create an MKS data dictionary for CSV output.

        :param temp_vals: Temperature array.
        :type temp_vals: np.ndarray
        :param tc: Critical temperature.
        :type tc: float
        :param mu: Viscosity array.
        :type mu: np.ndarray
        :param surface_tension: Surface tension array.
        :type surface_tension: np.ndarray
        :param hv: Heat of vaporization array.
        :type hv: np.ndarray
        :param pv: Vapor pressure array.
        :type pv: np.ndarray
        :param rho: Density array.
        :type rho: np.ndarray
        :param cp: Specific heat array.
        :type cp: np.ndarray
        :param thermal_conductivity: Thermal conductivity array.
        :type thermal_conductivity: np.ndarray
        :return: Dictionary with converted properties and labels.
        :rtype: dict
        """
        return {
            self.labels["temperature"]: temp_vals.to("K").magnitude,
            self.labels["critical_temp"]: tc.to("K").magnitude
            + np.zeros_like(temp_vals.magnitude),
            self.labels["viscosity"]: mu.to("Pa*s").magnitude,
            self.labels["surface_tension"]: surface_tension.to("N/m").magnitude,
            self.labels["heat_vaporization"]: hv.to("J/kg").magnitude,
            self.labels["vapor_pressure"]: pv.to("Pa").magnitude,
            self.labels["density"]: rho.to("kg/m^3").magnitude,
            self.labels["specific_heat"]: cp.to("J/(kg*K)").magnitude,
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
    if not hasattr(fuel, "comps") or not hasattr(fuel, "y0"):
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
        components = fuel.comps.names

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

    def validate_temperature_range(temp_vals, t_freeze, tc, is_mixture=True):
        """Validate and adjust temperature range based on freezing and critical temperatures.

        :param temp_vals: Array of temperature values.
        :type temp_vals: np.ndarray
        :param t_freeze: Freezing temperature.
        :type t_freeze: float
        :param tc: Critical temperature.
        :type tc: float
        :param is_mixture: Whether this is for mixture properties.
        :type is_mixture: bool
        :return: Tuple of (temp_min_allowed, temp_max_allowed, adjusted_temp_vals).
        :rtype: tuple
        """
        temp_min_allowed = nearest_temp(t_freeze)
        temp_max_allowed = tc

        # Handle minimum temperature warnings
        if np.any(temp_vals < temp_min_allowed):
            temp_min_allowed = nearest_ceil(temp_vals, temp_min_allowed)
            compound_type = "mixture" if is_mixture else "compound"
            print("!" * 88)
            print(
                "   Warning: Some compounds have freezing temperatures above the estimated"
            )
            print(
                f"   freezing temperature of the {compound_type} ({t_freeze:.2f} K). All properties calculated"
            )
            print(
                f"   below {temp_min_allowed} K will be set using a temperature of {temp_min_allowed} K."
            )
            print("!" * 88)

        # Handle maximum temperature warnings for mixtures
        if is_mixture:
            temp_max_allowed = min(fuel.comps.tc)
            if np.any(temp_vals > temp_max_allowed):
                temp_max_allowed = nearest_floor(temp_vals, temp_max_allowed)
                print("!" * 88)
                print(
                    "   Warning: Some compounds have critical temperatures below the estimated"
                )
                print(
                    f"   critical temperature of the mixture ({tc:.2f} K). All properties will be"
                )
                print(f"   calculated up to {temp_max_allowed} K.")
                print("!" * 88)

        # Filter temperature array to allowed range
        adjusted_temp_vals = temp_vals[
            (temp_vals >= temp_min_allowed) & (temp_vals <= temp_max_allowed)
        ]

        return temp_min_allowed, temp_max_allowed, adjusted_temp_vals

    def calculate_mixture_properties(temp_vals, fuel):
        """Calculate mixture properties for a range of temperatures.

        :param temp_vals: Array of temperature values.
        :type temp_vals: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.Fuel
        :return: Tuple of property arrays (mu, surface_tension, hv, pv, rho, cp, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = Units.Quantity(np.zeros_like(temp_vals.magnitude), "Pa*s")
        surface_tension = Units.Quantity(np.zeros_like(temp_vals.magnitude), "N/m")
        hv = Units.Quantity(np.zeros_like(temp_vals.magnitude), "J/kg")
        pv = Units.Quantity(np.zeros_like(temp_vals.magnitude), "Pa")
        rho = Units.Quantity(np.zeros_like(temp_vals.magnitude), "kg/m^3")
        cp = Units.Quantity(np.zeros_like(temp_vals.magnitude), "J/(kg*K)")
        thermal_conductivity = Units.Quantity(
            np.zeros_like(temp_vals.magnitude), "W/(m*K)"
        )

        for k, temp in enumerate(temp_vals):
            yi = fuel.y0
            xi = fuel.comps.y2x(yi)

            # Standard mixing rules for properties
            rho[k] = fuel.mix.rho(yi, temp)  # kg/m^3
            mu[k] = fuel.mix.mu(yi, temp)  # Pa*s
            pv[k] = fuel.mix.pvap(yi, temp)  # Pa
            surface_tension[k] = fuel.mix.sigma(yi, temp)  # N/m
            thermal_conductivity[k] = fuel.mix.k(yi, temp)

            # Generic mixing rules for latent heat and specific heat
            hv[k] = fl.correlate.mixture.arithmetic(xi, fuel.comps.hv(temp))  # J/kg
            cp[k] = fl.correlate.mixture.arithmetic(
                xi, fuel.comps.cp(temp, basis="mass")
            )  # J/kg/K

        return mu, surface_tension, hv, pv, rho, cp, thermal_conductivity

    def calculate_component_properties(temp_vals, fuel, comp_idx):
        """Calculate individual component properties for a range of temperatures.

        :param temp_vals: Array of temperature values.
        :type temp_vals: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.Fuel
        :param comp_idx: Index of the component.
        :type comp_idx: int
        :return: Tuple of property arrays (mu, surface_tension, hv, pv, rho, cp, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = Units.Quantity(np.zeros_like(temp_vals.magnitude), "Pa*s")
        surface_tension = Units.Quantity(np.zeros_like(temp_vals.magnitude), "N/m")
        hv = Units.Quantity(np.zeros_like(temp_vals.magnitude), "J/kg")
        pv = Units.Quantity(np.zeros_like(temp_vals.magnitude), "Pa")
        rho = Units.Quantity(np.zeros_like(temp_vals.magnitude), "kg/m^3")
        cp = Units.Quantity(np.zeros_like(temp_vals.magnitude), "J/(kg*K)")
        thermal_conductivity = Units.Quantity(
            np.zeros_like(temp_vals.magnitude), "W/(m*K)"
        )

        for k, temp in enumerate(temp_vals):
            rho[k] = fuel.comps.rho(temp, comp_idx=comp_idx)  # kg/m^3
            mu[k] = fuel.comps.mu(temp, comp_idx=comp_idx)  # Pa*s
            pv[k] = fuel.comps.pvap(temp, comp_idx=comp_idx)  # Pa
            surface_tension[k] = fuel.comps.sigma(temp, comp_idx=comp_idx)  # N/m
            thermal_conductivity[k] = fuel.comps.k(temp, comp_idx=comp_idx)
            hv[k] = fuel.comps.hv(temp, comp_idx=comp_idx)  # J/kg
            cp[k] = fuel.comps.cp(temp, comp_idx=comp_idx, basis="mass")  # J/kg/K

        return mu, surface_tension, hv, pv, rho, cp, thermal_conductivity

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
        num_temp_vals = int(((temp_max - temp_min) / temp_step).magnitude) + 1
        temp_vals = Units.Quantity(
            np.linspace(temp_min.magnitude, temp_max.magnitude, num_temp_vals), "K"
        )

        # Estimate freezing point and critical temp of mixture
        t_freeze = fl.correlate.mixture.arithmetic(
            fuel.comps.y2x(fuel.y0), fuel.comps.tm
        )
        tc = fl.correlate.mixture.arithmetic(fuel.comps.y2x(fuel.y0), fuel.comps.tc)

        print(f"\nEstimated mixture freezing temp: {t_freeze:.2f} K")
        print(f"Min freezing temp min(Tm_i): {min(fuel.comps.tm):.2f} K")
        print(f"Max freezing temp max(Tm_i): {max(fuel.comps.tm):.2f} K")
        print(f"Estimated mixture critical temp: {tc:.2f} K")
        print(f"Min critical temp min(Tc_i): {min(fuel.comps.tc):.2f} K")
        print(f"Max critical temp max(Tc_i): {max(fuel.comps.tc):.2f} K")

        # Validate and adjust temperature range
        temp_min_allowed, temp_max_allowed, temp_vals = validate_temperature_range(
            temp_vals, t_freeze, tc, is_mixture=True
        )

    for comp_idx, compound in enumerate(components):
        if not export_mix:
            # Get component-specific temperature limits
            t_freeze = fuel.comps.tm[comp_idx]
            tc = fuel.comps.tc[comp_idx]
            temp_min_allowed = nearest_temp(t_freeze)

            # Create temperature array up to critical temperature
            max_temp_vals = Units.Quantity(
                np.array([
                    (nearest_temp(tc) - temp_step).magnitude,
                    nearest_temp(tc).magnitude,
                    (nearest_temp(tc) + temp_step).magnitude,
                ]),
                "K",
            )
            temp_nearest_floor = nearest_floor(max_temp_vals, tc)
            num_temp_vals = (
                int(((temp_nearest_floor - temp_min_allowed) / temp_step).magnitude) + 1
            )
            temp_vals = Units.Quantity(
                np.linspace(
                    temp_min_allowed.magnitude,
                    temp_nearest_floor.magnitude,
                    num_temp_vals,
                ),
                "K",
            )
            temp_vals = Units.Quantity(
                np.append(temp_vals.magnitude, tc.to("K").magnitude), "K"
            )
            temp_max_allowed = tc
        # Calculate GCM properties for a range of temperatures
        comp_text = "" if export_mix else f"for {compound}"
        print(
            f"\nCalculating properties {comp_text} over {len(temp_vals)} temperatures from {temp_min_allowed} K to {temp_max_allowed} K..."
        )

        if export_mix:
            (
                mu,
                surface_tension,
                hv,
                pv,
                rho,
                cp,
                thermal_conductivity,
            ) = calculate_mixture_properties(temp_vals, fuel)
        else:
            (
                mu,
                surface_tension,
                hv,
                pv,
                rho,
                cp,
                thermal_conductivity,
            ) = calculate_component_properties(temp_vals, fuel, comp_idx)

        # Create data dictionary with converted units
        data = converter.create_data_dict(
            temp_vals, tc, mu, surface_tension, hv, pv, rho, cp, thermal_conductivity
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
            "Index": range(len(fuel.comps.names)),
            "Component": fuel.comps.names,
            "Mass Fraction": fuel.y0,
            "Mole Fraction": fuel.comps.y2x(fuel.y0),
            converter.labels["molecular_weight"]: fuel.comps.mw.to("kg/mol").magnitude,
        }
        export_properties_to_csv(composition_file, composition_data)
