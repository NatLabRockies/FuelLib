import os
from typing import Literal

import astropy.units as u
import numpy as np
import pandas as pd

import fuellib as fl
from fuellib._data_locator import get_fueldata_props_dir
from fuellib.units import convert_temperature

FUELDATA_PROPS_DIR = get_fueldata_props_dir()


def get_pred_and_data(
    fuel_name: str,
    prop_name: Literal[
        "Density", "VaporPressure", "Viscosity", "SurfaceTension", "ThermalConductivity"
    ],
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    # Get the fuel properties based on the GCM
    fuel = fl.fuel(fuel_name)

    data_file = f"{fuel_name}.csv"
    data = pd.read_csv(os.path.join(FUELDATA_PROPS_DIR, data_file))

    # Extract unit from column
    temp_unit = str(data["Temperature"].iloc[0])
    prop_unit = str(data[prop_name].iloc[0])

    # Separate properties and associated temperatures from data
    temp_data = data.Temperature.iloc[1:].to_numpy(dtype=float)
    prop_data = data[prop_name].iloc[1:].to_numpy(dtype=float)

    # Filter out invalid (NaN) entries and add units to values
    valid_idxs = ~np.isnan(prop_data)
    temp_data = convert_temperature(u.Quantity(temp_data[valid_idxs], temp_unit), "K")
    prop_data = u.Quantity(prop_data[valid_idxs], prop_unit)

    # Compile predictions for the given property
    pred = u.Quantity(np.zeros(len(temp_data)), prop_unit)
    for i, T in enumerate(temp_data):
        Y_li = fuel.Y_0

        if prop_name == "Density":
            pred[i] = fuel.mixture_density(Y_li, T, unit=prop_unit)

        elif prop_name == "VaporPressure":
            # Mixture vapor pressure (returns pv in Pa)
            pred[i] = fuel.mixture_vapor_pressure(Y_li, T, unit=prop_unit)

        elif prop_name == "Viscosity":
            pred[i] = fuel.mixture_kinematic_viscosity(Y_li, T, unit=prop_unit)

        elif prop_name == "SurfaceTension":
            pred[i] = fuel.mixture_surface_tension(Y_li, T, unit=prop_unit)

        elif prop_name == "ThermalConductivity":
            pred[i] = fuel.mixture_thermal_conductivity(Y_li, T, unit=prop_unit)

        else:
            raise ValueError(f"Unsupported property name: {prop_name}")

    return temp_data, prop_data, pred


# Backward-compatible alias for older call sites.
getPredAndData = get_pred_and_data
