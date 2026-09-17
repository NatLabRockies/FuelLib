import os

import numpy as np
import pandas as pd

import fuellib as fl
from fuellib._data_locator import get_fueldata_props_dir
from fuellib.fuel import FloatArray, PintArray
from fuellib.units import PintUnits

FUELDATA_PROPS_DIR = get_fueldata_props_dir()


def get_pred_and_data(
    fuel_name: str, prop_name: str
) -> tuple[PintArray, PintArray, PintArray]:
    # Get the fuel properties based on the GCM
    fuel = fl.Fuel(fuel_name)

    data_file = f"{fuel_name}.csv"
    data = pd.read_csv(os.path.join(FUELDATA_PROPS_DIR, data_file))

    t_vals: FloatArray = data.Temperature.iloc[1:].to_numpy(dtype=float)
    t_units: str = data.Temperature.iloc[0]
    data_temps: PintArray = PintUnits.Quantity(t_vals, t_units).to("K")

    data_vals: FloatArray = data[prop_name].iloc[1:].to_numpy(dtype=float)
    data_units: str = data[prop_name].iloc[0]
    data_props: PintArray = PintUnits.Quantity(data_vals, data_units)

    valid_idxs = ~np.isnan(data_props)
    data_temps = data_temps[valid_idxs]
    data_props = data_props[valid_idxs]

    pred_props = PintUnits.Quantity(np.zeros_like(data_props), data_units)
    for i, t in enumerate(data_temps):
        if prop_name == "Density":
            pred_props[i] = fuel.mixture_density(fuel.Y_0, t)
        if prop_name == "VaporPressure":
            pred_props[i] = fuel.mixture_vapor_pressure(fuel.Y_0, t)
        if prop_name == "Viscosity":
            pred_props[i] = fuel.mixture_kinematic_viscosity(fuel.Y_0, t)
        if prop_name == "SurfaceTension":
            pred_props[i] = fuel.mixture_surface_tension(fuel.Y_0, t)
        if prop_name == "ThermalConductivity":
            pred_props[i] = fuel.mixture_thermal_conductivity(fuel.Y_0, t)

    return data_temps, data_props, pred_props


# Backward-compatible alias for older call sites.
getPredAndData = get_pred_and_data
