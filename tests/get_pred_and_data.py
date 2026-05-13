import os
import numpy as np
import pandas as pd

import fuellib as fl
from fuellib._data_locator import get_fueldata_props_dir

FUELDATA_PROPS_DIR = get_fueldata_props_dir()


def get_pred_and_data(fuel_name, prop_name):
    # Get the fuel properties based on the GCM
    fuel = fl.fuel(fuel_name)

    data_file = f"{fuel_name}.csv"
    data = pd.read_csv(os.path.join(FUELDATA_PROPS_DIR, data_file), skiprows=[1])

    # Separate properties and associated temperatures from data
    T_data = data.Temperature[data[prop_name].notna()].to_numpy(dtype=float)
    prop_data = data[prop_name].dropna().to_numpy()

    # Vector for predictions
    pred = np.zeros_like(T_data)

    # Vectors for temperature (convert from C to K)
    T_pred = fl.C2K(T_data)

    for i in range(0, len(T_pred)):
        Y_li = fuel.Y_0

        if prop_name == "Density":
            # Mixture density (returns rho in kg/m^3)
            pred[i] = fuel.mixture_density(Y_li, T_pred[i])
            # Convert density to CGS (g/cm^3)
            pred[i] *= 1.0e-03

        if prop_name == "VaporPressure":
            # Mixture vapor pressure (returns pv in Pa)
            pred[i] = fuel.mixture_vapor_pressure(Y_li, T_pred[i])
            # Convert vapor pressure to kPa
            pred[i] *= 1.0e-03

        if prop_name == "Viscosity":
            pred[i] = fuel.mixture_kinematic_viscosity(Y_li, T_pred[i])
            # Convert viscosity to mm^2/s
            pred[i] *= 1.0e06

        if prop_name == "SurfaceTension":
            pred[i] = fuel.mixture_surface_tension(Y_li, T_pred[i])

        if prop_name == "ThermalConductivity":
            pred[i] = fuel.mixture_thermal_conductivity(Y_li, T_pred[i])

    return T_data, prop_data, pred


# Backward-compatible alias for older call sites.
getPredAndData = get_pred_and_data
