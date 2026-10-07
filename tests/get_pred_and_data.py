import numpy as np

import fuellib as fl
from fuellib.utils import Units


def get_pred_and_data(fuel_name, prop_name):
    # Get the fuel properties based on the GCM
    fuel = fl.Fuel(fuel_name)

    data = fuel.properties_data
    assert data is not None, f"No propertiesData for {fuel_name}"
    data = data[(data["Property"] == prop_name) & data["Temp"].notna()]

    data_temps = Units.Quantity(
        data["Temp"].to_numpy(dtype=float), data["Temp_Units"].iloc[0]
    ).to("K")
    data_props = Units.Quantity(
        data["Property_Value"].to_numpy(dtype=float), data["Property_Units"].iloc[0]
    )
    data_units = str(data_props.units)

    valid_idxs = ~np.isnan(data_props)
    data_temps = data_temps[valid_idxs]
    data_props = data_props[valid_idxs]
    pred_props = Units.Quantity(np.zeros_like(data_props.magnitude), data_units)

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
