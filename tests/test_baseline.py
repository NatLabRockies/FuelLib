import os
import sys
import numpy as np
import pandas as pd
import test_functions as fxns

"""
Script for calculating baseline FuelLib mixture property predictions for CI testing
Use this to update threshold values in CI test as model improves
"""


# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)
from paths import *
import FuelLib as fl

# Directories for tests and baseline predictions
test_dir = os.path.dirname(__file__)
baseline_dir = os.path.join(test_dir, "baselinePredictions")

# Fuel for GCM and data for validation (see fuelData/propertiesData for fuels)
# Options: 'decane','dodecane', 'heptane', 'posf10264', 'posf10325', 'posf10289'
fuel_names = ["heptane", "decane", "dodecane", "posf10264", "posf10325", "posf10289"]

# Properties to test
prop_names = [
    "Density",
    "Viscosity",
    "VaporPressure",
    "SurfaceTension",
    "ThermalConductivity",
]

# Property units
prop_units = {
    "Temperature": "C",
    "Density": "g/cm^3",
    "Viscosity": "mm^2/s",
    "VaporPressure": "kPa",
    "SurfaceTension": "N/m",
    "ThermalConductivity": "W/m/K",
}


def get_unit_for_column(col_name):
    for prop in prop_units:
        if prop in col_name:
            return prop_units[prop]
    return ""


# Loop through each fuel and generate csv of baseline property predictions
for fuel_name in fuel_names:
    export_name = os.path.join(baseline_dir, f"{fuel_name}.csv")
    df_combined = None

    for prop in prop_names:
        T, data, pred = fxns.getPredAndData(fuel_name, prop)

        # Create a dataframe for this property
        df_prop = pd.DataFrame(
            {"Temperature": T, prop: pred, f"Error_{prop}": np.abs(data - pred)}
        )

        if df_combined is None:
            # Initialize combined dataframe
            df_combined = df_prop
        else:
            # Merge on Temperature using outer join to ensure all temperatures are kept
            df_combined = pd.merge(df_combined, df_prop, on="Temperature", how="outer")

    # Sort by Temperature (optional, but nice for clean output)
    df_combined = df_combined.sort_values(by="Temperature").reset_index(drop=True)

    # Generate units list in correct order
    units = [get_unit_for_column(col) for col in df_combined.columns]

    # Create MultiIndex columns (name + unit)
    df_combined.columns = pd.MultiIndex.from_arrays([df_combined.columns, units])

    # Save final table
    df_combined.to_csv(export_name, index=False)
