"""Generate and fetch baseline predictions for fuel properties."""

import pandas as pd
import numpy as np
from pathlib import Path
from fuellib import correlate, Units, Fuel

data_dir = Path(__file__).parent
mixture_file = data_dir / "mixture_baseline.csv"
mixture_data = pd.DataFrame(
    columns=[
        "Fuel",
        "Temp",
        "Temp_Units",
        "Property",
        "Property_Units",
        "Baseline_Value",
        "Baseline_Error",
    ]
)

fuel_names = {"decane", "dodecane", "heptane", "posf10264", "posf10325", "posf10289"}
method_map = {
    "density": correlate.mixture.density,
    "viscosity": correlate.mixture.kinematic_viscosity_dutt,
    "vaporpressure": correlate.mixture.saturated_vapor_pressure,
    "dynamicviscosity": correlate.mixture.dynamic_viscosity_dutt,
    "surfacetension": correlate.mixture.surface_tension,
    "thermalconductivity": correlate.mixture.thermal_conductivity_latini,
    "cp": correlate.components.molar_specific_heat_capacity,
    "freezepoint": correlate.mixture.freeze_point_boehm,
    "flashpoint": correlate.mixture.flash_point_alibashki,
    "heatofcombustion": correlate.mixture.heat_of_combustion,
}


def update_baseline() -> None:
    """Update the baseline predictions in the mixture data."""
    for fuel_name in fuel_names:
        fuel = Fuel(fuel_name)
        props_file = Path(fuel.fuelDataPropsDir) / f"{fuel_name}.csv"

        if not props_file.exists():
            raise FileNotFoundError(f"Fuel data file not found: {props_file}")

        props_data = pd.read_csv(props_file)
        for prop_name in props_data["Property"].unique():
            prop_data = props_data[props_data["Property"] == prop_name]
            method = method_map.get(prop_name.replace(" ", "").strip().lower(), None)
            if method is None:
                print(f"No method found for property '{prop_name}'; skipping.")
                continue

            for row in prop_data.itertuples():
                # Temperature-independent properties (e.g., freeze point) have no Temp
                # (pandas parses "NaN" as float nan), so do not pass T to the method.
                kwargs = {}
                if not pd.isna(row.Temp):
                    kwargs["T"] = Units.Quantity(row.Temp, row.Temp_Units)
                prop_val = Units.Quantity(row.Property_Value, row.Property_Units)
                pred_val = method(fuel=fuel, **kwargs).to(row.Property_Units)

                pred_mag = pred_val.magnitude
                if isinstance(pred_mag, np.ndarray):
                    if pred_mag.size != 1:
                        msg = f"Expected a single value for property '{prop_name}' of fuel '{fuel_name}', but got an array of size {pred_mag.size}."
                        raise ValueError(msg)
                    pred_mag = pred_mag.item()

                mixture_data.loc[len(mixture_data)] = [
                    fuel_name,
                    row.Temp,
                    row.Temp_Units,
                    prop_name,
                    row.Property_Units,
                    pred_mag,
                    pred_mag - prop_val.magnitude,
                ]

    mixture_data.to_csv(mixture_file, index=False)


if __name__ == "__main__":
    update_baseline()
