#!/usr/bin/env python3
"""Generate updated baseline predictions for Fuel properties."""

import sys

from pathlib import Path

import pandas as pd
from fuellib import Fuel
from fuellib.utils import types

baseline_dir = Path(__file__).parent
if str(baseline_dir.parent) not in sys.path:
    sys.path.insert(0, str(baseline_dir.parent))

from get_pred_and_data import get_pred_and_data


def _prep_quantity(quantity: types.Quantity1D) -> list[str | float]:
    """Convert a Quantity1D to a list of float values with units in 0th index."""
    return [str(quantity.units)] + quantity.magnitude.tolist()


fuel_names = ["decane"]
properties = {
    "Density": "g/cm^3",
    "Viscosity": "mm^2/s",
    "VaporPressure": "kPa",
    "SurfaceTension": "N/m",
    "ThermalConductivity": "W/m/K",
}


def main():
    """Generate updated baseline predictions for Fuel properties."""
    for fuel_name in fuel_names:
        out_file = baseline_dir / f"{fuel_name}_test.csv"

        df_combined = pd.DataFrame()
        for prop_name, prop_unit in properties.items():
            T, data, pred = get_pred_and_data(fuel_name, prop_name)

            df_prop = pd.DataFrame({
                "Temperature": _prep_quantity(T.to("celsius")),
                prop_name: _prep_quantity(data.to(prop_unit)),
                f"Error_{prop_name}": _prep_quantity(
                    data.to(prop_unit) - pred.to(prop_unit)
                ),
            })

            if df_combined.empty:
                df_combined = df_prop
            else:
                df_combined = pd.merge(
                    df_combined, df_prop, on="Temperature", how="outer"
                )

        is_numeric = pd.to_numeric(df_combined["Temperature"], errors="coerce").notna()
        units_row = df_combined[~is_numeric]
        data_rows = df_combined[is_numeric].sort_values(
            by="Temperature", key=lambda s: s.astype(float)
        )
        df_combined = pd.concat([units_row, data_rows], ignore_index=True)

        df_combined.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
