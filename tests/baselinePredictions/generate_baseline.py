"""Generate and fetch baseline predictions for fuel properties.

Usage examples::

    python generate_baseline.py                          # regenerate everything
    python generate_baseline.py -f decane heptane        # all properties for these fuels
    python generate_baseline.py -p density               # density for every fuel
    python generate_baseline.py -f decane -p density cp  # selected fuel/property pairs

Existing baseline rows matching the selected fuel(s) and property(s) are deleted
and replaced by the newly generated values; all other rows are kept.
"""

from collections.abc import Iterable

import argparse

import pandas as pd
import numpy as np
from pathlib import Path
from fuellib import correlate, Units, Fuel

data_dir = Path(__file__).parent
mixture_file = data_dir / "mixture_baseline.csv"

columns = [
    "Fuel",
    "Temp",
    "Temp_Units",
    "Property",
    "Property_Units",
    "Baseline_Value",
    "Baseline_Error",
]
fuel_names = {
    # n-alkanes
    "decane",
    "dodecane",
    "heptane",
    # Iso-alkanes
    "2-methyloctane",
    # Cycloalkanes
    "methylcyclohexane",
    # Aromatics
    "toluene",
    # Multicomponent fuels
    "posf10264",
    "posf10289",
    "posf10325",
}

method_map = {
    "density": correlate.mixture.density,
    "viscosity": correlate.mixture.kinematic_viscosity_nannoolal,
    "vaporpressure": correlate.mixture.saturated_vapor_pressure,
    "dynamicviscosity": correlate.mixture.dynamic_viscosity_dutt,
    "surfacetension": correlate.mixture.surface_tension,
    "thermalconductivity": correlate.mixture.thermal_conductivity_latini,
    "cp": correlate.components.molar_specific_heat_capacity,
    "freezepoint": correlate.mixture.freeze_point_boehm,
    "flashpoint": correlate.mixture.flash_point_alibakhshi,
    "heatofcombustion": correlate.mixture.heat_of_combustion,
    "derivedcetanenumber": correlate.mixture.derived_cetane_number,
    "smokepoint": correlate.mixture.smoke_point,
}


def _normalize(prop_name: str) -> str:
    """Normalize a property name to match the keys of ``method_map``."""
    return prop_name.replace(" ", "").strip().lower()


def _format_baseline_float(value: float) -> str:
    """Return a round-trip-safe decimal representation of a float."""
    return repr(float(value))


def update_baseline(
    fuels: Iterable[str] | None = None,
    properties: Iterable[str] | None = None,
    useRefProperties: bool = False,
) -> None:
    """Update the baseline predictions in the mixture data.

    Args:
        fuels: Fuels to regenerate. If None, all fuels in ``fuel_names`` are used.
        properties: Properties to regenerate. If None, all properties in ``method_map``
            are used.
        useRefProperties: Whether to use reference properties when generating the
            baseline. Defaults to False.
    """
    selected_fuels = sorted(set(fuels)) if fuels else sorted(fuel_names)
    selected_props = {_normalize(p) for p in properties} if properties else None

    unknown_fuels = set(selected_fuels) - fuel_names
    if unknown_fuels:
        msg = (
            f"Unknown fuel(s) {sorted(unknown_fuels)}. "
            f"Valid options: {sorted(fuel_names)}."
        )
        raise ValueError(msg)

    if selected_props is not None:
        unknown = selected_props - method_map.keys()
        if unknown:
            msg = (
                f"Unknown property(s) {sorted(unknown)}. "
                f"Valid options: {sorted(method_map)}."
            )
            raise ValueError(msg)

    # Load existing baselines and drop the rows that will be regenerated.
    if mixture_file.exists():
        existing = pd.read_csv(mixture_file, float_precision="round_trip")
        fuel_mask = (
            existing["Fuel"].isin(selected_fuels)
            if fuels
            else pd.Series(True, index=existing.index)
        )
        prop_mask = (
            existing["Property"].map(_normalize).isin(selected_props)
            if selected_props is not None
            else pd.Series(True, index=existing.index)
        )
        existing = existing[~(fuel_mask & prop_mask)]
    else:
        existing = pd.DataFrame(columns=columns)

    new_rows = []
    for fuel_name in selected_fuels:
        fuel = Fuel(fuel_name, useRefProperties=useRefProperties)
        if fuel.propData is None:
            msg = f"No property data found for fuel '{fuel_name}'."
            raise ValueError(msg)

        props_data = fuel.propData
        for prop_name in props_data["Property"].unique():
            prop_data = props_data[props_data["Property"] == prop_name]
            if (
                selected_props is not None
                and _normalize(prop_name) not in selected_props
            ):
                continue
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

                new_rows.append([
                    fuel_name,
                    row.Temp,
                    row.Temp_Units,
                    prop_name,
                    row.Property_Units,
                    pred_mag,
                    pred_mag - prop_val.magnitude,
                ])

    new_data = pd.DataFrame(new_rows, columns=columns)
    frames = [df for df in (existing, new_data) if not df.empty]
    mixture_data = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns)
    )
    # Do not apply the format to input temperatures: their shortest decimal form
    # is more readable, and they are not calculated regression values.
    output_data = mixture_data.copy()
    for column in ("Baseline_Value", "Baseline_Error"):
        output_data[column] = output_data[column].map(_format_baseline_float)
    output_data.to_csv(mixture_file, index=False)
    print(
        f"Wrote {len(new_data)} regenerated row(s) to {mixture_file} "
        f"({len(mixture_data)} total)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-f",
        "--fuel",
        nargs="+",
        default=None,
        choices=sorted(fuel_names),
        metavar="FUEL",
        help=f"Fuel(s) to regenerate (default: all). Options: {', '.join(sorted(fuel_names))}.",
    )
    parser.add_argument(
        "-p",
        "--property",
        nargs="+",
        default=None,
        type=_normalize,
        choices=sorted(method_map),
        metavar="PROPERTY",
        help=f"Property(s) to regenerate (default: all). Options: {', '.join(sorted(method_map))}.",
    )
    parser.add_argument(
        "-r",
        "--refProps",
        action="store_true",
        help="Whether to use reference properties when generating the baseline (default: False).",
    )
    args = parser.parse_args()
    update_baseline(
        fuels=args.fuel,
        properties=args.property,
        useRefProperties=args.refProps,
    )
