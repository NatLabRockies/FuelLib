"""Regression tests for baseline-fixture generation."""

import pandas as pd

from baselinePredictions import generate_baseline
from fuellib import Fuel, Units, correlate


def test_generated_values_round_trip_without_losing_float_precision(
    monkeypatch, tmp_path
) -> None:
    """The CSV must preserve the exact binary64 values used by the test suite."""
    output_file = tmp_path / "mixture_baseline.csv"
    monkeypatch.setattr(generate_baseline, "mixture_file", output_file)

    generate_baseline.update_baseline(
        fuels=["dodecane"],
        properties=["density"],
    )

    fuel = Fuel("dodecane", useRefProperties=False)
    prop_data = fuel.propData
    assert prop_data is not None
    source_row = next(
        prop_data.loc[
            (prop_data["Property"] == "Density") & (prop_data["Temp"] == -9.55)
        ].itertuples()
    )
    expected_value = (
        correlate.mixture
        .density(
            fuel=fuel,
            T=Units.Quantity(source_row.Temp, source_row.Temp_Units),
        )
        .to(source_row.Property_Units)
        .magnitude
    )
    expected_error = expected_value - source_row.Property_Value

    written = pd.read_csv(output_file, float_precision="round_trip")
    generated_row = written.loc[written["Temp"] == -9.55].iloc[0]
    assert generated_row.Baseline_Value == expected_value
    assert generated_row.Baseline_Error == expected_error
    expected_csv_row = (
        "dodecane,-9.55,celsius,Density,g/cm^3,"
        f"{float(expected_value)!r},{float(expected_error)!r}"
    )
    assert expected_csv_row in output_file.read_text()
