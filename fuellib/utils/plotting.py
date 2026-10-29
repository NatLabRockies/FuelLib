"""Plotting utilities."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .types import DistillationResult, Units

if TYPE_CHECKING:
    from ..fuel import Fuel

# Percent recovered for each `DistillationResult` field.
DISTILLATION_PERCENTS = {
    "T0": 0.0,
    "T5": 5.0,
    "T10": 10.0,
    "T30": 30.0,
    "T50": 50.0,
    "T70": 70.0,
    "T90": 90.0,
    "Tf": 100.0,
}


def plot_distillation_curve(
    result: DistillationResult,
    fuel: "Fuel | None" = None,
    out: str | Path | None = None,
    *,
    ax: Axes | None = None,
    units: str = "celsius",
    label: str = "FuelLib",
    data_label: str = "Data",
) -> Axes:
    """Plot a distillation curve, with experimental data if available.

    Experimental data are read from the ``Distillation`` rows of
    ``fuel.propData``, where ``Temp`` is the measured temperature and
    ``Property_Value`` is the percent recovered.

    Args:
        result: Predicted distillation curve.
        fuel: Fuel whose experimental distillation data should be overlaid.
            Defaults to None (no experimental data).
        out: Path or string specifying where to save the plot.
            Defaults to None (do not save the plot).
        ax: Axes to plot on. Defaults to a new figure and axes.
        units: Temperature units for the plot. Defaults to "celsius".
        label: Legend label for the predicted curve. Defaults to "FuelLib".
        data_label: Legend label for experimental data. Defaults to "Data".

    Returns:
        The axes containing the plot.
    """
    out = Path(out) if out is not None else None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    percents = list(DISTILLATION_PERCENTS.values())
    temps = [getattr(result, f).m_as(units) for f in DISTILLATION_PERCENTS]
    ax.plot(percents, temps, "-o", color="#2980B9", linewidth=3, label=label)

    data = fuel.propData if fuel is not None else None
    if data is not None:
        data = data[
            (data["Property"] == "Distillation")
            & data["Temp"].notna()
            & data["Property_Value"].notna()
        ]
        if len(data) > 0:
            T_data = Units.Quantity(
                data["Temp"].to_numpy(dtype=float), data["Temp_Units"].iloc[0]
            ).m_as(units)
            pct_data = data["Property_Value"].to_numpy(dtype=float)
            ax.scatter(
                pct_data,
                T_data,
                marker="s",
                color="#333333",
                s=60,
                zorder=5,
                label=data_label,
            )

    unit_label = f"{Units.Unit(units):~P}"
    ax.set_xlabel("Volume recovered [%]")
    ax.set_ylabel(f"Temperature [{unit_label}]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if out is not None:
        plt.savefig(out)

    return ax
