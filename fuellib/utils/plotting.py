"""Plotting utilities for FuelLib.

Functions to plot the composition, mixture properties and simulated D86 distillation
curves of fuels. The ``fl-plt-comp``, ``fl-plt-props`` and ``fl-simulate`` commands
are thin wrappers around these functions. The colormaps and fuel colors are in
:mod:`fuellib.utils.color_maps`.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.ticker import MultipleLocator

import fuellib as fl

from . import types
from .color_maps import FL_BLUE_4, FUEL_COLORS, cycle_color
from .types import Units

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..simulate.distillation.greenfield import D86Result

# ---------------------------------------------------------------------------------
# Fuel styles
# ---------------------------------------------------------------------------------

FUEL_MARKERS = {
    "decane": "o",
    "posf10325": "o",
    "dodecane": "s",
    "posf10289": "s",
    "heptane": "D",
    "posf10264": "D",
    "posf11498": "^",
    "jet-a": "v",
    "hefa": "p",
}
"""Marker styles of fuel data, matched by substring of the lowercase fuel name."""

_MARKER_CYCLE = ("o", "s", "D", "^", "v", "p")
"""Marker styles of successive D86 curves on the same axes."""

_FAMILY_COLORS = {
    "n-alkane": "#063C61",
    "iso-alkane": "#2980B9",
    "cyclo-alkane": "#91BCD8",
    "alkene": "#663399",
    "aromatic": "#7f7f7f",
}
"""Colors of the hydrocarbon families in the composition plot."""

_DEFAULT_PROPERTIES = (
    "Density",
    "Viscosity",
    "VaporPressure",
    "SurfaceTension",
    "ThermalConductivity",
)
"""Properties plotted by :func:`plot_mixture_properties` by default."""

_DEFAULT_T_RANGES = {
    "Density": Units.Quantity([-40, 40], "celsius"),
    "Viscosity": Units.Quantity([-40, 100], "celsius"),
    "VaporPressure": Units.Quantity([0, 125], "celsius"),
    "SurfaceTension": Units.Quantity([-10, 40], "celsius"),
    "ThermalConductivity": Units.Quantity([0, 60], "celsius"),
}
"""Temperature range of the predictions of each property without measured data."""

_DEFAULT_UNITS = {
    "Density": "g/cm^3",
    "Viscosity": "mm^2/s",
    "VaporPressure": "kPa",
    "SurfaceTension": "N/m",
    "ThermalConductivity": "W/m/K",
}
"""Units of each property in the plot if there are no measured data."""

_D86_T_STEP = 50.0
"""Spacing of the major temperature ticks and rounding of the temperature limits."""
_D86_T_MIN = Units.Quantity(298.15, "K")
"""Standard temperature at the bottom of the D86 temperature axis."""
_D86_FIGSIZE = (10.0, 6.5)
"""Fixed size in inches of the D86 figure, wide enough for a legend of several fuels."""


def format_legend_label(fuel_name: str) -> str:
    """Format a fuel name for a plot legend.

    HEFA labels are uppercase, POSF prefixes are omitted, and other names are
    capitalized.

    Args:
        fuel_name: Name of the fuel.

    Returns:
        Legend label of the fuel.
    """
    if "hefa" in fuel_name.lower():
        return fuel_name.upper()
    if "posf" in fuel_name.lower():
        return fuel_name[4:].upper()
    return fuel_name.capitalize()


def get_line_spec(
    fuel_name: str, index: int = 0, cmap: str | Colormap = FL_BLUE_4
) -> tuple[str, str]:
    """Get the line color and marker style of a fuel.

    Fuels in ``FUEL_COLORS`` have fixed colors and fuels in ``FUEL_MARKERS`` have
    fixed marker styles. Other fuels cycle through the colormap by index and use
    circles.

    Args:
        fuel_name: Name of the fuel.
        index: Index of the fuel in the plot, used to cycle through the colormap.
            Defaults to 0.
        cmap: Colormap or name of a registered colormap for fuels without a fixed
            color.
            Defaults to ``FL_BLUE_4``.

    Returns:
        Color and marker style.
    """
    name = fuel_name.lower()
    # Prefer the longest matching key so "dodecane" does not match "decane"
    matches = [key for key in FUEL_MARKERS if key in name]
    marker = FUEL_MARKERS[max(matches, key=len)] if matches else "o"
    color = FUEL_COLORS.get(name)
    if color is None:
        color = cycle_color(index, cmap)
    return color, marker


def _resolve_legend_labels(
    fuel_names: Sequence[str], leg: Sequence[str] | None = None
) -> list[str]:
    """Get the custom or default legend labels of the fuels.

    Args:
        fuel_names: Names of the fuels.
        leg: Literal legend label for each fuel.
            Defaults to the formatted fuel names.

    Returns:
        Legend label of each fuel.

    Raises:
        ValueError: If ``leg`` does not have exactly one label for each fuel.
    """
    if leg is None:
        return [format_legend_label(fuel_name) for fuel_name in fuel_names]
    if len(leg) == len(fuel_names):
        return list(leg)

    msg = "leg must contain exactly one label for each fuel."
    raise ValueError(msg)


# ---------------------------------------------------------------------------------
# Fuel composition
# ---------------------------------------------------------------------------------


def plot_composition(
    fuel_name: str,
    fuel_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    save: bool = True,
    display: bool = False,
) -> None:
    """Plot the composition of a fuel by carbon number and hydrocarbon family.

    Args:
        fuel_name: Name of the fuel to plot.
        fuel_data_dir: Directory containing fuel data.
            Defaults to the bundled data.
        output_dir: Directory for the saved plot.
            Defaults to the current directory.
        title: Title of the plot, or "none" to disable it.
            Defaults to "Fuel Composition".
        save: Whether to save the plot.
            Defaults to True.
        display: Whether to display the plot with ``plt.show()``.
            Defaults to False.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # Handle title
    plot_title = None
    if title is None:
        plot_title = "Fuel Composition"
    elif title.lower() != "none":
        plot_title = title

    if fuel_data_dir is None:
        fuel_data_dir = fl.get_fueldata_dir()

    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the fuel
    fuel = fl.Fuel(fuel_name, fuelDataDir=fuel_data_dir)

    # Create DataFrame with compound data and carbon numbers from fuel object
    df = pd.DataFrame({
        "Compound": fuel.compounds,
        "Weight %": fuel.Y_0.magnitude * 100,
        "Family": fuel.hc_type,
        "nC": fuel.nC,
    })

    # Get unique families from the fuel data in canonical order
    canonical_order = ["n-alkane", "iso-alkane", "cyclo-alkane", "aromatic", "alkene"]
    unique_families = list(np.unique(fuel.hc_type))
    family_names = [f for f in canonical_order if f in unique_families]

    # Remove rows with weight % <= 0.01
    df = df[df["Weight %"] > 0.01]

    # Calculate family weights
    family_weights = df.groupby("Family")["Weight %"].sum()

    # Print composition table
    print(f"\n{'=' * 50}")
    print("Relative Weight % of Each Compound Family")
    print(f"Fuel: {fuel_name}")
    print("=" * 50)
    for family in family_names:
        if family in family_weights.index:
            weight = family_weights[family]
            print(f"  {family:<20} {weight:>8.2f}%")
    print("-" * 50)
    print(f"  {'Total':<20} {family_weights.sum():>8.2f}%")
    print("=" * 50 + "\n")

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Bar chart grouped by carbon number, colored by hydrocarbon type
    spacing = [-0.2985, -0.099, 0.099, 0.2985]
    nC_values = sorted(df["nC"].unique())

    # Get unique families that are in the filtered data
    families_in_data = [f for f in family_names if f in df["Family"].values]

    # Create bars for each family at each carbon number
    for k, family in enumerate(families_in_data):
        df_family = df[df["Family"] == family]

        # Group by carbon number and sum weights
        family_by_nC = df_family.groupby("nC")["Weight %"].sum()

        ax1.bar(
            family_by_nC.index + spacing[k],
            family_by_nC.values,
            label=family,
            alpha=1,
            color=_FAMILY_COLORS.get(family, "#7f7f7f"),
            width=0.2,
        )

    ax1.set_xlabel("Carbon Number", fontsize=16)
    ax1.set_ylabel("Weight %", fontsize=16)
    ax1.set_xticks(nC_values)
    ax1.set_xticklabels(
        [int(n) if n == int(n) else f"{n:.1f}" for n in nC_values], fontsize=14
    )
    ax1.set_xlim(min(nC_values) - 0.5, max(nC_values) + 0.5)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Pie chart of family composition
    # Only include families that have weight > 0, in canonical order
    families_present = [
        f for f in family_names if f in family_weights.index and family_weights[f] > 0
    ]
    family_weights_sorted = family_weights[families_present]

    # Create pie chart without labels/percentages (we'll add them outside)
    wedges, _texts = ax2.pie(
        family_weights_sorted,
        labels=None,
        autopct=None,
        startangle=140,
        colors=[
            _FAMILY_COLORS.get(family, "#7f7f7f")
            for family in family_weights_sorted.index
        ],
    )

    # Add percentages outside the pie with arrows
    for wedge, value, family in zip(
        wedges, family_weights_sorted.values, family_weights_sorted.index
    ):
        angle = (wedge.theta2 + wedge.theta1) / 2
        radius = 1.3
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))

        # Determine horizontal alignment based on position
        ha = "left" if x > 0 else "right"

        # Add annotation with arrow
        ax2.annotate(
            f"{value:.1f}%",
            xy=(np.cos(np.radians(angle)), np.sin(np.radians(angle))),
            xytext=(x, y),
            ha=ha,
            va="center",
            fontsize=14,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.5},
        )

    ax2.axis("equal")

    # Adjust layout to make room for legend BEFORE adding it
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    # Add a single figure-level legend for all families
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=_FAMILY_COLORS.get(family, "#7f7f7f"))
        for family in family_weights_sorted.index
    ]
    fig.legend(
        legend_handles,
        family_weights_sorted.index,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=13,
        frameon=True,
    )

    # Add title if specified
    if plot_title:
        fig.suptitle(plot_title, fontsize=16, fontweight="bold")

    # Save the plot if requested
    if save:
        plot_file = os.path.join(output_dir, f"composition_{fuel_name}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"Composition plot saved to {plot_file}")

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------------
# Mixture properties
# ---------------------------------------------------------------------------------


def _property_label(prop_name: str, prop_units: str) -> str:
    """Create the axis label of a property in the units of the plot.

    Args:
        prop_name: Name of the property.
        prop_units: Units of the property in the plot.

    Returns:
        Axis label.
    """
    label = "Vapor Pressure" if prop_name == "VaporPressure" else prop_name
    return rf"{label} [$\mathrm{{{prop_units}}}$]"


def _predictions_and_data(
    fuel_name: str,
    prop_name: str,
    fuel_data_dir: str | Path,
    output_units: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Get the predicted and measured values of a property of a fuel.

    The predictions cover the range of the measured temperatures, or a default range
    of the property without measured data.

    Args:
        fuel_name: Name of the fuel.
        prop_name: Name of the property.
        fuel_data_dir: Directory containing fuel data.
        output_units: Units of the returned property values.
            Defaults to the units of the measured data, or the default units of the
            property without measured data.

    Returns:
        Measured temperatures in degC, measured property values, predicted
        temperatures in degC, predicted property values, and the units of the
        property values.
    """
    fuel = fl.Fuel(fuel_name, fuelDataDir=fuel_data_dir)

    # Try to load experimental data
    T_data = Units.Quantity([], "celsius")
    prop_units = _DEFAULT_UNITS.get(prop_name, "dimensionless")
    prop_data = Units.Quantity([], prop_units)
    if fuel.propData is not None:
        try:
            data = fuel.propData
            data = data[
                (data["Property"] == prop_name)
                & data["Temp"].notna()
                & data["Property_Value"].notna()
            ]
            if len(data) > 0:
                temp_units = data["Temp_Units"].iloc[0]
                prop_units = data["Property_Units"].iloc[0]
                T_data = Units.Quantity(data["Temp"].to_numpy(), temp_units)
                prop_data = Units.Quantity(
                    data["Property_Value"].to_numpy(), prop_units
                )
        except (OSError, KeyError, ValueError):
            pass

    # Generate predictions over temperature range
    # First check if experimental data exists - use its range if available
    num_prediction_points = (
        25 if prop_name == "SurfaceTension" and fuel.num_compounds > 20 else 100
    )
    if len(T_data) > 1:
        # Use the measured temperature range when it defines an interval.
        T_pred = Units.Quantity(
            np.linspace(
                T_data.magnitude.min(),
                T_data.magnitude.max(),
                num_prediction_points,
            ),
            T_data.units,
        ).to("K")
    else:
        # Use property-specific default range
        temp_range = _DEFAULT_T_RANGES.get(
            prop_name, Units.Quantity([0, 100], "celsius")
        )
        T_pred = Units.Quantity(
            np.linspace(
                temp_range.magnitude[0],
                temp_range.magnitude[1],
                num_prediction_points,
            ),
            temp_range.units,
        ).to("K")

    pred = Units.Quantity(np.zeros_like(T_pred.magnitude), prop_units)
    Y_li = fuel.Y_0

    for i, T in enumerate(T_pred):
        try:
            if prop_name == "Density":
                pred[i] = fuel.mixture_density(Y_li, T).to(prop_units)
            elif prop_name == "VaporPressure":
                pred[i] = fuel.mixture_vapor_pressure(Y_li, T).to(prop_units)
            elif prop_name == "Viscosity":
                pred[i] = fuel.mixture_kinematic_viscosity(Y_li, T).to(prop_units)
            elif prop_name == "SurfaceTension":
                pred[i] = fuel.mixture_surface_tension(Y_li, T).to(prop_units)
            elif prop_name == "ThermalConductivity":
                pred[i] = fuel.mixture_thermal_conductivity(Y_li, T).to(prop_units)
        except (ValueError, TypeError, RuntimeError):
            pred.magnitude[i] = np.nan

    plot_units = output_units if output_units is not None else prop_units
    return (
        T_data.to("celsius").magnitude,
        prop_data.to(plot_units).magnitude,
        T_pred.to("celsius").magnitude,
        pred.to(plot_units).magnitude,
        plot_units,
    )


def plot_mixture_properties(
    fuel_names: str | Sequence[str],
    property_names: Sequence[str] | None = None,
    fuel_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    leg: Sequence[str] | None = None,
    save: bool = True,
    display: bool = False,
    cmap: str | Colormap = FL_BLUE_4,
) -> None:
    """Plot mixture properties of fuels over a temperature range.

    Args:
        fuel_names: Fuel name or names to plot.
        property_names: Properties to plot, from Density, Viscosity, VaporPressure,
            SurfaceTension, and ThermalConductivity.
            Defaults to all of them.
        fuel_data_dir: Directory containing fuel data.
            Defaults to the bundled data.
        output_dir: Directory for the saved plot.
            Defaults to the current directory.
        title: Title of the plot.
            Defaults to None.
        leg: Literal legend label for each fuel.
            Defaults to the formatted fuel names.
        save: Whether to save the plot.
            Defaults to True.
        display: Whether to display the plot with ``plt.show()``.
            Defaults to False.
        cmap: Colormap or name of a registered colormap for the fuels without a fixed
            color (see :func:`get_line_spec`).
            Defaults to ``FL_BLUE_4``.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    if fuel_data_dir is None:
        fuel_data_dir = fl.get_fueldata_dir()

    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle single fuel_name or list
    names = [fuel_names] if isinstance(fuel_names, str) else list(fuel_names)
    legend_labels = _resolve_legend_labels(names, leg)

    # Default properties if not specified
    properties = _DEFAULT_PROPERTIES if property_names is None else property_names

    # Create figure with subplots
    n_props = len(properties)
    figW = 4.25 * n_props
    fig, axes = plt.subplots(
        1, n_props, figsize=(figW, 5.5), constrained_layout=True, squeeze=False
    )
    ax = axes[0]

    # Plot properties for each fuel
    for i, prop_name in enumerate(properties):
        plot_units = None
        prop_units = ""
        for fuel_idx, fuel_name in enumerate(names):
            T_data, prop_data, T_pred, pred, prop_units = _predictions_and_data(
                fuel_name, prop_name, fuel_data_dir, output_units=plot_units
            )
            if plot_units is None:
                plot_units = prop_units
            line_color, marker_style = get_line_spec(fuel_name, fuel_idx, cmap)
            legend_label = legend_labels[fuel_idx]

            # Plot predictions
            ax[i].plot(
                T_pred,
                pred,
                "-",
                color=line_color,
                label=f"FuelLib: {legend_label}",
                linewidth=4,
            )

            # Plot experimental data if available
            if len(prop_data) > 0:
                # Label the data with the fuel name
                ax[i].scatter(
                    T_data,
                    prop_data,
                    marker=marker_style,
                    label=f"Data: {legend_label}",
                    facecolors=line_color,
                    s=75,
                    zorder=5,
                )

        # Format subplot
        ax[i].set_xlabel("T [°C]", fontsize=18)
        ax[i].set_ylabel(_property_label(prop_name, prop_units), fontsize=18)
        ax[i].tick_params(labelsize=18)
        ax[i].grid(alpha=0.3)

    # Add legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=len(names), fontsize=18
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # Save the plot if requested
    if save:
        fuel_str = "_".join(names)
        plot_file = os.path.join(output_dir, f"mixture_properties_{fuel_str}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Mixture properties plot saved to {plot_file}")

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------------
# D86 distillation curves
# ---------------------------------------------------------------------------------


def plot_d86(
    result: "D86Result",
    ax: "Axes | None" = None,
    *,
    label: str | None = None,
    temperature_units: str = "K",
    data: tuple[types.Quantity1D, types.Quantity1D] | None = None,
    color: str | None = None,
    marker: str | None = None,
    data_label: str | None = None,
) -> "Axes":
    """Plot a simulated D86 curve (D86 temperature vs. volume percent recovered).

    Only the distillation period (recovered volume > 0) is plotted. Pass the returned
    axes to further calls to overlay several curves; each new curve takes the next
    color of ``fl_blue_4`` and the temperature limits cover all curves. The
    temperature axis starts at the standard temperature, or at the tick below the
    data if they are colder.

    Args:
        result: Simulated D86 curve.
        ax: Axes to plot on.
            Defaults to the axes of a new figure.
        label: Legend entry of the curve. The legend is shown if any curve on the axes
            has a label.
            Defaults to None.
        temperature_units: Units of the temperature axis (e.g. ``"degC"``).
            Defaults to ``"K"``.
        data: Measured recovered volume fractions and D86 temperatures, plotted as
            symbols in the color of the curve.
            Defaults to None.
        color: Color of the curve and the data symbols.
            Defaults to the next color of ``fl_blue_4``.
        marker: Marker style of the measured data.
            Defaults to the next style of the cycle.
        data_label: Legend entry of the measured data.
            Defaults to ``"Data: {label}"``.

    Returns:
        Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=_D86_FIGSIZE, constrained_layout=True)
    vol = 100.0 * result.volume_recovered.to("dimensionless").magnitude
    T = result.T_vapor.to(temperature_units).magnitude
    recovering = vol > 0.0
    # Data are scatter collections, so the number of lines counts the curves
    n_curves = len(ax.lines)
    if color is None:
        color = cycle_color(n_curves)
    if marker is None:
        marker = _MARKER_CYCLE[n_curves % len(_MARKER_CYCLE)]
    if data_label is None and label is not None:
        data_label = f"Data: {label}"
    ax.plot(
        vol[recovering],
        T[recovering],
        color=color,
        linestyle="-",
        linewidth=4.0,
        label=label,
    )
    if data is not None:
        vol_data, T_data = data
        ax.scatter(
            100.0 * vol_data.to("dimensionless").magnitude,
            T_data.to(temperature_units).magnitude,
            facecolors=color,
            marker=marker,
            s=75.0,
            zorder=5,
            clip_on=False,  # Show the symbols at 0% and 100% in full
            label=data_label,
        )

    # Labels
    ax.set_xlabel("Volume recovered [%]", fontsize=18)
    ax.set_ylabel(f"Temperature [{Units.Unit(temperature_units):~P}]", fontsize=18)

    # Limits: full recovery range; temperatures from the standard temperature (or the
    # lowest tick below the data) up to the next major tick above the data
    ax.set_xlim(0.0, 100.0)
    T_lo, T_hi = ax.dataLim.intervaly
    T_min = float(_D86_T_MIN.to(temperature_units).magnitude)
    if T_lo < T_min:
        T_min = _D86_T_STEP * np.floor(T_lo / _D86_T_STEP)
    ax.set_ylim(
        T_min,
        T_min + _D86_T_STEP * (np.floor((T_hi - T_min) / _D86_T_STEP) + 1.0),
    )

    # Ticks
    ax.xaxis.set_major_locator(MultipleLocator(20.0))
    ax.yaxis.set_major_locator(MultipleLocator(_D86_T_STEP, offset=T_min))
    ax.tick_params(labelsize=18)

    # Grid and legend below the axes, replaced when overlaying another curve
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    if fig is not None:
        for legend in list(fig.legends):
            legend.remove()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                fontsize=18,
                ncol=len(ax.lines),
            )
    return ax


def _d86_data(fuel: "fl.Fuel") -> tuple[types.Quantity1D, types.Quantity1D] | None:
    """Get the measured D86 curve of a fuel from its properties data.

    Distillation rows store the D86 temperature in ``Temp`` and the recovered volume
    in ``Property_Value``.

    Args:
        fuel: Fuel object.

    Returns:
        Recovered volume fractions and D86 temperatures, or None if the fuel has no
        distillation data.
    """
    if fuel.propData is None:
        return None
    df = fuel.propData[fuel.propData["Property"] == "Distillation"]
    if df.empty:
        return None
    # pint does not define "percentage"
    vol_units = df["Property_Units"].iloc[0].replace("percentage", "percent")
    vol = Units.Quantity(df["Property_Value"].to_numpy(float), vol_units)
    T = Units.Quantity(df["Temp"].to_numpy(float), df["Temp_Units"].iloc[0])
    return vol, T


def plot_distillation(
    fuels: Sequence["fl.Fuel | str"],
    fuel_data_dir: str | Path | None = None,
    title: str | None = None,
    leg: Sequence[str] | None = None,
    temperature_units: str = "celsius",
    cmap: str | Colormap = FL_BLUE_4,
) -> "Figure":
    """Simulate and plot the ASTM D86 distillation curves of fuels.

    The curves are simulated with the model of Greenfield et al. (1998) (see
    :mod:`fuellib.simulate.distillation.greenfield`) and overlaid on the same axes with
    the colors, symbols and legend labels of :func:`plot_mixture_properties`, together
    with the measured D86 curves if available. The figure is neither saved nor shown.

    Args:
        fuels: Fuel objects or names of fuels to load.
        fuel_data_dir: Directory containing the data of the fuels given by name.
            Defaults to the bundled data.
        title: Title of the plot.
            Defaults to None.
        leg: Literal legend label for each fuel.
            Defaults to the formatted fuel names.
        temperature_units: Units of the temperature axis.
            Defaults to ``"celsius"``.
        cmap: Colormap or name of a registered colormap for the fuels without a fixed
            color (see :func:`get_line_spec`).
            Defaults to ``FL_BLUE_4``.

    Returns:
        Figure with the D86 curves.
    """
    # Imported here since the simulation module imports this module
    from ..simulate.distillation import greenfield

    loaded = [
        fl.Fuel(fuel, fuelDataDir=fuel_data_dir) if isinstance(fuel, str) else fuel
        for fuel in fuels
    ]
    legend_labels = _resolve_legend_labels([fuel.name for fuel in loaded], leg)

    fig, ax = plt.subplots(figsize=_D86_FIGSIZE, constrained_layout=True)
    for i, (fuel, legend_label) in enumerate(zip(loaded, legend_labels, strict=True)):
        color, marker = get_line_spec(fuel.name, i, cmap)
        plot_d86(
            greenfield.distill(fuel),
            ax,
            label=f"FuelLib: {legend_label}",
            temperature_units=temperature_units,
            data=_d86_data(fuel),
            color=color,
            marker=marker,
            data_label=f"Data: {legend_label}",
        )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig
