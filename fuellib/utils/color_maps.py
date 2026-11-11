"""Colormaps and fuel colors of FuelLib.

The colormaps ``fl_blue_4`` and ``fl_pastel_18`` are registered with Matplotlib when
FuelLib is imported, so they can be used by name (e.g.
``plt.get_cmap("fl_pastel_18")``).
"""

import matplotlib
from matplotlib.colors import Colormap, ListedColormap, to_hex

PRIMARY_BLUE = "#2980B9"
"""Primary blue of FuelLib."""
GRAY_50 = "#7F7F7F"
"""50% gray."""
GRAY_DARK = "#333333"
"""Dark gray."""
PURPLE = "#663399"
"""Purple."""

FL_BLUE_4 = ListedColormap([PRIMARY_BLUE, GRAY_50, GRAY_DARK, PURPLE], name="fl_blue_4")
"""FuelLib palette of four colors: primary blue, 50% gray, dark gray, and purple."""

FL_PASTEL_18 = ListedColormap(
    [
        "midnightblue",
        "royalblue",
        "skyblue",
        "teal",
        "cadetblue",
        "darkseagreen",
        "darkred",
        "firebrick",
        "indianred",
        "darkorchid",
        "mediumorchid",
        "orchid",
        "darkolivegreen",
        "olivedrab",
        "yellowgreen",
        "darkgoldenrod",
        "goldenrod",
        "darkkhaki",
    ],
    name="fl_pastel_18",
)
"""FuelLib palette of 18 colors: six families (blue, teal, red, purple, green, and
gold) of three shades each, from dark to light."""

for _cmap in (FL_BLUE_4, FL_PASTEL_18):
    matplotlib.colormaps.register(_cmap, force=True)

FUEL_COLORS = {
    "posf10264": PRIMARY_BLUE,
    "posf10325": GRAY_50,
    "posf10289": GRAY_DARK,
    "posf11498": PURPLE,
    "heptane": PRIMARY_BLUE,
    "decane": GRAY_50,
    "dodecane": GRAY_DARK,
}
"""Colors of specific fuels by lowercase fuel name, from the ``fl_blue_4`` colors."""


def cycle_color(index: int, cmap: str | Colormap = FL_BLUE_4) -> str:
    """Get a color of a colormap by index, wrapping around at the end of the colormap.

    Intended for qualitative colormaps such as ``fl_blue_4`` and ``fl_pastel_18``.

    Args:
        index: Index of the color.
        cmap: Colormap or name of a registered colormap.
            Defaults to ``FL_BLUE_4``.

    Returns:
        Color as a hex string.
    """
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    return to_hex(cmap(index % cmap.N))
