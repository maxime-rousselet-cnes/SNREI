from typing import Optional

from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cmocean import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator
from numpy import Inf, linspace, maximum, minimum, ndarray, round
from pyshtools.expand import MakeGridDH


def natural_projection(
    ax: GeoAxes,
    harmonics: ndarray[float],
    saturation_threshold: Optional[float] = 50,
    decimals: int = 4,
):
    """
    Displays a projection of a given harmonic quantity on the given matplotlib Axes.
    """

    # Gets quantity in spatial domain.
    spatial_result: ndarray[float] = round(
        a=MakeGridDH(harmonics, sampling=2),
        decimals=decimals,
    )

    # Projects.
    contour = ax.pcolormesh(
        linspace(start=0, stop=360, num=len(spatial_result[0])),
        linspace(start=90, stop=-90, num=len(spatial_result)),
        maximum(
            -Inf if saturation_threshold is None else -saturation_threshold,
            minimum(Inf if saturation_threshold is None else saturation_threshold, spatial_result),
        ),
        transform=PlateCarree(),
        cmap=cm.balance,
        norm=TwoSlopeNorm(vcenter=0, vmin=-saturation_threshold, vmax=saturation_threshold),
    )
    # generates coastlines.
    ax.coastlines()

    # Generates parallels and labels.
    gl = ax.gridlines(crs=PlateCarree(), draw_labels=True, linewidth=2, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = False
    gl.xlocator = FixedLocator([-90, 90, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 15, "color": "gray"}
    gl.xlabel_style = {"color": "red", "weight": "bold"}

    return contour
