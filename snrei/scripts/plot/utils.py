from typing import Any, Optional

from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cmocean import cm
from matplotlib.colors import TwoSlopeNorm
from numpy import array, inf, maximum, minimum, ndarray, round

from ...functions import make_grid
from ...utils import BoundaryCondition, Direction

REFERENCE_RED = (176.0 / 255.0, 0.0, 39.0 / 255.0)
COLORS = (
    array(
        [
            (176, 0, 39),
            (153, 204, 255),
            (102, 178, 255),
            (51, 153, 255),
            (0, 128, 255),
            (0, 102, 204),
            (176, 0, 39),
            (255, 178, 102),
            (255, 153, 51),
            (255, 108, 0),
            (204, 102, 0),
            (176, 0, 39),
            (0, 128, 255),
            (255, 108, 0),
            (204, 0, 204),
        ]
    )
    / 255.0
)
SYMBOLS_PER_BOUNDARY_CONDITION = {
    BoundaryCondition.load: "'",
    BoundaryCondition.shear: "*",
    BoundaryCondition.potential: "",
}
SYMBOLS_PER_DIRECTION = {Direction.radial: "h", Direction.tangential: "l", Direction.potential: "k"}


def get_grid(harmonics: ndarray[float], latitudes: ndarray[float], longitudes: ndarray[float], decimals: int = 4) -> ndarray[float]:
    """
    Projects spherical harmonics on a (latitude x longitude) grid.
    """
    return round(
        a=make_grid(harmonics=harmonics, latitudes=latitudes, longitudes=longitudes),
        decimals=decimals,
    )


def natural_projection(
    ax: GeoAxes,
    saturation_threshold: float,
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    harmonics: Optional[ndarray[float]] = None,
    map: Optional[ndarray[float]] = None,
    mask: Optional[ndarray[float]] = None,
) -> Any:
    """
    Displays a projection of a given harmonic quantity on the given matplotlib Axes.
    """

    # Gets quantity in spatial domain.
    spatial_result = (map if not map is None else get_grid(harmonics=harmonics, latitudes=latitudes, longitudes=longitudes)) * (
        mask if not mask is None else 1.0
    )
    # Projects.
    contour = ax.pcolormesh(
        longitudes,
        latitudes,
        maximum(
            -inf if saturation_threshold is None else -saturation_threshold,
            minimum(inf if saturation_threshold is None else saturation_threshold, spatial_result),
        ),
        transform=PlateCarree(central_longitude=0),
        cmap=cm.balance,
        norm=TwoSlopeNorm(vcenter=0, vmin=-saturation_threshold, vmax=saturation_threshold),
    )
    # Generates coastlines.
    ax.coastlines()

    # Generates parallels and labels.
    gl = ax.gridlines(crs=PlateCarree(central_longitude=0), draw_labels=True, linewidth=2, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"color": "gray"}

    return contour
