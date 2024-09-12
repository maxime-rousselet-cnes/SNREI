from typing import Any, Optional

from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cmocean import cm
from matplotlib.colors import TwoSlopeNorm
from numpy import argsort, array, flip, inf, maximum, minimum, ndarray, round, where
from numpy.ma import MaskedArray

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


def get_grid(harmonics: ndarray[float], n_max: int, decimals: int = 4) -> ndarray[float]:
    """
    Projects spherical harmonics on a (latitude x longitude) grid.
    """
    return round(
        a=make_grid(harmonics=harmonics, n_max=n_max),
        decimals=decimals,
    )


def get_grid(harmonics: ndarray[float], n_max: int, decimals: int = 4) -> ndarray[float]:
    """
    Projects spherical harmonics on a (latitude x longitude) grid.
    """
    return round(
        a=make_grid(harmonics=harmonics, n_max=n_max),
        decimals=decimals,
    )


def natural_projection(
    ax: GeoAxes,
    saturation_threshold: float,
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    n_max: int,
    signal_threshold: float,
    harmonics: Optional[ndarray[float]] = None,
    map: Optional[ndarray[float]] = None,
    mask: Optional[ndarray[float]] = None,
) -> tuple[Any, ndarray[float]]:
    """
    Displays a projection of a given harmonic quantity on the given matplotlib Axes.
    Ensures that longitudes and data are wrapped correctly, and areas outside the mask are shaded in grey.
    """

    # Ensures longitudes are in the range [-180, 180].
    longitudes = where(longitudes > 180, longitudes - 360, longitudes)

    # Sorts the longitudes and corresponding data.
    sort_idx = flip(argsort(longitudes))
    longitudes = longitudes[sort_idx]
    spatial_result = map if map is not None else get_grid(harmonics=harmonics, n_max=n_max)
    spatial_result = spatial_result[:, sort_idx]

    # Applies the mask: masks areas outside your region of interest.
    if mask is not None:
        mask *= abs(spatial_result) < signal_threshold
        mask = mask[:, sort_idx]
        spatial_result = MaskedArray(spatial_result, mask=(mask == 0))

    # Applies saturation threshold.
    limited_result = maximum(
        -inf if saturation_threshold is None else -saturation_threshold,
        minimum(inf if saturation_threshold is None else saturation_threshold, spatial_result),
    )

    # Plots.
    contour = ax.pcolormesh(
        longitudes,
        latitudes,
        limited_result,
        transform=PlateCarree(central_longitude=0),
        cmap=cm.balance,
        norm=TwoSlopeNorm(vcenter=0, vmin=-saturation_threshold, vmax=saturation_threshold),
        shading="auto",
    )
    contour.cmap.set_bad(color="grey")
    ax.coastlines()
    gl = ax.gridlines(crs=PlateCarree(central_longitude=0), draw_labels=True, linewidth=2, color="gray", alpha=0.5, linestyle="--")
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = False

    return contour, mask
