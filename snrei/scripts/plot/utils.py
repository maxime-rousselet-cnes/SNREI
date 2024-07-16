from typing import Any

from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from cmocean import cm
from matplotlib.colors import TwoSlopeNorm
from numpy import inf, linspace, maximum, minimum, ndarray, round
from pyshtools.expand import MakeGridDH

from ...utils import BoundaryCondition, Direction

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
        a=MakeGridDH(harmonics, sampling=2, lmax=n_max),
        decimals=decimals,
    )


def natural_projection(
    ax: GeoAxes, harmonics: ndarray[float], saturation_threshold: float, n_max: int, mask: ndarray[float]
) -> Any:
    """
    Displays a projection of a given harmonic quantity on the given matplotlib Axes.
    """

    # Gets quantity in spatial domain.
    spatial_result = get_grid(harmonics=harmonics, n_max=n_max) * mask

    # Projects.
    contour = ax.pcolormesh(
        linspace(start=0, stop=360, num=len(spatial_result[0])),
        linspace(start=90, stop=-90, num=len(spatial_result)),
        maximum(
            -inf if saturation_threshold is None else -saturation_threshold,
            minimum(inf if saturation_threshold is None else saturation_threshold, spatial_result),
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
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"color": "gray"}

    return contour
