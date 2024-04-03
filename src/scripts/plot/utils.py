from pathlib import Path

import matplotlib.pyplot as plt
from cartopy import crs
from matplotlib import ticker
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from numpy import Inf, linspace, maximum, minimum, ndarray, round
from pyshtools.expand import MakeGridDH

from ...utils import BoundaryCondition, Direction

SYMBOLS_PER_BOUNDARY_CONDITION = {Direction.radial: "h", Direction.tangential: "l", Direction.potential: "k"}
SYMBOLS_PER_DIRECTION = {BoundaryCondition.load: "'", BoundaryCondition.shear: "*", BoundaryCondition.potential: ""}


def get_degrees_indices(degrees: list[int], degrees_to_plot: list[int]) -> list[int]:
    """
    Returns the indices of the wanted degrees in the list of degrees.
    """
    return [list(degrees).index(degree) for degree in degrees_to_plot]


def plot_harmonics_on_natural_projection(
    harmonics: ndarray[float],
    figure_subpath: Path,
    name: str,
    title: str,
    label: str,
    ocean_mask: ndarray[float] | float,
    min_saturation: float,
    max_saturation: float,
    num_colormesh_bins: int,
) -> None:
    """
    Creates a world map figure of the given harmonics. Eventually exclude areas with a given mask.
    Plots a saturated version and a casual version.
    """
    for min_saturation_value, max_saturation_value in [[None, None], [min_saturation, max_saturation]]:
        fig = plt.figure(
            figsize=(16, 9),
        )
        ax = fig.add_subplot(1, 1, 1, projection=crs.Robinson(central_longitude=180))
        plt.title(
            title + (" saturated" if not (min_saturation_value is None and max_saturation_value is None) else ""), fontsize=20
        )
        ax.set_global()
        spatial_result = round(
            a=MakeGridDH(harmonics, sampling=2) * ocean_mask,
            decimals=3,
        )
        contour = ax.pcolormesh(
            linspace(start=0, stop=360, num=len(spatial_result[0])),
            linspace(start=90, stop=-90, num=len(spatial_result)),
            maximum(
                -Inf if min_saturation_value is None else min_saturation_value,
                minimum(Inf if max_saturation_value is None else max_saturation_value, spatial_result),
            ),
            transform=crs.PlateCarree(),
            cmap="RdBu_r",
            # levels=num_colormesh_bins,
            norm=SymLogNorm(vcenter=0),  # TwoSlopeNorm(vcenter=0), TODO.
        )
        ax.coastlines()
        cbar = plt.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.07)
        tick_locator = ticker.MaxNLocator(nbins=num_colormesh_bins)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label=label, size=16)
        plt.savefig(
            figure_subpath.joinpath(
                name + ("_saturated" if not (min_saturation_value is None and max_saturation_value is None) else "") + ".png"
            )
        )
        plt.clf()
