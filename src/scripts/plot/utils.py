from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from cartopy import crs, feature
from matplotlib import colors
from numpy import Inf, array, linspace, maximum, minimum, ndarray, ones, round, vstack
from pyshtools.expand import MakeGridDH

from ...utils import (
    BoundaryCondition,
    Direction,
    RunHyperParameters,
    data_Frederikse_path,
    extract_temporal_load_signal,
    figures_path,
)

SYMBOLS_PER_BOUNDARY_CONDITION = {BoundaryCondition.load: "'", BoundaryCondition.shear: "*", BoundaryCondition.potential: ""}
SYMBOLS_PER_DIRECTION = {Direction.radial: "h", Direction.tangential: "l", Direction.potential: "k"}
SMALT_BLUE = (0.0, 79.0 / 255.0, 145.0 / 255.0)
VALENCIA_RED = (217.0 / 255.0, 80.0 / 255.0, 70.0 / 255.0)
MOON_YELLOW = (236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0)
KELLY_GREEN = (66.0 / 255.0, 142.0 / 255.0, 14.0 / 255.0)
WHITE = (255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0)


def option_color(option: RunHyperParameters) -> str:
    """
    Chose a linestyle for the given run option.
    """
    return (
        SMALT_BLUE
        if (not option.use_long_term_anelasticity) and (not option.use_short_term_anelasticity)
        else (
            KELLY_GREEN
            if not option.use_long_term_anelasticity
            else (MOON_YELLOW if not option.use_short_term_anelasticity else VALENCIA_RED)
        )
    )


def options_label(option: RunHyperParameters) -> str:
    """
    Builds a label string corresponding to the given run option.
    """
    return " ".join(
        (
            "with long-term visc." if option.use_long_term_anelasticity else "",
            "with short-term visc." if option.use_short_term_anelasticity else "",
        )
    )


def plot_harmonics_on_natural_projection(
    harmonics: ndarray[float],
    figure_subpath: Path,
    name: str,
    title: str,
    label: str,
    ocean_mask: ndarray[float] | float,
    min_saturation: Optional[float],
    max_saturation: Optional[float],
    decimals: int = 4,
    logscale: bool = True,
    figsize: tuple[int, int] = (10, 10),
    negative_saturated_color: tuple[float, float, float] = SMALT_BLUE,
    positive_saturated_color: tuple[float, float, float] = VALENCIA_RED,
    zero_modified_color: tuple[float, float, float] = WHITE,
    exp_scale_factor: float = 1.0,
    brightness_saturation_factor: float = 0.85,
) -> None:
    """
    Creates a world map figure of the given harmonics. Eventually exclude areas with a given mask.
    Plots a saturated version and a casual version.
    """
    for min_saturation_value, max_saturation_value in [[None, None], [min_saturation, max_saturation]]:
        fig = plt.figure(
            figsize=figsize,
        )
        ax = fig.add_subplot(1, 1, 1, projection=crs.Robinson(central_longitude=180))
        plt.title(title + (" saturated" if not (min_saturation_value is None and max_saturation_value is None) else ""))
        ax.set_global()
        spatial_result: ndarray[float] = round(
            a=MakeGridDH(harmonics, sampling=2) * ocean_mask,
            decimals=decimals,
        )
        positive_range_colors = array(
            [
                linspace(zero_code, positive_saturated_code * brightness_saturation_factor, 256)
                for zero_code, positive_saturated_code in zip(zero_modified_color, positive_saturated_color)
            ]
            + [
                ones(256),
            ]
        ).T
        zero_modified_colors = array([list(zero_modified_color) + [1]])
        negative_range_colors = array(
            [
                linspace(negative_saturated_code * brightness_saturation_factor, zero_code, 255)
                for zero_code, negative_saturated_code in zip(zero_modified_color, negative_saturated_color)
            ]
            + [
                ones(255),
            ]
        ).T
        colorbar_values = vstack((negative_range_colors, zero_modified_colors, positive_range_colors)) ** exp_scale_factor
        contour = ax.pcolormesh(
            linspace(start=0, stop=360, num=len(spatial_result[0])),
            linspace(start=90, stop=-90, num=len(spatial_result)),
            maximum(
                -Inf if min_saturation_value is None else min_saturation_value,
                minimum(Inf if max_saturation_value is None else max_saturation_value, spatial_result),
            ),
            transform=crs.PlateCarree(),
            cmap=colors.LinearSegmentedColormap.from_list(
                name="two_solopes_colorbar_grey_zero", colors=colorbar_values
            ),  # "RdBu_r",
            norm=(
                colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=min_saturation_value, vmax=max_saturation_value, base=10)
                if logscale
                else colors.TwoSlopeNorm(vcenter=0)
            ),
        )
        ax.coastlines(color="grey", linewidth=2)
        ax.add_feature(feature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor="grey"))
        cbar = plt.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.06)
        cbar.set_label(label=label)
        plt.savefig(
            figure_subpath.joinpath(
                name + ("_saturated" if not (min_saturation_value is None and max_saturation_value is None) else "") + ".png"
            )
        )
        plt.close()


def plot_temporal_load_signal(
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    figure_subpath_string: str = "load_functions",
    path: Path = data_Frederikse_path,
    name: str = "global_basin_timeseries.csv",
) -> None:
    """
    Plots The uniform elastic load signal history.
    """
    plt.figure(figsize=figsize)
    dates, barystatic = extract_temporal_load_signal(path=path, name=name)
    plt.plot(dates, barystatic, linewidth=linewidth)
    plt.title("Load history")
    plt.xlabel("date (y)")
    plt.ylabel("Relative sea level rise  (mm)")
    plt.grid()
    figure_subpath = figures_path.joinpath(figure_subpath_string)
    figure_subpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_subpath.joinpath(path.name + ".png"))
    plt.close()
