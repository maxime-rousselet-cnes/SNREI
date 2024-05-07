from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from cartopy import crs, feature
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from numpy import (
    Inf,
    array,
    concatenate,
    linspace,
    maximum,
    minimum,
    ndarray,
    ones,
    round,
    vstack,
)
from pyshtools.expand import MakeGridDH

from ...utils import (
    BoundaryCondition,
    Direction,
    GMSL_data_path,
    LoadSignalHyperParameters,
    RunHyperParameters,
    extract_temporal_load_signal,
    extract_trends_GRACE,
    figures_path,
    load_load_signal_hyper_parameters,
    map_sampling,
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
    return (
        "elastic"
        if (not option.use_long_term_anelasticity) and (not option.use_short_term_anelasticity)
        else "with "
        + " and ".join(
            (["long-term"] if option.use_long_term_anelasticity else [])
            + (["short-term"] if option.use_short_term_anelasticity else [])
        )
        + " visc."
    )


def get_colorbar_values(
    negative_saturated_color: tuple[float, float, float] = SMALT_BLUE,
    positive_saturated_color: tuple[float, float, float] = VALENCIA_RED,
    zero_modified_color: tuple[float, float, float] = WHITE,
    exp_scale_factor: float = 1.0,
    saturation_factor: float = 1.0,
) -> ndarray:
    """
    Generatesn array of colors for the map colorbar.
    """
    positive_range_colors = array(
        [
            linspace(zero_code, positive_saturated_code, 256)
            for zero_code, positive_saturated_code in zip(zero_modified_color, positive_saturated_color)
        ]
    ).T
    zero_modified_colors = array([list(zero_modified_color)])
    negative_range_colors = array(
        [
            linspace(negative_saturated_code, zero_code, 255)
            for zero_code, negative_saturated_code in zip(zero_modified_color, negative_saturated_color)
        ]
    ).T
    return maximum(
        0.0,
        minimum(
            1.0,
            colors.hsv_to_rgb(
                colors.rgb_to_hsv(vstack((negative_range_colors, zero_modified_colors, positive_range_colors)))
                * array(object=[[1.0, saturation_factor, exp_scale_factor]])
            ),
        ),
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
    saturation_factor: float = 1.0,
    continents: bool = False,
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
        colorbar_values = get_colorbar_values(
            negative_saturated_color=negative_saturated_color,
            positive_saturated_color=positive_saturated_color,
            zero_modified_color=zero_modified_color,
            exp_scale_factor=exp_scale_factor,
            saturation_factor=saturation_factor,
        )
        contour = ax.pcolormesh(
            linspace(start=0, stop=360, num=len(spatial_result[0])),
            linspace(start=90, stop=-90, num=len(spatial_result)),
            maximum(
                -Inf if min_saturation_value is None else min_saturation_value,
                minimum(Inf if max_saturation_value is None else max_saturation_value, spatial_result),
            ),
            transform=crs.PlateCarree(),
            cmap=colors.LinearSegmentedColormap.from_list(
                name="two_solopes_colorbar_grey_zero", colors=concatenate((colorbar_values, ones(shape=(512, 1))), axis=1)
            ),  # "RdBu_r",
            norm=(
                colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=min_saturation_value, vmax=max_saturation_value, base=10)
                if logscale
                else colors.TwoSlopeNorm(vcenter=0)
            ),
        )
        ax.coastlines(color="grey", linewidth=2)
        if not continents:
            ax.add_feature(feature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor="grey"))
        cbar = plt.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.06)
        cbar.set_label(label=label)
        plt.savefig(
            figure_subpath.joinpath(
                name + ("_saturated" if not (min_saturation_value is None and max_saturation_value is None) else "") + ".png"
            )
        )
        plt.close()


def plot_load_signal(
    figsize: tuple[int, int] = (8, 8),
    linewidth: int = 2,
    figure_subpath_string: str = "load_functions",
    data_path: Path = GMSL_data_path,
    filename: str = "Frederikse/global_basin_timeseries.csv",
    decimals: int = 4,
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    negative_saturated_color: tuple[float, float, float] = SMALT_BLUE,
    positive_saturated_color: tuple[float, float, float] = VALENCIA_RED,
    zero_modified_color: tuple[float, float, float] = WHITE,
    exp_scale_factor: float = 1.0,
    saturation_factor: float = 1.0,
    label: str = "mm/yr",
    min_saturation: Optional[float] = -50,
    max_saturation: Optional[float] = 50,
    logscale: bool = False,
) -> None:
    """
    Plots The uniform elastic load signal history.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig)
    plot = fig.add_subplot(gs[0, 0])

    dates, mean_barystatic = extract_temporal_load_signal(path=data_path, name="mean", filename=filename, zero=False)
    _, upper_barystatic = extract_temporal_load_signal(path=data_path, name="upper", filename=filename, zero=False)
    _, lower_barystatic = extract_temporal_load_signal(path=data_path, name="lower", filename=filename, zero=False)

    map = extract_trends_GRACE(
        name=load_signal_hyper_parameters.GRACE, load_signal_hyper_parameters=load_signal_hyper_parameters
    )
    spatial_result = round(
        a=map_sampling(map, n_max=load_signal_hyper_parameters.n_max, harmonic_domain=False)[0], decimals=decimals
    )

    plot.fill_between(dates, lower_barystatic, upper_barystatic)
    plot.plot(dates, mean_barystatic, linewidth=linewidth, color=(0, 0, 0))
    plot.set_xlabel("date (y)")
    plot.set_title("Barystatic  (mm)")
    plot.grid()

    ax = fig.add_subplot(gs[:, 1:], projection=crs.Robinson(central_longitude=180))
    ax.set_title("GRACE's solution 2003-2023 trend")
    ax.set_global()
    colorbar_values = get_colorbar_values(
        negative_saturated_color=negative_saturated_color,
        positive_saturated_color=positive_saturated_color,
        zero_modified_color=zero_modified_color,
        exp_scale_factor=exp_scale_factor,
        saturation_factor=saturation_factor,
    )
    contour = ax.pcolormesh(
        linspace(start=0, stop=360, num=len(spatial_result[0])),
        linspace(start=90, stop=-90, num=len(spatial_result)),
        maximum(
            -Inf if min_saturation is None else min_saturation,
            minimum(Inf if max_saturation is None else max_saturation, spatial_result),
        ),
        transform=crs.PlateCarree(),
        cmap=colors.LinearSegmentedColormap.from_list(
            name="two_solopes_colorbar_grey_zero", colors=concatenate((colorbar_values, ones(shape=(512, 1))), axis=1)
        ),  # "RdBu_r",
        norm=(
            colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=min_saturation, vmax=max_saturation, base=10)
            if logscale
            else colors.TwoSlopeNorm(vcenter=0)
        ),
    )
    ax.coastlines(color="grey", linewidth=2)
    cbar = plt.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.06)
    cbar.set_label(label=label)

    figure_subpath = figures_path.joinpath(figure_subpath_string)
    figure_subpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_subpath.joinpath(data_path.name + ".png"))
    plt.close()
