# Computes anelastic-modified harmonic load signal for a given description and options.
# Gets already computed Love numbers, builds load signal from data, computes anelastic induced harmonic load signal and
# saves it.
# Saves the corresponding figures (spatial domain) in the specified subfolder.

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from cartopy import crs
from matplotlib import ticker
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from numpy import linspace, ndarray, round
from pyshtools.expand import MakeGridDH

from utils import (
    BOOLEANS,
    SignalHyperParameters,
    anelastic_harmonic_induced_load_signal,
    build_elastic_load_signal,
    figures_path,
    format_ocean_mask,
    load_base_model,
    parameters_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()

from numpy import linspace, maximum, minimum


def plot_harmonics_on_natural_projection(
    harmonics: ndarray[float],
    title: str,
    figure_subpath: Path,
    name: str,
    n_max: int,
    num: int = 10,
    ocean_mask_filename: Optional[str] = None,
    v_min: Optional[int] = None,
    v_max: Optional[int] = None,
    label: str = "cm/y",
) -> None:
    """
    Creates a world map figure of the given harmonics. Eventually exclude areas with a given mask. Eventually saturates the
    color scale.
    """

    fig = plt.figure(
        figsize=(16, 9),
    )
    ax = fig.add_subplot(1, 1, 1, projection=crs.Robinson(central_longitude=180))
    plt.title(title, fontsize=20)
    ax.set_global()
    spatial_result = round(
        a=MakeGridDH(harmonics, sampling=2)
        * format_ocean_mask(ocean_mask_filename=ocean_mask_filename, n_max=min(n_max, len(harmonics[0]))),
        decimals=3,
    )
    contour = ax.pcolormesh(
        linspace(start=0, stop=360, num=len(spatial_result[0])),
        linspace(start=90, stop=-90, num=len(spatial_result)),
        spatial_result if v_min is None else maximum(v_min, minimum(v_max, spatial_result)),
        transform=crs.PlateCarree(),
        cmap="RdBu_r",
        # levels=num,
        norm=SymLogNorm(vcenter=0),  # TwoSlopeNorm(vcenter=0),
    )
    ax.coastlines()
    cbar = plt.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.07)
    tick_locator = ticker.MaxNLocator(nbins=num)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label=label, size=16)
    plt.savefig(figure_subpath.joinpath(name))
    plt.show(block=False)
    plt.close()


def anelastic_induced_harmonic_load_trend(
    real_description_id: str,
    figure_subpath_string: str,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
) -> None:
    """
    Computes anelastic-modified harmonic load signal for a given description and options.
    Gets already computed Love numbers, builds load signal from data, computes anelastic induced harmonic load signal and
    saves it.
    Saves the corresponding figures (spatial domain) in the specified subfolder.
    """
    # Builds frequential signal.
    dates, frequencies, (elastic_trend, _, frequencial_elastic_load_signal, harmonic_weights) = build_elastic_load_signal(
        signal_hyper_parameters=signal_hyper_parameters, get_harmonic_weights=True
    )

    # Computes anelastic induced harmonic load signal.
    (
        path,
        _,
        _,
        _,
        harmonic_trends,
        _,
    ) = anelastic_harmonic_induced_load_signal(
        harmonic_weights=harmonic_weights,
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        dates=dates,
        frequencies=frequencies,
        frequencial_elastic_normalized_load_signal=frequencial_elastic_load_signal / elastic_trend,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id).joinpath(path.name)
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Results.
    for saturation in BOOLEANS:
        v_min = None if not saturation else -5.0
        v_max = None if not saturation else 5.0
        # Inputs.
        plot_harmonics_on_natural_projection(
            harmonics=harmonic_weights,
            title=signal_hyper_parameters.weights_map,
            figure_subpath=figure_subpath.parent.parent,
            n_max=signal_hyper_parameters.n_max,
            name=signal_hyper_parameters.weights_map + ("_saturated" if saturation else ""),
            ocean_mask_filename=signal_hyper_parameters.ocean_mask,
            v_min=v_min,
            v_max=v_max,
        )
        # Output.
        plot_harmonics_on_natural_projection(
            harmonics=harmonic_trends,
            title="anelastic induced loads : trends since " + str(signal_hyper_parameters.first_year_for_trend),
            figure_subpath=figure_subpath,
            n_max=signal_hyper_parameters.n_max,
            name=signal_hyper_parameters.weights_map
            + "_"
            + signal_hyper_parameters.signal
            + "_trend_anelastic"
            + ("_saturated" if saturation else ""),
            ocean_mask_filename=signal_hyper_parameters.ocean_mask,
            v_min=v_min,
            v_max=v_max,
        )
        # Difference.
        plot_harmonics_on_natural_projection(
            harmonics=harmonic_trends - harmonic_weights,
            title="anelastic induced loads differences with elastic : trends since "
            + str(signal_hyper_parameters.first_year_for_trend),
            figure_subpath=figure_subpath,
            n_max=signal_hyper_parameters.n_max,
            name=signal_hyper_parameters.weights_map
            + "_"
            + signal_hyper_parameters.signal
            + "_trend_diff_with_elastic"
            + ("_saturated" if saturation else ""),
            ocean_mask_filename=signal_hyper_parameters.ocean_mask,
            v_min=v_min,
            v_max=v_max,
        )


if __name__ == "__main__":
    anelastic_induced_harmonic_load_trend(
        real_description_id=(args.real_description_id if args.real_description_id else "test"),
        figure_subpath_string=args.subpath if args.subpath else "spatial_load_signal",
    )

    # TODO. 04: loop on it.
