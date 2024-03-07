# Computes anelastic-modified load signal for a given description and options.
# Gets already computed Love numbers, builds load signal from data, computes anelastic induced load signal and saves it.
# Saves the corresponding figures in the specified subfolder.

import argparse

import matplotlib.pyplot as plt
from numpy import transpose

from utils import (
    SignalHyperParameters,
    anelastic_induced_load_signal,
    build_elastic_load_signal,
    figures_path,
    load_base_model,
    parameters_path,
    signal_induced_trend_from_dates,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def anelastic_induced_load_trend(
    real_description_id: str,
    figure_subpath_string: str,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
    degrees_to_plot: list[int] = [2, 3, 5, 10, 20],
) -> None:
    """
    Computes anelastic-modified load signal for a given description and options.
    Gets already computed Love numbers, builds load signal from data, computes anelastic induced load signal and saves it.
    Saves the corresponding figures in the specified subfolder.
    """
    # Builds frequential signal.
    dates, frequencies, (elastic_trend, temporal_elastic_load_signal, frequencial_elastic_load_signal, _) = (
        build_elastic_load_signal(signal_hyper_parameters=signal_hyper_parameters, get_harmonic_weights=False)
    )

    # Computes anelastic induced load signal.
    (_, degrees, temporal_load_signal_per_degree, _, _) = anelastic_induced_load_signal(
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        dates=dates,
        frequencies=frequencies,
        frequencial_elastic_load_signal=frequencial_elastic_load_signal,
    )

    # Gets trends.
    (
        trend_dates,
        elastic_signal,
        signals,
        signal_trends,
    ) = signal_induced_trend_from_dates(
        elastic_trend=elastic_trend,
        signal_hyper_parameters=signal_hyper_parameters,
        elastic_signal=temporal_elastic_load_signal,
        signal=transpose(a=temporal_load_signal_per_degree),
        dates=dates,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    figure_subpath.mkdir(parents=True, exist_ok=True)
    degrees_indices = [list(degrees).index(degree) for degree in degrees_to_plot]

    # Whole signal.
    plt.figure(figsize=(16, 9))
    plt.plot(dates, temporal_elastic_load_signal, label="elastic")
    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
        plt.plot(
            dates,
            temporal_load_signal_per_degree[i_degree],
            label="degree " + str(degree),
        )
    plt.legend()
    plt.xlabel("time (y)")
    plt.grid()
    plt.title("anelastic induced load signal")
    plt.legend()
    plt.savefig(figure_subpath.joinpath("anelastic_induced_load_signal.png"))
    plt.show(block=False)

    # Trend since first_year_for_trend.
    plt.figure(figsize=(16, 9))
    plt.plot(trend_dates, elastic_signal, label="elastic : trend = " + str(round(number=elastic_trend, ndigits=5)) + "(mm/y)")
    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
        plt.plot(
            trend_dates,
            signals[:, i_degree],
            label=("degree " + str(degree))
            + " : trend difference with elastic = "
            + str(round(number=signal_trends[i_degree], ndigits=5))
            + "(mm/y)",
        )
    plt.legend()
    plt.xlabel("time (y)")
    plt.grid()
    plt.title("anelastic induced load signal : trend since " + str(signal_hyper_parameters.first_year_for_trend))
    plt.legend()
    plt.savefig(figure_subpath.joinpath("anelastic_induced_load_signal_trend.png"))
    plt.show()


if __name__ == "__main__":
    anelastic_induced_load_trend(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_high-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio0.1-1.0"
        ),
        figure_subpath_string=args.subpath if args.subpath else "load_signal",
    )
