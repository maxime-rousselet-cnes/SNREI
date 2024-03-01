# Computes viscoelastic-modified load signal for a given description and options.
# Gets already computed Love numbers, builds load signal from data, computes viscoelastic induced load signal and saves it.
# Saves the corresponding figures in the specified subfolder.

import argparse

import matplotlib.pyplot as plt

from utils import (
    SignalHyperParameters,
    build_load_signal,
    figures_path,
    load_base_model,
    parameters_path,
    signal_trend,
    viscoelastic_load_signal,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def viscoelastic_load_trend(
    real_description_id: str,
    figure_subpath_string: str,
    last_year: int = 2018,
    last_years_for_trend: int = 15,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
    degrees_to_plot: list[int] = [1, 2, 3, 5, 10, 20],
) -> None:
    """
    Computes viscoelastic-modified load signal for a given description and options.
    Gets already computed Love numbers, builds load signal from data, computes viscoelastic induced load signal and saves it.
    Saves the corresponding figures in the specified subfolder.
    """
    # Builds frequential signal.
    dates, frequencies, frequencial_load_signal, _ = build_load_signal(
        signal_hyper_parameters=signal_hyper_parameters, get_harmonic_weights=False
    )

    _, degrees, load_signal, last_years_dates, load_signal_last_years, load_signal_trends = signal_trend(
        signal_computing=viscoelastic_load_signal,
        harmonic_weights=None,
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        frequencies=frequencies,
        frequencial_load_signal=frequencial_load_signal,
        dates=dates,
        last_year=last_year,
        last_years_for_trend=last_years_for_trend,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Results.
    degrees_indices = [list(degrees).index(degree) for degree in degrees_to_plot]
    plt.figure(figsize=(16, 10))
    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
        plt.plot(
            dates,
            load_signal[:, i_degree],
            label="elastic" if i_degree == 0 else "degree " + str(degree),
        )
    plt.legend()
    plt.xlabel("time (y)")
    plt.grid()
    plt.title("viscoelastic induced load signal")
    plt.legend()
    plt.savefig(figure_subpath.joinpath("viscoelastic_induced_load_signal.png"))
    plt.show(block=False)

    # Trend since last_year - last_years_for_trend.
    plt.figure(figsize=(16, 10))
    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
        plt.plot(
            last_years_dates,
            load_signal_last_years[:, i_degree],
            label=("elastic" if i_degree == 0 else "degree " + str(degree))
            + " - trend = "
            + str(load_signal_trends[i_degree])
            + "(mm/y)",
        )
    plt.legend()
    plt.xlabel("time (y)")
    plt.grid()
    plt.title("viscoelastic induced load signal - trend since " + str(last_year - last_years_for_trend))
    plt.legend()
    plt.savefig(figure_subpath.joinpath("viscoelastic_induced_load_signal_trend.png"))
    plt.show()


if __name__ == "__main__":
    viscoelastic_load_trend(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_high-viscosity-asthenosphere-anelastic-lithosphere_Benjamin-variable-asymptotic_ratio0.1-1.0"
        ),
        figure_subpath_string=args.subpath if args.subpath else "load_signal",
    )
