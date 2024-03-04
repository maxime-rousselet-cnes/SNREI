# Computes viscoelastic-modified harmonic load signal for a given description and options.
# Gets already computed Love numbers, builds load signal from data, computes viscoelastic induced harmonic load signal and
# saves it.
# Saves the corresponding figures (spatial domain) in the specified subfolder.

import argparse

import matplotlib.pyplot as plt
from numpy import linspace
from pyshtools.expand import MakeGridDH

from utils import (
    SignalHyperParameters,
    build_load_signal,
    figures_path,
    load_base_model,
    parameters_path,
    signal_trend,
    viscoelastic_harmonic_load_signal,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def viscoelastic_harmonic_load_trend(
    real_description_id: str,
    figure_subpath_string: str,
    last_year: int = 2018,
    last_years_for_trend: int = 15,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
) -> None:
    """
    Computes viscoelastic-modified harmonic load signal for a given description and options.
    Gets already computed Love numbers, builds load signal from data, computes viscoelastic induced harmonic load signal and
    saves it.
    Saves the corresponding figures (spatial domain) in the specified subfolder.
    """
    # Builds frequential signal.
    dates, elastic_load_signal, frequencies, frequencial_load_signal, harmonic_weights = build_load_signal(
        signal_hyper_parameters=signal_hyper_parameters, get_harmonic_weights=True
    )

    _, _, _, _, _, load_signal_trends = signal_trend(
        signal_computing=viscoelastic_harmonic_load_signal,
        harmonic_weights=harmonic_weights,
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        elastic_load_signal=elastic_load_signal,
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
    plt.figure()
    spatial_results = MakeGridDH(load_signal_trends, sampling=2)
    plt.colorbar(
        plt.imshow(spatial_results),
        boundaries=linspace(
            start=min([min(row) for row in spatial_results]), stop=max([max(row) for row in spatial_results]), num=10
        ),
    )
    plt.title("Viscoelastic induced loads differences with elastic - trends since " + str(last_year - last_years_for_trend))
    plt.savefig(figure_subpath.joinpath(signal_hyper_parameters.weights_map + "_" + signal_hyper_parameters.signal + ".png"))
    plt.show()


if __name__ == "__main__":
    viscoelastic_harmonic_load_trend(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_low-viscosity-asthenosphere-anelastic-lithosphere_Benjamin-variable-asymptotic_ratio0.1-1.0"
        ),
        figure_subpath_string=args.subpath if args.subpath else "spatial_load_signal",
    )
# TODO. 04: loop on it.
