# Computes anelastic-modified load signal by iterating on:
#   - models: A real description is used per triplet of:
#       - 'elasticity_model_name'
#       - 'anelasticity_model_name'
#       - 'attenuation_model_name'
#   - asymptotic_ratios, when the options allow it.
# Gets already computed Love numbers, builds load signal from data, computes anelastic induced load signal and saves it.
# Saves the corresponding figures in the specified subfolder.

import argparse
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt

from utils import (
    SignalHyperParameters,
    anelastic_load_signal,
    build_load_signal,
    figures_path,
    gets_id_asymptotic_ratios,
    id_from_model_names,
    load_base_model,
    load_Love_numbers_hyper_parameters,
    parameters_path,
    real_description_from_parameters,
    signal_trend,
)

# TODO.

parser = argparse.ArgumentParser()
parser.add_argument("--initial_real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def anelastic_load_trend_for_asymptotic_ratios(
    initial_real_description_id: str,
    figure_subpath_string: str,
    asymptotic_ratios: list[list[float]],
    elasticity_model_names: Optional[list[str]] = None,
    anelasticity_model_names: Optional[list[str]] = None,
    attenuation_model_names: Optional[list[str]] = None,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
    degrees_to_plot: list[int] = [1, 2, 3, 5, 10, 20],
) -> None:
    """
    Computes anelastic-modified load signal by iterating on:
        - models: A real description is used per triplet of:
            - 'elasticity_model_name'
            - 'anelasticity_model_name'
            - 'attenuation_model_name'
        - asymptotic_ratios, when the options allow it.
    Gets already computed Love numbers, builds load signal from data, computes anelastic induced load signal and saves it.
    Saves the corresponding figures in the specified subfolder.
    """
    # Builds frequential signal.
    dates, elastic_load_signal, frequencies, frequencial_load_signal, _ = build_load_signal(
        signal_hyper_parameters=signal_hyper_parameters, get_harmonic_weights=False
    )

    # Gets description.
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=load_Love_numbers_hyper_parameters(),
        real_description_id=initial_real_description_id,
        load_description=False,
        save=False,
    )

    # Builds dummy lists for unmodified models.
    if not elasticity_model_names:
        elasticity_model_names = [initial_real_description.elasticity_model_name]
    if not anelasticity_model_names:
        anelasticity_model_names = [initial_real_description.anelasticity_model_name]
    if not attenuation_model_names:
        attenuation_model_names = [initial_real_description.attenuation_model_name]
    dummy_ratios = [asymptotic_ratios[0]]

    # Loops on model files.
    for elasticity_model_name, anelasticity_model_name, attenuation_model_name in product(
        elasticity_model_names, anelasticity_model_names, attenuation_model_names
    ):
        # Loops on asymptotic_ratio.
        for asymptotic_ratios_per_layer in (
            asymptotic_ratios if signal_hyper_parameters.bounded_attenuation_functions else dummy_ratios
        ):

            real_description_id = gets_id_asymptotic_ratios(
                real_description_id=id_from_model_names(
                    id="",
                    real_description=initial_real_description,
                    elasticity_model_name=elasticity_model_name,
                    anelasticity_model_name=anelasticity_model_name,
                    attenuation_model_name=attenuation_model_name + "-variable-asymptotic_ratio",
                ),
                asymptotic_ratios_per_layer=asymptotic_ratios_per_layer,
            )

            # Gets anelastic induced signal.
            _, degrees, load_signal, last_years_dates, load_signal_last_years, load_signal_trends = signal_trend(
                signal_computing=anelastic_load_signal,
                harmonic_weights=None,
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
            degrees_indices = [list(degrees).index(degree) for degree in degrees_to_plot]
            plt.figure(figsize=(16, 9))
            for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                plt.plot(
                    dates,
                    load_signal[:, i_degree],
                    label="elastic" if i_degree == 0 else "degree " + str(degree),
                )
            plt.legend()
            plt.xlabel("time (y)")
            plt.grid()
            plt.title("anelastic induced load signal")
            plt.legend()
            plt.savefig(figure_subpath.joinpath("anelastic_induced_load_signal.png"))
            plt.close()

            # Trend since last_year - last_years_for_trend.
            plt.figure(figsize=(16, 9))
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
            plt.title("anelastic induced load signal - trend since " + str(last_year - last_years_for_trend))
            plt.legend()
            plt.savefig(figure_subpath.joinpath("anelastic_induced_load_signal_trend.png"))
            plt.close()


if __name__ == "__main__":
    anelastic_load_trend_for_asymptotic_ratios(
        initial_real_description_id=(
            args.initial_real_description_id
            if args.initial_real_description_id
            else "PREM_high-viscosity-asthenosphere-anelastic-lithosphere_Benjamin"
        ),
        figure_subpath_string=args.subpath if args.subpath else "load_signal",
        asymptotic_ratios=[[1.0, 1.0], [0.5, 1.0], [0.2, 1.0], [0.1, 1.0], [0.05, 1.0]],
        anelasticity_model_names=[
            "high-viscosity-asthenosphere-anelastic-lithosphere",
            "low-viscosity-asthenosphere-anelastic-lithosphere",
            "high-viscosity-asthenosphere-elastic-lithosphere",
            "low-viscosity-asthenosphere-elastic-lithosphere",
        ],
    )
