import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import linspace, log10, ndarray

from utils import (
    Earth_radius,
    Integration,
    LoveNumbersHyperParameters,
    RealDescription,
    frequencies_to_periods,
    load_base_model,
    parameters_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--figure_path_string", type=str, required=True, help="wanted path to save figure")

args = parser.parse_args()


def plot_mu_profiles(
    figure_path_string: str,
    period_values: list[float] = [18.6, 100, 1000],
):
    """
    Generates figures of mu_1/Q_0, and real and imaginary parts of mu with and without low viscosity Asthenosphere model and
    with and without bounded attenuation functions.
    """
    # Initializes.
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name="Love_numbers_hyper_parameters", path=parameters_path, base_model_type=LoveNumbersHyperParameters
    )
    Love_numbers_hyper_parameters.load()
    results: dict[tuple[bool, bool, float, str], dict[int, ndarray]] = {}
    x_tabs: dict[tuple[bool, int], ndarray] = {}

    # Preprocesses.
    for use_Asthenosphere_model, bounded_attenuation_functions in product([False, True], [False, True]):
        # Gets corresponding real description.
        real_description = RealDescription(
            id="test-model",
            below_ICB_layers=Love_numbers_hyper_parameters.real_description_parameters.below_ICB_layers,
            below_CMB_layers=Love_numbers_hyper_parameters.real_description_parameters.below_CMB_layers,
            splines_degree=Love_numbers_hyper_parameters.real_description_parameters.splines_degree,
            radius_unit=(
                Love_numbers_hyper_parameters.real_description_parameters.radius_unit
                if Love_numbers_hyper_parameters.real_description_parameters.radius_unit
                else Earth_radius
            ),
            real_crust=Love_numbers_hyper_parameters.real_description_parameters.real_crust,
            n_splines_base=10,
            profile_precision=10,
            radius=(
                Love_numbers_hyper_parameters.real_description_parameters.radius_unit
                if Love_numbers_hyper_parameters.real_description_parameters.radius_unit
                else Earth_radius
            ),
            anelasticity_model_from_name="test-low-viscosity-Asthenosphere" if use_Asthenosphere_model else "test",
            load_description=False,
            save=False,
        )
        for frequency in frequencies:
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=True,
                use_attenuation=True,
                bounded_attenuation_functions=bounded_attenuation_functions,
            )
            for k_layer, layer in enumerate(integration.description_layers):
                x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                for part in ["real", "imag"]:
                    if k_layer == 0:
                        results[use_Asthenosphere_model, bounded_attenuation_functions, frequency, part] = {}
                    results[use_Asthenosphere_model, bounded_attenuation_functions, frequency, part][k_layer] = layer.evaluate(
                        x=x, variable="mu_" + part
                    )
                x_tabs[use_Asthenosphere_model, k_layer] = x

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(14, 10), sharex=True)
        for frequency, period, plot in zip(frequencies, period_values, plots):
            for use_Asthenosphere_model, bounded_attenuation_functions in product([True, False], [True, False]):
                color = (
                    0.0 if bounded_attenuation_functions else 1.0,
                    0.0,
                    0.0 if use_Asthenosphere_model else 1.0,
                )
                linestyle = (
                    ":"
                    if (use_Asthenosphere_model and bounded_attenuation_functions)
                    else ("" if use_Asthenosphere_model else "-") + ("." if bounded_attenuation_functions else "-")
                )
                plot.plot(
                    results[use_Asthenosphere_model, bounded_attenuation_functions, frequency, part][2]
                    * real_description.elasticity_unit,
                    (1.0 - x_tabs[use_Asthenosphere_model, 2]) * real_description.radius_unit / 1e3,
                    color=color,
                    linestyle=linestyle,
                    label=(
                        "base model"
                        if (not use_Asthenosphere_model) and (not bounded_attenuation_functions)
                        else (
                            ("with LV Asth. " if use_Asthenosphere_model else "")
                            + ("with bounded f" if bounded_attenuation_functions else "")
                        )
                    ),
                )
                plot.legend(loc="lower left")
                for k_layer in results[use_Asthenosphere_model, bounded_attenuation_functions, frequency, part].keys():
                    if k_layer < 3:
                        continue
                    plot.plot(
                        results[use_Asthenosphere_model, bounded_attenuation_functions, frequency, part][k_layer]
                        * real_description.elasticity_unit,
                        (1.0 - x_tabs[use_Asthenosphere_model, k_layer]) * real_description.radius_unit / 1e3,
                        color=color,
                        linestyle=linestyle,
                    )
            plot.set_xlabel("$\mu_{" + part + "}$ (Pa)")
            plot.set_ylabel("Depth (km)")
            plot.invert_yaxis()
            plot.grid()
            plot.set_title("$T=" + str(period) + "$ (y)")
        plt.savefig(figure_path.joinpath("mu_" + part + ".png"))
        plt.show()


if __name__ == "__main__":
    plot_mu_profiles(figure_path_string=args.figure_path_string)
