import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import linspace, log10, ndarray, ones, round

from utils import (
    AttenuationDescription,
    Earth_radius,
    Integration,
    LoveNumbersHyperParameters,
    RealDescription,
    attenuation_descriptions_path,
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
    tau_M_values: list[float] = [1.0 / 12.0, 1.0, 5.0, 20.0, 100.0],
):
    """
    Generates figures of mu_1/Q_0, and real and imaginary parts of mu for several tau_M values.
    """
    # Initializes.
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name="Love_numbers_hyper_parameters", path=parameters_path, base_model_type=LoveNumbersHyperParameters
    )
    Love_numbers_hyper_parameters.load()
    attenuation_description = AttenuationDescription(
        radius_unit=(
            Love_numbers_hyper_parameters.real_description_parameters.radius_unit
            if Love_numbers_hyper_parameters.real_description_parameters.radius_unit
            else Earth_radius
        ),
        real_crust=Love_numbers_hyper_parameters.real_description_parameters.real_crust,
        n_splines_base=10,
        model_filename="Benjamin",
        id="test-tau_M",
        load_description=False,
    )
    results: dict[tuple[float, float, str, int], ndarray] = {}
    x_tabs: dict[int, ndarray] = {}

    # Preprocesses.
    for tau_M_years in tau_M_values:
        # Modifies tau_M value
        attenuation_description.description_layers[0].splines["tau_M"] = (
            attenuation_description.description_layers[0].splines["tau_M"][0],
            tau_M_years * ones(shape=attenuation_description.description_layers[0].splines["tau_M"][1].shape),
            attenuation_description.description_layers[0].splines["tau_M"][2],
        )
        # Saves new description.
        attenuation_description.save(path=attenuation_descriptions_path)
        # Gets corresponding real description.
        real_description = RealDescription(
            id="test-tau_M",
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
            attenuation_description_from_id="test-tau_M",
            load_description=False,
            save=False,
        )
        for frequency in frequencies:
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=False,
                use_attenuation=True,
                bounded_attenuation_functions=True,
            )
            for k_layer, layer in enumerate(integration.description_layers):
                for part in ["real", "imag"]:
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                    results[tau_M_years, frequency, part, k_layer] = layer.evaluate(x=x, variable="mu_" + part)
                    x_tabs[k_layer] = x

    # Plots.
    plt.figure(figsize=(8, 10))
    plot = plt.subplot()
    for k_layer in range(2, len(integration.description_layers)):
        layer = integration.description_layers[k_layer]
        variable_values = real_description.variable_values_per_layer[k_layer]
        x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
        plot.semilogx(
            variable_values["mu_1"] / variable_values["Qmu"] * real_description.elasticity_unit,
            (1.0 - x) * real_description.radius_unit / 1e3,
            color=(0.0, 0.0, 1.0),
        )
    plot.set_xlabel("$\mu_{real} / Q_{mu}$ (Pa)")
    plot.set_ylabel("Depth (km)")
    plot.invert_yaxis()
    plt.savefig(figure_path.joinpath("mu_on_Q.png"))
    plt.show()

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(14, 10), sharex=True)
        for frequency, period, plot in zip(frequencies, period_values, plots):
            for k_layer in range(2, len(integration.description_layers)):
                for i_tau_M, tau_M_years in enumerate(tau_M_values):
                    plot.plot(
                        results[tau_M_years, frequency, part, k_layer] * real_description.elasticity_unit,
                        (1.0 - x_tabs[k_layer]) * real_description.radius_unit / 1e3,
                        color=(i_tau_M / len(tau_M_values), 0.0, 1.0),
                        label="$\\tau _M=$" + str(round(a=tau_M_years, decimals=4)) + " (y)",
                    )
                if k_layer == 2:
                    plot.legend(loc="lower left")
            plot.set_xlabel("$\mu_{" + part + "}$ (Pa)")
            plot.set_ylabel("Depth (km)")
            plot.invert_yaxis()
            plot.grid()
            plot.set_title("$T=" + str(period) + "$ (y)")
        plt.savefig(figure_path.joinpath("mu_" + part + ".png"))
        plt.show()


if __name__ == "__main__":
    plot_mu_profiles(figure_path_string=args.figure_path_string)
