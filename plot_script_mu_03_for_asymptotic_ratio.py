# Creates figures for elastic modulus mu for given asymptotic ratios.
#
# Generates 2 figures, with respect to depth:
#   - mu real part, a plot per period and a curb per asymptotic ratio.
#   - mu imaginary part, a plot per period and a curb per asymptotic ratio.

import argparse

import matplotlib.pyplot as plt
from numpy import linspace, log10

from utils import (
    Integration,
    Model,
    RealDescription,
    attenuation_models_path,
    figures_path,
    frequencies_to_periods,
    load_base_model,
    load_Love_numbers_hyper_parameters,
    real_description_from_parameters,
    save_base_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--initial_real_description_id", type=str, help="wanted ID for the real description to load")
parser.add_argument(
    "--load_initial_description", action="store_true", help="Option to tell if the description should be loaded"
)
parser.add_argument("--with_anelasticity", action="store_true", help="Option to tell if the description should be loaded")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


def plot_mu_profiles_for_asymptotic_ratio(
    initial_real_description_id: str,
    load_description: bool,
    with_anelasticity: bool,
    figure_subpath_string: str,
    asymptotic_ratio_values: list[list[float]] = [[1.0, 1.0], [0.5, 1.0], [0.2, 1.0], [0.1, 1.0], [0.05, 1.0]],
    period_values: list[float] = [18.6, 100, 1000, 10000],
):
    """
    Generates figures of real and imaginary parts of mu for different models.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    Love_numbers_hyper_parameters.use_anelasticity = with_anelasticity
    path = figures_path.joinpath(figure_subpath_string).joinpath(initial_real_description_id)
    integrations: dict[int, dict[float, Integration]] = {}
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    path.mkdir(parents=True, exist_ok=True)
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=initial_real_description_id,
        load_description=load_description,
        save=False,
    )
    attenuation_model: Model = load_base_model(
        name=initial_real_description.attenuation_model_name, path=attenuation_models_path, base_model_type=Model
    )
    temp_name_attenuation_model = initial_real_description.attenuation_model_name + "-variable-asymptotic_ratio"

    # Preprocesses.
    for i_ratio, asymptotic_ratio_values_per_layer in enumerate(asymptotic_ratio_values):
        for k_layer, asymptotic_ratio in enumerate(asymptotic_ratio_values_per_layer):
            attenuation_model.polynomials["tau_M"][k_layer][0] = 0.0
            attenuation_model.polynomials["asymptotic_attenuation"][k_layer][0] = 1.0 - asymptotic_ratio
        save_base_model(obj=attenuation_model, name=temp_name_attenuation_model, path=attenuation_models_path)
        real_description: RealDescription = real_description_from_parameters(
            Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
            real_description_id=initial_real_description_id,
            load_description=False,
            attenuation_model_from_name=temp_name_attenuation_model,
            save=True,
        )
        integrations[i_ratio] = {}
        for frequency in frequencies:
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
                use_attenuation=True,
                bounded_attenuation_functions=True,
            )
            integrations[i_ratio][frequency] = integration

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(16, 12), sharex=True)
        # Iterates on frequencies.
        for frequency, period, plot in zip(frequencies, period_values, plots):
            # Iterates on models.
            for i_ratio, asymptotic_ratio_values_per_layer in enumerate(asymptotic_ratio_values):
                layer = integrations[i_ratio][frequency].description_layers[2]
                x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                plot.plot(
                    layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                    (1.0 - x) * real_description.radius_unit / 1e3,
                    color=(colors[i_ratio % len(colors)]),
                    label="_".join([str(asymptotic_ratio) for asymptotic_ratio in asymptotic_ratio_values_per_layer]),
                )
                # Iterates on layers.
                for k_layer in range(
                    3,
                    len(integrations[i_ratio][frequency].description_layers),
                ):
                    layer = integrations[i_ratio][frequency].description_layers[k_layer]
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                    plot.plot(
                        layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                        (1.0 - x) * real_description.radius_unit / 1e3,
                        color=(colors[i_ratio % len(colors)]),
                    )
            plot.legend(loc="lower left")
            plot.set_xlabel("$\mu_{" + part + "}$ (Pa)")
            plot.set_ylabel("Depth (km)")
            plot.invert_yaxis()
            plot.grid()
            plot.set_title("$T=" + str(period) + "$ (y)")
        plt.savefig(path.joinpath("mu_" + part + ".png"))
        plt.show()


if __name__ == "__main__":
    plot_mu_profiles_for_asymptotic_ratio(
        initial_real_description_id=args.initial_real_description_id if args.initial_real_description_id else "base-model",
        load_description=args.load_initial_description if args.load_initial_description else False,
        with_anelasticity=args.with_anelasticity if args.with_anelasticity else False,
        figure_subpath_string=args.subpath if args.subpath else "mu_for_asymptotic_ratio",
    )
