# Creates figures for elastic modulus mu for given options.
#
# Generates 2 figures, with respect to depth:
#   - mu real part, a plot per period and a curb per model.
#   - mu imaginary part, a plot per period and a curb per model.

import argparse
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
from numpy import linspace, log10

from utils import (
    Integration,
    RealDescription,
    figures_path,
    frequencies_to_periods,
    load_Love_numbers_hyper_parameters,
    real_description_from_parameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--initial_real_description_id", type=str, help="wanted ID for the real description to load")
parser.add_argument(
    "--load_initial_description", action="store_true", help="Option to tell if the description should be loaded"
)
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


def plot_mu_profiles_for_descriptions(
    initial_real_description_id: str,
    load_description: bool,
    figure_subpath_string: str,
    elasticity_model_names: Optional[list[str]] = None,
    anelasticity_model_names: Optional[list[str]] = None,
    attenuation_model_names: Optional[list[str]] = None,
    period_values: list[float] = [18.6, 100, 1000],
):
    """
    Generates figures of real and imaginary parts of mu for different models.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    path = figures_path.joinpath(figure_subpath_string).joinpath(initial_real_description_id)
    integrations: dict[tuple[str, str, str], dict[float, Integration]] = {}
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    path.mkdir(parents=True, exist_ok=True)
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=initial_real_description_id,
        load_description=load_description,
        save=False,
    )

    # Builds dummy lists for unmodified models.
    if not elasticity_model_names:
        elasticity_model_names = [initial_real_description.elasticity_model_name]
    if not anelasticity_model_names:
        anelasticity_model_names = [initial_real_description.anelasticity_model_name]
    if not attenuation_model_names:
        attenuation_model_names = [initial_real_description.attenuation_model_name]

    # Preprocesses.
    for elasticity_model_name, anelasticity_model_name, attenuation_model_name in product(
        elasticity_model_names, anelasticity_model_names, attenuation_model_names
    ):
        real_description: RealDescription = real_description_from_parameters(
            Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
            real_description_id=initial_real_description_id,
            load_description=False,
            elasticity_model_from_name=elasticity_model_name,
            anelasticity_model_from_name=anelasticity_model_name,
            attenuation_model_from_name=attenuation_model_name,
            save=False,
        )
        integrations[elasticity_model_name, anelasticity_model_name, attenuation_model_name] = {}
        for frequency in frequencies:
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
                use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
                bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
            )
            integrations[elasticity_model_name, anelasticity_model_name, attenuation_model_name][frequency] = integration

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(16, 12), sharex=True)
        # Iterates on frequencies.
        for frequency, period, plot in zip(frequencies, period_values, plots):
            # Iterates on models.
            for i_model, (elasticity_model_name, anelasticity_model_name, attenuation_model_name) in enumerate(
                product(elasticity_model_names, anelasticity_model_names, attenuation_model_names)
            ):
                layer = integrations[elasticity_model_name, anelasticity_model_name, attenuation_model_name][
                    frequency
                ].description_layers[2]
                x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                plot.plot(
                    layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                    (1.0 - x) * real_description.radius_unit / 1e3,
                    color=(colors[i_model % len(colors)]),
                    label="_".join((elasticity_model_name, anelasticity_model_name, attenuation_model_name)),
                )
                # Iterates on layers.
                for k_layer in range(
                    3,
                    len(
                        integrations[elasticity_model_name, anelasticity_model_name, attenuation_model_name][
                            frequency
                        ].description_layers
                    ),
                ):
                    layer = integrations[elasticity_model_name, anelasticity_model_name, attenuation_model_name][
                        frequency
                    ].description_layers[k_layer]
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                    plot.plot(
                        layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                        (1.0 - x) * real_description.radius_unit / 1e3,
                        color=(colors[i_model % len(colors)]),
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
    plot_mu_profiles_for_descriptions(
        initial_real_description_id=args.initial_real_description_id if args.initial_real_description_id else "base-model",
        load_description=args.load_initial_description if args.load_initial_description else False,
        figure_subpath_string=args.subpath if args.subpath else "mu_for_descriptions",
        anelasticity_model_names=["test", "test-low-viscosity-Asthenosphere"],
    )
