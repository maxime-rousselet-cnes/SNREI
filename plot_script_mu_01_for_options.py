# Creates figures for elastic modulus mu for a given description.
#
# Generates 3 figures, with respect to depth:
#   - mu_0 / Q_0 ratio.
#   - mu real part, a plot per period and a curb per option.
#   - mu imaginary part, a plot per period and a curb per option.

import argparse
from itertools import product

import matplotlib.pyplot as plt
from numpy import linspace, log10

from utils import (
    Integration,
    figures_path,
    frequencies_to_periods,
    load_Love_numbers_hyper_parameters,
    real_description_from_parameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="wanted ID for the real description to load")
parser.add_argument("--load_description", action="store_true", help="Option to tell if the description should be loaded")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def plot_mu_profiles_for_options(
    real_description_id: str,
    load_description: bool,
    figure_subpath_string: str,
    period_values: list[float] = [18.6, 100.0, 1000.0],
):
    """
    Generates figures of mu_0/Q_0, and real and imaginary parts of mu for several period values.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    path = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    booleans = [True, False]
    options_list = list(product(booleans, booleans, booleans))
    integrations: dict[float, dict[tuple[bool, bool, bool], Integration]] = {}
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    path.mkdir(parents=True, exist_ok=True)
    real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=real_description_id,
        load_description=load_description,
        save=False,
    )

    # Preprocesses.
    for frequency in frequencies:
        integrations[frequency] = {}
        for use_anelasticity, use_attenuation, bounded_attenuation_functions in options_list:
            if (not use_attenuation) and bounded_attenuation_functions:
                continue
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
                bounded_attenuation_functions=bounded_attenuation_functions,
            )
            integrations[frequency][use_anelasticity, use_attenuation, bounded_attenuation_functions] = integration

    # Plots mu_0 / Q_0.
    plt.figure(figsize=(8, 10))
    plot = plt.subplot()
    # Iterates on layers.
    for k_layer in range(2, len(integrations[frequency][True, True, True].description_layers)):
        # Iterates on options.
        for use_anelasticity, use_attenuation, bounded_attenuation_functions in options_list:
            if (not use_attenuation) and bounded_attenuation_functions:
                continue
            base_label = (
                "elastic"
                if (not use_anelasticity and not use_attenuation)
                else ("with anelasticity " if use_anelasticity else "")
                + ("with " + ("bounded " if bounded_attenuation_functions else "") + "attenuation" if use_attenuation else "")
            )
            layer = integrations[frequency][
                use_anelasticity, use_attenuation, bounded_attenuation_functions
            ].description_layers[k_layer]
            variable_values = real_description.variable_values_per_layer[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
            plot.semilogx(
                variable_values["mu_0"] / variable_values["Qmu"] * real_description.elasticity_unit,
                (1.0 - x) * real_description.radius_unit / 1e3,
                color=(
                    1.0 if use_anelasticity else 0.0,
                    1.0 if bounded_attenuation_functions else 0.0,
                    1.0 if use_attenuation else 0.0,
                ),
                linestyle=(
                    ":"
                    if (not use_anelasticity and not use_attenuation)
                    else ("-" if use_anelasticity else "") + ("-" if use_attenuation else ".")
                ),
                label=base_label,
            )
        if k_layer == 2:
            plot.legend(loc="lower left")
    plot.set_xlabel("$\mu_{real} / Q_{mu}$ (Pa)")
    plot.set_ylabel("Depth (km)")
    plot.invert_yaxis()
    plt.savefig(path.joinpath("mu_on_Q.png"))
    plt.show()

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(16, 12), sharex=True)
        # Iterates on frequencies.
        for frequency, period, plot in zip(frequencies, period_values, plots):
            # Iterates on layers.
            for k_layer in range(2, len(integrations[frequency][True, True, True].description_layers)):
                # Iterates on options.
                for use_anelasticity, use_attenuation, bounded_attenuation_functions in options_list:
                    if (not use_attenuation) and bounded_attenuation_functions:
                        continue
                    base_label = (
                        "elastic"
                        if (not use_anelasticity and not use_attenuation)
                        else ("with anelasticity " if use_anelasticity else "")
                        + (
                            "with " + ("bounded " if bounded_attenuation_functions else "") + "attenuation"
                            if use_attenuation
                            else ""
                        )
                    )
                    layer = integrations[frequency][
                        use_anelasticity, use_attenuation, bounded_attenuation_functions
                    ].description_layers[k_layer]
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                    plot.plot(
                        layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                        (1.0 - x) * real_description.radius_unit / 1e3,
                        color=(
                            1.0 if use_anelasticity else 0.0,
                            0.5 if bounded_attenuation_functions else 0.0,
                            1.0 if use_attenuation else 0.0,
                        ),
                        linestyle=(
                            ":"
                            if (not use_anelasticity and not use_attenuation)
                            else ("-" if use_anelasticity else "") + ("-" if use_attenuation else ".")
                        ),
                        label=base_label,
                    )
                if k_layer == 2:
                    plot.legend(loc="lower left")
            plot.set_xlabel("$\mu_{" + part + "}$ (Pa)")
            plot.set_ylabel("Depth (km)")
            plot.invert_yaxis()
            plot.grid()
            plot.set_title("$T=" + str(period) + "$ (y)")
        plt.savefig(path.joinpath("mu_" + part + ".png"))
        plt.show()


if __name__ == "__main__":
    plot_mu_profiles_for_options(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_high-viscosity-asthenosphere-elastic-lithosphere_Benjamin"
        ),
        load_description=args.load_description if args.load_description else False,
        figure_subpath_string=args.subpath if args.subpath else "mu_for_options",
    )
