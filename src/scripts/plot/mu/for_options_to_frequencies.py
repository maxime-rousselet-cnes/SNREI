# Creates figures for elastic modulus mu for a given description.
#
# Generates 3 figures, with respect to depth:
#   - mu_0 / Q_0 ratio.
#   - mu real part, a plot per period and a curb per option.
#   - mu imaginary part, a plot per period and a curb per option.

import argparse
from itertools import product

import matplotlib.pyplot as plt
from numpy import linspace, log10, zeros

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
    T_min: float = 4e-5,  # (y).
    T_max: float = 3e5,  # (y).
    n_points: int = 100,
):
    """
    Generates figures of mu/mu_0, and real and imaginary parts of mu for several options.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    path = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    booleans = [True, False]
    options_list = list(product(booleans, booleans))
    path.mkdir(parents=True, exist_ok=True)
    real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=real_description_id,
        load_description=load_description,
        save=False,
    )
    periods = 10 ** linspace(log10(T_min), log10(T_max), n_points)
    frequencies = frequencies_to_periods(periods)

    _, plots = plt.subplots(2, 1, figsize=(18.3, 9), sharex=True)
    for use_anelasticity, use_attenuation in options_list:
        if (not use_anelasticity) and (not use_attenuation):
            continue
        result_tab = {
            "real": zeros(shape=frequencies.shape, dtype=float),
            "imag": zeros(shape=frequencies.shape, dtype=float),
        }
        for i_f, frequency in enumerate(frequencies):
            # Preprocesses.
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
                bounded_attenuation_functions=use_attenuation,
            )
            mu_real = integration.description_layers[integration.below_CMB_layers].evaluate(
                x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_real"
            ) / integration.description_layers[integration.below_CMB_layers].evaluate(
                x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_0"
            )
            mu_imag = integration.description_layers[integration.below_CMB_layers].evaluate(
                x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_imag"
            ) / integration.description_layers[integration.below_CMB_layers].evaluate(
                x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_0"
            )
            result_tab["real"][i_f] = mu_real / (mu_real**2 + mu_imag**2)
            result_tab["imag"][i_f] = -mu_imag / (mu_real**2 + mu_imag**2)
        # Plots mu_real and mu_imag.
        for part in ["real", "imag"]:
            plot = plots[0 if part == "real" else 1]
            base_label = ("long-term anelasticity " if use_anelasticity else "") + (
                "transient regime" if use_attenuation else ""
            )
            plot.plot(
                periods,
                result_tab[part],
                color=(
                    0.0,
                    0.0,
                    1.0,
                ),
                linestyle=(":" if (use_anelasticity and not use_attenuation) else ("-." if use_anelasticity else "--")),
                label=base_label,
            )
            if part == "imag":
                plot.legend(loc="upper left")
            plot.set_ylabel(part + " part")
            # plot.set_xscale("log")
            plot.set_xlabel("Period (y)")
            plot.grid()
    plots[0].set_title("$\mu_0 / \mu$")
    plots[0].set_xscale("log")
    plots[1].set_yscale("symlog")
    plt.savefig(path.joinpath("mu_0_on_mu.png"))
    plt.show()


if __name__ == "__main__":
    plot_mu_profiles_for_options(
        real_description_id=(args.real_description_id if args.real_description_id else "PREM-constant-mu-Q30_uniform_uniform"),
        load_description=args.load_description if args.load_description else False,
        figure_subpath_string=args.subpath if args.subpath else "mu_for_options",
    )
