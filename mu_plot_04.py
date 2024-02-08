import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import linspace, log10

from utils import (
    Earth_radius,
    Integration,
    RealDescription,
    frequencies_to_periods,
    real_descriptions_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("real_description_id", type=str, required=True, help="wanted ID for the real description to load")
parser.add_argument("figure_path_string", type=str, required=True, help="wanted path to save figure")

args = parser.parse_args()


def plot_mu_profiles(
    real_description_id: str,
    figure_path_string: str,
    period_values: list[int] = [18.6, 100, 1000],
):
    """
    Generates figures of mu_1/Q_0, and real and imaginary parts of mu for several period values.
    """
    # Initializes.
    booleans = [True, False]
    real_description = RealDescription(
        id=real_description_id,
        below_ICB_layers=1,
        below_CMB_layers=2,
        splines_degree=1,
        radius_unit=Earth_radius,
        real_crust=False,
        n_splines_base=10,
        profile_precision=10000,
        radius=Earth_radius,
    )
    options_list = list(product(booleans, booleans))
    integrations: dict[float, dict[tuple[bool, bool], Integration]] = {}
    real_description.load(path=real_descriptions_path)
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)

    # Preprocesses.
    for frequency in frequencies:
        integrations[frequency]: dict[tuple[bool, bool], Integration] = {}
        for use_anelasticity, use_attenuation in options_list:
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
            )
            integrations[frequency][use_anelasticity, use_attenuation] = integration

    # Plots.
    plt.figure(figsize=(8, 10))
    plot = plt.subplot()
    for k_layer in range(2, len(integrations[frequency][True, True].description_layers)):
        for use_anelasticity, use_attenuation in options_list:
            base_label = (
                "elastic"
                if (not use_anelasticity and not use_attenuation)
                else ("with anelasticity " if use_anelasticity else "") + ("with attenuation" if use_attenuation else "")
            )
            layer = integrations[frequency][use_anelasticity, use_attenuation].description_layers[k_layer]
            variable_values = real_description.variable_values_per_layer[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
            plot.semilogx(
                variable_values["mu_1"] / variable_values["Qmu"] * real_description.elasticity_unit,
                (1.0 - x) * real_description.radius_unit / 1e3,
                color=(1.0 if use_anelasticity else 0.0, 0.0, 1.0 if use_attenuation else 0.0),
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
    plt.savefig(figure_path.joinpath("mu_on_Q.png"))
    plt.show()

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(14, 10), sharex=True)
        for frequency, period, plot in zip(frequencies, period_values, plots):
            for k_layer in range(2, len(integrations[frequency][True, True].description_layers)):
                for use_anelasticity, use_attenuation in options_list:
                    base_label = (
                        "elastic"
                        if (not use_anelasticity and not use_attenuation)
                        else ("with anelasticity " if use_anelasticity else "")
                        + ("with attenuation" if use_attenuation else "")
                    )
                    layer = integrations[frequency][use_anelasticity, use_attenuation].description_layers[k_layer]
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                    plot.plot(
                        layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                        (1.0 - x) * real_description.radius_unit / 1e3,
                        color=(1.0 if use_anelasticity else 0.0, 0.0, 1.0 if use_attenuation else 0.0),
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
            plot.set_title("$T=" + str(period) + "$ (y)")
        plt.savefig(figure_path.joinpath("mu_" + part + ".png"))
        plt.show()


if __name__ == "__main__":
    plot_mu_profiles(real_description_id=args.real_description_id, figure_path_string=args.figure_path_string)
