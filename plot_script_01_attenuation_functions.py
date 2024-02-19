# Creates figures for attenuation functions f_r and f_i.
#
# Generates 3 figures per layer of interest:
#   - unbounded f_r and f_i.
#   - bounded f_r and f_i for different tau_M values
#   - bounded f_r and f_i for different asymptotic_ratio values.

import argparse

import matplotlib.pyplot as plt
from numpy import linspace, log10, ndarray, round

from utils import (
    SECONDS_PER_YEAR,
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
parser.add_argument("--attenuation_model", type=str, help="wanted ID for the real description to load")
parser.add_argument("--load_description", action="store_true", help="Option to tell if the description should be loaded")
parser.add_argument("--subpath", type=str, help="wanted sub-path to save figures")
args = parser.parse_args()


def plot_attenuation_functions(
    attenuation_model_name: str,
    figure_subpath_string: str,
    tau_M_years_values: dict[int, list[float]] = {0: [1.0 / 12, 1.0, 5.0, 20.0, 100.0, 500.0, 1000.0]},
    asymptotic_ratio_values: dict[int, list[float]] = {0: [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]},
):
    """
    Generates plots of attenuation functions f_r and f_i. A figure for unbounded attenuation functions, a second one for
    bounded attenuation functions with given tau_M values and a third one for bounded attenuation functions with given
    asymptotic_ratio values.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    Love_numbers_hyper_parameters.use_attenuation = True
    Love_numbers_hyper_parameters.bounded_attenuation_functions = False
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=None,
        load_description=False,
        attenuation_model_from_name=attenuation_model_name,
        save=False,
    )
    attenuation_model: Model = load_base_model(
        name=initial_real_description.attenuation_model_name, path=attenuation_models_path, base_model_type=Model
    )
    temp_name_attenuation_model = initial_real_description.attenuation_model_name + "-variable-tau_M"
    path = figures_path.joinpath(figure_subpath_string).joinpath(initial_real_description.attenuation_model_name)
    T_tab = 10 ** linspace(
        0,
        2
        + log10(
            SECONDS_PER_YEAR * max([max(tau_M_years_list_values) for _, tau_M_years_list_values in tau_M_years_values.items()])
        ),
        100,
    )  # (s).

    # Iterates on layers on interest.
    for k_layer, tau_M_years_list_values in tau_M_years_values.items():
        # Creates directory for layer.
        layer_name = attenuation_model.layer_names[k_layer]
        subpath = path.joinpath(layer_name)
        subpath.mkdir(parents=True, exist_ok=True)
        k_layer_real_description = [
            description_layer.name.split(sep="-")[-1] for description_layer in initial_real_description.description_layers
        ].index(layer_name)

        # Gets unbounded f.
        f_r_unbounded, f_i_unbounded = get_attenuation_function(
            T_tab=T_tab,
            real_description=initial_real_description,
            use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
            bounded_attenuation_functions=False,
            k_layer=k_layer_real_description,
        )

        # Plots unbounded f.
        plt.plot(T_tab, f_r_unbounded, label="$f_r$")
        plt.plot(T_tab, f_i_unbounded, label="$f_i$")
        plt.legend()
        plt.xscale("log")
        plt.xlabel("Period (s)")
        plt.savefig(subpath.joinpath("unbounded_attenuation_functions.png"))
        plt.show()

        # Iterates on tau_M values.
        _, plots = plt.subplots(2, 1, sharex=True, figsize=(8, 11))
        for tau_M_years in tau_M_years_list_values:
            # Modifies tau_M value in model.
            tau_M = 1.0 / frequencies_to_periods(frequencies=tau_M_years)
            attenuation_model.polynomials["tau_M"][k_layer][0] = tau_M_years
            attenuation_model.polynomials["asymptotic_attenuation"][k_layer][0] = 0.0
            save_base_model(obj=attenuation_model, name=temp_name_attenuation_model, path=attenuation_models_path)
            real_description = real_description_from_parameters(
                Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                real_description_id=None,
                load_description=False,
                attenuation_model_from_name=temp_name_attenuation_model,
                save=False,
            )

            # Gets bounded f.
            f_r, f_i = get_attenuation_function(
                T_tab=T_tab,
                real_description=real_description,
                use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
                bounded_attenuation_functions=True,
                k_layer=k_layer_real_description,
            )

            plots[0].plot(T_tab, f_r)
            plots[1].plot(T_tab, f_i, label="$\\tau _M=$" + str(round(a=tau_M_years, decimals=4)) + " (y)")
            plots[0].scatter([tau_M] * 15, linspace(start=min(f_r), stop=max(f_r), num=15), s=5)
            plots[1].scatter([tau_M] * 15, linspace(start=min(f_i), stop=max(f_i), num=15), s=5)

        plots[0].plot(T_tab, f_r_unbounded)
        plots[1].plot(T_tab, f_i_unbounded, label="unbounded")

        plots[0].set_ylabel("$f_r$")
        plots[0].grid()

        plots[1].set_ylabel("$f_i$")
        plots[1].grid()

        plt.xlabel("Period (s)")
        plt.xscale("log")
        plt.legend()
        plt.savefig(subpath.joinpath("bounded_attenuation_functions_for_tau_M.png"))
        plt.show()

        # Iterates on asymptotic_ratio values.
        _, plots = plt.subplots(2, 1, sharex=True, figsize=(12, 15))
        for asymptotic_ratio in asymptotic_ratio_values[k_layer]:
            # Modifies tau_M value in model.
            attenuation_model.polynomials["tau_M"][k_layer][0] = 0.0
            attenuation_model.polynomials["asymptotic_attenuation"][k_layer][0] = 1.0 - asymptotic_ratio
            save_base_model(obj=attenuation_model, name=temp_name_attenuation_model, path=attenuation_models_path)
            real_description = real_description_from_parameters(
                Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                real_description_id=None,
                load_description=False,
                attenuation_model_from_name=temp_name_attenuation_model,
                save=False,
            )

            # Gets bounded f.
            f_r, f_i = get_attenuation_function(
                T_tab=T_tab,
                real_description=real_description,
                use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
                bounded_attenuation_functions=True,
                k_layer=k_layer_real_description,
            )

            plots[0].plot(T_tab, f_r)
            plots[1].plot(
                T_tab,
                f_i,
                label="$\\mu_\\infty / \\mu_0=$"
                + str(asymptotic_ratio)
                + " : $\\tau _M=$"
                + str(
                    round(
                        a=real_description.description_layers[k_layer_real_description].evaluate(
                            x=attenuation_model.r_limits[k_layer_real_description] / real_description.radius_unit,
                            variable="tau_M",
                        )
                        / SECONDS_PER_YEAR
                        * real_description.period_unit,
                        decimals=2,
                    )
                )
                + " (y)",
            )

        plots[0].plot(T_tab, f_r_unbounded)
        plots[1].plot(T_tab, f_i_unbounded, label="unbounded")

        plots[0].set_ylabel("$f_r$")
        plots[0].grid()

        plots[1].set_ylabel("$f_i$")
        plots[1].grid()

        plt.xlabel("Period (s)")
        plt.xscale("log")
        plt.legend()
        plt.savefig(subpath.joinpath("bounded_attenuation_functions_for_ratio.png"))
        plt.show()


def get_attenuation_function(
    T_tab: ndarray, real_description: RealDescription, use_anelasticity: bool, bounded_attenuation_functions: bool, k_layer: int
) -> tuple[ndarray, ndarray]:
    """
    Evaluates attenuation functions of a given description at every frequency of a given list.
    """
    f_r, f_i = [], []
    for T_value in T_tab:
        integration = Integration(
            real_description=real_description,
            log_frequency=log10(real_description.period_unit / T_value),
            use_anelasticity=use_anelasticity,
            use_attenuation=True,
            bounded_attenuation_functions=bounded_attenuation_functions,
        )
        f_r += [
            integration.description_layers[k_layer].evaluate(x=integration.description_layers[k_layer].x_inf, variable="f_r")
        ]
        f_i += [
            integration.description_layers[k_layer].evaluate(x=integration.description_layers[k_layer].x_inf, variable="f_i")
        ]
    return f_r, f_i


if __name__ == "__main__":
    plot_attenuation_functions(
        attenuation_model_name=args.attenuation_model,
        figure_subpath_string=args.subpath if args.subpath else "attenuation_functions",
    )
