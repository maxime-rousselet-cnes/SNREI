import matplotlib.pyplot as plt
from numpy import linspace, log10, ndarray, round

from ...utils import (
    SECONDS_PER_YEAR,
    AnelasticityDescription,
    Integration,
    Model,
    ModelPart,
    RunHyperParameters,
    create_model_variation,
    figures_path,
    find_tau_M,
    frequencies_to_periods,
    load_base_model,
    load_Love_numbers_hyper_parameters,
    models_path,
)


def period_abscissa_values(n_points_period: int, tau_M_years_values: list[float]) -> ndarray[float]:
    """
    Creates an array of period values (s) convenient for bounded attenuation functions plotting.
    """
    return 10 ** linspace(
        0,
        2 + log10(SECONDS_PER_YEAR * max(tau_M_years_values)),
        n_points_period,
    )


def plot_attenuation_functions(
    short_term_anelasticity_model_name: str,
    figure_subpath_string: str = "attenuation_functions",
    tau_M_years_values: dict[int, list[float]] = [1.0 / 12, 1.0, 5.0, 20.0, 100.0, 500.0, 1000.0],
    asymptotic_mu_ratio_values: dict[int, list[float]] = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
    n_points_period: int = 50,
    above_CMB_layer_index: int = 0,
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
):
    """
    Generates plots of attenuation functions f_r and f_i for a given short-term anelasticity model.
    2 figures are generated:
        - a figure that includes a curb per tau_M value.
        - a figure that includes a curb per asymptotic_mu_ratio value.
    Each figure includes real/imaginary part and unbounded version of attenuation functions f_i and f_r.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    Love_numbers_hyper_parameters.run_hyper_parameters.use_short_term_anelasticity = True
    figures_subpath = figures_path.joinpath(figure_subpath_string).joinpath(short_term_anelasticity_model_name)
    short_term_anelasticity_model: Model = load_base_model(
        name=short_term_anelasticity_model_name, path=models_path[ModelPart.short_term_anelasticity], base_model_type=Model
    )
    tau_M_years = {
        "tau_M": tau_M_years_values,
        "asymptotic_mu_ratio": [
            find_tau_M(
                omega_m=short_term_anelasticity_model.polynomials["omega_m"][above_CMB_layer_index][0],
                alpha=short_term_anelasticity_model.polynomials["alpha"][above_CMB_layer_index][0],
                asymptotic_mu_ratio=asymptotic_mu_ratio,
                Q_mu=short_term_anelasticity_model.polynomials["Q_mu"][above_CMB_layer_index][0],
            )
            for asymptotic_mu_ratio in asymptotic_mu_ratio_values
        ],
    }
    periods = {
        "tau_M": period_abscissa_values(n_points_period=n_points_period, tau_M_years_values=tau_M_years_values),
        "asymptotic_mu_ratio": period_abscissa_values(
            n_points_period=n_points_period,
            tau_M_years_values=tau_M_years["asymptotic_mu_ratio"],
        ),
    }  # (s).

    # Loops on constrained variable.
    for variable, constrained_values in zip(["tau_M", "asymptotic_mu_ratio"], [tau_M_years_values, asymptotic_mu_ratio_values]):
        # Iterates on tau_M values.
        _, plots = plt.subplots(2, 1, sharex=True, figsize=figsize)
        Love_numbers_hyper_parameters.run_hyper_parameters.use_bounded_attenuation_functions = True
        for tau_M_years, variable_value in zip(tau_M_years[variable], constrained_values):
            # Modifies tau_M value in model.
            model: Model = load_base_model(
                name=short_term_anelasticity_model_name,
                path=models_path[ModelPart.short_term_anelasticity],
                base_model_type=Model,
            )
            create_model_variation(
                model_part=ModelPart.short_term_anelasticity,
                base_model=model,
                base_model_name=short_term_anelasticity_model_name,
                parameter_values_per_layer=[
                    (variable, layer_name, [variable_value]) for layer_name in short_term_anelasticity_model.layer_names
                ],
                create=True,
            )
            anelasticity_description = AnelasticityDescription(
                anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
                short_term_anelasticity_model_name=short_term_anelasticity_model_name,
                save=False,
            )
            # Gets bounded f.
            f_r, f_i = get_attenuation_function(
                periods=periods[variable],
                anelasticity_description=anelasticity_description,
                run_hyper_parameters=Love_numbers_hyper_parameters.run_hyper_parameters,
                above_CMB_layer_index=above_CMB_layer_index,
            )
            # Plots.
            tau_M_seconds = 1.0 / frequencies_to_periods(frequencies=tau_M_years)
            plots[0].plot(
                periods[variable],
                f_r,
                linewidth=linewidth,
            )
            plots[1].plot(
                periods[variable],
                f_i,
                label=("" if variable == "tau_M" else "$\\mu(\\omega=0)/\\mu_0=$" + str(round(a=variable_value, decimals=4)))
                + "$\\tau _M=$"
                + str(round(a=tau_M_years, decimals=4))
                + " (y)",
                linewidth=linewidth,
            )
            plots[0].scatter([tau_M_seconds] * 15, linspace(start=min(f_r), stop=max(f_r), num=15), s=5)
            plots[1].scatter([tau_M_seconds] * 15, linspace(start=min(f_i), stop=max(f_i), num=15), s=5)
        # Gets unbounded f.
        Love_numbers_hyper_parameters.run_hyper_parameters.use_bounded_attenuation_functions = False
        f_r_unbounded, f_i_unbounded = get_attenuation_function(
            periods=periods[variable],
            anelasticity_description=anelasticity_description,
            run_hyper_parameters=Love_numbers_hyper_parameters.run_hyper_parameters,
            above_CMB_layer_index=above_CMB_layer_index,
        )
        plots[0].plot(
            periods[variable],
            f_r_unbounded,
            linewidth=linewidth,
        )
        plots[1].plot(
            periods[variable],
            f_i_unbounded,
            label="unbounded",
            linewidth=linewidth,
        )
        # Layout.
        plots[0].set_ylabel("$f_r$")
        plots[0].grid()
        plots[1].set_ylabel("$f_i$")
        plots[1].grid()
        plt.xlabel("Period (s)")
        plt.xscale("log")
        plt.legend()
        plt.savefig(figures_subpath.joinpath("attenuation_functions_for_" + variable + ".png"))
        plt.clf()


def get_attenuation_function(
    periods: ndarray,
    anelasticity_description: AnelasticityDescription,
    run_hyper_parameters: RunHyperParameters,
    above_CMB_layer_index: int,
) -> tuple[ndarray, ndarray]:
    """
    Evaluates attenuation functions of a given description at every frequency of a given list.
    """
    f_r, f_i = [], []
    for T_value in periods:
        integration = Integration(
            anelasticity_description=anelasticity_description,
            log_frequency=log10(anelasticity_description.period_unit / T_value),
            use_long_term_anelasticity=run_hyper_parameters.use_long_term_anelasticity,
            use_short_term_anelasticity=True,
            use_bounded_attenuation_functions=run_hyper_parameters.use_bounded_attenuation_functions,
        )
        f_r += [
            integration.description_layers[anelasticity_description.below_CMB_layers + above_CMB_layer_index].evaluate(
                x=integration.description_layers[anelasticity_description.below_CMB_layers + above_CMB_layer_index].x_inf,
                variable="f_r",
            )
        ]
        f_i += [
            integration.description_layers[anelasticity_description.below_CMB_layers + above_CMB_layer_index].evaluate(
                x=integration.description_layers[anelasticity_description.below_CMB_layers + above_CMB_layer_index].x_inf,
                variable="f_i",
            )
        ]
    return f_r, f_i
