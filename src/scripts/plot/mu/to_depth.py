from typing import Optional

import matplotlib.pyplot as plt
from numpy import linspace, log10

from ....utils import (
    OPTIONS,
    AnelasticityDescription,
    Integration,
    LoveNumbersHyperParameters,
    RunHyperParameters,
    figures_path,
    frequencies_to_periods,
    load_Love_numbers_hyper_parameters,
)
from ..utils import option_color, options_label


def plot_mu_profiles_for_options_for_periods_to_depth(
    load_description: bool = False,
    forced_anelasticity_description_id: Optional[str] = None,
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_name: Optional[str] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    overwrite_descriptions: bool = False,
    figure_subpath_string: str = "mu/to_depth",
    period_values: list[float] = [10, 100, 1000],
    options: list[RunHyperParameters] = OPTIONS[:3],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
) -> None:
    """
    Generates 2 figures:
        - mu real part.
        - mu imaginary part.
    Each figure has a plot per period value and shows mu with respect to depth. Inverted axis.
    """
    # Initializes.
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    anelasticity_description = AnelasticityDescription(
        anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
        load_description=load_description,
        id=forced_anelasticity_description_id,
        save=False,
        overwrite_descriptions=overwrite_descriptions,
        elasticity_name=elasticity_model_name,
        long_term_anelasticity_name=long_term_anelasticity_model_name,
        short_term_anelasticity_name=short_term_anelasticity_model_name,
    )
    figures_subpath = figures_path.joinpath(figure_subpath_string).joinpath(anelasticity_description.id)
    figures_subpath.mkdir(parents=True, exist_ok=True)

    # Plots mu_real and mu_imag. A figure per loop.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=figsize, sharex=True, sharey=True)
        # Iterates on frequencies. A plot per loop.
        for frequency, period, plot in zip(frequencies, period_values, plots):
            # Iterates on options. A curb per loop.
            for option in options + [
                RunHyperParameters(
                    use_long_term_anelasticity=False, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False
                )
            ]:
                integration = Integration(
                    anelasticity_description=anelasticity_description,
                    log_frequency=log10(frequency / anelasticity_description.frequency_unit),
                    use_long_term_anelasticity=option.use_long_term_anelasticity,
                    use_short_term_anelasticity=option.use_short_term_anelasticity,
                    use_bounded_attenuation_functions=option.use_bounded_attenuation_functions,
                )
                label = options_label(option=option)
                color = option_color(option=option)
                # Iterates on layers.
                for k_layer in range(anelasticity_description.below_CMB_layers, len(integration.description_layers)):
                    layer = integration.description_layers[k_layer]
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)

                    if k_layer == anelasticity_description.below_CMB_layers:
                        plot.plot(
                            layer.evaluate(x=x, variable="mu_" + part) * anelasticity_description.elasticity_unit,
                            (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                            color=color,
                            label=label,
                            linewidth=linewidth,
                        )
                    else:
                        plot.plot(
                            layer.evaluate(x=x, variable="mu_" + part) * anelasticity_description.elasticity_unit,
                            (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                            color=color,
                            linewidth=linewidth,
                        )
            if frequency == frequencies[0] and legend:
                plot.legend()
            plot.set_xlabel("$\mu_{" + part + "}$ (Pa)")
            plot.grid()
            # plot.set_xscale("log")
            plot.set_title("$T=" + str(period) + "$ (y)")
        plot.set_ylabel("Depth (km)")
        plot.invert_yaxis()
        plt.savefig(figures_subpath.joinpath("mu_" + part + ".png"))
        plt.close()


def plot_mu_profiles_for_options_for_periods_to_depth_per_description(
    anelasticity_description_ids: list[Optional[str]] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    figure_subpath_string: str = "mu/to_depth",
    period_values: list[float] = [10, 100, 1000],
    options: list[RunHyperParameters] = OPTIONS[:3],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
) -> None:
    """
    Generates 2 figures per description ID:
        - mu real part.
        - mu imaginary part.
    Each figure has a plot per period value and shows mu with respect to depth. Inverted axis.
    """

    for anelasticity_description_id in anelasticity_description_ids:
        plot_mu_profiles_for_options_for_periods_to_depth(
            load_description=True,
            forced_anelasticity_description_id=anelasticity_description_id,
            Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
            figure_subpath_string=figure_subpath_string,
            period_values=period_values,
            options=options,
            figsize=figsize,
            linewidth=linewidth,
            legend=legend,
        )
