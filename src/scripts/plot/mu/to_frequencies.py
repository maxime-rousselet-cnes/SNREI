from typing import Optional

import matplotlib.pyplot as plt
from numpy import concatenate, linspace, log10, ndarray, zeros

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


def plot_mu_profiles_for_options_to_periods(
    load_description: bool = False,
    forced_anelasticity_description_id: Optional[str] = None,
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_name: Optional[str] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    overwrite_descriptions: bool = False,
    figure_subpath_string: str = "mu/to_periods",
    layer_name: str = "MANTLE_LVZ__ASTHENOSPHERE__MANTLE_LVZ",
    T_min: float = 0.5,
    T_max: float = 2.5e3,
    n_period_points: int = 100,
    options: list[RunHyperParameters] = OPTIONS[:3],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
    grid: bool = True,
    periods_to_print: list[float] = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
):
    """
    Generates a figure with two plots:
        - mu_0 / mu real part.
        - mu_0 / mu imaginary part.
    """
    # Initializes.
    period_values: ndarray[float] = concatenate((10 ** linspace(log10(T_min), log10(T_max), n_period_points), periods_to_print))
    period_values.sort()
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
    layer_names = [description_layer.name for description_layer in anelasticity_description.description_layers]
    layer_index = layer_names.index(layer_name)

    _, plots = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # Iterates on options. A curb per loop.
    for option in options + [
        RunHyperParameters(
            use_long_term_anelasticity=False, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False
        )
    ]:
        label = options_label(option=option)
        color = option_color(option=option)
        mu = zeros(shape=frequencies.shape, dtype=complex)
        print(option)
        for i_f, (frequency, period) in enumerate(zip(frequencies, period_values)):
            # Preprocesses.
            integration = Integration(
                anelasticity_description=anelasticity_description,
                log_frequency=log10(frequency / anelasticity_description.frequency_unit),
                use_long_term_anelasticity=option.use_long_term_anelasticity,
                use_short_term_anelasticity=option.use_short_term_anelasticity,
                use_bounded_attenuation_functions=option.use_bounded_attenuation_functions,
            )
            mu_0 = integration.description_layers[layer_index].evaluate(
                x=integration.description_layers[layer_index].x_inf, variable="mu_0"
            )
            mu[i_f] = (
                integration.description_layers[layer_index].evaluate(
                    x=integration.description_layers[layer_index].x_inf, variable="mu_real"
                )
                + integration.description_layers[layer_index].evaluate(
                    x=integration.description_layers[layer_index].x_inf, variable="mu_imag"
                )
                * 1.0j
            ) / mu_0
            if period in periods_to_print:
                print("----period (y):", period)
                print("--------mu_0:", mu_0 * anelasticity_description.elasticity_unit)
                print("--------mu/mu_0:", mu[i_f])
        # Plots mu_real and mu_imag.
        for part, plot in zip(["real", "imag"], plots):
            plot.plot(
                period_values,
                (1.0 / mu).real if part == "real" else (1.0 / mu).imag,
                color=color,
                label=label,
                linewidth=linewidth,
            )
            if part == "imag" and legend:
                plot.legend(loc="upper left")
            plot.set_ylabel(part + " part")
            if grid:
                plot.grid(True)
        plot.set_xlabel("Period (y)")
    plot.set_xscale("log")
    plt.suptitle("$\mu_0 / \mu$")
    plt.savefig(figures_subpath.joinpath("mu_0_on_mu.png"))
    plt.clf()


def plot_mu_profiles_for_options_to_periods_per_description(
    anelasticity_description_ids: list[Optional[str]] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    figure_subpath_string: str = "mu/to_periods",
    layer_name: str = "MANTLE_LVZ__ASTHENOSPHERE__MANTLE_LVZ",
    T_min: float = 0.5,
    T_max: float = 2.5e3,
    n_period_points: int = 100,
    options: list[RunHyperParameters] = OPTIONS[:3],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
    grid: bool = True,
    periods_to_print: list[float] = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
) -> None:
    """
    Generates 2 figures per description ID:
        - mu real part.
        - mu imaginary part.
    Each figure has a plot per period value and shows mu with respect to depth. Inverted axis.
    """

    for anelasticity_description_id in anelasticity_description_ids:
        plot_mu_profiles_for_options_to_periods(
            load_description=True,
            forced_anelasticity_description_id=anelasticity_description_id,
            Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
            figure_subpath_string=figure_subpath_string,
            layer_name=layer_name,
            T_min=T_min,
            T_max=T_max,
            n_period_points=n_period_points,
            options=options,
            figsize=figsize,
            linewidth=linewidth,
            legend=legend,
            grid=grid,
            periods_to_print=periods_to_print,
        )
