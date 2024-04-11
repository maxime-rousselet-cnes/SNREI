from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
from numpy import linspace, log10, zeros

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
from ..utils import option_linestyle, options_label


def plot_mu_profiles_for_options_to_periods(
    load_description: bool = False,
    forced_anelasticity_description_id: Optional[str] = None,
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_name: Optional[str] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    overwrite_descriptions: bool = False,
    figure_subpath_string: str = "mu/to_periods",
    T_min: float = 1.0,
    T_max: float = 2.5e3,
    n_period_points: int = 100,
    options: list[RunHyperParameters] = OPTIONS[:3],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
):
    """
    Generates a figure with two plots:
        - mu/mu_0 real part.
        - mu/mu_0 imaginary part.
    Each figure has a plot per period value and shows mu with respect to depth. Inverted axis.
    """
    # Initializes.
    period_values = 10 ** linspace(log10(T_min), log10(T_max), n_period_points)
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    anelasticity_description = AnelasticityDescription(
        anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
        load_description=load_description,
        id=forced_anelasticity_description_id,
        save=False,
        overwrite_descriptions=overwrite_descriptions,
        elasticity_model_name=elasticity_model_name,
        long_term_anelasticity_model_name=long_term_anelasticity_model_name,
        short_term_anelasticity_model_name=short_term_anelasticity_model_name,
    )
    figures_subpath = figures_path.joinpath(figure_subpath_string).joinpath(anelasticity_description.id)
    figures_subpath.mkdir(parents=True, exist_ok=True)

    _, plots = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # Iterates on options. A curb per loop.
    for option in options + [
        RunHyperParameters(
            use_long_term_anelasticity=False, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False
        )
    ]:
        label = (
            "elastic"
            if (not option.use_long_term_anelasticity and not option.use_short_term_anelasticity)
            else (options_label(option=option))
        )
        linestyle = (
            "-"
            if (not option.use_long_term_anelasticity and not option.use_short_term_anelasticity)
            else option_linestyle(option=option)
        )
        mu = {
            "real": zeros(shape=frequencies.shape, dtype=float),
            "imag": zeros(shape=frequencies.shape, dtype=float),
        }

        for i_f, frequency in enumerate(frequencies):
            # Preprocesses.
            integration = Integration(
                anelasticity_description=anelasticity_description,
                log_frequency=log10(frequency / anelasticity_description.frequency_unit),
                use_long_term_anelasticity=option.use_long_term_anelasticity,
                use_short_term_anelasticity=option.use_short_term_anelasticity,
                use_bounded_attenuation_functions=option.use_bounded_attenuation_functions,
            )
            # Plots mu_real and mu_imag.
            for part in ["real", "imag"]:
                mu[part][i_f] = integration.description_layers[integration.below_CMB_layers].evaluate(
                    x=integration.x_CMB, variable="mu_" + part
                ) / integration.description_layers[integration.below_CMB_layers].evaluate(x=integration.x_CMB, variable="mu_0")
        # Plots mu_real and mu_imag.
        for part, plot in zip(["real", "imag"], plots):
            plot.plot(
                period_values,
                mu[part],
                color=(
                    0.0,
                    0.0,
                    1.0,
                ),
                linestyle=linestyle,
                label=label,
                linewidth=linewidth,
            )
            if part == "imag":
                plot.legend(loc="upper left")
            plot.set_ylabel(part + " part")
            plot.grid()
            # plot.set_xscale("log")
        plot.set_xlabel("Period (y)")
    plot.set_xscale("log")
    plot.set_yscale("symlog")
    plt.suptitle("$\mu_0 / \mu$", fontsize=20)
    plt.savefig(figures_subpath.joinpath("mu_on_mu_0.png"))
    plt.clf()
