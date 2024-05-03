from typing import Optional

import matplotlib.pyplot as plt
from numpy import linspace, log10

from ...utils import (
    OPTIONS,
    AnelasticityDescription,
    Integration,
    LoveNumbersHyperParameters,
    RunHyperParameters,
    figures_path,
    load_Love_numbers_hyper_parameters,
)
from .utils import option_color, options_label


def plot_viscosity_profiles_for_descriptions_to_depth(
    elasticity_model_name: str = "PREM",
    long_term_anelasticity_model_names: Optional[list[str]] = None,
    short_term_anelasticity_model_names: Optional[list[str]] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    overwrite_descriptions: bool = False,
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
    grid: bool = True,
) -> None:
    """
    Generates a figure of the Maxwell viscosity with respect to depth. Inverted axis.
    """

    _, ax = plt.figure(figsize=figsize)
    # Iterates on models. A curb per loop.
    for elasticity_model_name, long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
        [elasticity_model_name],
        long_term_anelasticity_model_names,
        short_term_anelasticity_model_names,
    ):
        anelasticity_description = AnelasticityDescription(
            anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
            save=False,
            overwrite_descriptions=overwrite_descriptions,
            elasticity_name=elasticity_model_name,
            long_term_anelasticity_name=long_term_anelasticity_model_name,
            short_term_anelasticity_name=short_term_anelasticity_model_name,
        )
        color = option_color(option=option)  # TODO.
        # Iterates on layers.
        for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
            layer = anelasticity_description.description_layers[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)

            if k_layer == anelasticity_description.below_CMB_layers:
                plt.plot(
                    layer.evaluate(x=x, variable="eta_m") * anelasticity_description.elasticity_unit,
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    label=long_term_anelasticity_model_name,
                    linewidth=linewidth,
                )
            else:
                plt.plot(
                    layer.evaluate(x=x, variable="eta_m") * anelasticity_description.elasticity_unit,
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    linewidth=linewidth,
                )
        if legend:
            plt.legend()
        if grid:
            plt.grid()
        plt.xlabel("$\eta_m$ (Pa.s)")
        plt.xscale("log")
        plt.ylabel("Depth (km)")
        ax.invert_yaxis()
        plt.savefig(figures_path.joinpath("eta_m.png"))
        plt.close()
