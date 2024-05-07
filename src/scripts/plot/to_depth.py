from typing import Optional

import matplotlib.pyplot as plt
from numpy import linspace

from ...utils import (
    AnelasticityDescription,
    LoveNumbersHyperParameters,
    figures_path,
    load_Love_numbers_hyper_parameters,
)


def plot_viscosity_profiles_for_descriptions_to_depth(
    elasticity_model_name: str = "PREM",
    long_term_anelasticity_model_names: Optional[list[str]] = ["VM7", "Lambeck_2017", "Caron", "Lau_2016", "VM5a"],
    short_term_anelasticity_model_names: Optional[list[str]] = [
        "Benjamin_Q_PAR3P",
        "Benjamin_Q_PREM",
        "Benjamin_Q_QL6",
        "Benjamin_Q_QM1",
        "Benjamin_Q_Resovsky",
    ],
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    overwrite_descriptions: bool = False,
    figsize: tuple[int, int] = (7, 5),
    linewidth: int = 2,
    legend: bool = True,
    grid: bool = True,
    viscosity_colors: Optional[list[tuple[float, float, float]]] = [
        (126 / 255, 135 / 255, 176 / 255),
        (3 / 255, 134 / 255, 0 / 255),
        (184 / 255, 134 / 255, 6 / 255),
        (253 / 255, 0 / 255, 2 / 255),
        (66 / 255, 105 / 255, 228 / 255),
    ],
    Q_colors: Optional[list[tuple[float, float, float]]] = [
        (200 / 255, 0 / 255, 200 / 255),
        (1 / 255, 215 / 255, 101 / 255),
        (73 / 255, 216 / 255, 216 / 255),
        (35 / 255, 35 / 255, 35 / 255),
        (241 / 255, 18 / 255, 11 / 255),
    ],
) -> None:
    """
    Generates a figure of the Maxwell viscosity and attenuation Q with respect to depth.
    """
    figures_path.mkdir(exist_ok=True, parents=True)
    _, plots = plt.subplots(1, 2, figsize=figsize)

    # Iterates on viscosity models. A curb per loop.
    for long_term_anelasticity_model_name, color in zip(long_term_anelasticity_model_names, viscosity_colors):
        anelasticity_description = AnelasticityDescription(
            anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
            save=False,
            overwrite_descriptions=overwrite_descriptions,
            elasticity_name=elasticity_model_name,
            long_term_anelasticity_name=long_term_anelasticity_model_name,
            short_term_anelasticity_name=short_term_anelasticity_model_names[0],
        )
        # Iterates on layers.
        for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
            layer = anelasticity_description.description_layers[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)
            if k_layer == anelasticity_description.below_CMB_layers:
                plots[0].plot(
                    layer.evaluate(x=x, variable="eta_m") * anelasticity_description.viscosity_unit,
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    label=long_term_anelasticity_model_name,
                    linewidth=linewidth,
                    linestyle=":",
                )
            else:
                plots[0].plot(
                    layer.evaluate(x=x, variable="eta_m") * anelasticity_description.viscosity_unit,
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    linewidth=linewidth,
                    linestyle=":",
                )
                first_value_current_layer = (
                    layer.evaluate(x=layer.x_inf, variable="eta_m") * anelasticity_description.viscosity_unit
                )
                plots[0].plot(
                    [last_value_previous_layer, first_value_current_layer],
                    [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                    color=color,
                    linewidth=linewidth,
                    linestyle=":",
                )
            last_value_previous_layer = (
                layer.evaluate(x=layer.x_sup, variable="eta_m") * anelasticity_description.viscosity_unit
            )
    plots[0].set_xlabel("$\eta_m$ (Pa.s)")
    if legend:
        plots[0].legend()
    if grid:
        plots[0].grid()
    plots[0].set_xscale("log")
    plots[0].invert_yaxis()
    plots[1].set_ylabel("Depth (km)")

    # Iterates on attenuation models. A curb per loop.
    for short_term_anelasticity_model_name, color in zip(short_term_anelasticity_model_names, Q_colors):
        anelasticity_description = AnelasticityDescription(
            anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
            save=False,
            overwrite_descriptions=overwrite_descriptions,
            elasticity_name=elasticity_model_name,
            long_term_anelasticity_name=long_term_anelasticity_model_names[0],
            short_term_anelasticity_name=short_term_anelasticity_model_name,
        )
        # Iterates on layers.
        for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
            layer = anelasticity_description.description_layers[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)
            # Q_mu
            if k_layer == anelasticity_description.below_CMB_layers:
                plots[1].plot(
                    layer.evaluate(x=x, variable="Q_mu"),
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    label=short_term_anelasticity_model_name[9:],
                    linewidth=linewidth,
                    linestyle=":",
                )
            else:
                plots[1].plot(
                    layer.evaluate(x=x, variable="Q_mu"),
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    linewidth=linewidth,
                    linestyle=":",
                )
                first_value_current_layer = layer.evaluate(x=layer.x_inf, variable="Q_mu")
                plots[1].plot(
                    [last_value_previous_layer, first_value_current_layer],
                    [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                    color=color,
                    linewidth=linewidth,
                    linestyle=":",
                )
            last_value_previous_layer = layer.evaluate(x=layer.x_sup, variable="Q_mu")
    plots[1].set_xlabel("$Q_{mu}$")
    if legend:
        plots[1].legend()
    if grid:
        plots[1].grid()
    plots[1].invert_yaxis()
    plots[1].set_xscale("log")
    plots[1].set_ylabel("Depth (km)")

    ax = plots[1].twiny()
    ax.set_xlabel("$\mu_0$ (Pa)")
    # Iterates on layers.
    for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
        layer = anelasticity_description.description_layers[k_layer]
        x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)
        # mu_0
        if k_layer == anelasticity_description.below_CMB_layers:
            ax.plot(
                layer.evaluate(x=x, variable="mu_0") * anelasticity_description.elasticity_unit,
                (1 - x) * anelasticity_description.radius_unit / 1e3,
                color=(0, 0, 0),
                label="$\mu_0$",
                linewidth=linewidth,
            )
        ax.plot(
            layer.evaluate(x=x, variable="mu_0") * anelasticity_description.elasticity_unit,
            (1 - x) * anelasticity_description.radius_unit / 1e3,
            color=(0, 0, 0),
            linewidth=linewidth,
        )
    if legend:
        ax.legend(loc="center left")
    if grid:
        ax.grid()

    plt.savefig(figures_path.joinpath("eta_m_and_Q_mu.png"))
    plt.close()
