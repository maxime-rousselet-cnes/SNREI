from typing import Optional

from matplotlib.axes import Axes
from matplotlib.pyplot import setp, show, subplots, tight_layout
from numpy import linspace, log10

from ...utils import (
    AnelasticityDescription,
    Integration,
    LoveNumbersHyperParameters,
    RunHyperParameters,
    frequencies_to_periods,
    load_Love_numbers_hyper_parameters,
)
from .utils import COLORS


def generate_figure_3(
    elasticity_model_name: Optional[str] = "PREM",
    long_term_anelasticity_model_name: Optional[str] = "VM7",
    short_term_anelasticity_model_name: Optional[str] = "Benjamin_Q_Resovsky",
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    periods: list[float] = [10.0 / 365.0, 10.0, 100.0, 1000.0],
    run_hyper_parameters_list: list[RunHyperParameters] = [
        RunHyperParameters(use_long_term_anelasticity=True, use_short_term_anelasticity=True, use_bounded_attenuation_functions=True),
        RunHyperParameters(use_long_term_anelasticity=False, use_short_term_anelasticity=True, use_bounded_attenuation_functions=True),
        RunHyperParameters(use_long_term_anelasticity=True, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False),
        RunHyperParameters(use_long_term_anelasticity=False, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False),
    ],
) -> None:
    """
    2024's article.
    """

    # Creates subplots.
    axes: list[list[Axes]]
    _, axes = subplots(
        len(periods),
        2,
        figsize=(10.0, 8.0),
        sharey=True,
    )
    axes[0][0].invert_yaxis()

    # Initializes.
    anelasticity_description = AnelasticityDescription(
        anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
        save=False,
        overwrite_descriptions=False,
        elasticity_name=elasticity_model_name,
        long_term_anelasticity_name=long_term_anelasticity_model_name,
        short_term_anelasticity_name=short_term_anelasticity_model_name,
    )
    # Panel A.
    # Iterates on viscosity models. A curb per loop.
    for i_period, period in enumerate(periods):
        for i_parameters, run_hyper_parameters in enumerate(run_hyper_parameters_list):
            integration = Integration(
                anelasticity_description=anelasticity_description,
                log_frequency=log10(
                    frequencies_to_periods(period) / anelasticity_description.frequency_unit
                ),  # Base 10 logarithm of the unitless frequency.
                use_long_term_anelasticity=run_hyper_parameters.use_long_term_anelasticity,
                use_short_term_anelasticity=run_hyper_parameters.use_short_term_anelasticity,
                use_bounded_attenuation_functions=run_hyper_parameters.use_bounded_attenuation_functions,
            )
            linewidth = 4 if run_hyper_parameters.use_long_term_anelasticity and run_hyper_parameters.use_short_term_anelasticity else 2
            color = COLORS[i_parameters]
            # Iterates on layers.
            last_value_previous_layer = integration.description_layers[integration.below_CMB_layers].evaluate(
                x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_0"
            ) / (
                integration.description_layers[integration.below_CMB_layers].evaluate(
                    x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_real"
                )
                + 1.0j
                * integration.description_layers[integration.below_CMB_layers].evaluate(
                    x=integration.description_layers[integration.below_CMB_layers].x_inf, variable="mu_imag"
                )
            )
            for k_layer in range(integration.below_CMB_layers, len(integration.description_layers)):
                layer = integration.description_layers[k_layer]
                x = linspace(start=layer.x_inf, stop=layer.x_sup, num=integration.spline_number)
                first_value_current_layer = layer.evaluate(x=layer.x_inf, variable="mu_0") / (
                    layer.evaluate(x=layer.x_inf, variable="mu_real") + 1.0j * layer.evaluate(x=layer.x_inf, variable="mu_imag")
                )
                variable = layer.evaluate(x=x, variable="mu_0") / (
                    layer.evaluate(x=x, variable="mu_real") + 1.0j * layer.evaluate(x=x, variable="mu_imag")
                )
                for i_part, part in enumerate(["Real part", "Imaginary part"]):
                    axes[i_period][i_part].plot(
                        variable.real if part == "Real part" else variable.imag,
                        (1.0 - x) * integration.radius_unit / 1e3,
                        color=color,
                        label=run_hyper_parameters.string() if k_layer == integration.below_CMB_layers else None,
                        linewidth=linewidth,
                    )
                    axes[i_period][i_part].plot(
                        [
                            last_value_previous_layer.real if part == "Real part" else last_value_previous_layer.imag,
                            first_value_current_layer.real if part == "Real part" else first_value_current_layer.imag,
                        ],
                        [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                        color=color,
                        linewidth=linewidth,
                    )
                last_value_previous_layer = layer.evaluate(x=layer.x_sup, variable="mu_0") / (
                    layer.evaluate(x=layer.x_sup, variable="mu_real") + 1.0j * layer.evaluate(x=layer.x_sup, variable="mu_imag")
                )

    # Adds "A" and "B" labels in the top-left corners of each subplot inside boxes.
    for ax_real, ax_imag, panel_real, panel_imag in zip(
        [sub_axes[0] for sub_axes in axes], [sub_axes[1] for sub_axes in axes], ["A", "C", "E", "G"], ["B", "D", "F", "H"]
    ):
        ax: Axes
        for ax, panel, part in zip([ax_real, ax_imag], [panel_real, panel_imag], ["real", "imag"]):
            ax.text(
                0.15,
                0.2,
                panel,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
            )
            ax.tick_params(axis="both", which="both", length=6, direction="inout")
            # ax.set_xscale("log")
    ax_real.set_xlabel("$Re(\mu_0/\mu)$")
    ax_imag.set_xlabel("$Im(\mu_0/\mu)$")

    # Shares axes by columns.
    for column in range(2):
        for ax in axes[:-1, column]:
            ax.sharex(axes[-1, column])

    # Adds legends.
    axes[0, 0].set_ylabel("Depth (km)")
    axes[1, 0].set_ylabel("Depth (km)")
    axes[0, 0].legend(loc="lower right", frameon=False)
    tight_layout()
    show()
