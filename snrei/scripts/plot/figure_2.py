from typing import Optional

from matplotlib.axes import Axes
from matplotlib.pyplot import show, subplots, tight_layout
from numpy import linspace, ones

from ...utils import AnelasticityDescription, LoveNumbersHyperParameters, load_Love_numbers_hyper_parameters
from .utils import COLORS


def generate_figure_2(
    elasticity_model_name: Optional[str] = "PREM",
    long_term_anelasticity_model_names: Optional[list[str]] = ["VM7", "Lambeck_2017", "Caron", "Lau_2016", "VM5a", "Mao_Zhong"],
    short_term_anelasticity_model_names: Optional[list[str]] = [
        "Benjamin_Q_Resovsky",
        "Benjamin_Q_PAR3P",
        "Benjamin_Q_PREM",
        "Benjamin_Q_QL6",
        "Benjamin_Q_QM1",
    ],
    asthenosphere_eta_m: float = 3e19,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
) -> None:
    """
    2024's article.
    """

    # Creates subplots.
    ax1: Axes
    ax2: Axes
    _, (ax1, ax2) = subplots(1, 2, figsize=(10.0, 8.0), sharey=True)
    ax = ax2.twiny()

    # Panel A.
    # Iterates on viscosity models. A curb per loop.
    for i_model, long_term_anelasticity_model_name in enumerate(long_term_anelasticity_model_names):
        anelasticity_description = AnelasticityDescription(
            anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
            save=False,
            overwrite_descriptions=False,
            elasticity_name=elasticity_model_name,
            long_term_anelasticity_name=long_term_anelasticity_model_name,
            short_term_anelasticity_name=short_term_anelasticity_model_names[0],
        )
        linewidth = 4 if i_model == 0 else 2
        color = COLORS[i_model]
        # Iterates on layers.
        last_value_previous_layer = (
            anelasticity_description.description_layers[anelasticity_description.below_CMB_layers].evaluate(
                x=anelasticity_description.description_layers[anelasticity_description.below_CMB_layers].x_inf, variable="eta_m"
            )
            * anelasticity_description.viscosity_unit
        )
        name_previous_layer = anelasticity_description.description_layers[anelasticity_description.below_CMB_layers - 1]
        for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
            layer = anelasticity_description.description_layers[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)
            ax1.plot(
                layer.evaluate(x=x, variable="eta_m") * anelasticity_description.viscosity_unit,
                (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                color=color,
                label=long_term_anelasticity_model_name if k_layer == anelasticity_description.below_CMB_layers else None,
                linewidth=linewidth,
            )
            first_value_current_layer = layer.evaluate(x=layer.x_inf, variable="eta_m") * anelasticity_description.viscosity_unit
            ax1.plot(
                [last_value_previous_layer, first_value_current_layer],
                [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                color=color,
                linewidth=linewidth,
            )
            last_value_previous_layer = layer.evaluate(x=layer.x_sup, variable="eta_m") * anelasticity_description.viscosity_unit
            # Eventually draws the asthenosphere variation.
            if "ASTHENOSPHERE" in layer.name:
                ax1.plot(
                    asthenosphere_eta_m * ones(shape=x.shape),
                    (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                    color=color,
                    label=long_term_anelasticity_model_name if k_layer == anelasticity_description.below_CMB_layers else None,
                    linewidth=linewidth,
                    linestyle="--",
                )
                if not "ASTHENOSPHERE" in name_previous_layer:
                    ax1.plot(
                        [last_value_previous_layer, asthenosphere_eta_m],
                        [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                        color=color,
                        label=long_term_anelasticity_model_name if k_layer == anelasticity_description.below_CMB_layers else None,
                        linewidth=linewidth,
                        linestyle="--",
                    )
            elif "ASTHENOSPHERE" in name_previous_layer:
                ax1.plot(
                    [last_value_previous_layer, asthenosphere_eta_m],
                    [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                    color=color,
                    label=long_term_anelasticity_model_name if k_layer == anelasticity_description.below_CMB_layers else None,
                    linewidth=linewidth,
                    linestyle="--",
                )
            name_previous_layer = layer.name

    # Panel B.
    # Iterates on viscosity models. A curb per loop.
    for i_model, short_term_anelasticity_model_name in enumerate(short_term_anelasticity_model_names):
        anelasticity_description = AnelasticityDescription(
            anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
            save=False,
            overwrite_descriptions=False,
            elasticity_name=elasticity_model_name,
            long_term_anelasticity_name=long_term_anelasticity_model_names[0],
            short_term_anelasticity_name=short_term_anelasticity_model_name,
        )
        color = COLORS[i_model + 6]
        linewidth = 4 if i_model == 0 else 2
        last_value_previous_layer = anelasticity_description.description_layers[anelasticity_description.below_CMB_layers].evaluate(
            x=anelasticity_description.description_layers[anelasticity_description.below_CMB_layers].x_inf, variable="Q_mu"
        )
        for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
            layer = anelasticity_description.description_layers[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)
            ax2.plot(
                layer.evaluate(x=x, variable="Q_mu"),
                (1.0 - x) * anelasticity_description.radius_unit / 1e3,
                color=color,
                label=short_term_anelasticity_model_name.split("_")[2] if k_layer == anelasticity_description.below_CMB_layers else None,
                linewidth=linewidth,
            )
            first_value_current_layer = layer.evaluate(x=layer.x_inf, variable="Q_mu")
            ax2.plot(
                [last_value_previous_layer, first_value_current_layer],
                [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
                color=color,
                linewidth=linewidth,
            )
            last_value_previous_layer = layer.evaluate(x=layer.x_sup, variable="Q_mu")

    # mu.
    # Iterates on layers.
    last_value_previous_layer = (
        anelasticity_description.description_layers[anelasticity_description.below_CMB_layers].evaluate(
            x=anelasticity_description.description_layers[anelasticity_description.below_CMB_layers].x_inf, variable="mu_0"
        )
        * anelasticity_description.elasticity_unit
    )
    for k_layer in range(anelasticity_description.below_CMB_layers, len(anelasticity_description.description_layers)):
        layer = anelasticity_description.description_layers[k_layer]
        x = linspace(start=layer.x_inf, stop=layer.x_sup, num=anelasticity_description.spline_number)
        ax.plot(
            layer.evaluate(x=x, variable="mu_0") * anelasticity_description.elasticity_unit,
            (1.0 - x) * anelasticity_description.radius_unit / 1e3,
            color=COLORS[0],
            label="$\mu_0$" if k_layer == anelasticity_description.below_CMB_layers else None,
            linewidth=2,
            linestyle="--",
        )
        first_value_current_layer = layer.evaluate(x=layer.x_inf, variable="mu_0") * anelasticity_description.elasticity_unit
        ax.plot(
            [last_value_previous_layer, first_value_current_layer],
            [(1.0 - layer.x_inf) * anelasticity_description.radius_unit / 1e3] * 2,
            color=COLORS[0],
            linewidth=2,
            linestyle="--",
        )
        last_value_previous_layer = layer.evaluate(x=layer.x_sup, variable="mu_0") * anelasticity_description.elasticity_unit

    # Adds "A" and "B" labels in the top-left corners of each subplot inside boxes.
    for ax_i, panel in zip([ax1, ax2], ["A", "B"]):
        ax_i.text(
            0.1,
            1.05,
            panel,
            transform=ax_i.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

    # Adds horizontal lines for mantle principal layers:
    for k_layer, name in zip([2, 18, 24, 27, 30], ["Lower Mantle", "Upper Mantle", "Asthenosphere", "Lithosphere", "Crust"]):
        layer = anelasticity_description.description_layers[k_layer]
        y = (1 - layer.x_inf) * anelasticity_description.radius_unit / 1e3
        ax1.axhline(y=y, linewidth=1, color=(0.5, 0.5, 0.5), linestyle="--" if name == "Asthenosphere" or name == "Lithosphere" else "-")
        ax2.axhline(y=y, linewidth=1, color=(0.5, 0.5, 0.5))
        ax1.text(x=1e18, y=y - 10.0, s=name)

    # Adds legends.
    ax1.set_xlabel("$\eta_m$ (Pa.s)")
    ax1.set_xlim(left=9e17, right=1e24)
    ax2.set_xlabel("$Q_{\mu}$")
    ax1.invert_yaxis()
    ax.set_xscale("log")
    ax1.set_xscale("log")
    ax1.set_ylabel("Depth (km)")
    ax.set_xlabel("$\mu_0$ (Pa)")
    ax.tick_params(axis="both", which="both", length=6, direction="inout")
    ax1.tick_params(axis="both", which="both", length=6, direction="inout")
    ax2.tick_params(axis="both", which="both", length=6, direction="inout")
    ax.legend(loc="lower left", frameon=False)
    ax1.legend(loc="center left", frameon=False)
    ax2.legend(loc="center left", frameon=False)
    tight_layout()
    show()
