from itertools import product

import matplotlib.pyplot as plt
from numpy import imag, ndarray, real, where

from ...functions import get_degrees_indices
from ...utils import (
    BOOLEANS,
    OPTIONS,
    BoundaryCondition,
    Direction,
    Result,
    RunHyperParameters,
    figures_path,
    frequencies_to_periods,
    get_run_folder_name,
    load_base_model,
    load_Love_numbers_hyper_parameters,
    results_path,
)
from .utils import (
    SYMBOLS_PER_BOUNDARY_CONDITION,
    SYMBOLS_PER_DIRECTION,
    option_color,
    options_label,
)


def plot_Love_numbers_for_options_for_descriptions_per_type(
    anelasticity_description_ids: list[str],
    figure_subpath_string: str = "Love_numbers_comparative_for_descriptions",
    use_bounded_attenuation_functions: bool = load_Love_numbers_hyper_parameters().run_hyper_parameters.use_bounded_attenuation_functions,
    options: list[RunHyperParameters] = OPTIONS,
    degrees_to_plot: list[int] = [2, 10],
    directions: list[Direction] = [Direction.radial, Direction.tangential, Direction.potential],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.load],
    T_min_zoom_in: float = 1.0,  # (y)
    T_max_zoom_in: float = 2.5e3,  # (y)
    # Should have as many elements as 'degrees_to_plot'.
    degrees_linestyles: list[str] = [":", "-"],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
    model_numbers_only: bool = True,
    title_fontsize: int = 20,
    grid: bool = True,
):
    """
    Generates figures of Love numbers.
    A grid of plots is generated per Love number type (direction/boundary condition: h, h', k, k', etc...).
    In this grid, every column correspond to an anelasticity description (thus, arheological model).
    Lines correspond to real/imaginary part.
    Every plot contains several curves:
        - a color is used per degree
        - a style is used per option: with/without long/short term anelasticity.
    """
    # Initializes.
    options = [
        option
        for option in options
        if option.use_bounded_attenuation_functions
        == (option.use_short_term_anelasticity and use_bounded_attenuation_functions)
    ]
    anelastic_results: dict[tuple[str, bool, bool], Result] = {
        (anelasticity_description_id, option.use_long_term_anelasticity, option.use_short_term_anelasticity): Result()
        for anelasticity_description_id, option in product(anelasticity_description_ids, options)
    }
    T_values: dict[tuple[str, bool, bool], ndarray] = {}
    elastic_results: dict[str, Result] = {
        anelasticity_description_id: Result() for anelasticity_description_id in anelasticity_description_ids
    }
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(
        "with" + ("" if use_bounded_attenuation_functions else "out") + "_bounded_attenuation_functions"
    )
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Loads results.
    for anelasticity_description_id in anelasticity_description_ids:
        for option in options:
            result_subpath = results_path.joinpath(
                get_run_folder_name(anelasticity_description_id=anelasticity_description_id, run_id=option.run_id())
            )
            anelastic_results[
                anelasticity_description_id, option.use_long_term_anelasticity, option.use_short_term_anelasticity
            ].load(name="anelastic_Love_numbers", path=result_subpath)
            T_values[anelasticity_description_id, option.use_long_term_anelasticity, option.use_short_term_anelasticity] = (
                frequencies_to_periods(frequencies=load_base_model(name="frequencies", path=result_subpath))
            )
        elastic_results[anelasticity_description_id].load(name="elastic_Love_numbers", path=result_subpath.parent.parent)
    degrees: list[int] = load_base_model(name="degrees", path=result_subpath.parent.parent)
    degrees_indices = get_degrees_indices(degrees=degrees, degrees_to_plot=degrees_to_plot)

    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            symbol = SYMBOLS_PER_DIRECTION[direction] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition]
            for zoom_in in BOOLEANS:
                # A grid of plots per Love number type and with/without zoom in.
                _, plots = plt.subplots(2, len(anelasticity_description_ids), figsize=figsize, sharex="col", sharey="row")
                for part in ["real", "imaginary"]:
                    for i_plot, (anelasticity_description_id, plot) in enumerate(
                        zip(anelasticity_description_ids, plots[0 if part == "real" else 1])
                    ):
                        plot.set_xscale("log")
                        elastic_values = elastic_results[anelasticity_description_id].values[direction][boundary_condition]
                        for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                            linestyle = degrees_linestyles[degrees_to_plot.index(degree)]
                            for option in options:
                                # Gets corresponding data.
                                complex_result_values = anelastic_results[
                                    anelasticity_description_id,
                                    option.use_long_term_anelasticity,
                                    option.use_short_term_anelasticity,
                                ].values[direction][boundary_condition]
                                T = T_values[
                                    anelasticity_description_id,
                                    option.use_long_term_anelasticity,
                                    option.use_short_term_anelasticity,
                                ]
                                # Eventually restricts frequency range.
                                min_frequency_index = -1 if not zoom_in else where(T >= T_min_zoom_in)[0][-1]
                                max_frequency_index = 0 if not zoom_in else where(T <= T_max_zoom_in)[0][0]
                                # Plots.
                                color = option_color(option=option)
                                result_values = (
                                    real(complex_result_values[i_degree])
                                    if part == "real"
                                    else imag(complex_result_values[i_degree])
                                ) / real(elastic_values[i_degree][0])
                                plot.plot(
                                    T[max_frequency_index:min_frequency_index],
                                    result_values[max_frequency_index:min_frequency_index],
                                    label="n = " + str(degree) + ": " + options_label(option=option),
                                    color=color,
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                )
                        # Layout.
                        if grid:
                            plot.grid()
                        if part == "real":
                            plot.set_title(
                                "Model "
                                + str(i_plot)
                                + (
                                    ""
                                    if model_numbers_only
                                    else "\n" + anelasticity_description_id.replace("__", "\n").replace("/", "\n")
                                )
                            )
                        if anelasticity_description_id == anelasticity_description_ids[0]:
                            plot.set_ylabel(part + " part")
                        if part == "imaginary":
                            plot.set_xlabel("T (y)")
                        if legend and part == "imaginary" and anelasticity_description_id == anelasticity_description_ids[0]:
                            plot.legend()
                plt.suptitle("$" + symbol + "/" + symbol + "^E$", fontsize=title_fontsize)
                # Saves.
                plt.savefig(figure_subpath.joinpath(symbol + (" zoom in " if zoom_in else "") + ".png"))
                plt.close()


def plot_Love_numbers_for_options_per_description_per_type(
    anelasticity_description_ids: list[str],
    use_bounded_attenuation_functions: bool = load_Love_numbers_hyper_parameters().run_hyper_parameters.use_bounded_attenuation_functions,
    options: list[RunHyperParameters] = OPTIONS,
    degrees_to_plot: list[int] = [2, 10],
    directions: list[Direction] = [Direction.radial, Direction.tangential, Direction.potential],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.load],
    T_min_zoom_in: float = 1.0,  # (y)
    T_max_zoom_in: float = 2.5e3,  # (y)
    degrees_colors: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
    figsize: tuple[int, int] = (10, 10),
    linewidth: int = 2,
    legend: bool = True,
    model_numbers_only: bool = True,
    title_fontsize: int = 20,
    grid: bool = True,
):
    """
    Generates figures of Love numbers.
    A column of 2 plots is generated per Love number type (direction/boundary condition: h, h', k, k', etc...) and per real
    description ID.
    These 2 plots correspond to real/imaginary part.
    Every plot contains several curves:
        - a color is used per degree
        - a style is used per option: with/without long/short term anelasticity.
    """
    for anelasticity_description_id in anelasticity_description_ids:
        plot_Love_numbers_for_options_for_descriptions_per_type(
            anelasticity_description_ids=[anelasticity_description_id],
            figure_subpath_string=anelasticity_description_id + "/Love_numbers",
            use_bounded_attenuation_functions=use_bounded_attenuation_functions,
            options=options,
            degrees_to_plot=degrees_to_plot,
            directions=directions,
            boundary_conditions=boundary_conditions,
            T_min_zoom_in=T_min_zoom_in,
            T_max_zoom_in=T_max_zoom_in,
            degrees_colors=degrees_colors,
            figsize=figsize,
            linewidth=linewidth,
            legend=legend,
            model_numbers_only=model_numbers_only,
            title_fontsize=title_fontsize,
            grid=grid,
        )
