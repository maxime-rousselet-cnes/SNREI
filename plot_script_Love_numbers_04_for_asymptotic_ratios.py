# Generates panels of Love number graphs for a given description.
#
# A figure is generated per Love number direction and per boundary condition.
# Every figure consist of a panel of Love number graphs, a graph per asymptotic ratio value.
# Every graphs shows Love numbers real/imaginary parts ratio to elastic version for some given degrees with respect to
# frequencies.

import argparse

import matplotlib.pyplot as plt
from numpy import imag, ndarray, real, where

from utils import (
    BOOLEANS,
    SYMBOLS_PER_BOUNDARY_CONDITION,
    SYMBOLS_PER_DIRECTION,
    BoundaryCondition,
    Direction,
    Result,
    figures_path,
    frequencies_to_periods,
    gets_id_asymptotic_ratios,
    gets_run_id,
    load_base_model,
    results_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--initial_real_description_id", type=str, help="wanted ID for the real description to load")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
parser.add_argument("--with_anelasticity", action="store_true", help="Option to tell if the description should be loaded")
args = parser.parse_args()


def plot_comparative_Love_numbers_for_asymptotic_ratios_for_descriptions(
    initial_real_description_id: str,
    figure_subpath_string: str,
    asymptotic_ratios: list[list[int]] = [
        [1.0, 1.0],
        [0.2, 1.0],
    ],  # [0.5, 1.0], [0.2, 1.0], [0.1, 1.0], [0.05, 1.0]],
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
    directions: list[Direction] = [Direction.potential],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.load],
    use_anelasticity: bool = True,
):
    """
    Generates comparative figures of Love numbers for different asymptotic ratios and descriptions.
    """
    # Initializes.
    figure_path = (
        figures_path.joinpath(figure_subpath_string)
        .joinpath(initial_real_description_id)
        .joinpath("with" + ("" if use_anelasticity else "out") + "_anelasticity")
    )
    figure_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, Result] = {
        gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer): Result()
        for asymptotic_ratios_per_layer in asymptotic_ratios
    }
    T_values: dict[str, ndarray] = {}
    elastic_results: dict[str, Result] = {
        gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer): Result()
        for asymptotic_ratios_per_layer in asymptotic_ratios
    }

    # Loads results.
    for asymptotic_ratios_per_layer in asymptotic_ratios:
        real_description_id = gets_id_asymptotic_ratios(
            real_description_id=initial_real_description_id, asymptotic_ratios_per_layer=asymptotic_ratios_per_layer
        )
        path = results_path.joinpath(real_description_id)
        subpath = path.joinpath("runs").joinpath(
            gets_run_id(
                use_anelasticity=use_anelasticity,
                bounded_attenuation_functions=True,
                use_attenuation=True,
            )
        )
        results[gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer)].load(
            name="anelastic_Love_numbers", path=subpath
        )
        T_values[gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer)] = frequencies_to_periods(
            frequencies=load_base_model(name="frequencies", path=subpath)
        )
        elastic_results[gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer)].load(
            name="elastic_Love_numbers", path=path
        )
    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]

    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            symbol = SYMBOLS_PER_DIRECTION[direction.value] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            for zoom_in in BOOLEANS:
                _, plots = plt.subplots(len(asymptotic_ratios), 2, figsize=(16, 9), sharex=True)
                for plot_line, asymptotic_ratios_per_layer in enumerate(asymptotic_ratios):
                    for part in ["real", "imaginary"]:
                        # Gets corresponding data.
                        complex_result_values = results[
                            gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer)
                        ].values[direction][boundary_condition]
                        T = T_values[gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer)]
                        elastic_values = elastic_results[
                            gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer)
                        ].values[direction][boundary_condition]
                        plot = plots[plot_line][0 if part == "real" else 1]
                        # Eventually restricts frequency range.
                        min_frequency_index = -1 if not zoom_in else where(T >= 1e0)[0][-1]
                        max_frequency_index = 0 if not zoom_in else where(T <= 2.5e3)[0][0]
                        for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                            color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                            result_values = (
                                real(complex_result_values[i_degree])
                                if part == "real"
                                else imag(complex_result_values[i_degree])
                            ) / real(elastic_values[i_degree][0])
                            plot.semilogx(
                                T[max_frequency_index:min_frequency_index],
                                result_values[max_frequency_index:min_frequency_index],
                                label="n = " + str(degree),
                                color=color,
                            )
                        # plot.set_xscale("log")
                        plot.legend(loc="upper left")
                        if plot_line == 0:
                            plot.set_title(part + " part")
                        elif plot_line == len(asymptotic_ratios) - 1:
                            plot.set_xlabel("T (y)")
                            plot.set_xscale("log")
                        if part == "real":
                            plot.set_ylabel(gets_id_asymptotic_ratios(asymptotic_ratios_per_layer=asymptotic_ratios_per_layer))
                        plot.grid()
                plt.suptitle("$" + symbol + "/" + symbol + "^E$", fontsize=20)
                plt.savefig(figure_path.joinpath(symbol + (" zoom in" if zoom_in else "") + ".png"))


if __name__ == "__main__":
    plot_comparative_Love_numbers_for_asymptotic_ratios_for_descriptions(
        initial_real_description_id=(
            args.initial_real_description_id
            if args.initial_real_description_id
            else "PREM_low-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio"
        ),
        use_anelasticity=args.with_anelasticity if args.with_anelasticity else False,
        figure_subpath_string=args.subpath if args.subpath else "Love_numbers_for_asymptotic_ratios",
    )
