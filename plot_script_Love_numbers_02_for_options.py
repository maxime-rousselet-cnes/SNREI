# Generates panels of Love number graphs for a given description.
#
# A figure is generated per Love number direction and per boundary condition.
# Every figure consist of a panel of Love number graphs, a graph per option set.
# Every graphs shows Love numbers real and imaginary parts ratio to elastic version for some given degrees with respect to
# frequencies.

import argparse
from itertools import product

import matplotlib.pyplot as plt
from numpy import imag, ndarray, real

from utils import (
    BOOLEANS,
    SYMBOLS_PER_BOUNDARY_CONDITION,
    SYMBOLS_PER_DIRECTION,
    BoundaryCondition,
    Direction,
    Result,
    figures_path,
    frequencies_to_periods,
    gets_run_id,
    load_base_model,
    results_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="wanted ID for the real description to load")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def plot_comparative_Love_numbers_for_options(
    real_description_id: str,
    figure_subpath_string: str,
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
    directions: list[Direction] = [Direction.radial],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.potential],
):
    """
    Generates comparative figures of Love numbers for different options.
    """
    # Initializes.
    figure_path = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    figure_path.mkdir(parents=True, exist_ok=True)
    options_list = list(product(BOOLEANS, BOOLEANS, BOOLEANS))
    results: dict[tuple[bool, bool, bool], Result] = {
        (use_anelasticity, bounded_attenuation_functions, use_attenuation): Result()
        for use_anelasticity, bounded_attenuation_functions, use_attenuation in options_list
    }
    T_values: dict[tuple[bool, bool, bool], ndarray] = {}
    elastic_results = Result()

    # Loads results.
    path = results_path.joinpath(real_description_id)
    for use_anelasticity, bounded_attenuation_functions, use_attenuation in options_list:
        if (not use_attenuation) and (bounded_attenuation_functions or not use_anelasticity):
            continue
        subpath = path.joinpath("runs").joinpath(
            gets_run_id(
                use_anelasticity=use_anelasticity,
                bounded_attenuation_functions=bounded_attenuation_functions,
                use_attenuation=use_attenuation,
            )
        )
        results[use_anelasticity, bounded_attenuation_functions, use_attenuation].load(
            name="anelastic_Love_numbers", path=subpath
        )
        T_values[use_anelasticity, bounded_attenuation_functions, use_attenuation] = frequencies_to_periods(
            frequencies=load_base_model(name="frequencies", path=subpath)
        )
    elastic_results.load(name="elastic_Love_numbers", path=path)
    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]

    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            symbol = SYMBOLS_PER_DIRECTION[direction.value] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            _, plots = plt.subplots(5, 2, figsize=(16, 10), sharex=True)
            plot_line = 0
            for use_anelasticity, bounded_attenuation_functions, use_attenuation in options_list:
                if (not use_attenuation) and (bounded_attenuation_functions or not use_anelasticity):
                    continue
                complex_result_values = results[use_anelasticity, bounded_attenuation_functions, use_attenuation].values[
                    direction
                ][boundary_condition]
                T = T_values[use_anelasticity, bounded_attenuation_functions, use_attenuation]
                elastic_values = elastic_results.values[direction][boundary_condition]
                for part in ["real", "imaginary"]:
                    plot = plots[plot_line][0 if part == "real" else 1]
                    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                        color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                        result_values = (
                            real(complex_result_values[i_degree]) if part == "real" else imag(complex_result_values[i_degree])
                        ) / real(elastic_values[i_degree][0])
                        plot.plot(T, result_values, label="n = " + str(degree), color=color)
                    # plot.set_xscale("log")
                    plot.legend(loc="upper left")
                    if plot_line == 0:
                        plot.set_title(part + " part")
                    elif plot_line == 4:
                        plot.set_xlabel("T (y)")
                        plot.set_xscale("log")
                    if part == "real":
                        plot.set_ylabel(
                            " ".join(
                                (
                                    "Maxwell" if use_anelasticity else "",
                                    "bounded" if bounded_attenuation_functions else "",
                                    "att." if use_attenuation else "",
                                )
                            )
                        )
                    plot.grid()
                plot_line += 1
                plt.suptitle("$" + symbol + "/" + symbol + "^E$", fontsize=20)
                plt.savefig(figure_path.joinpath(symbol + ".png"))


if __name__ == "__main__":
    plot_comparative_Love_numbers_for_options(
        real_description_id=args.real_description_id if args.real_description_id else "base-model",
        figure_subpath_string=args.subpath if args.subpath else "Love_numbers_for_options",
    )
