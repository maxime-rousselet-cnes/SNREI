import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import array, imag, ndarray, real

from utils import (
    SYMBOLS_PER_BOUNDARY_CONDITION,
    SYMBOLS_PER_DIRECTION,
    BoundaryCondition,
    Direction,
    Result,
    frequencies_to_periods,
    load_base_model,
    results_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--figure_path_string", type=str, required=True, help="wanted path to save figure")

args = parser.parse_args()


def plot_comparative_Love_numbers(
    figure_path_string: str,
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
):
    """
    Generates a figure of attenuation functions f_r and f_i.
    """
    # Initializes.
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)
    options_list = list(product([False, True], [False, True]))
    results: dict[tuple[bool, bool], Result] = {options: Result() for options in options_list}
    elastic_results: dict[tuple[bool, bool], Result] = {options: Result() for options in options_list}
    frequency_values: dict[tuple[bool, bool], ndarray] = {options: array(()) for options in options_list}
    T_values: dict[tuple[bool, bool], ndarray] = {options: array(()) for options in options_list}

    # Loads results.
    for use_Asthenosphere_model, bounded_attenuation_functions in options_list:
        path = results_path.joinpath(
            "with"
            + ("" if use_Asthenosphere_model else "out")
            + "_Asthenosphere_model"
            + "_with"
            + ("" if bounded_attenuation_functions else "out")
            + "_bounded_attenuation_functions"
        )
        degrees: list[int] = load_base_model(name="degrees", path=path)
        degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]
        sub_path = path.joinpath("runs").joinpath("anelasticity_True__attenuation_True")
        results[use_Asthenosphere_model, bounded_attenuation_functions].load(name="anelastic_Love_numbers", path=sub_path)
        frequency_values[use_Asthenosphere_model, bounded_attenuation_functions] = load_base_model(
            name="frequencies", path=sub_path
        )
        T_values[use_Asthenosphere_model, bounded_attenuation_functions] = frequencies_to_periods(
            frequencies=frequency_values[use_Asthenosphere_model, bounded_attenuation_functions]
        )
        elastic_results[use_Asthenosphere_model, bounded_attenuation_functions].load(name="elastic_Love_numbers", path=path)

    # Plots Love numbers.
    for direction in Direction:
        for boundary_condition in BoundaryCondition:
            _, plots = plt.subplots(4, 2, figsize=(14, 8), sharex=True)
            symbol = SYMBOLS_PER_DIRECTION[direction.value] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            for part in ["real", "imaginary"]:
                for use_Asthenosphere_model, bounded_attenuation_functions in options_list:
                    plot = plots[int(use_Asthenosphere_model) + 2 * int(bounded_attenuation_functions)][
                        0 if part == "real" else 1
                    ]
                    complex_result_values = results[use_Asthenosphere_model, bounded_attenuation_functions].values[direction][
                        boundary_condition
                    ]
                    T = T_values[use_Asthenosphere_model, bounded_attenuation_functions]
                    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                        color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                        result_values = (
                            real(
                                complex_result_values[i_degree]
                                / elastic_results[use_Asthenosphere_model, False].values[direction][boundary_condition][
                                    i_degree
                                ][0]
                            )
                            if part == "real"
                            else imag(complex_result_values[i_degree]) / degree
                        )
                        plot.plot(T, result_values, label="n = " + str(degree), color=color)
                        plot.set_xscale("log")
                    if use_Asthenosphere_model:
                        if not bounded_attenuation_functions:
                            plot.set_title(part + " part" + (" ratio to elastic" if part == "real" else ""))
                            if part == "real":
                                plot.legend(loc="upper left")
                        else:
                            plot.set_xlabel("T (y)")
                    if part == "real":
                        plot.set_ylabel(
                            ("with LV Asth. " if use_Asthenosphere_model else "")
                            + ("with bounded f " if bounded_attenuation_functions else "")
                        )
            plt.suptitle("$" + symbol + "$", fontsize=20)
            plt.savefig(figure_path.joinpath(symbol + ".png"))


if __name__ == "__main__":
    plot_comparative_Love_numbers(figure_path_string=args.figure_path_string)
