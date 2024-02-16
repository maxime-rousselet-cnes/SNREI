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
parser.add_argument("--real_description_id", type=str, required=True, help="wanted ID for the real description to load")
parser.add_argument("--figure_path_string", type=str, required=True, help="wanted path to save figure")

args = parser.parse_args()


def plot_comparative_Love_numbers(
    real_description_id: str,
    figure_path_string: str,
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
):
    """
    Generates a figure of attenuation functions f_r and f_i.
    """
    # Initializes.
    figure_path = Path(figure_path_string)
    booleans = [True, False]
    path = results_path.joinpath(real_description_id)
    options_list = list(product(booleans, booleans))
    figure_path.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[bool, bool], Result] = {options: Result() for options in options_list}
    frequency_values: dict[tuple[bool, bool], ndarray] = {options: array(()) for options in options_list}
    T_values: dict[tuple[bool, bool], ndarray] = {options: array(()) for options in options_list}

    # Loads results.
    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]
    for use_anelasticity, use_attenuation in options_list:
        if use_anelasticity or use_attenuation:
            sub_path = path.joinpath("runs").joinpath(
                "anelasticity_" + str(use_anelasticity) + "__attenuation_" + str(use_attenuation)
            )
            results[use_anelasticity, use_attenuation].load(name="anelastic_Love_numbers", path=sub_path)
            frequency_values[use_anelasticity, use_attenuation] = load_base_model(name="frequencies", path=sub_path)
            T_values[use_anelasticity, use_attenuation] = frequencies_to_periods(
                frequencies=frequency_values[use_anelasticity, use_attenuation]
            )
        else:
            results[use_anelasticity, use_attenuation].load(name="elastic_Love_numbers", path=path)

    # Plots Love numbers.
    for direction in Direction:
        for boundary_condition in BoundaryCondition:
            _, plots = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
            symbol = (
                ("" if direction == Direction.radial else "n ")
                + SYMBOLS_PER_DIRECTION[direction.value]
                + "_n"
                + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            )
            for part in ["real", "imaginary"]:
                for use_anelasticity, use_attenuation in options_list:
                    if use_anelasticity or use_attenuation:
                        plot = plots[int(use_anelasticity) + 2 * int(use_attenuation) - 1][0 if part == "real" else 1]
                        complex_result_values = results[use_anelasticity, use_attenuation].values[direction][boundary_condition]
                        T = T_values[use_anelasticity, use_attenuation]
                        for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                            color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                            result_values = (
                                real(
                                    complex_result_values[i_degree]
                                    / results[False, False].values[direction][boundary_condition][i_degree][0]
                                )
                                if part == "real"
                                else imag(complex_result_values[i_degree]) / degree
                            )
                            plot.plot(T, result_values, label="n = " + str(degree), color=color)
                            plot.set_xscale("log")
                        if use_anelasticity:
                            if not use_attenuation:
                                plot.set_title(part + " part" + (" ratio to elastic" if part == "real" else ""))
                                if part == "real":
                                    plot.legend(loc="upper left")
                            else:
                                plot.set_xlabel("T (y)")
                        if part == "real":
                            plot.set_ylabel(
                                ("with anelasticity " if use_anelasticity else "")
                                + ("with attenuation " if use_attenuation else "")
                            )
            plt.suptitle("$" + symbol + "$", fontsize=20)
            plt.savefig(figure_path.joinpath(symbol + ".png"))


if __name__ == "__main__":
    plot_comparative_Love_numbers(real_description_id=args.real_description_id, figure_path_string=args.figure_path_string)
