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

part_names = ["low", "mid", "high"]


def plot_comparative_sampling_Love_numbers(
    figure_path_string: str,
    real_description_id_parts: tuple[str, str] = ("p", "ns"),
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
    directions: list[Direction] = [Direction.radial],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.potential],
    parts=["real"],
):
    """
    Generates a figure of attenuation functions f_r and f_i.
    """
    # Initializes.
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)
    booleans = [True, False]
    part_name_products = list(product(part_names, part_names))
    options_list = list(product(booleans, booleans))
    results: dict[str, dict[str, dict[tuple[bool, bool], Result]]] = {
        part_name_1: {part_name_2: {options: Result() for options in options_list} for part_name_2 in part_names}
        for part_name_1 in part_names
    }
    frequency_values: dict[str, dict[str, dict[tuple[bool, bool]], ndarray]] = {
        part_name_1: {part_name_2: {options: array(()) for options in options_list} for part_name_2 in part_names}
        for part_name_1 in part_names
    }
    T_values: dict[str, dict[str, dict[tuple[bool, bool], ndarray]]] = {
        part_name_1: {part_name_2: {options: array(()) for options in options_list} for part_name_2 in part_names}
        for part_name_1 in part_names
    }

    # Loads results.
    for part_name_1, part_name_2 in part_name_products:
        path = results_path.joinpath(
            "_".join((part_name_1, real_description_id_parts[0], part_name_2, real_description_id_parts[1]))
        )
        for use_anelasticity, use_attenuation in options_list:
            if use_anelasticity or use_attenuation:
                sub_path = path.joinpath("runs").joinpath(
                    "anelasticity_" + str(use_anelasticity) + "__attenuation_" + str(use_attenuation)
                )
                results[part_name_1][part_name_2][use_anelasticity, use_attenuation].load(
                    name="anelastic_Love_numbers", path=sub_path
                )
                frequency_values[part_name_1][part_name_2][use_anelasticity, use_attenuation] = load_base_model(
                    name="frequencies", path=sub_path
                )
                T_values[part_name_1][part_name_2][use_anelasticity, use_attenuation] = frequencies_to_periods(
                    frequencies=frequency_values[part_name_1][part_name_2][use_anelasticity, use_attenuation]
                )
            else:
                results[part_name_1][part_name_2][use_anelasticity, use_attenuation].load(
                    name="elastic_Love_numbers", path=path
                )

    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]

    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            symbol = (
                ("" if direction == Direction.radial else "n ")
                + SYMBOLS_PER_DIRECTION[direction.value]
                + "_n"
                + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            )
            for part in parts:
                for use_anelasticity, use_attenuation in options_list:
                    if use_anelasticity or use_attenuation:
                        _, plots = plt.subplots(len(part_names), len(part_names), figsize=(18, 12), sharex=True, sharey=True)
                        for i_plot, (part_name_1, part_name_2) in enumerate(part_name_products):
                            plot = plots[i_plot // len(part_names)][i_plot % len(part_names)]
                            complex_result_values = results[part_name_1][part_name_2][use_anelasticity, use_attenuation].values[
                                direction
                            ][boundary_condition]
                            elastic_values = results[part_name_1][part_name_2][False, False].values[direction][
                                boundary_condition
                            ]
                            T = T_values[part_name_1][part_name_2][use_anelasticity, use_attenuation]
                            for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                                color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                                result_values = (
                                    real(complex_result_values[i_degree] / elastic_values[i_degree][0])
                                    if part == "real"
                                    else imag(complex_result_values[i_degree]) / degree
                                )
                                plot.plot(T, result_values, label="n = " + str(degree), color=color)
                            # plot.set_xscale("log")
                            plot.legend(loc="upper left")
                            if part_name_1 == part_names[0]:
                                plot.set_title(" ".join((part_name_2, real_description_id_parts[1])))
                            elif part_name_1 == part_names[-1]:
                                plot.set_xlabel("T (y)")
                                plot.set_xscale("log")
                            if part_name_2 == part_names[0]:
                                plot.set_ylabel(" ".join((part_name_1, real_description_id_parts[0])))
                        base_title = (
                            part
                            + " part"
                            + (" ratio to elastic" if part == "real" else "")
                            + (" with anelasticity" if use_anelasticity else "")
                            + (" with attenuation" if use_attenuation else "")
                        )
                        plt.suptitle("$" + symbol + "$ " + base_title, fontsize=20)
                        plt.savefig(figure_path.joinpath(symbol + " " + base_title + ".png"))


if __name__ == "__main__":
    plot_comparative_sampling_Love_numbers(figure_path_string=args.figure_path_string)
