# Generates panels of Love number graphs.
#
# A figure is generated per Love number direction, per boundary condition and per imaginary/real part.
# Every figure consist of a panel of Love number graphs, a graph per option set.
# Every graphs shows Love numbers for some given degrees with respect to frequencies.

import argparse
from itertools import product

import matplotlib.pyplot as plt
from numpy import imag, ndarray, real

from utils import (
    SAMPLINGS,
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
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()
part_names = list(SAMPLINGS.keys())


def plot_comparative_Love_numbers_for_sampling(
    figure_subpath_string: str,
    real_description_id_parts: tuple[str, str] = ("p", "ns"),
    first_part_names: list[str] = part_names,
    second_part_names: list[str] = part_names,
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
    directions: list[Direction] = [Direction.radial],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.potential],
    parts: list[str] = ["real"],
    use_anelasticity: bool = True,
    use_attenuation: bool = True,
    bounded_attenuation_functions: bool = True,
):
    """
    Generates comparative figures of Love numbers for different sampling options.
    """
    # Initializes.
    run_id = gets_run_id(
        use_anelasticity=use_anelasticity,
        bounded_attenuation_functions=bounded_attenuation_functions,
        use_attenuation=use_attenuation,
    )
    figure_path = figures_path.joinpath(figure_subpath_string).joinpath(run_id)
    figure_path.mkdir(parents=True, exist_ok=True)
    part_name_products = list(product(first_part_names, second_part_names))
    results: dict[tuple[str, str], Result] = {
        (part_name_1, part_name_2): Result() for part_name_1, part_name_2 in part_name_products
    }
    T_values: dict[tuple[str, str], ndarray] = {}
    elastic_results: dict[tuple[str, str], Result] = {
        (part_name_1, part_name_2): Result() for part_name_1, part_name_2 in part_name_products
    }

    # Loads results.
    for part_name_1, part_name_2 in part_name_products:
        path = results_path.joinpath(
            "_".join((part_name_1, real_description_id_parts[0], part_name_2, real_description_id_parts[1]))
        )
        subpath = path.joinpath("runs").joinpath(run_id)
        results[part_name_1, part_name_2].load(name="anelastic_Love_numbers", path=subpath)
        T_values[part_name_1, part_name_2] = frequencies_to_periods(
            frequencies=load_base_model(name="frequencies", path=subpath)
        )
        elastic_results[part_name_1, part_name_2].load(name="elastic_Love_numbers", path=path)
    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]

    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            symbol = SYMBOLS_PER_DIRECTION[direction.value] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            for part in parts:
                _, plots = plt.subplots(len(part_names), len(part_names), figsize=(16, 12), sharex=True, sharey=True)
                for i_plot, (part_name_1, part_name_2) in enumerate(part_name_products):
                    plot = plots[i_plot // len(part_names)][i_plot % len(part_names)]
                    complex_result_values = results[part_name_1, part_name_2].values[direction][boundary_condition]
                    elastic_values = elastic_results[part_name_1, part_name_2].values[direction][boundary_condition]
                    T = T_values[part_name_1, part_name_2]
                    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                        color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                        result_values = (
                            real(complex_result_values[i_degree]) if part == "real" else imag(complex_result_values[i_degree])
                        ) / real(elastic_values[i_degree][0])
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
                    plot.grid()
                base_title = part + " part"
                plt.suptitle("$" + symbol + "/" + symbol + "^E$", fontsize=20)
                plt.savefig(figure_path.joinpath(symbol + " " + base_title + ".png"))


if __name__ == "__main__":
    plot_comparative_Love_numbers_for_sampling(
        figure_subpath_string=args.subpath if args.subpath else "Love_numbers_for_sampling"
    )
