# Generates panels of Love number graphs for some given different descriptions.
#
# A figure is generated per Love number direction, per boundary condition and per real/imaginary part.
# Every figure consist of a panel of Love number graphs, a graph per asymptotic ratio value.
# Every graphs shows Love numbers ratio to elastic version for some given degrees with respect to frequencies,
# A column of graph per given description.

import argparse

import matplotlib.pyplot as plt
from numpy import imag, ndarray, real

from utils import (
    SYMBOLS_PER_BOUNDARY_CONDITION,
    SYMBOLS_PER_DIRECTION,
    BoundaryCondition,
    Direction,
    Result,
    figures_path,
    frequencies_to_periods,
    gets_run_id_asymptotic_ratios,
    load_base_model,
    results_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def plot_comparative_Love_numbers_for_asymptotic_ratios_for_descriptions(
    real_description_ids: list[str],
    figure_subpath_string: str,
    asymptotic_ratios: list[list[int]] = [[1.0, 1.0], [0.5, 1.0], [0.2, 1.0], [0.1, 1.0], [0.05, 1.0]],
    degrees_to_plot: list[int] = [2, 3, 4, 5, 10],
    directions: list[Direction] = [Direction.radial],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.potential],
    use_anelasticity: bool = True,
):
    """
    Generates comparative figures of Love numbers for different asymptotic ratios and descriptions.
    """
    # Initializes.
    figure_path = figures_path.joinpath(figure_subpath_string)
    figure_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[list[int], Result]] = {
        real_description_id: {asymptotic_ratios_per_layer: Result() for asymptotic_ratios_per_layer in asymptotic_ratios}
        for real_description_id in real_description_ids
    }
    T_values: dict[str, dict[list[int], ndarray]] = {real_description_id: {} for real_description_id in real_description_ids}
    elastic_results: dict[str, dict[list[int], Result]] = {
        real_description_id: {asymptotic_ratios_per_layer: Result() for asymptotic_ratios_per_layer in asymptotic_ratios}
        for real_description_id in real_description_ids
    }

    # Loads results.
    for real_description_id in real_description_ids:
        path = results_path.joinpath(real_description_id)
        for asymptotic_ratios_per_layer in asymptotic_ratios:
            subpath = path.joinpath("runs").joinpath(
                gets_run_id_asymptotic_ratios(
                    use_anelasticity=use_anelasticity, asymptotic_ratios_per_layer=asymptotic_ratios_per_layer
                )
            )
            results[real_description_id][asymptotic_ratios_per_layer].load(name="anelastic_Love_numbers", path=subpath)
            T_values[real_description_id][asymptotic_ratios_per_layer] = frequencies_to_periods(
                frequencies=load_base_model(name="frequencies", path=subpath)
            )
            elastic_results[real_description_id][asymptotic_ratios_per_layer].load(name="elastic_Love_numbers", path=path)
    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]

    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            for part in ["real", "imaginary"]:
                symbol = (
                    SYMBOLS_PER_DIRECTION[direction.value] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
                )
                _, plots = plt.subplots(len(asymptotic_ratios), len(real_description_ids), figsize=(16, 10), sharex=True)
                plot_line = 0
                for asymptotic_ratios_per_layer in asymptotic_ratios:
                    for real_description_id in real_description_ids:
                        complex_result_values = results[real_description_id][asymptotic_ratios_per_layer].values[direction][
                            boundary_condition
                        ]
                        T = T_values[real_description_id][asymptotic_ratios_per_layer]
                        elastic_values = elastic_results[real_description_id][asymptotic_ratios_per_layer].values[direction][
                            boundary_condition
                        ]
                        plot = plots[plot_line][real_description_ids.index(real_description_id)]
                        for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                            color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                            result_values = (
                                real(complex_result_values[i_degree])
                                if part == "real"
                                else imag(complex_result_values[i_degree])
                            ) / real(elastic_values[i_degree][0])
                            plot.plot(T, result_values, label="n = " + str(degree), color=color)
                        # plot.set_xscale("log")
                        plot.legend(loc="upper left")
                        if plot_line == 0:
                            plot.set_title(real_description_id)
                        elif plot_line == 4:
                            plot.set_xlabel("T (y)")
                            plot.set_xscale("log")
                        if real_description_id == real_description_ids[0]:
                            plot.set_ylabel(
                                gets_run_id_asymptotic_ratios(
                                    use_anelasticity=use_anelasticity, asymptotic_ratios_per_layer=asymptotic_ratios_per_layer
                                )
                            )
                        plot.grid()
                    plot_line += 1
                    plt.suptitle("$" + symbol + "/" + symbol + "^E$ " + part + " part", fontsize=20)
                    plt.savefig(figure_path.joinpath(symbol + " " + part + " part.png"))


if __name__ == "__main__":
    plot_comparative_Love_numbers_for_asymptotic_ratios_for_descriptions(
        real_description_ids=["base-model", "PREM_test-low-viscosity-Asthenosphere_Benjamin"],
        figure_subpath_string=args.subpath if args.subpath else "Love_numbers_for_asymptotic_ratios_for_descriptions",
    )