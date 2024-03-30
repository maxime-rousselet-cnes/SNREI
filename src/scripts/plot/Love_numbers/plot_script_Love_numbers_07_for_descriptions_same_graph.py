# Generates panels of Love number graphs.
#
# A figure is generated per Love number direction and per boundary condition.
# Every figure consist of a panel of Love number graphs, a graph for real part and another one for imaginary part.
# Every graphs shows Love numbers (real or imaginary part) ratio to elastic version for some given degrees with respect to
# frequencies, for a set of descriptions.

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
    gets_run_id,
    load_base_model,
    results_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def plot_comparative_Love_numbers_for_descriptions_same_graph(
    real_description_ids: list[str],
    linestyle_per_description: list[str],
    key_word_per_description: list[str],
    figure_subpath_string: str,
    use_anelasticity: bool = False,
    use_attenuation: bool = True,
    bounded_attenuation_functions: bool = True,
    degrees_to_plot: list[int] = [2, 10],
    directions: list[Direction] = [Direction.potential],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.load],
):
    """
    Generates comparative figures of Love numbers for different options.
    """
    # Initializes.
    figure_path = figures_path.joinpath(figure_subpath_string)
    results: dict[str, Result] = {real_description_id: Result() for real_description_id in real_description_ids}
    T_values: dict[str, ndarray] = {}
    elastic_results: dict[str, Result] = {real_description_id: Result() for real_description_id in real_description_ids}

    # Loads results.
    for real_description_id in real_description_ids:
        path = results_path.joinpath(real_description_id)
        subpath = path.joinpath("runs").joinpath(
            gets_run_id(
                use_anelasticity=use_anelasticity,
                bounded_attenuation_functions=bounded_attenuation_functions and use_attenuation,
                use_attenuation=use_attenuation,
            )
        )
        results[real_description_id].load(name="anelastic_Love_numbers", path=subpath)
        T_values[real_description_id] = frequencies_to_periods(frequencies=load_base_model(name="frequencies", path=subpath))
        elastic_results[real_description_id].load(name="elastic_Love_numbers", path=path)
    degrees: list[int] = load_base_model(name="degrees", path=path)
    degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]
    # Plots Love numbers.
    for direction in directions:
        for boundary_condition in boundary_conditions:
            symbol = SYMBOLS_PER_DIRECTION[direction.value] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition.value]
            for zoom_in in BOOLEANS:
                _, plots = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
                for part in ["real", "imaginary"]:
                    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
                        for real_description_id, linestyle, key_word in zip(
                            real_description_ids, linestyle_per_description, key_word_per_description
                        ):
                            # Gets corresponding data.
                            complex_result_values = results[real_description_id].values[direction][boundary_condition]
                            T = T_values[real_description_id]
                            elastic_values = elastic_results[real_description_id].values[direction][boundary_condition]
                            # Eventually restricts frequency range.
                            min_frequency_index = -1 if not zoom_in else where(T >= 1e0)[0][-1]
                            max_frequency_index = 0 if not zoom_in else where(T <= 2.5e3)[0][0]
                            plot = plots[0 if part == "real" else 1]
                            color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                            result_values = (
                                real(complex_result_values[i_degree])
                                if part == "real"
                                else imag(complex_result_values[i_degree])
                            ) / real(elastic_values[i_degree][0])
                            plot.plot(
                                T[max_frequency_index:min_frequency_index],
                                result_values[max_frequency_index:min_frequency_index],
                                label="n = " + str(degree) + " : " + key_word,
                                color=color,
                                linestyle=linestyle,
                            )
                    # plot.set_xscale("log")
                    plot.legend()
                    plot.set_title(part + " part")
                    plot.set_xlabel("T (y)")
                    plot.set_xscale("log")
                    plot.grid()
                plt.suptitle("$" + symbol + "/" + symbol + "^E$", fontsize=20)
                path = figure_path.joinpath(real_description_id)
                path.mkdir(parents=True, exist_ok=True)
                plt.savefig(path.joinpath(symbol + (" zoom in " if zoom_in else "") + ".png"))
                plt.show()


if __name__ == "__main__":
    plot_comparative_Love_numbers_for_descriptions_same_graph(
        real_description_ids=[
            "PREM_low-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio0.2-1.0",
            "PREM_failed-low-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio0.2-1.0",
        ],
        linestyle_per_description=["--", ":"],
        key_word_per_description=["corrected", "failed"],
        figure_subpath_string=args.subpath if args.subpath else "Love_numbers_for_descriptions_same_graph",
    )
