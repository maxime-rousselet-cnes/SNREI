# Generates panels of Love number graphs for a given description.
#
# A figure is generated per Love number direction and per boundary condition.
# Every figure consist of a panel of Love number graphs, a graph for real part and another one for imaginary part.
# Every graphs shows Love numbers (real or imaginary part) ratio to elastic version for some given degrees with respect to
# frequencies, for a set of options.

import argparse
from itertools import product

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
parser.add_argument("--real_description_id", type=str, help="wanted ID for the real description to load")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()


def plot_comparative_Love_numbers_for_options_same_graph(
    real_description_id: str,
    figure_subpath_string: str,
    bounded_attenuation_functions: bool = True,
    degrees_to_plot: list[int] = [2, 10],
    directions: list[Direction] = [Direction.potential],
    boundary_conditions: list[BoundaryCondition] = [BoundaryCondition.load],
):
    """
    Generates comparative figures of Love numbers for different options.
    """
    # Initializes.
    figure_path = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    options_list = list(product(BOOLEANS, BOOLEANS))
    results: dict[tuple[bool, bool, bool], Result] = {
        (use_anelasticity, use_attenuation): Result() for use_anelasticity, use_attenuation in options_list
    }
    T_values: dict[tuple[bool, bool, bool], ndarray] = {}
    elastic_results = Result()

    # Loads results.
    path = results_path.joinpath(real_description_id)
    for use_anelasticity, use_attenuation in options_list:
        if (not use_attenuation) and (not use_anelasticity):
            continue
        subpath = path.joinpath("runs").joinpath(
            gets_run_id(
                use_anelasticity=use_anelasticity,
                bounded_attenuation_functions=use_attenuation and bounded_attenuation_functions,
                use_attenuation=use_attenuation,
            )
        )
        results[use_anelasticity, use_attenuation].load(name="anelastic_Love_numbers", path=subpath)
        T_values[use_anelasticity, use_attenuation] = frequencies_to_periods(
            frequencies=load_base_model(name="frequencies", path=subpath)
        )
    elastic_results.load(name="elastic_Love_numbers", path=path)
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
                        for use_anelasticity, use_attenuation in options_list:
                            if (not use_attenuation) and (not use_anelasticity):
                                continue
                            # Gets corresponding data.
                            complex_result_values = results[use_anelasticity, use_attenuation].values[direction][
                                boundary_condition
                            ]
                            T = T_values[use_anelasticity, use_attenuation]
                            elastic_values = elastic_results.values[direction][boundary_condition]
                            # Eventually restricts frequency range.
                            min_frequency_index = -1 if not zoom_in else where(T >= 1e0)[0][-1]
                            max_frequency_index = 0 if not zoom_in else where(T <= 2.5e3)[0][0]
                            plot = plots[0 if part == "real" else 1]
                            color = (degrees_to_plot.index(degree) / len(degrees_to_plot), 0.0, 1.0)
                            linestyle = (
                                ":" if use_anelasticity and not use_attenuation else ("--" if not use_anelasticity else "-.")
                            )
                            result_values = (
                                real(complex_result_values[i_degree])
                                if part == "real"
                                else imag(complex_result_values[i_degree])
                            ) / real(elastic_values[i_degree][0])
                            plot.plot(
                                T[max_frequency_index:min_frequency_index],
                                result_values[max_frequency_index:min_frequency_index],
                                label="n = "
                                + str(degree)
                                + " : "
                                + " ".join(
                                    (
                                        "long-term viscosity" if use_anelasticity else "",
                                        "attenuation" if use_attenuation else "",
                                    )
                                ),
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
                path = figure_path.joinpath(("" if bounded_attenuation_functions else "un") + "bounded")
                path.mkdir(parents=True, exist_ok=True)
                plt.savefig(path.joinpath(symbol + (" zoom in " if zoom_in else "") + ".png"))
                plt.show()


if __name__ == "__main__":
    plot_comparative_Love_numbers_for_options_same_graph(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_low-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio1.0-1.0"
        ),
        figure_subpath_string=args.subpath if args.subpath else "Love_numbers_for_options_same_graph",
    )
