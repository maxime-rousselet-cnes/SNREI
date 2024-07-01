from matplotlib.axes import Axes
from matplotlib.pyplot import show, subplots, suptitle

from ...functions import get_degrees_indices
from ...utils import BoundaryCondition, Direction, Love_numbers_path, Result
from .utils import SYMBOLS_PER_BOUNDARY_CONDITION, SYMBOLS_PER_DIRECTION


def generate_Love_numbers_plot(
    anelastic_id: str,
    elastic_id: str,
    degrees_to_plot: list[int] = [1],
    direction: Direction = Direction.radial,
    boundary_condition: BoundaryCondition = BoundaryCondition.load,
    linewidth: int = 2,
    grid: bool = True,
):
    """
    Generates figures of Love numbers.
    Plots real/imaginary part of Love numbers with respect to frequency.
    Every plot contains several curves, a color is used per degree.
    """
    # Loads result.
    anelastic_Love_numbers = Result()
    anelastic_Love_numbers.load(name=anelastic_id, path=Love_numbers_path)
    elastic_Love_numbers = Result()
    elastic_Love_numbers.load(name=elastic_id, path=Love_numbers_path)

    degrees_indices = get_degrees_indices(
        degrees=elastic_Love_numbers.axes["degrees"], degrees_to_plot=degrees_to_plot
    )

    # Initializes.
    symbol = SYMBOLS_PER_DIRECTION[direction] + "_n" + SYMBOLS_PER_BOUNDARY_CONDITION[boundary_condition]
    _, plots = subplots(1, 2, sharex=True)
    plot: Axes

    # Plots Love numbers.
    for part, plot in zip(["real", "imaginary"], plots):
        plot.set_xscale("log")
        for i_degree, degree in zip(degrees_indices, degrees_to_plot):
            # Gets corresponding data.
            result_values = anelastic_Love_numbers.values[direction][boundary_condition]
            # Plots.
            result_values = (
                result_values.real[i_degree] if part == "real" else result_values.imag[i_degree]
            ) / elastic_Love_numbers.values[direction][boundary_condition].real[i_degree]
            plot.plot(
                1.0 / anelastic_Love_numbers.axes["frequencies"].real,  # (yr^-1 -> yr)
                result_values,
                label="n = " + str(degree),
                linewidth=linewidth,
            )
        plot.set_title(part + " part")
        plot.set_xlabel("T (yr)")
        plot.legend(frameon=False)
        plot.tick_params(axis="x", direction="inout")
        plot.tick_params(axis="y", direction="inout")
        if grid:
            plot.grid()
    suptitle("$" + symbol + "/" + symbol + "^E$")
    show()
