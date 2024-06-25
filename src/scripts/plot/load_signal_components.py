from pathlib import Path

from cartopy.crs import Robinson
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import ndarray

from ...utils import (
    harmonic_geoid_trends_path,
    harmonic_load_signal_trends_before_degree_one_replacement_path,
    harmonic_load_signal_trends_path,
    harmonic_radial_displacement_trends_path,
    harmonic_residual_trends_path,
    load_complex_array_from_binary,
)
from .utils import natural_projection

ROW_PATHS: dict[str, Path] = {
    "load signal before degree one replacement": harmonic_load_signal_trends_before_degree_one_replacement_path,
    "load signal after degree one replacement": harmonic_load_signal_trends_path,
    "geoid height": harmonic_geoid_trends_path,
    "radial displacement": harmonic_radial_displacement_trends_path,
    "residuals": harmonic_residual_trends_path,
}


def generate_load_signal_components_figure(
    elastic_load_signal_id: str = "0",
    anelastic_load_signal_id: str = "2",
    rows: list[str] = [
        "geoid height",
        "radial displacement",
        "residuals",
    ],
):
    """
    Generates a figure showing maps for all components of a computed load signal.
    Columns:
        - elastic signal.
        - anelastic signal.
        - difference between the anelastic and the elastic signal.
    Rows:
        - load signal before degree one replacement.
        - load signal after degree one replacement (L).
        - geoid height (G).
        - radial displacement (R).
        - residuals L + D - (G - R) where D is the scale factor.

    """

    # Gets results.
    trend_harmonic_components_per_column: dict[str, dict[str, ndarray]] = {
        column: {row: load_complex_array_from_binary(name=load_signal_id, path=ROW_PATHS[row]) for row in rows}
        for column, load_signal_id in zip(["elastic", "anelastic"], [elastic_load_signal_id, anelastic_load_signal_id])
    }

    # Figure's configuration.
    fig = figure(layout="compressed")
    row_number = len(rows)
    ax: list[GeoAxes] = []

    # Loops on all plots to generate.
    for i_row, row in enumerate(rows):

        # Elastic and anelastic results...
        for i_column, (column, trend_harmonic_components) in enumerate(trend_harmonic_components_per_column.items()):
            # Generates plot.
            current_ax: GeoAxes = fig.add_subplot(
                row_number, 3, 3 * i_row + i_column + 1, projection=Robinson(central_longitude=180)
            )
            contour = natural_projection(ax=current_ax, harmonics=trend_harmonic_components[row].real)
            ax += [current_ax]
            # Adds layout.
            if i_row == 0:
                current_ax.set_title(column)
                # Eventually memorizes the contour for scale.
                colorbar_contour = contour
            if i_column == 0:
                current_ax.set_ylabel(row)

        # ...and their difference.
        current_ax: GeoAxes = fig.add_subplot(row_number, 3, 3 * i_row + 3, projection=Robinson(central_longitude=180))
        current_ax.set_title("difference")
        contour = natural_projection(
            ax=current_ax,
            harmonics=trend_harmonic_components_per_column["anelastic"][row].real
            - trend_harmonic_components_per_column["elastic"][row].real,
        )
        ax += [current_ax]

    # Ends figure configuration.
    cbar = fig.colorbar(colorbar_contour, ax=ax, orientation="horizontal", shrink=0.5, extend="both")
    cbar.set_label(label="mm/yr")
    show()
