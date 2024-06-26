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
SATURATION_THRESHOLDS: dict[str, float] = {
    "load signal before degree one replacement": 50.0,
    "load signal after degree one replacement": 50.0,
    "geoid height": 2.0,
    "radial displacement": 2.0,
    "residuals": 20.0,
}


def generate_load_signal_components_figure(
    elastic_load_signal_id: str = "0",
    anelastic_load_signal_id: str = "2",
    rows: list[str] = ["load signal before degree one replacement", "load signal after degree one replacement"],
    difference: bool = True,
):
    """
    Generates a figure showing maps for all components of a computed load signal.
    Columns:
        - elastic signal.
        - anelastic signal.
        - (OPTIONAL) difference between the anelastic and the elastic signal.
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

    # Loops on all plots to generate.
    for i_row, row in enumerate(rows):

        ax: list[GeoAxes] = []

        # Elastic and anelastic results...
        for i_column, (column, trend_harmonic_components) in enumerate(trend_harmonic_components_per_column.items()):
            # Generates plot.
            current_ax: GeoAxes = fig.add_subplot(
                row_number,
                3 if difference else 2,
                (3 if difference else 2) * i_row + i_column + 1,
                projection=Robinson(central_longitude=180),
            )
            contour = natural_projection(
                ax=current_ax,
                harmonics=trend_harmonic_components[row].real,
                saturation_threshold=SATURATION_THRESHOLDS[row],
            )
            ax += [current_ax]
            # Adds layout.
            current_ax.set_title(column)
            # Eventually memorizes the contour for scale.
            if column == "anelastic":
                colorbar_contour = contour

        # ...and their difference.
        if difference:
            current_ax: GeoAxes = fig.add_subplot(
                row_number, 3, 3 * i_row + 3, projection=Robinson(central_longitude=180)
            )
            if i_row == 0:
                current_ax.set_title("difference")
            contour = natural_projection(
                ax=current_ax,
                harmonics=trend_harmonic_components_per_column["anelastic"][row].real
                - trend_harmonic_components_per_column["elastic"][row].real,
                saturation_threshold=SATURATION_THRESHOLDS[row],
            )
            ax += [current_ax]

        # Ends figure configuration.
        cbar = fig.colorbar(colorbar_contour, ax=ax, orientation="horizontal", shrink=0.5, extend="both")
        cbar.set_label(label=row + " - mm/yr")

    show()
