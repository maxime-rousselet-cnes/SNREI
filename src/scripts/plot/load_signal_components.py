from pathlib import Path

from cartopy.crs import Robinson
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import ndarray, zeros

from ...functions import mean_on_mask
from ...utils import (
    get_ocean_mask,
    harmonic_geoid_trends_path,
    harmonic_load_signal_trends_before_degree_one_replacement_path,
    harmonic_load_signal_trends_path,
    harmonic_radial_displacement_trends_path,
    harmonic_residual_trends_path,
    load_complex_array_from_binary,
    load_load_signal_hyper_parameters,
    redefine_n_max,
)
from .utils import get_grid, natural_projection

ROW_PATHS: dict[str, Path] = {
    "harmonic": harmonic_residual_trends_path,  # harmonic_load_signal_trends_path,
    "geoid height": harmonic_geoid_trends_path,
    "radial displacement": harmonic_radial_displacement_trends_path,
    "residuals": harmonic_residual_trends_path,
}
SATURATION_THRESHOLDS: dict[str, float] = {
    "load signal": 50.0,
    "harmonic": 2.0,
    "geoid height": 2.0,
    "radial displacement": 2.0,
    "residuals": 10.0,
}


def select_degrees(harmonics: dict[str, ndarray[complex]], row: str) -> ndarray[float]:
    """
    Subfunction to avoid copy-paste.
    """
    result = harmonics[row].real
    if ("C" in row) or ("S" in row):
        result_mask = zeros(shape=result.shape)
        for harmonic_name in row.split(" "):
            symbols = harmonic_name.split("_")
            sign, degree, order = symbols[0], int(symbols[1]), int(symbols[2])
            result_mask[0 if sign == "C" else 1, degree, order] = 1.0
        result = result_mask * result
    return result


def generate_load_signal_components_figure(
    elastic_load_signal_id: str = "0",
    anelastic_load_signal_id: str = "2",
    rows: list[str] = ["C_2_1"],
    difference: bool = False,
    continents: bool = False,
):
    """
    Generates a figure showing maps for all components of a computed load signal.
    Columns:
        - elastic signal.
        - anelastic signal.
        - (OPTIONAL) difference between the anelastic and the elastic signal.
    Row options:
        - load signal
        - C_i_j
        - C_i_j C_k_l ...
        - geoid height (G).
        - radial displacement (R).
        - residuals L + D - (G - R) where D is the scale factor.

    """

    # Gets results.
    trend_harmonic_components_per_column: dict[str, dict[str, ndarray]] = {
        column: {
            row: load_complex_array_from_binary(
                name=load_signal_id,
                path=(
                    harmonic_load_signal_trends_before_degree_one_replacement_path
                    if "before" in row
                    else (
                        ROW_PATHS["harmonic"] if ("after" in row) or ("C" in row) or ("S" in row) else ROW_PATHS[row]
                    )
                ),
            )
            for row in rows
        }
        for column, load_signal_id in zip(["elastic", "anelastic"], [elastic_load_signal_id, anelastic_load_signal_id])
    }
    if difference:
        trend_harmonic_components_per_column["difference"] = {
            row: trend_harmonic_components_per_column["anelastic"][row]
            - trend_harmonic_components_per_column["elastic"][row]
            for row in rows
        }
    load_signal_hyper_parameters = load_load_signal_hyper_parameters()
    n_max = redefine_n_max(
        n_max=load_signal_hyper_parameters.n_max, harmonics=trend_harmonic_components_per_column["elastic"][rows[0]]
    )
    mask = get_ocean_mask(name=load_signal_hyper_parameters.ocean_mask, n_max=n_max)

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
                harmonics=select_degrees(harmonics=trend_harmonic_components, row=row),
                saturation_threshold=(
                    SATURATION_THRESHOLDS["load signal"]
                    if "load signal" in row
                    else (
                        SATURATION_THRESHOLDS["harmonic"]
                        if ("C" in row) or ("S" in row)
                        else SATURATION_THRESHOLDS[row]
                    )
                )
                / (5.0 if "difference" in column else 1.0),
                n_max=n_max,
                mask=mask if not continents else 1.0,
            )
            ax += [current_ax]
            # Adds layout.
            current_ax.set_title(
                column
                + ": "
                + str(
                    mean_on_mask(
                        mask=mask,
                        grid=get_grid(
                            harmonics=select_degrees(harmonics=trend_harmonic_components, row=row), n_max=n_max
                        ),
                    ),
                )
            )
            if not continents:
                current_ax.add_feature(
                    NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor="grey")
                )
            # Eventually memorizes the contour for scale.
            if column == "anelastic":
                cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", shrink=0.5, extend="both")
                cbar.set_label(label=row + " (mm/yr)")
            elif column == "difference":
                cbar = fig.colorbar(contour, ax=current_ax, orientation="horizontal", shrink=0.5, extend="both")
                cbar.set_label(label="(mm/yr)")

    show()
