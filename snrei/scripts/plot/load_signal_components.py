from typing import Optional

from cartopy.crs import Robinson
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import inf, ndarray, zeros

from ...functions import mean_on_mask
from ...utils import (
    LoadSignalHyperParameters,
    build_elastic_load_signal_components,
    harmonic_load_signal_trends_path,
    load_complex_array_from_binary,
    load_load_signal_hyper_parameters,
)
from .utils import get_grid, natural_projection


def select_degrees(harmonics: dict[str, ndarray[complex]], row_name: str, row_components: Optional[str]) -> ndarray[float]:
    """
    Subfunction to avoid copy-paste.
    """
    result = harmonics[row_name, row_components].real
    if not row_components is None:
        row_components: str
        result_mask = zeros(shape=result.shape)
        for harmonic_name in row_components.split(" "):
            symbols = harmonic_name.split("_")
            sign, degree, order = symbols[0], int(symbols[1]), int(symbols[2])
            result_mask[0 if sign == "C" else 1, degree, order] = 1.0
        result = result_mask * result
    return result


def generate_load_signal_components_figure(
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    rows: list[tuple[str, Optional[str], float]] = [
        ("step_2", None, 50.0, 10.0),
        ("step_3", None, 50.0, 10.0),
        # ("continental_leakage", None, 5.0),
        # ("oceanic_leakage", None, 5.0),
    ],  # Plot row names, components and their corresponding saturation thresholds
    elastic_load_signal_id: str = "0",
    anelastic_load_signal_id: str = "2",
    difference: bool = True,
    continents: bool = False,
):
    """
    Generates a figure showing maps for all components of a computed load signal.
    Columns:
        - elastic signal.
        - anelastic signal.
        - (OPTIONAL) difference between the anelastic and the elastic signal.
    One may select specific harmonic components of a signal. The corresponding way to write row is:
        ("residual, "C_0_1 C_1_1 S_1_1", 10.0, 2.0) for 10.0 mm/yr absolute saturation threshold and 2.0 for difference threshold.
    """

    # Gets results.
    trend_harmonic_components_per_column: dict[tuple[str, Optional[str]], dict[str, ndarray]] = {
        column: {
            (row_name, row_components): load_complex_array_from_binary(name=load_signal_id, path=harmonic_load_signal_trends_path.joinpath(row_name))
            for row_name, row_components, _, _ in rows
        }
        for column, load_signal_id in zip(["elastic", "anelastic"], [elastic_load_signal_id, anelastic_load_signal_id])
    }
    if difference:
        trend_harmonic_components_per_column["difference"] = {
            (row_name, row_components): trend_harmonic_components_per_column["anelastic"][row_name, row_components]
            - trend_harmonic_components_per_column["elastic"][row_name, row_components]
            for row_name, row_components, _, _ in rows
        }
    (
        load_signal_hyper_parameters.n_max,
        _,
        _,
        _,
        _,
        ocean_land_buffered_mask,
        latitudes,
        longitudes,
    ) = build_elastic_load_signal_components(load_signal_hyper_parameters=load_signal_hyper_parameters)

    # Figure's configuration.
    fig = figure(layout="compressed", figsize=(17, 15))
    row_number = len(rows)

    # Loops on all plots to generate.
    for i_row, (row_name, row_components, row_saturation_threshold, difference_saturation_threshold) in enumerate(rows):

        ax: list[GeoAxes] = []

        # Elastic and anelastic results...
        for i_column, (column, trend_harmonic_components) in enumerate(trend_harmonic_components_per_column.items()):
            # Generates plot.
            current_ax: GeoAxes = fig.add_subplot(
                row_number,
                3 if difference else 2,
                (3 if difference else 2) * i_row + i_column + 1,
                projection=Robinson(central_longitude=0),
            )
            contour = natural_projection(
                ax=current_ax,
                saturation_threshold=difference_saturation_threshold if column == "difference" else row_saturation_threshold,
                latitudes=latitudes,
                longitudes=longitudes,
                harmonics=select_degrees(harmonics=trend_harmonic_components, row_name=row_name, row_components=row_components),
                mask=ocean_land_buffered_mask if not continents else 1.0,
                n_max=load_signal_hyper_parameters.n_max,
            )
            ax += [current_ax]
            # Adds layout.
            current_ax.set_title(
                column
                + ": "
                + str(
                    mean_on_mask(
                        signal_threshold=inf,
                        mask=ocean_land_buffered_mask,
                        grid=get_grid(
                            harmonics=select_degrees(harmonics=trend_harmonic_components, row_name=row_name, row_components=row_components),
                            n_max=load_signal_hyper_parameters.n_max,
                        ),
                        latitudes=latitudes,
                        n_max=load_signal_hyper_parameters.n_max,
                    ),
                )
            )
            if not continents:
                current_ax.add_feature(NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor="grey"))
            # Eventually memorizes the contour for scale.
            if column == "anelastic":
                cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.9, extend="both")
                cbar.set_label(label=row_name + ": " + str(row_components) + " (mm/yr)")
            elif column == "difference":
                cbar = fig.colorbar(contour, ax=current_ax, orientation="vertical", shrink=0.8, extend="both")
                cbar.set_label(label="(mm/yr)")

    show()
