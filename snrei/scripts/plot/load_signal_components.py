from typing import Optional

from cartopy.crs import Robinson
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import linspace, ndarray, zeros

from ...functions import mean_on_mask
from ...utils import (
    get_ocean_mask,
    harmonic_load_signal_trends_path,
    load_complex_array_from_binary,
    load_load_signal_hyper_parameters,
    redefine_n_max,
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
    latitudes: ndarray[float] = linspace(90, -90, 181),
    longitudes: ndarray[float] = linspace(0, 360, 361),
    rows: list[tuple[str, Optional[str], float]] = [
        ("step_2", None, 50.0),
        ("step_3", None, 50.0),
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
    One may select specific harmonic components of a signal. The corresponding way to write row name is:
        "residual: C_0_1 C_1_1 S_1_1"
    """

    # Gets results.
    trend_harmonic_components_per_column: dict[tuple[str, Optional[str]], dict[str, ndarray]] = {
        column: {
            (row_name, row_components): load_complex_array_from_binary(name=load_signal_id, path=harmonic_load_signal_trends_path.joinpath(row_name))
            for row_name, row_components, _ in rows
        }
        for column, load_signal_id in zip(["elastic", "anelastic"], [elastic_load_signal_id, anelastic_load_signal_id])
    }
    if difference:
        trend_harmonic_components_per_column["difference"] = {
            (row_name, row_components): trend_harmonic_components_per_column["anelastic"][row_name, row_components]
            - trend_harmonic_components_per_column["elastic"][row_name, row_components]
            for row_name, row_components, _ in rows
        }
    load_signal_hyper_parameters = load_load_signal_hyper_parameters()
    n_max = redefine_n_max(
        n_max=load_signal_hyper_parameters.n_max,
        harmonics=trend_harmonic_components_per_column["elastic"][rows[0][:2]],
    )
    # TODO: get it from build...
    mask = get_ocean_mask(name="0", n_max=n_max)

    # Figure's configuration.
    fig = figure(layout="compressed")
    row_number = len(rows)

    # Loops on all plots to generate.
    for i_row, (row_name, row_components, row_saturation_threshold) in enumerate(rows):

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
                saturation_threshold=row_saturation_threshold,
                latitudes=latitudes,
                longitudes=longitudes,
                harmonics=select_degrees(harmonics=trend_harmonic_components, row_name=row_name, row_components=row_components),
                mask=mask if not continents else 1.0,
                n_max=load_signal_hyper_parameters.n_max,
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
                            harmonics=select_degrees(harmonics=trend_harmonic_components, row_name=row_name, row_components=row_components),
                            n_max=n_max,
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
                cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", shrink=0.5, extend="both")
                cbar.set_label(label=row_name + ": " + str(row_components) + " (mm/yr)")
            elif column == "difference":
                cbar = fig.colorbar(contour, ax=current_ax, orientation="horizontal", shrink=0.5, extend="both")
                cbar.set_label(label="(mm/yr)")

    show()
