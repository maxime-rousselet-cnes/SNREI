from typing import Optional

from cartopy.crs import Robinson
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import ndarray, zeros
from pyGFOToolbox.processing.filter.filter_ddk import _pool_apply_DDK_filter

from ...functions import make_grid, mean_on_mask
from ...utils import (
    LoadSignalHyperParameters,
    build_elastic_load_signal_components,
    collection_SH_data_from_map,
    harmonic_load_signal_trends_path,
    load_complex_array_from_binary,
    load_load_signal_hyper_parameters,
    map_from_collection_SH_data,
    map_sampling,
)
from .utils import get_grid, natural_projection


def select_degrees(harmonics: ndarray[float], row_components: Optional[str]) -> ndarray[float]:
    """
    Subfunction to avoid copy-paste.
    """
    result = harmonics
    if not row_components is None:
        row_components: str
        result_mask = zeros(shape=result.shape)
        for harmonic_name in row_components.split(" "):
            symbols = harmonic_name.split("_")
            sign, degree, order = symbols[0], int(symbols[1]), int(symbols[2])
            result_mask[0 if sign == "C" else 1, degree, order] = 1.0
        result = result_mask * result
    return result


def apply_optional_filter(grid: ndarray[float], ddk_level: Optional[int], n_max: int) -> ndarray[float]:
    """"""
    if not ddk_level:
        return grid
    else:
        return map_from_collection_SH_data(
            collection_data=_pool_apply_DDK_filter(
                grace_monthly_sh=collection_SH_data_from_map(
                    spatial_load_signal=grid,
                    n_max=n_max,
                ),
                ddk_filter_level=ddk_level,
            ),
            n_max=n_max,
        )


def generate_load_signal_components_figure(
    rows: list[tuple[str, Optional[str], float, float, float, Optional[int]]],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
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
    id_list = [elastic_load_signal_id, anelastic_load_signal_id]
    trend_harmonic_components_per_column: dict[tuple[str, Optional[str]], dict[str, ndarray]] = {
        column: {
            (row_name, row_components, ddk_level): load_complex_array_from_binary(
                name=load_signal_id, path=harmonic_load_signal_trends_path.joinpath(row_name)
            ).real
            for row_name, row_components, _, _, _, ddk_level in rows
        }
        for column, load_signal_id in zip(["elastic", "anelastic"], id_list)
    }
    if difference:
        trend_harmonic_components_per_column["difference"] = {
            (row_name, row_components, ddk_level): trend_harmonic_components_per_column["anelastic"][row_name, row_components, ddk_level]
            - trend_harmonic_components_per_column["elastic"][row_name, row_components, ddk_level]
            for row_name, row_components, _, _, _, ddk_level in rows
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
    for i_row, (
        row_name,
        row_components,
        elastic_saturation_threshold,
        anelastic_saturation_threshold,
        difference_saturation_threshold,
        ddk_level,
    ) in enumerate(rows):

        # Elastic and anelastic results...
        for i_column, ((column, trend_harmonic_components), threshold) in enumerate(
            zip(
                trend_harmonic_components_per_column.items(),
                [elastic_saturation_threshold, anelastic_saturation_threshold, difference_saturation_threshold],
            )
        ):
            # Generates plot.
            current_ax: GeoAxes = fig.add_subplot(
                row_number,
                3 if difference else 2,
                (3 if difference else 2) * i_row + i_column + 1,
                projection=Robinson(central_longitude=0),
            )
            mask = (
                1.0
                if continents
                else (
                    ocean_land_buffered_mask
                    if not ("residual" in row_name)
                    else (
                        ocean_land_buffered_mask
                        * (
                            load_signal_hyper_parameters.signal_threshold
                            > abs(
                                get_grid(
                                    harmonics=load_complex_array_from_binary(
                                        name=id_list[i_column % 2], path=harmonic_load_signal_trends_path.joinpath("step_3")
                                    ).real,
                                    n_max=load_signal_hyper_parameters.n_max,
                                )
                            )
                        )
                    )
                )
            )
            grid = apply_optional_filter(
                grid=(1.0 if not ("residual" in row_name) else mask)
                * make_grid(harmonics=trend_harmonic_components[row_name, row_components, ddk_level].real, n_max=load_signal_hyper_parameters.n_max),
                ddk_level=ddk_level,
                n_max=load_signal_hyper_parameters.n_max,
            )
            harmonics = select_degrees(
                harmonics=map_sampling(map=grid, n_max=load_signal_hyper_parameters.n_max, harmonic_domain=True)[0], row_components=row_components
            )
            contour, mask = natural_projection(
                ax=current_ax,
                saturation_threshold=threshold,
                latitudes=latitudes,
                longitudes=longitudes,
                harmonics=harmonics,
                mask=mask,
                n_max=load_signal_hyper_parameters.n_max,
                signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
            )
            # Adds layout.
            current_ax.set_title(
                column
                + ": "
                + str(
                    mean_on_mask(
                        mask=mask,
                        grid=make_grid(harmonics=harmonics, n_max=load_signal_hyper_parameters.n_max),
                        latitudes=latitudes,
                        n_max=load_signal_hyper_parameters.n_max,
                        signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                    ),
                )
            )
            if not continents:
                current_ax.add_feature(NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor="grey"))
            # Eventually memorizes the contour for scale.
            cbar = fig.colorbar(contour, ax=current_ax, orientation="vertical", shrink=0.9, extend="both")
            cbar.set_label(label=row_name + ": " + str(row_components) + " (mm/yr) " + str(ddk_level))

    show()
