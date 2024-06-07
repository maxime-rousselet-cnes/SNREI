from copy import deepcopy
from typing import Any, Callable

from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from numpy import array, errstate, exp, log, nan_to_num, ndarray, pi, sin, sum, where

from .constants import (
    EARTHQUAKE_CORNERS,
    OTHER_PARAMETERS_NUMBER,
    PARAMETER_BOUNDS,
    PERIODIC_PARAMETERS_NUMBER,
    SEISMIC_PARAMETERS_NUMBER,
)


def select_area(
    time_dependent_maps: ndarray,
    lat: ndarray,
    lon: ndarray,
    upper_left_corner: tuple[float, float],
    lower_right_corner: tuple[float, float],
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Select a rectangular subset.
    """
    latitude_indices = (lat <= upper_left_corner[0]) * (lat >= lower_right_corner[0])
    longitude_indices = (lon >= upper_left_corner[1]) * (lon <= lower_right_corner[1])
    return time_dependent_maps[:, latitude_indices][:, :, longitude_indices], latitude_indices, longitude_indices


def get_parameter(parameters: Parameters, index: int, parameter_base_name: str) -> float:
    """
    Gets 'parameter_base_name' parameter for index-th function.
    """
    return parameters[parameter_base_name + "_" + str(index)]


def co_seismic_signal(t: ndarray[float], parameters: Parameters, i_earthquake: int) -> ndarray[float]:
    """
    Describe a co-seismic signal on EWH time-serie.
    """
    return get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="co_seismic_amplitude") * array(
        object=t >= get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="earthquake_t_0"),
        dtype=float,
    )


def post_seismic_signal(t: ndarray[float], parameters: Parameters, i_earthquake: int) -> ndarray[float]:
    """
    Describe a post-seismic signal on EWH time-serie.
    """
    with errstate(invalid="ignore", divide="ignore"):
        return array(
            object=t >= get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="earthquake_t_0"),
            dtype=float,
        ) * nan_to_num(
            x=(
                get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="post_seismic_exp_amplitude")
                * exp(
                    -(t - get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="earthquake_t_0"))
                    / get_parameter(
                        parameters=parameters, index=i_earthquake, parameter_base_name="post_seismic_exp_relaxation_time"
                    )
                )
                + get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="post_seismic_log_amplitude")
                * log(
                    1
                    + (t - get_parameter(parameters=parameters, index=i_earthquake, parameter_base_name="earthquake_t_0"))
                    / get_parameter(
                        parameters=parameters, index=i_earthquake, parameter_base_name="post_seismic_log_relaxation_time"
                    )
                )
            ),
            nan=0.0,
        )


def sinusoidal_signal(t: ndarray[float], parameters: Parameters, harmonic: int) -> ndarray[float]:
    """
    Describes a sinusoidal signal, such as the annual or semi-annual signal.
    """
    return get_parameter(parameters=parameters, index=harmonic, parameter_base_name="sinusoidal_amplitude") * sin(
        2
        * pi
        * harmonic
        * (t - t[0] - get_parameter(parameters=parameters, index=harmonic, parameter_base_name="sinusoidal_delay"))
    )


def secular_signal(t: ndarray[float], parameters: Parameters) -> ndarray[float]:
    """
    Describes a secular signal.
    """
    return parameters["secular_slope"] * (t - t[0]) + parameters["additive_constant"]


def single_earthquake_signal(t: ndarray[float], parameters: Parameters, i_earthquake: int) -> ndarray[float]:
    """
    Describes a co-seismic and a post-seismic signal. Can be used to fit level-3 GRACE data. Does not contain periodic signal.
    """
    return co_seismic_signal(t=t, parameters=parameters, i_earthquake=i_earthquake) + post_seismic_signal(
        t=t, parameters=parameters, i_earthquake=i_earthquake
    )


def multiple_earthquake_signal(t: ndarray[float], parameters: Parameters) -> ndarray[float]:
    """
    Describes a signal with multiple earthquakes contributions. Can be used to fit level-3 GRACE data. Does not contain
    periodic signal.
    """
    return sum(
        a=[
            single_earthquake_signal(t=t, parameters=parameters, i_earthquake=i_earthquake)
            for i_earthquake in range(
                (len(parameters.items()) - (OTHER_PARAMETERS_NUMBER + 2 * PERIODIC_PARAMETERS_NUMBER))
                // SEISMIC_PARAMETERS_NUMBER
            )
        ],
        axis=0,
    )


def full_earthquake_signal(t: ndarray[float], parameters: Parameters) -> ndarray[float]:
    """
    Describes a geophysical signal with annual and semi-annual variations and earthquakes signatures. Can be used to fit
    level-3 GRACE data.
    """
    return (
        secular_signal(t=t, parameters=parameters)
        + sinusoidal_signal(t=t, parameters=parameters, harmonic=1)
        + sinusoidal_signal(t=t, parameters=parameters, harmonic=2)
        + multiple_earthquake_signal(t=t, parameters=parameters)
    )


def fit_expression(
    times: ndarray[float], time_serie: ndarray, expression: Callable[[Any], Any], parameters: Parameters
) -> dict[str, float]:
    """
    Fits a function of time to data.
    """
    result: MinimizerResult = minimize(
        fcn=lambda params, x, y: (expression(x, params) - y) ** 2,
        params=parameters,
        args=(times, time_serie),
        method="differential_evolution",
    )
    params: Parameters = result.params
    return params.valuesdict()


def fit_earthquakes(
    time_dependent_maps: ndarray,
    times: ndarray,
    lat: ndarray,
    lon: ndarray,
    corners: dict[str, dict[str, tuple[float, float]]] = EARTHQUAKE_CORNERS,
) -> dict[str, dict[str, list[list[dict[str, float]]]]]:
    """
    Fits all the parameters needed to describe the earthquakes on GRACE level-3 data.
    """

    # Gets restricted areas where fitting is needed.
    restricted_areas_data = {
        area: select_area(
            time_dependent_maps=time_dependent_maps,
            lat=lat,
            lon=lon,
            upper_left_corner=sub_dict_corners["upper_left"],
            lower_right_corner=sub_dict_corners["lower_right"],
        )
        for area, sub_dict_corners in corners.items()
    }

    # Fits data accordingly to earthquake signature expressions.
    fitted_parameters_per_area = {}
    for area, (time_dependent_map, selected_latitudes_indices, selected_longitudes_indices) in restricted_areas_data.items():

        # Creates a parameterization class instance to optimize.
        parameters = Parameters()
        for parameter_name in PARAMETER_BOUNDS.keys():
            if "sinusoidal" in parameter_name:
                for harmonic in range(1, 3):
                    parameters.add(
                        name=parameter_name + "_" + str(harmonic),
                        min=PARAMETER_BOUNDS[parameter_name]["lower_bound"],
                        max=PARAMETER_BOUNDS[parameter_name]["upper_bound"],
                    )
            elif ("earthquake" in parameter_name) or ("seismic" in parameter_name):
                for i_earthquake in range(4 if area == "Sumatra" else 1):
                    parameters.add(
                        name=parameter_name + "_" + str(i_earthquake),
                        min=PARAMETER_BOUNDS[parameter_name]["lower_bound"],
                        max=PARAMETER_BOUNDS[parameter_name]["upper_bound"],
                    )
            else:
                parameters.add(
                    name=parameter_name,
                    min=PARAMETER_BOUNDS[parameter_name]["lower_bound"],
                    max=PARAMETER_BOUNDS[parameter_name]["upper_bound"],
                )
        fitted_parameters_per_area[area] = {
            "fitted_parameters": [
                [
                    fit_expression(
                        times=times,
                        time_serie=time_serie,
                        expression=full_earthquake_signal,
                        parameters=parameters,
                    )
                    for time_serie in line_time_series
                ]
                for line_time_series in time_dependent_map.transpose((1, 2, 0))
            ],
            "latitudes": selected_latitudes_indices,
            "longitudes": selected_longitudes_indices,
        }

    return fitted_parameters_per_area


def remove_earthquakes(
    time_dependent_maps: ndarray,
    times: ndarray,
    lat: ndarray,
    lon: ndarray,
    corners: dict[str, dict[str, tuple[float, float]]] = EARTHQUAKE_CORNERS,
) -> tuple[dict[str, dict[str, list[list[dict[str, float]]]]], ndarray]:
    """
    Removes earthquakes signal from GRACE level-3 data.
    """

    # Fits the signal parameters on the wanted earthqakes areas.
    fitted_parameters_per_area = fit_earthquakes(
        time_dependent_maps=time_dependent_maps, times=times, lat=lat, lon=lon, corners=corners
    )

    # Remove modelized signal from real data.
    corrected_time_dependent_maps = deepcopy(time_dependent_maps)
    for _, area_fitted_parameters in fitted_parameters_per_area.items():
        for latitude, fitted_parameters_line in zip(
            where(area_fitted_parameters["latitudes"])[0], area_fitted_parameters["fitted_parameters"]
        ):
            for longitude, fitted_parameters in zip(where(area_fitted_parameters["longitudes"])[0], fitted_parameters_line):
                corrected_time_dependent_maps[:, latitude, longitude] -= multiple_earthquake_signal(
                    t=times, parameters=fitted_parameters
                )

    return fitted_parameters_per_area, corrected_time_dependent_maps
