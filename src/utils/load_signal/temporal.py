from pathlib import Path
from typing import Optional

import netCDF4
from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    conjugate,
    cos,
    expand_dims,
    flip,
    linspace,
    log2,
    mean,
    ndarray,
    newaxis,
    ones,
    pi,
    prod,
    real,
    round,
    transpose,
    unique,
    vstack,
    where,
    zeros,
)
from numpy.linalg import pinv
from pandas import read_csv
from pyshtools.expand import MakeGridDH, SHExpandDH
from scipy import interpolate
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm

from ..classes import BoundaryCondition, Direction, Result, SignalHyperParameters
from ..constants import SECONDS_PER_YEAR
from ..database import load_base_model, save_base_model
from ..Love_numbers import gets_run_id
from ..paths import data_path, results_path


def anelastic_induced_load_signal_per_degree():
    return


def extract_ocean_load_data(path: Path = data_path) -> tuple[ndarray[float], ndarray[float]]:
    """
    Opens Frederikse et al.'s file and formats its data. Mean load in equivalent water height with respect to time.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath("data_Frederikse").joinpath("global_basin_timeseries.csv"), sep=",")
    dates = df["Unnamed: 0"].values
    # Formats.
    sum_values = array(
        object=[float(item.split(",")[0] + "." + item.split(",")[1]) for item in df["Sum of contributors [mean]"].values],
        dtype=float,
    )
    steric_values = array(
        object=[float(item.split(",")[0] + "." + item.split(",")[1]) for item in df["Steric [mean]"].values],
        dtype=float,
    )
    barystatic = sum_values - steric_values
    return dates, barystatic - barystatic[0]


def erase_area(
    map: ndarray[float],
    lat: ndarray[float],
    lon: ndarray[float],
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
):
    """
    ERASES a rectangle area on a (latitude, longitude) map.
    """
    return map * (
        1
        - expand_dims(a=flip(m=(lat > min_latitude) * (lat < max_latitude), axis=0), axis=1)
        * expand_dims(a=(lon > (min_longitude % 360)) * (lon < (max_longitude % 360)), axis=0)
    )


def extract_land_ocean_mask(path: Path = data_path) -> ndarray:
    """
    Opens NASA's nc file for land/sea mask and formats its data.
    """
    # Gets raw data.
    ds = netCDF4.Dataset(path.joinpath("IMERG_land_sea_mask.nc"))
    map = flip(m=ds.variables["landseamask"], axis=0).data

    # reject big lakes.
    lat = array(object=[latitude for latitude in ds.variables["lat"]])
    lon = array(object=[longitude for longitude in ds.variables["lon"]])
    # Lake Superior.
    map = erase_area(
        map=map,
        lat=lat,
        lon=lon,
        min_latitude=41.375963,
        max_latitude=50.582521,
        min_longitude=-93.748270,
        max_longitude=-75.225322,
    )
    # Lake Victoria.
    map = erase_area(
        map=map,
        lat=lat,
        lon=lon,
        min_latitude=-2.809322,
        max_latitude=0.836983,
        min_longitude=31.207700,
        max_longitude=34.530942,
    )
    # Caspian Sea.
    map = erase_area(
        map=map,
        lat=lat,
        lon=lon,
        min_latitude=35.569650,
        max_latitude=47.844035,
        min_longitude=44.303403,
        max_longitude=60.937192,
    )

    return map


def extract_GRACE_trends(filename: str, path: Path = data_path.joinpath("trends_GRACE")) -> ndarray:
    """
    Opens and formats GRACE trends CSV datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(filename), skiprows=3 if filename.split(".")[-1] == "xyz" else 11, sep=";")
    return flip(m=[[value for value in df["EWH"][df["lat"] == lat]] for lat in unique(ar=df["lat"])], axis=0)[1:, 1:]


def extract_ocean_mean_mask(filename: str, path: Path = data_path) -> ndarray:
    """
    Opens and formats ocean mask CSV datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(filename), sep=",")
    return flip(m=[[value for value in df["mask"][df["lat"] == lat]] for lat in unique(ar=df["lat"])], axis=0)[1:, 1:]


def normalize_map(
    map: ndarray,
) -> ndarray:
    """
    Sets global mean as zero and max as one by homothety.
    """
    n_t = len(map) * len(map[0])
    sum_map = sum(sum(map))
    max_map = max([max(row) for row in map])
    return map / (max_map - sum_map / (n_t)) + 1.0 / (1.0 - max_map * n_t / sum_map)


def get_trend(
    dates: ndarray[float], load_signal_hyper_parameters: SignalHyperParameters, signal: ndarray[float]
) -> tuple[ndarray[bool], ndarray[float], ndarray[float], tuple[float, float]]:
    """
    Returns the trend and the selected dates, indices and signal to compute so.
    """
    trend_indices = dates >= load_signal_hyper_parameters.first_year_for_trend
    return (
        trend_indices,
        dates[trend_indices],
        signal[trend_indices],
        signal_trend(
            trend_dates=dates[trend_indices],
            signal=signal[trend_indices],
        ),
    )


def build_uniform_elastic_load_signal(
    dates: ndarray[float],
    load: ndarray[float],
    signal_hyper_parameters: SignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], float, float]:
    """
    Builds an artificial signal history that has mean value, antisymetry and no Gibbs effect.
    """
    # Linearly extends the signal for last years.
    trend_indices = dates >= signal_hyper_parameters.first_year_for_trend
    extend_part_slope, extend_part_constant = signal_trend(
        trend_dates=dates[trend_indices],
        signal=load[trend_indices],
    )
    extend_dates = arange(dates[-1] + 1, signal_hyper_parameters.last_year_for_trend + 1)
    extend_part = extend_part_slope * extend_dates + extend_part_constant
    # Creates cubic spline for antisymetry.
    mean_slope = extend_part[-1] / signal_hyper_parameters.spline_time
    spline = lambda T: mean_slope / (2.0 * signal_hyper_parameters.spline_time**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    # Builds signal history / Creates a constant step at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(signal_hyper_parameters.zero_duration)),
            load,
            extend_part,
            spline(T=arange(start=-signal_hyper_parameters.spline_time, stop=0)),
        )
    )
    # Applies antisymetry.
    extended_time_serie = concatenate((extended_time_serie_past, [0], -flip(m=extended_time_serie_past)))
    # Deduces dates axis.
    n_extended_signal = len(extended_time_serie)
    extended_dates = arange(stop=n_extended_signal) - n_extended_signal // 2
    # Interpolates at sufficient sampling for no Gibbs effect.
    n_log_min_no_Gibbs = round(ceil(log2(n_extended_signal)))
    half_signal_period = max(extended_dates)
    n_signal = int(2 ** (n_log_min_no_Gibbs + signal_hyper_parameters.anti_Gibbs_effect_factor))
    dates = linspace(-half_signal_period, stop=half_signal_period, num=n_signal)

    return (
        dates,
        interpolate.splev(x=dates, tck=interpolate.splrep(x=extended_dates, y=extended_time_serie, k=3)),  # Signal.
        2.0 * half_signal_period / n_signal,  # Time step.
        extend_part_slope,  # Trend.
    )


def re_sample_map(map: ndarray[float], n_max: int) -> ndarray[float]:
    """
    Redefined a (latitude, longitude) map definition.
    """
    return MakeGridDH(
        SHExpandDH(
            map,
            sampling=2,
            lmax_calc=min(n_max, (len(map) - 1) // 2),
        ),
        sampling=2,
    )


def build_elastic_load_signal(
    signal_hyper_parameters: SignalHyperParameters, get_harmonic_weights: bool = False
) -> tuple[ndarray[float], ndarray[float], tuple[float, ndarray[float], ndarray[complex], Optional[ndarray[float]]] | Path]:
    """
    Builds load history in frequential domain, eventually in frequential-harmonic domain.
    Returns:
        - dates
        - frequencies
        - For SLR/GRACE load history, a frequential-harmonic load history: i.e. path of the folder containing a function of
        omega per harmonic in (.JSON) files.for Frederikse et al.'s ocean uniform load, a tuple of :
            - temporal elastic signal trend
            - a uniform temporal elastic load history
            - a uniform frequential elastic load history
            - static harmonic weights if needed.
    TODO :
    # Gets temporal load trend.
    _, _, _, (elastic_load_signal_trend, _) = get_trend(
        dates=dates, load_signal_hyper_parameters=load_signal_hyper_parameters, signal=temporal_elastic_load_signal
    ) as -2-th output plz.
    """
    if signal_hyper_parameters.signal == "ocean_load":
        # Builds frequencial signal.
        dates, load = extract_ocean_load_data()
        dates, temporal_elastic_load_signal, time_step, elastic_trend = build_uniform_elastic_load_signal(
            dates=dates,
            load=load,
            signal_hyper_parameters=signal_hyper_parameters,
        )  # (y).
        frequencial_elastic_load_signal = fft(x=temporal_elastic_load_signal)
        frequencies = fftfreq(n=len(frequencial_elastic_load_signal), d=time_step)
        spatial_weights = (
            None
            if not get_harmonic_weights
            else re_sample_map(
                map=(
                    normalize_map(map=extract_land_ocean_mask())
                    if signal_hyper_parameters.weights_map == "mask"
                    else extract_GRACE_trends(filename=signal_hyper_parameters.GRACE)
                ),
                n_max=signal_hyper_parameters.n_max,
            )
        )
        # Eventually gets harmonics.
        return (
            dates,
            frequencies,
            (
                elastic_trend,
                temporal_elastic_load_signal,
                frequencial_elastic_load_signal,
                (
                    None
                    if not get_harmonic_weights
                    else SHExpandDH(
                        spatial_weights,
                        sampling=2,
                        lmax_calc=min(signal_hyper_parameters.n_max - 1, (len(spatial_weights) - 1) // 2),
                    )
                ),
            ),
        )
    else:
        # TODO: Get GRACE's data history ?
        pass


def build_hermitian(signal: ndarray[complex]) -> ndarray[complex]:
    """
    For a given signal defined for positive values, builds the corresponding extended signal that has hermitian symetry.
    """
    return concatenate((conjugate(flip(m=signal)), signal))


def get_trend_dates(
    dates: ndarray[float],
    signal_hyper_parameters: SignalHyperParameters,
) -> tuple[ndarray[float], ndarray[int]]:
    """
    Returns trend indices and trend dates.
    """
    shift_dates = dates + signal_hyper_parameters.spline_time + signal_hyper_parameters.last_year_for_trend
    trend_indices = where(
        (shift_dates <= signal_hyper_parameters.last_year_for_trend - 1)
        * (shift_dates >= signal_hyper_parameters.first_year_for_trend)
    )[0]
    return trend_indices, shift_dates[trend_indices]


def signal_trend(trend_dates: ndarray[float], signal: ndarray[float]) -> tuple[float, float]:
    """
    Returns signal's trend: mean slope and additive constant during last years (LSE).
    """
    # Assemble matrix A.
    A = vstack(
        [
            trend_dates,
            ones(len(trend_dates)),
        ]
    ).T
    # Direct least square regression using pseudo-inverse.
    return pinv(A).dot(signal[:, newaxis]).flatten()  # Turn the signal into a column vector.


def signal_induced_trend_from_dates(
    # Signal parameters.
    elastic_trend: float,
    signal_hyper_parameters: SignalHyperParameters,
    elastic_signal: ndarray[float],  # One-dimensional.
    signal: ndarray[float],  # May have multiple dimnesions. Temporal dimension should correspond to axis 0.
    dates: ndarray[float],
    # Trend parameters.
) -> tuple[ndarray[float], ndarray[float], ndarray[float], ndarray[float]]:
    """
    Gets some signal computing function result and computes its trend difference with elastic trend.
    """
    # Dates preprocessing.
    trend_indices, trend_dates = get_trend_dates(
        dates=dates,
        signal_hyper_parameters=signal_hyper_parameters,
    )

    # Gets last years trend differences with elastic trend.
    signal_shape = signal.shape
    signal.reshape((signal_shape[0], -1))
    signal_trends = zeros(shape=(1) if len(signal_shape) == 1 else prod(a=signal_shape[1:]))
    signal_origins = zeros(shape=signal_trends.shape)
    # Eventually loops on components.
    for i_component, signal_component in enumerate([signal] if len(signal_shape) == 1 else transpose(a=signal)):
        anelastic_component_signal_trend, _ = signal_trend(trend_dates=trend_dates, signal=signal_component[trend_indices])
        # Difference with elastic.
        signal_trends[i_component] = anelastic_component_signal_trend - elastic_trend
        signal_origins[i_component] = signal_component[trend_indices][0]

    return (
        trend_dates,
        elastic_signal[trend_indices] - elastic_signal[trend_indices][0],
        signal[trend_indices] - array(object=signal_origins),
        (signal_trends[0] if len(signal_shape) == 1 else signal_trends.reshape(signal_shape[1:])),
    )


def interpolate_Love_numbers(
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    frequencies: ndarray[float],  # (y^-1).
) -> tuple[Result, Result, ndarray[int], Path]:
    """
    Interpolates load Love numbers in frequency (h'n, l'n, 1 + k'n).
    """
    # Gets Love numbers.
    base_path = results_path.joinpath(real_description_id)
    path = base_path.joinpath("runs").joinpath(
        gets_run_id(
            use_anelasticity=signal_hyper_parameters.use_anelasticity,
            bounded_attenuation_functions=signal_hyper_parameters.bounded_attenuation_functions,
            use_attenuation=signal_hyper_parameters.use_attenuation,
        )
    )
    elastic_Love_numbers, anelastic_Love_numbers = Result(), Result()
    elastic_Love_numbers.load(name="elastic_Love_numbers", path=base_path)
    degrees = array(load_base_model(name="degrees", path=base_path))
    anelastic_Love_numbers.load(name="anelastic_Love_numbers", path=path)
    Love_number_frequencies = SECONDS_PER_YEAR * array(load_base_model("frequencies", path=path))  # (y^-1).

    # Interpolates Love numbers on signal frequencies as hermitian signal.
    symmetric_Love_number_frequencies = concatenate((-flip(m=Love_number_frequencies), Love_number_frequencies))
    return (
        Result(
            values={
                direction: {
                    BoundaryCondition.load: [
                        interpolate.interp1d(
                            x=symmetric_Love_number_frequencies,
                            y=build_hermitian(
                                signal=(1.0 if direction == Direction.potential else 0.0)
                                + (Love_numbers_for_degree / (degree if direction != Direction.radial else 1.0))
                            ),
                            kind="linear",
                        )(x=frequencies)
                        for Love_numbers_for_degree, degree in zip(
                            anelastic_Love_numbers.values[direction][BoundaryCondition.load],
                            degrees,
                        )
                    ]
                }
                for direction in Direction
            }
        ),
        elastic_Love_numbers,
        degrees,
        path,
    )


def anelastic_induced_load_signal(
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    dates: ndarray[float],  # (y).
    frequencies: ndarray[float],  # (y^-1).
    frequencial_elastic_load_signal: ndarray[complex],
) -> tuple[Path, ndarray[int], ndarray[float], ndarray[complex], Result]:
    """
    Gets already computed Love numbers, computes anelastic induced load signal per degree and save it in (.JSON) file.
    """
    # Interpolates Love numbers on signal frequencies as hermitian signal.
    hermitian_Love_numbers, elastic_Love_numbers, degrees, path = interpolate_Love_numbers(
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        frequencies=frequencies,
    )

    # Computes anelastic induced signal in frequencial domain.
    frequencial_load_signal_per_degree = array(
        object=[
            frequencial_elastic_load_signal * (1.0 + (elastic_k_for_degree[0] / degree)) / anelastic_coefficient
            for anelastic_coefficient, elastic_k_for_degree, degree in zip(
                hermitian_Love_numbers.values[Direction.potential][BoundaryCondition.load],
                elastic_Love_numbers.values[Direction.potential][BoundaryCondition.load],
                degrees,
            )
        ],
        dtype=complex,
    )

    # Computes anelastic induced signal in temporal domain.
    temporal_load_signal_per_degree = array(
        object=[
            real(ifft(x=frequencial_load_signal_for_degree))
            for frequencial_load_signal_for_degree in frequencial_load_signal_per_degree
        ],
        dtype=float,
    )

    # Saves needed information.
    save_base_model(obj=dates, name="signal_dates", path=path)
    save_base_model(obj=frequencies, name="signal_frequencies", path=path)
    save_base_model(obj={"imag": frequencial_load_signal_per_degree.imag}, name="frequencial_load_signal_per_degree", path=path)
    save_base_model(obj=temporal_load_signal_per_degree, name="temporal_load_signal_per_degree", path=path)

    return (
        path,
        degrees,
        temporal_load_signal_per_degree,
        frequencial_load_signal_per_degree,
        hermitian_Love_numbers,  # May needed when this function is called before surface deformation computing.
    )


def interpolate_on_all_degrees(
    load_signal_per_degree: ndarray[complex], degrees: ndarray[int], all_degrees: ndarray[int]
) -> ndarray[complex]:
    """
    Interpolate a 2-D (degrees, frequencies/time) signal on its first dimension (degrees).
    """
    return array(
        object=[
            interpolate.splev(x=all_degrees, tck=interpolate.splrep(x=degrees, y=load_signal_for_time, k=3), ext=0.0)
            for load_signal_for_time in transpose(a=load_signal_per_degree)
        ],
        dtype=float,
    )


def build_harmonic_name(i_order_sign: int, i_degree: int, i_order: int) -> str:
    """
    Builds a conventional name for a harmonic given its 3 indices.
    """
    return ("C" if i_order_sign == 0 else "S") + "_" + str(i_degree) + "_" + str(i_order)


def format_ocean_mask(ocean_mask_filename: str, n_max: int) -> ndarray[float]:
    """
    Gets the wanted ocean mask and adjusts it.
    """
    if ocean_mask_filename is None:
        return 1.0
    else:
        if ocean_mask_filename.split(".")[-1] == "csv":
            ocean_mask = extract_ocean_mean_mask(filename=ocean_mask_filename)
        else:
            ocean_mask = extract_land_ocean_mask() / 100
        return round(
            a=re_sample_map(
                map=ocean_mask,
                n_max=n_max - 1,
            )
        )


# TODO. Compute differenly to consider GRACE's solution filtering effect on continents.
def ocean_mean(harmonic_weights: ndarray[float], ocean_mask_filename: str, n_max: int) -> float:
    """
    Computes mean value over ocean surface. Uses a given mask.
    """
    grid = MakeGridDH(harmonic_weights, sampling=2)
    dS = expand_dims(a=cos(linspace(start=-pi / 2, stop=pi / 2, num=len(grid))), axis=1)
    resampled_ocean_mask = format_ocean_mask(ocean_mask_filename=ocean_mask_filename, n_max=min(n_max, len(grid) // 2))
    return mean(mean(grid * resampled_ocean_mask * dS))


def anelastic_harmonic_induced_load_signal(
    harmonic_weights: Optional[ndarray[float]],
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    dates: ndarray[float],
    frequencies: ndarray[float],  # (y^-1).
    frequencial_elastic_normalized_load_signal: ndarray[complex],
) -> tuple[Path, ndarray[int], Path, Path, ndarray[float], Result]:
    """
    Computes the spatially dependent anelastic induced harmonic load and saves it in a (.JSON) file.
    """
    if signal_hyper_parameters.signal == "ocean_load":

        # Gets Love numbers, computes anelastic induced load signal and saves.
        (
            path,
            degrees,
            temporal_normalized_load_signal_per_degree,
            frequencial_normalized_load_signal_per_degree,
            hermitian_Love_numbers,
        ) = anelastic_induced_load_signal(
            real_description_id=real_description_id,
            signal_hyper_parameters=signal_hyper_parameters,
            dates=dates,
            frequencies=frequencies,
            frequencial_elastic_load_signal=frequencial_elastic_normalized_load_signal,
        )

        # Preprocesses.
        all_degrees = arange(stop=len(harmonic_weights[0]))
        harmonic_frequencial_subpath = path.joinpath("anelastic_harmonic_induced_frequencial_load_signal")
        harmonic_temporal_subpath = path.joinpath("anelastic_harmonic_induced_temporal_load_signal")
        harmonic_trends = zeros(shape=harmonic_weights.shape)
        trend_indices, trend_dates = get_trend_dates(dates=dates, signal_hyper_parameters=signal_hyper_parameters)

        # Interpolates in degrees, for each frequency.
        frequencial_load_signals = interpolate_on_all_degrees(
            load_signal_per_degree=frequencial_normalized_load_signal_per_degree.imag, degrees=degrees, all_degrees=all_degrees
        )
        temporal_load_signals = interpolate_on_all_degrees(
            load_signal_per_degree=temporal_normalized_load_signal_per_degree, degrees=degrees, all_degrees=all_degrees
        )

        # Loops on harmonics:
        for i_order_sign, weights_per_degree in enumerate(harmonic_weights):
            for i_degree, weights_per_order in tqdm(
                total=len(weights_per_degree) - i_order_sign,
                desc="C" if i_order_sign == 0 else "S",
                iterable=zip(range(i_order_sign, len(weights_per_degree)), weights_per_degree[i_order_sign:]),
            ):  # Because S_n0 does not exist.
                for i_order, harmonic_weight in zip(
                    range(i_order_sign, i_degree + 1), weights_per_order[i_order_sign : i_degree + 1]
                ):
                    harmonic_name = build_harmonic_name(i_order_sign=i_order_sign, i_degree=i_degree, i_order=i_order)
                    # Computes result.
                    harmonic_frequencial_load_signal = frequencial_load_signals[:, i_degree] * harmonic_weight
                    harmonic_temporal_load_signal = temporal_load_signals[:, i_degree] * harmonic_weight
                    # Saves in (.JSON) file.
                    save_base_model(
                        obj=harmonic_frequencial_load_signal,
                        name=harmonic_name,
                        path=harmonic_frequencial_subpath,
                    )
                    save_base_model(
                        obj=harmonic_temporal_load_signal,
                        name=harmonic_name,
                        path=harmonic_temporal_subpath,
                    )
                    # Compute harmonic trend.
                    harmonic_trends[i_order_sign, i_degree, i_order] = signal_trend(
                        trend_dates=trend_dates, signal=harmonic_temporal_load_signal[trend_indices]
                    )[0]

        for harmonic_map, name in zip([harmonic_weights, harmonic_trends], ["input", "viscoelastic"]):
            print(
                ocean_mean(
                    harmonic_weights=harmonic_map,
                    ocean_mask_filename=signal_hyper_parameters.ocean_mask,
                    n_max=signal_hyper_parameters.n_max,
                ),
                name + " ocean mean",
            )

        return (
            path,
            degrees,
            harmonic_temporal_subpath,
            harmonic_frequencial_subpath,
            harmonic_trends,
            hermitian_Love_numbers,
        )
    else:  # TODO: manage GRACE's full data. harmonic_weights = None.
        pass
