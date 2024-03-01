from pathlib import Path
from typing import Callable, Optional

import netCDF4
from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    conjugate,
    expand_dims,
    flip,
    linspace,
    log2,
    ndarray,
    real,
    round,
    sum,
    transpose,
    where,
    zeros,
)
from pyshtools.expand import MakeGridDH, SHExpandDH
from scipy import interpolate
from scipy.fft import fft, fftfreq, ifft

from .classes import BoundaryCondition, Direction, Result, SignalHyperParameters
from .constants import SECONDS_PER_YEAR
from .database import load_base_model, save_base_model
from .Love_numbers import gets_run_id
from .paths import data_path, results_path


def extract_ocean_load_data(path: Path = data_path) -> tuple[ndarray[float], ndarray[float]]:
    """
    Opens Frederik et al.'s file and formats its data. Mean load in equivalent water height with respect to time.
    """
    with open(path.joinpath("datafrederik")) as file:
        lines = array([[float(value) for value in line.rstrip().split("\t")] for line in list(file)[:-1]])
    return lines[:, 0], lines[:, 1] - min(lines[:, 1])


def extract_land_ocean_harmonic_weights(n_max: int, path: Path = data_path) -> ndarray:
    """
    Opens NASA's nc file for land/sea mask and formats its data.
    """
    # Gets raw data.
    ds = netCDF4.Dataset(path.joinpath("IMERG_land_sea_mask.nc"))
    mask = flip(ds.variables["landseamask"], axis=0).data
    # Sets mean as zero and max as one by homothety.
    sum_mask = sum(sum(mask))
    max_mask = max([max(row) for row in mask])
    n_t = len(ds.variables["lon"]) * len(ds.variables["lat"])
    weights = n_t * mask / (n_t * max_mask - sum_mask) + sum_mask / (sum_mask - max_mask * n_t)
    # Gets harmionics.
    return SHExpandDH(weights, sampling=2, lmax_calc=min(n_max, (len(weights) - 1) // 2))


def build_single_load_signal(
    load: ndarray[float],
    signal_hyper_parameters: SignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], float]:
    """
    Builds an artificial signal history that has mean value, antisymetry and no Gibbs effect.
    """
    # Creates cubic spline for antisymetry.
    mean_slope = load[-1] / signal_hyper_parameters.spline_time
    spline = lambda T: mean_slope / (2.0 * signal_hyper_parameters.spline_time**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    # Builds signal history / Creates a constant step at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(signal_hyper_parameters.zero_duration)),
            load,
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
    signal_splines = interpolate.splrep(x=extended_dates, y=extended_time_serie, k=3)

    return (
        dates,
        interpolate.splev(x=dates, tck=signal_splines),  # Signal.
        2.0 * half_signal_period / n_signal,  # Time step.
    )


def build_hermitian(signal: ndarray[complex]) -> ndarray[complex]:
    """
    For a given signal defined for positive values, builds the extended signal from it that has hermitian symetry.
    """
    return concatenate((conjugate(flip(m=signal)), signal))


def build_load_signal(
    signal_hyper_parameters: SignalHyperParameters, get_harmonic_weights: bool = False
) -> tuple[ndarray[float], ndarray[float], ndarray[float], Optional[ndarray[float]]]:
    """
    Builds load history in frequential domain, eventually in frequential-harmonic domain.
    Returns:
        for Frederik et al.'s ocean mean load: mean frequential load history, described in space by static harmonic weights.
        for SLR/GRACE load history: frequential-harmonic load history.
    """
    # Builds frequencial signal.
    if signal_hyper_parameters.signal == "ocean_load":
        dates, time_domain_signal, time_step = build_single_load_signal(
            load=extract_ocean_load_data()[1],
            signal_hyper_parameters=signal_hyper_parameters,
        )  # (y).
        frequencial_load_signal = fft(x=time_domain_signal)
        frequencies = fftfreq(n=len(frequencial_load_signal), d=time_step)
        return (
            dates,
            frequencies,
            frequencial_load_signal,
            (
                None
                if not get_harmonic_weights  # Eventually gets weights map.
                else (
                    extract_land_ocean_harmonic_weights(n_max=signal_hyper_parameters.n_max)
                    if signal_hyper_parameters.weights_map == "mask"
                    else ()  # TODO: Gets GRACE's trend.
                )
            ),
        )
    else:
        # TODO: Get SLR's data?
        pass


def viscoelastic_load_signal(
    _: Optional[ndarray[float]],  # Un-used parameter for the function's description to be formatted.
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    frequencies: ndarray[float],  # (y^-1).
    frequencial_load_signal: ndarray[complex],
    dates: ndarray[float],
) -> tuple[Path, ndarray[int], ndarray[complex], ndarray[complex], Result]:
    """
    Gets already computed Love numbers, computes viscoelastic load signal and save it in (.JSON) file.
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

    # Interpolates Love numbers as hermitian signal.
    symmetric_Love_number_frequencies = concatenate((-flip(m=Love_number_frequencies), Love_number_frequencies))
    hermitian_Love_numbers = Result(
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
    )

    # Computes viscoelastic induced signal in frequencial domain.
    frequential_load_signal_per_degree = array(
        object=[
            frequencial_load_signal * (1.0 + (elastic_k_for_degree[0] / degree)) / anelastic_coefficient
            for anelastic_coefficient, elastic_k_for_degree, degree in zip(
                hermitian_Love_numbers.values[Direction.potential][BoundaryCondition.load],
                elastic_Love_numbers.values[Direction.potential][BoundaryCondition.load],
                degrees,
            )
        ],
        dtype=complex,
    )

    # Computes viscoelastic induced signal in temporal domain.
    load_signal_per_degree = array(
        object=[real(ifft(x=frequential_load_signal)) for frequential_load_signal in frequential_load_signal_per_degree],
        dtype=float,
    )

    # Saves.
    save_base_model(obj=dates, name="signal_dates", path=path)
    save_base_model(obj=frequencies, name="signal_frequencies", path=path)
    save_base_model(obj={"imag": frequential_load_signal_per_degree.imag}, name="frequential_load_signal_per_degree", path=path)
    save_base_model(obj=load_signal_per_degree, name="load_signal_per_degree", path=path)

    return (
        path,
        degrees,
        transpose(a=load_signal_per_degree),
        transpose(a=frequential_load_signal_per_degree),
        hermitian_Love_numbers,
    )


def signal_trend(
    signal_computing: Callable[
        [Optional[ndarray[float]], str, SignalHyperParameters, ndarray[float], ndarray[complex], ndarray[float]],
        tuple[Path, ndarray[int], ndarray[complex], ndarray[complex], Result],
    ],
    # Signal computing parameters.
    harmonic_weights: Optional[ndarray[float]],  # Un-used parameter for the function's description to be formatted.
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    frequencies: ndarray[float],  # (y^-1).
    frequencial_load_signal: ndarray[complex],
    dates: ndarray[float],
    # Trend parameters.
    last_year: int = 2018,
    last_years_for_trend: int = 15,
) -> tuple[Path, ndarray[int], ndarray[float], ndarray[float], ndarray[float], ndarray[float]]:
    """
    Gets some signal computing function result and computes its trend.
    """
    # Gets signal.
    path, degrees, load_signal, _, _ = signal_computing(
        harmonic_weights,
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        frequencies=frequencies,
        frequencial_load_signal=frequencial_load_signal,
        dates=dates,
    )

    # Gets last years trend.
    shift_dates = dates + signal_hyper_parameters.spline_time + last_year
    trend_indices = where((shift_dates <= last_year) * (shift_dates >= last_year - last_years_for_trend))[0]
    load_signal_last_years = load_signal[trend_indices] - expand_dims(a=load_signal[trend_indices[0]], axis=0)

    return (
        path,
        degrees,
        load_signal,
        shift_dates[trend_indices],
        load_signal_last_years,
        load_signal_last_years[-1] / last_years_for_trend,
    )
