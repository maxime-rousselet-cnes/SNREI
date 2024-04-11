from pathlib import Path
from typing import Optional

from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    flip,
    linspace,
    log2,
    ndarray,
    newaxis,
    ones,
    round,
    vstack,
    where,
    zeros,
)
from numpy.linalg import pinv
from scipy import interpolate
from scipy.fft import fft, fftfreq

from ..classes import (
    SECONDS_PER_YEAR,
    BoundaryCondition,
    Direction,
    LoadSignalHyperParameters,
)
from ..database import save_base_model
from ..Love_numbers import interpolate_Love_numbers
from .data import (
    extract_mask_csv,
    extract_mask_nc,
    extract_temporal_load_signal,
    extract_trends_GRACE,
    map_normalizing,
    map_sampling,
)


def get_trend_dates(
    signal_dates: ndarray[float] | list[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> tuple[ndarray[float], ndarray[int]]:
    """
    Returns trend indices and trend dates.
    """
    shift_dates = (
        array(object=signal_dates, dtype=float)
        + load_signal_hyper_parameters.spline_time
        + load_signal_hyper_parameters.last_year_for_trend
    )
    trend_indices = where(
        (shift_dates <= load_signal_hyper_parameters.last_year_for_trend - 1)
        * (shift_dates >= load_signal_hyper_parameters.first_year_for_trend)
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
    result: ndarray = pinv(A).dot(signal[:, newaxis])
    return result.flatten()  # Turn the signal into a column vector. (slope, additive_constant)


def build_elastic_load_signal(
    load_signal_hyper_parameters: LoadSignalHyperParameters, get_harmonic_weights: bool = False
) -> tuple[ndarray[float], ndarray[float], Path | tuple[float, ndarray[float], ndarray[complex], Optional[ndarray[float]]]]:
    """
    Builds load history in frequential domain, eventually in frequential-harmonic domain.
    Returns:
        - signal_dates
        - frequencies
        - For SLR/GRACE load history:
            - a frequential-harmonic load history: i.e. path of the folder containing a function of omega per harmonic in files.
          For Frederikse et al.'s ocean load history, a tuple of :
            - a frequential elastic load signal history
            - elastic load signal trend
            - static harmonic weights if needed.
    """
    if load_signal_hyper_parameters.load_signal == "ocean_load_Frederikse":
        # Builds frequencial signal.
        dates_Frederikse, load_signal_Frederikse = extract_temporal_load_signal()
        signal_dates, temporal_elastic_load_signal, time_step, elastic_load_signal_trend = build_elastic_load_signal_history(
            signal_dates=dates_Frederikse,
            load_signal=load_signal_Frederikse,
            load_signal_hyper_parameters=load_signal_hyper_parameters,
        )  # (y).
        frequencial_elastic_load_signal = fft(x=temporal_elastic_load_signal)
        frequencies = fftfreq(n=len(frequencial_elastic_load_signal), d=time_step)
        if load_signal_hyper_parameters.weights_map == "GRACE":
            map = extract_trends_GRACE(name=load_signal_hyper_parameters.GRACE)
        else:
            map = map_normalizing(
                map=(
                    extract_mask_csv(name=load_signal_hyper_parameters.ocean_mask)
                    if load_signal_hyper_parameters.ocean_mask.split(".")[-1] == "csv"
                    else extract_mask_nc(name=load_signal_hyper_parameters.ocean_mask)
                )
            )
        harmonic_weights = (
            None
            if not get_harmonic_weights
            else map_sampling(
                map=map,
                n_max=load_signal_hyper_parameters.n_max,
                harmonic_domain=True,
            )
        )
        # Eventually gets harmonics.
        return (
            signal_dates,
            frequencies,
            (frequencial_elastic_load_signal / elastic_load_signal_trend, elastic_load_signal_trend, harmonic_weights),
        )
    else:
        # TODO: Get GRACE's data history ?
        pass


def build_elastic_load_signal_history(
    signal_dates: ndarray[float],
    load_signal: ndarray[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], float, float]:
    """
    Builds an artificial load signal history that has zero mean value, antisymetry and no Gibbs effect.
    """
    # Linearly extends the signal for last years.
    trend_indices = signal_dates >= load_signal_hyper_parameters.first_year_for_trend
    elastic_load_signal_trend, elastic_load_signal_additive_constant = signal_trend(
        trend_dates=signal_dates[trend_indices],
        signal=load_signal[trend_indices],
    )
    extend_part_dates = arange(signal_dates[-1] + 1, load_signal_hyper_parameters.last_year_for_trend + 1)
    extend_part_load_signal = elastic_load_signal_trend * extend_part_dates + elastic_load_signal_additive_constant
    # Creates cubic spline for antisymetry.
    mean_slope = extend_part_load_signal[-1] / load_signal_hyper_parameters.spline_time
    spline = lambda T: mean_slope / (2.0 * load_signal_hyper_parameters.spline_time**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    # Builds signal history / Creates a constant step at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(load_signal_hyper_parameters.zero_duration)),
            load_signal,
            extend_part_load_signal,
            spline(T=arange(start=-load_signal_hyper_parameters.spline_time, stop=0)),
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
    n_signal = int(2 ** (n_log_min_no_Gibbs + load_signal_hyper_parameters.anti_Gibbs_effect_factor))
    signal_dates = linspace(-half_signal_period, stop=half_signal_period, num=n_signal)

    return (
        signal_dates,
        interpolate.splev(x=signal_dates, tck=interpolate.splrep(x=extended_dates, y=extended_time_serie, k=3)),  # Signal.
        2.0 * half_signal_period / n_signal,  # Time step.
        elastic_load_signal_trend,  # Trend.
    )


def anelastic_induced_load_signal_per_degree(
    anelasticity_description_id: str,
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    signal_dates: ndarray[float],  # (y).
    frequencies: ndarray[float],  # (y^-1).
    frequencial_elastic_normalized_load_signal: ndarray[complex],
    elastic_load_signal_trend: float,
) -> tuple[
    Path,
    ndarray[int],
    ndarray[complex],
]:
    """
    Gets already computed Love numbers, computes anelastic induced frequential load signal per degree and save it in a
    (.JSON) file.
    """
    # Interpolates Love numbers on signal frequencies as hermitian signal.
    hermitian_Love_number_fractions, elastic_Love_number_fractions, degrees, Love_numbers_path = interpolate_Love_numbers(
        anelasticity_description_id=anelasticity_description_id,
        target_frequencies=frequencies / SECONDS_PER_YEAR,
        option=load_signal_hyper_parameters.run_hyper_parameters,
        degrees=None,
        directions=[Direction.potential],
        boundary_conditions=[BoundaryCondition.load],
        function=lambda x: 1.0 / x,
    )

    # Computes anelastic induced signal in frequencial domain.
    frequencial_load_signal_per_degree: ndarray[complex] = array(
        object=[
            frequencial_elastic_normalized_load_signal * anelastic_fraction / elastic_fraction[0]
            for anelastic_fraction, elastic_fraction in zip(
                hermitian_Love_number_fractions.values[Direction.potential][BoundaryCondition.load],
                elastic_Love_number_fractions.values[Direction.potential][BoundaryCondition.load],
            )
        ],
        dtype=complex,
    )

    # Saves the needed informations.
    path: Path = Love_numbers_path.joinpath("load").joinpath(load_signal_hyper_parameters.load_signal)
    save_base_model(obj=elastic_load_signal_trend, name="elastic_load_signal_trend", path=path)
    save_base_model(
        obj={"real": frequencial_elastic_normalized_load_signal.real, "imag": frequencial_elastic_normalized_load_signal.imag},
        name="frequencial_elastic_normalized_load_signal",
        path=path,
    )
    subpath = path.joinpath("anelastic_induced_frequencial_load_signal_per_degree")
    subpath.mkdir(parents=True, exist_ok=True)
    frequencial_load_signal: ndarray[complex]
    for degree, frequencial_load_signal in zip(degrees, frequencial_load_signal_per_degree):
        save_base_model(
            obj={"real": frequencial_load_signal.real, "imag": frequencial_load_signal.imag}, name=str(degree), path=subpath
        )
    save_base_model(obj=signal_dates, name="signal_dates", path=path)

    return path, degrees, frequencial_load_signal_per_degree
