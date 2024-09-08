from re import compile

from matplotlib.pyplot import plot, show
from numpy import arange, array, concatenate, flip, ndarray, zeros
from scipy import interpolate
from scipy.fft import fft, ifft

from ...functions import signal_trend
from ..classes import (
    DENSITY_RATIO,
    EARTH_RADIUS,
    MILLI_ARC_SECOND_TO_RADIANS,
    OMEGA,
    BoundaryCondition,
    Direction,
    LoadSignalHyperParameters,
    Result,
    pole_data_path,
)
from ..load_signal import get_trend_dates


def build_polar_tide_history(
    initial_signal_dates: ndarray[float],
    initial_pole_signal: ndarray[float],
    signal_dates: ndarray[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> ndarray[float]:
    """ """

    # Gets recent trend for eventual linear prolongation.
    recent_trend_indices, recent_trend_dates = get_trend_dates(
        signal_dates=initial_signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        recent_trend=True,
        shift=False,
    )

    # Linearly extends the signal for last years.
    elastic_load_signal_recent_trend, elastic_load_signal_additive_constant = signal_trend(
        trend_dates=recent_trend_dates,
        signal=initial_pole_signal[recent_trend_indices],
    )
    dt = initial_signal_dates[1] - initial_signal_dates[0]
    extend_part_dates = arange(start=initial_signal_dates[-1] + dt, stop=load_signal_hyper_parameters.last_year_for_trend + dt, step=dt)
    extend_part_load_signal = elastic_load_signal_recent_trend * extend_part_dates + elastic_load_signal_additive_constant

    # Eventually includes a LIA effect.
    if load_signal_hyper_parameters.LIA:
        # zero plateau after LIA.
        initial_pole_signal = concatenate(
            (
                zeros(
                    shape=(
                        int(
                            (initial_signal_dates[0] - load_signal_hyper_parameters.LIA_end_date) / dt
                            + load_signal_hyper_parameters.LIA_time_years / dt
                        )
                    )
                ),
                initial_pole_signal,
            )
        )

    # Creates cubic spline for antisymetry.
    mean_slope = extend_part_load_signal[-1] / load_signal_hyper_parameters.spline_time_years
    spline = lambda T: mean_slope / (2.0 * load_signal_hyper_parameters.spline_time_years**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T

    # Builds signal history / Creates an initial plateau at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(int(load_signal_hyper_parameters.initial_plateau_time_years / dt))),
            initial_pole_signal,
            extend_part_load_signal,
            spline(T=arange(start=-load_signal_hyper_parameters.spline_time_years, stop=0, step=dt)),
        )
    )

    # Applies antisymetry.
    extended_time_serie = concatenate((extended_time_serie_past, [0], -flip(m=extended_time_serie_past)))

    # Deduces dates axis.
    n_extended_signal = len(extended_time_serie)
    dates = arange(start=0, stop=n_extended_signal * dt, step=dt) - (n_extended_signal * dt) // 2

    return interpolate.splev(
        x=signal_dates,
        tck=interpolate.splrep(x=dates, y=extended_time_serie, k=3),
    )


def polar_motion_correction(
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    signal_dates: ndarray[float],
    signal_frequencies: ndarray[float],
    Love_numbers: Result,
    g: float,  # Earth surface mean gravitational acceleration.
    target_frequencies: ndarray[float],
):
    """"""
    load_signal_hyper_parameters.LIA_amplitude_effect = 0.0

    file = open(pole_data_path.joinpath(load_signal_hyper_parameters.pole_data + ".txt"), "r")
    lines = file.readlines()
    dates, m1, m2 = [], [], []
    for line in lines[1:]:
        p = compile(r"\d+\.\d+")  # Compiles a pattern to capture float values.
        items = [float(i) for i in p.findall(line)]
        dates += [float(items[0])]
        m1 += [float(items[1]) * MILLI_ARC_SECOND_TO_RADIANS]  # (Unitless).
        m2 += [float(items[3]) * MILLI_ARC_SECOND_TO_RADIANS]  # (Unitless).

    m1_signal = build_polar_tide_history(
        initial_signal_dates=array(object=dates, dtype=float),
        initial_pole_signal=array(object=m1, dtype=float),
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )

    m2_signal = build_polar_tide_history(
        initial_signal_dates=array(object=dates, dtype=float),
        initial_pole_signal=array(object=m2, dtype=float),
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )

    frequencial_m1: ndarray[complex] = fft(m1_signal)
    frequencial_m2: ndarray[complex] = fft(m2_signal)

    # Filtering the Annual and Chandler wobbles.
    frequencial_m1[abs(signal_frequencies) > load_signal_hyper_parameters.wobble_filtering_frequency] = 0.0
    frequencial_m2[abs(signal_frequencies) > load_signal_hyper_parameters.wobble_filtering_frequency] = 0.0

    PHI_CONSTANT = 1.0  # -EARTH_RADIUS * OMEGA**2 / (2.0 * g)

    # Gets element in position 1 for degree 2.
    Phi_SE_PT_complex: ndarray[complex] = (
        PHI_CONSTANT * (Love_numbers.values[Direction.potential][BoundaryCondition.potential][1] - 1.0) * (frequencial_m1 - 1.0j * frequencial_m2)
    )

    # C_PT_SE_2_1, S_PT_SE_2_1.
    Stokes_to_EWH_factor = 5.0 / 3.0 / DENSITY_RATIO * EARTH_RADIUS / (Love_numbers.values[Direction.potential][BoundaryCondition.load][1])

    return (
        Stokes_to_EWH_factor * fft(ifft(Phi_SE_PT_complex).real),
        -Stokes_to_EWH_factor * fft(ifft(Phi_SE_PT_complex).imag),
    )
