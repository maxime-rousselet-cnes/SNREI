from numpy import arange, array, concatenate, flip, linspace, ndarray, zeros
from scipy import interpolate
from scipy.fft import fft, ifft

from ...functions import signal_trend
from ..classes import (
    MEAN_POLE_COEFFICIENTS,
    MEAN_POLE_T_0,
    MILLI_ARC_SECOND_TO_RADIANS,
    PHI_CONSTANT,
    STOKES_TO_EWH_CONSTANT,
    BoundaryCondition,
    Direction,
    LoadSignalHyperParameters,
    Result,
    load_load_signal_hyper_parameters,
    pole_data_path,
)
from ..load_signal import get_trend_dates


def get_polar_motion_time_series(
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    i: int = 1,
) -> tuple[ndarray[float], ndarray[float]]:
    """ """

    file = open(pole_data_path.joinpath(load_signal_hyper_parameters.pole_data + ".txt"), "r")
    lines = file.readlines()
    dates, m_1, m_2 = [], [], []
    for line in lines[1:]:
        items = [float(i) for i in [string for string in line.split(" ") if not (string in ["", "\n"])]]
        dates += [items[0]]
        m_1 += [
            (
                float(items[i])
                + (
                    0.0
                    if load_signal_hyper_parameters.pole_case == "mean"
                    else (-items[i + 1] / 2.0 if load_signal_hyper_parameters.pole_case == "lower" else items[i + 1] / 2.0)
                )
            )
        ]
        m_2 += [
            (
                float(items[i + 2])
                + (
                    0.0
                    if load_signal_hyper_parameters.pole_case == "mean"
                    else (-items[i + 3] / 2.0 if load_signal_hyper_parameters.pole_case == "lower" else items[i + 3] / 2.0)
                )
            )
        ]

    return array(object=dates, dtype=float), array(object=m_1, dtype=float), array(object=m_2, dtype=float)


def convolve(a: ndarray[float], v: ndarray[float]) -> ndarray[float]:
    """ """
    return array([sum([a[i + j] * v[i] for i in range(len(v))]) for j in range(len(a) - len(v) + 1)])


def polar_motion_correction(
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    signal_dates: ndarray[float],
    Love_numbers: Result,
    signal_frequencies: ndarray[float],
):
    """"""
    dates, m_1, m_2 = get_polar_motion_time_series(load_signal_hyper_parameters=load_signal_hyper_parameters)

    m_1_signal = build_polar_tide_history(
        initial_signal_dates=dates,
        initial_pole_signal=MILLI_ARC_SECOND_TO_RADIANS * m_1,
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        mean_trend=MEAN_POLE_COEFFICIENTS[load_signal_hyper_parameters.mean_pole_convention]["m_1"][1],
    )

    m_2_signal = build_polar_tide_history(
        initial_signal_dates=dates,
        initial_pole_signal=-MILLI_ARC_SECOND_TO_RADIANS * m_2,
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        mean_trend=-MEAN_POLE_COEFFICIENTS[load_signal_hyper_parameters.mean_pole_convention]["m_2"][1],
    )

    frequencial_m1: ndarray[complex] = fft(m_1_signal)
    frequencial_m2: ndarray[complex] = fft(m_2_signal)

    # Filtering the Annual and Chandler wobbles.
    if load_signal_hyper_parameters.filter_wobble:
        frequencial_m1[abs(signal_frequencies) > load_signal_hyper_parameters.wobble_filtering_frequency] = 0.0
        frequencial_m2[abs(signal_frequencies) > load_signal_hyper_parameters.wobble_filtering_frequency] = 0.0

    # Gets element in position 1 for degree 2.
    Phi_SE_PT_complex: ndarray[complex] = (
        -(PHI_CONSTANT if load_signal_hyper_parameters.phi_constant else 1.0)
        * (Love_numbers.values[Direction.potential][BoundaryCondition.potential][1] - 1)
        * (frequencial_m1 - 1.0j * frequencial_m2)  # Because 'Love_numbers' saves 1 + k.
    )

    # C_PT_SE_2_1, S_PT_SE_2_1.
    Stokes_to_EWH_factor = STOKES_TO_EWH_CONSTANT / (Love_numbers.values[Direction.potential][BoundaryCondition.load][1])  # Divides by 1 + k'.

    return (
        Stokes_to_EWH_factor * fft(ifft(Phi_SE_PT_complex).real),
        -Stokes_to_EWH_factor * fft(ifft(Phi_SE_PT_complex).imag),
    )


def build_polar_tide_history(
    initial_signal_dates: ndarray[float],
    initial_pole_signal: ndarray[float],
    signal_dates: ndarray[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    mean_trend: float,
) -> ndarray[float]:
    """ """
    dt = initial_signal_dates[1] - initial_signal_dates[0]

    # Gets recent trend for eventual linear prolongation.
    recent_trend_indices, recent_trend_dates = get_trend_dates(
        signal_dates=initial_signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        recent_trend=True,
        shift=False,
    )

    # Updates from IERS correction.
    initial_pole_signal -= mean_trend * (dates - MEAN_POLE_T_0[load_signal_hyper_parameters.mean_pole_convention])

    # Linearly extends the signal for last years.
    elastic_recent_trend, elastic_additive_constant = signal_trend(
        trend_dates=recent_trend_dates,
        signal=initial_pole_signal[recent_trend_indices],
    )

    extend_part_dates = arange(start=initial_signal_dates[-1] + dt, stop=load_signal_hyper_parameters.last_year_for_trend + dt, step=dt)
    extend_part = elastic_recent_trend * extend_part_dates + elastic_additive_constant

    # Eventually includes a LIA effect.
    if load_signal_hyper_parameters.LIA:
        # Gets full trend to build LIA signal model.
        full_trend, _ = signal_trend(
            trend_dates=initial_signal_dates,
            signal=initial_pole_signal,
        )
        # zero plateau after LIA.
        initial_pole_signal = (
            concatenate(
                (
                    linspace(
                        start=full_trend * load_signal_hyper_parameters.LIA_amplitude_effect,
                        stop=0,
                        num=int(load_signal_hyper_parameters.LIA_time_years / dt),
                    ),
                    zeros(shape=(int((initial_signal_dates[0] - load_signal_hyper_parameters.LIA_end_date) / dt))),
                    initial_pole_signal,
                )
            )
            - full_trend * load_signal_hyper_parameters.LIA_amplitude_effect
        )

    # Creates cubic spline for antisymetry.
    mean_slope = extend_part[-1] / load_signal_hyper_parameters.spline_time_years
    spline = lambda T: mean_slope / (2.0 * load_signal_hyper_parameters.spline_time_years**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T

    # Builds signal history / Creates an initial plateau at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(int(load_signal_hyper_parameters.initial_plateau_time_years / dt))),
            initial_pole_signal,
            extend_part,
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
