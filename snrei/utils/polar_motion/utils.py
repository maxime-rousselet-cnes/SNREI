from re import compile

from numpy import arange, array, concatenate, flip, linspace, ndarray, zeros
from scipy import interpolate
from scipy.fft import fft, ifft

from ...functions import signal_trend
from ..classes import (
    DENSITY_RATIO,
    EARTH_RADIUS,
    MILLI_ARC_SECOND_TO_RADIANS,
    BoundaryCondition,
    Direction,
    LoadSignalHyperParameters,
    Result,
    pole_data_path,
)
from ..load_signal import get_trend_dates

mean_pole_coeffs = {
    "IERS_2018_update": {"m_1": [55.0, 1.677], "m_2": [320.5, 3.460]},  # (mas/yr^index).
}
mean_pole_t_0 = {
    "IERS_2018_update": 2000.0,  # (yr).
}


def polar_motion_correction(
    load_signal_hyper_parameters: LoadSignalHyperParameters, signal_dates: ndarray[float], Love_numbers: Result, signal_frequencies: ndarray[float]
):
    """"""
    load_signal_hyper_parameters.LIA_amplitude_effect = 0.0

    file = open(pole_data_path.joinpath(load_signal_hyper_parameters.pole_data + ".txt"), "r")
    lines = file.readlines()
    dates, m_1, m_2 = [], [], []
    for line in lines[1:]:
        p = compile(r"\d+\.\d+")  # Compiles a pattern to capture float values.
        items = [float(i) for i in p.findall(line)]
        dates += [float(items[0])]
        m_1 += [
            (
                float(items[1])
                + (
                    0.0
                    if load_signal_hyper_parameters.pole_case == "mean"
                    else (-float(items[2]) / 2.0 if load_signal_hyper_parameters.pole_case == "lower" else float(items[2]) / 2.0)
                )
            )
            * MILLI_ARC_SECOND_TO_RADIANS
        ]  # (Unitless).
        m_2 += [
            (
                float(items[3])
                + (
                    0.0
                    if load_signal_hyper_parameters.pole_case == "mean"
                    else (-float(items[4]) / 2.0 if load_signal_hyper_parameters.pole_case == "lower" else float(items[4]) / 2.0)
                )
            )
            * MILLI_ARC_SECOND_TO_RADIANS
        ]  # (Unitless).

    m_1 = array(object=m_1, dtype=float)
    m_2 = array(object=m_2, dtype=float)
    dates = array(object=dates, dtype=float)

    mean_pole_m_1 = (
        sum(
            [
                coeff * (dates - mean_pole_t_0[load_signal_hyper_parameters.mean_pole_convention]) ** index
                for index, coeff in enumerate(mean_pole_coeffs[load_signal_hyper_parameters.mean_pole_convention]["m_1"])
            ]
        )
        * MILLI_ARC_SECOND_TO_RADIANS
    )
    mean_pole_m_2 = (
        sum(
            [
                coeff * (dates - mean_pole_t_0[load_signal_hyper_parameters.mean_pole_convention]) ** index
                for index, coeff in enumerate(mean_pole_coeffs[load_signal_hyper_parameters.mean_pole_convention]["m_2"])
            ]
        )
        * MILLI_ARC_SECOND_TO_RADIANS
    )

    m_1_signal = build_polar_tide_history(
        initial_signal_dates=dates,
        initial_pole_signal=m_1 + mean_pole_m_1,
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )

    m_2_signal = build_polar_tide_history(
        initial_signal_dates=dates,
        initial_pole_signal=m_2 + mean_pole_m_2,
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )

    frequencial_m1: ndarray[complex] = fft(m_1_signal)
    frequencial_m2: ndarray[complex] = fft(m_2_signal)

    # Filtering the Annual and Chandler wobbles.
    if load_signal_hyper_parameters.filter_wobble:
        frequencial_m1[abs(signal_frequencies) > load_signal_hyper_parameters.wobble_filtering_frequency] = 0.0
        frequencial_m2[abs(signal_frequencies) > load_signal_hyper_parameters.wobble_filtering_frequency] = 0.0

    # Gets element in position 1 for degree 2.
    Phi_SE_PT_complex: ndarray[complex] = (Love_numbers.values[Direction.potential][BoundaryCondition.potential][1] - 1.0) * (
        frequencial_m1 - 1.0j * frequencial_m2
    )

    # C_PT_SE_2_1, S_PT_SE_2_1.
    Stokes_to_EWH_factor = 5.0 / 3.0 / DENSITY_RATIO * EARTH_RADIUS / (Love_numbers.values[Direction.potential][BoundaryCondition.load][1])

    return (
        Stokes_to_EWH_factor * fft(ifft(Phi_SE_PT_complex).real),
        -Stokes_to_EWH_factor * fft(ifft(Phi_SE_PT_complex).imag),
    )


def build_polar_tide_history(
    initial_signal_dates: ndarray[float],
    initial_pole_signal: ndarray[float],
    signal_dates: ndarray[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> ndarray[float]:
    """ """

    # Gets past trend for mean motion correction.
    past_trend_indices, past_trend_dates = get_trend_dates(
        signal_dates=initial_signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        recent_trend=False,
        shift=False,
    )

    # Gets mean motion correction.
    elastic_past_trend, _ = signal_trend(
        trend_dates=past_trend_dates[past_trend_dates < load_signal_hyper_parameters.pole_secular_term_trend_end_date],
        signal=initial_pole_signal[past_trend_indices][past_trend_dates < load_signal_hyper_parameters.pole_secular_term_trend_end_date],
    )

    # Applies mean motion correction.
    initial_pole_signal -= elastic_past_trend * initial_signal_dates

    # Starts at zero.
    initial_pole_signal -= initial_pole_signal[0]

    # Gets recent trend for eventual linear prolongation.
    recent_trend_indices, recent_trend_dates = get_trend_dates(
        signal_dates=initial_signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        recent_trend=True,
        shift=False,
    )

    # Linearly extends the signal for last years.
    elastic_recent_trend, elastic_additive_constant = signal_trend(
        trend_dates=recent_trend_dates,
        signal=initial_pole_signal[recent_trend_indices],
    )

    dt = initial_signal_dates[1] - initial_signal_dates[0]
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
        initial_pole_signal = concatenate(
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
