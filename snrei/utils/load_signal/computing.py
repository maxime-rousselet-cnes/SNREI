from numpy import concatenate, errstate, multiply, nan_to_num, ndarray, ones, zeros
from scipy.fft import ifft

from ...functions import signal_trend
from ..classes import BoundaryCondition, Direction, LoadSignalHyperParameters, Result
from .utils import get_trend_dates


def anelastic_frequencial_harmonic_load_signal_computing(
    elastic_Love_numbers: Result,
    anelastic_Love_numbers: Result,
    signal_frequencies: ndarray[float],
    frequencial_elastic_load_signal: ndarray[complex],
) -> ndarray[complex]:
    """
    Gets already computed Love numbers and computes anelastic induced frequential-harmonic load signal.
    """

    # Computes anelastic induced signal in frequencial-harmonic domain.
    with errstate(invalid="ignore", divide="ignore"):
        # (1 + k_el) / (1 + k_anel) * {C, S}.
        return multiply(
            concatenate(
                (  # Adds a line of one values for degree zero.
                    ones(shape=(1, len(signal_frequencies))),
                    nan_to_num(
                        x=elastic_Love_numbers.values[Direction.potential][BoundaryCondition.load]
                        / anelastic_Love_numbers.values[Direction.potential][BoundaryCondition.load],
                        nan=0.0,
                    ),
                ),
                axis=0,
            ).T,
            frequencial_elastic_load_signal.transpose((0, 2, 3, 1)),
        ).transpose((0, 3, 1, 2))


def compute_harmonic_signal_trends(
    signal_dates: ndarray,
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    frequencial_harmonic_signal: ndarray[complex],
    recent_trend: bool = True,
) -> ndarray[float]:
    """
    Computes harmonic trends (C/S, degrees, orders) from frequencial harmonic data (C/S, degrees, orders, frequencies).
    """

    # Initializes.
    trend_indices, trend_dates = get_trend_dates(
        signal_dates=signal_dates, load_signal_hyper_parameters=load_signal_hyper_parameters, recent_trend=recent_trend
    )
    signal_trends = zeros(shape=frequencial_harmonic_signal.shape[:-1])

    # Computes trend for all harmonics.
    for i_sign, frequencial_harmonics_per_degree in enumerate(frequencial_harmonic_signal):
        for i_degree, frequencial_harmonic_load_signal_per_order in enumerate(frequencial_harmonics_per_degree):
            for i_order, frequencial_signal in enumerate(frequencial_harmonic_load_signal_per_order):
                signal: ndarray = ifft(frequencial_signal)[trend_indices]
                signal_trends[i_sign, i_degree, i_order] = signal_trend(
                    trend_dates=trend_dates,
                    signal=signal.real,
                )[0]

    return signal_trends
