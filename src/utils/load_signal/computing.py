from numpy import Inf, arange, array, expand_dims, multiply, ndarray, zeros
from scipy.fft import ifft

from ...functions import signal_trend
from ..classes import (
    ELASTIC_RUN_HYPER_PARAMETERS,
    SECONDS_PER_YEAR,
    BoundaryCondition,
    Direction,
    LoadSignalHyperParameters,
    LoveNumbersHyperParameters,
    Result,
)
from ..Love_numbers import (
    interpolate_anelastic_Love_numbers,
    interpolate_elastic_Love_numbers,
)
from .utils import get_trend_dates


def elastic_Love_numbers_computing(
    anelasticity_description_id: str,
    n_max: int,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
) -> Result:
    """
    Gets already computed Love numbers and computes elastic frequential-harmonic load signal.
    """

    # Interpolates elastic Love numbers on signal degrees.
    elastic_Love_numbers: Result = interpolate_elastic_Love_numbers(
        anelasticity_description_id=anelasticity_description_id,
        target_degrees=arange(n_max) + 1,
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        directions=[Direction.radial, Direction.potential],
        boundary_conditions=[BoundaryCondition.load],
    )

    # Elastic signal in harmonic domain.
    return elastic_Love_numbers


def anelastic_frequencial_harmonic_load_signal_computing(
    anelasticity_description_id: str,
    n_max: int,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
    signal_frequencies: ndarray[float],  # (yr^-1).
    frequencial_elastic_normalized_load_signal: ndarray[complex],
    elastic_Love_numbers: Result = None,
) -> tuple[ndarray[complex], Result]:
    """
    Gets already computed Love numbers and computes anelastic induced frequential-harmonic load signal.
    """

    # Interpolates anelastic Love numbers on signal degrees and frequencies as hermitian signal.
    anelastic_hermitian_Love_numbers: Result = interpolate_anelastic_Love_numbers(
        anelasticity_description_id=anelasticity_description_id,
        target_frequencies=signal_frequencies / SECONDS_PER_YEAR,  # (yr^-1) -> (Hz).
        target_degrees=arange(n_max) + 1,
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        directions=[Direction.radial, Direction.potential],
        boundary_conditions=[BoundaryCondition.load],
    )

    # Computes anelastic induced signal in frequencial-harmonic domain.
    return (  # (1 + k_el) / (1 + k_anel) * {C, S}.
        multiply(
            expand_dims(
                a=elastic_Love_numbers.values[Direction.potential][
                    BoundaryCondition.load
                ],
                axis=0,
            )
            / anelastic_hermitian_Love_numbers.values[Direction.potential][
                BoundaryCondition.load
            ].T,
            frequencial_elastic_normalized_load_signal.transpose((0, 2, 3, 1)),
        ).transpose((0, 3, 1, 2)),
        anelastic_hermitian_Love_numbers,
    )


def compute_signal_trends(
    signal_dates: ndarray,
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    frequencial_harmonic_load_signal: ndarray[complex],
) -> ndarray[float]:
    """
    Computes harmonic trends (C/S, degrees, orders) from frequencial harmonic data (C/S, degrees, orders, frequencies).
    """

    # Initializes.
    trend_indices, trend_dates = get_trend_dates(
        signal_dates=signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )
    signal_trends = zeros(shape=frequencial_harmonic_load_signal.shape[:-1])

    # Computes trend for all harmonics.
    for i_sign, frequencial_harmonics_per_degree in enumerate(
        frequencial_harmonic_load_signal
    ):
        for i_degree, frequencial_harmonic_load_signal_per_order in enumerate(
            frequencial_harmonics_per_degree
        ):
            for i_order, frequencial_signal in enumerate(
                frequencial_harmonic_load_signal_per_order
            ):
                signal: ndarray = ifft(frequencial_signal)[trend_indices]
                signal_trends[i_sign, i_degree, i_order] = signal_trend(
                    trend_dates=trend_dates,
                    signal=signal.real,
                )[0]

    return signal_trends
