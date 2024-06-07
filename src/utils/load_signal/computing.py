from numpy import Inf, arange, array, multiply, ndarray

from ..classes import (
    ELASTIC_RUN_HYPER_PARAMETERS,
    SECONDS_PER_YEAR,
    BoundaryCondition,
    Direction,
    LoveNumbersHyperParameters,
)
from ..Love_numbers import interpolate_Love_numbers


def anelastic_induced_frequencial_harmonic_load_signal_computing(
    anelasticity_description_id: str,
    n_max: int,
    Love_number_hyper_parameters: LoveNumbersHyperParameters,
    signal_frequencies: ndarray[float],  # (yr^-1).
    frequencial_elastic_normalized_load_signal: ndarray[complex],
) -> ndarray[complex]:
    """
    Gets already computed Love numbers and computes anelastic induced frequential-harmonic load signal.
    """

    # Interpolates anelastic Love numbers on signal degrees and frequencies as hermitian signal.
    anelastic_hermitian_Love_number_fractions = interpolate_Love_numbers(
        anelasticity_description_id=anelasticity_description_id,
        target_frequencies=signal_frequencies / SECONDS_PER_YEAR,  # (yr^-1) -> (Hz).
        target_degrees=arange(n_max) + 1,
        Love_number_hyper_parameters=Love_number_hyper_parameters,
        directions=[Direction.potential],
        boundary_conditions=[BoundaryCondition.load],
        function=lambda x: 1.0 / x,
    ).values[Direction.potential][BoundaryCondition.load]

    # Interpolates elastic Love numbers on signal degrees.
    elastic_hermitian_Love_number_fractions = interpolate_Love_numbers(
        anelasticity_description_id=anelasticity_description_id,
        target_frequencies=array(object=[Inf]),  # Infinite frequency for elastic case.
        target_degrees=arange(n_max) + 1,
        Love_number_hyper_parameters=Love_number_hyper_parameters | {"run_hyper_parameters": ELASTIC_RUN_HYPER_PARAMETERS},
        directions=[Direction.potential],
        boundary_conditions=[BoundaryCondition.load],
        function=lambda x: 1.0 / x,
    ).values[Direction.potential][BoundaryCondition.load]

    if len(frequencial_elastic_normalized_load_signal.shape) == 1:
        # Computes anelastic induced signal in harmonic domain.
        return multiply(
            anelastic_hermitian_Love_number_fractions / elastic_hermitian_Love_number_fractions,
            frequencial_elastic_normalized_load_signal,
        )
    else:
        # Computes anelastic induced signal in frequencial-harmonic domain.
        return multiply(
            anelastic_hermitian_Love_number_fractions.T / elastic_hermitian_Love_number_fractions.T,
            frequencial_elastic_normalized_load_signal.transpose((0, 2, 3, 1)),
        ).transpose((0, 3, 1, 2))
