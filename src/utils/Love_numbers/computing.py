from multiprocessing import Pool

from numpy import Inf, array, concatenate, linspace, log10, ndarray, round, unique

from ...functions import interpolate_all, precise_curvature
from ..classes import (
    AnelasticityDescription,
    Integration,
    RunHyperParameters,
    YSystemHyperParameters,
)
from ..rheological_formulas import frequencies_to_periods


def generate_log_frequency_initial_values(
    period_min_year: float,
    period_max_year: float,
    n_frequency_0: int,
    frequency_unit: float,
) -> ndarray[float]:
    """
    Generates an array of log-spaced frequency values.
    """
    return linspace(
        start=log10(frequencies_to_periods(period_max_year) / frequency_unit),
        stop=log10(frequencies_to_periods(period_min_year) / frequency_unit),
        num=n_frequency_0,
    )


def Love_numbers_computing(
    max_tol: float,
    decimals: int,
    y_system_hyper_parameters: YSystemHyperParameters,
    run_hyper_parameters: RunHyperParameters,
    degrees: list[int],
    log_frequency_initial_values: ndarray[float],
    anelasticity_description: AnelasticityDescription,
) -> tuple[ndarray, ndarray]:
    """
    Performs Love numbers computing (n, frequency) with given anelasticity description and hyper parameters.
    Returns log frequency array and Love numbers as an array.
    """

    # Initializes a Callable as a global variable to parallelize.
    global parallel_processing

    # Defines a function for parallel processing.
    def parallel_processing(n: int) -> tuple[ndarray[float], ndarray[complex]]:
        return Love_numbers_computing_subfunction(
            n=n,
            anelasticity_description=anelasticity_description,
            y_system_hyper_parameters=y_system_hyper_parameters,
            run_hyper_parameters=run_hyper_parameters,
            log_frequency_initial_values=(
                log_frequency_initial_values
                if (
                    run_hyper_parameters.use_long_term_anelasticity
                    or run_hyper_parameters.use_short_term_anelasticity
                )
                else array([Inf])
            ),
            max_tol=max_tol,
            decimals=decimals,
        )

    with Pool() as p:  # Processes for degrees.
        frequency_and_Love_numbers_tuples: list[
            tuple[ndarray[float], ndarray[complex]]
        ] = p.map(
            func=parallel_processing,
            iterable=degrees,
        )

    # Interpolates in frequency for all degrees.
    log_frequency_values_per_degree = [
        round(a=frequency_and_Love_numbers_tuple[0], decimals=decimals)
        for frequency_and_Love_numbers_tuple in frequency_and_Love_numbers_tuples
    ]
    Love_numbers_per_degree = [
        frequency_and_Love_numbers_tuple[1]
        for frequency_and_Love_numbers_tuple in frequency_and_Love_numbers_tuples
    ]
    log_frequency_all_values = unique(concatenate(log_frequency_values_per_degree))
    all_Love_numbers = interpolate_all(
        x_values_per_component=log_frequency_values_per_degree,
        function_values=Love_numbers_per_degree,
        x_shared_values=log_frequency_all_values,
    )

    return (
        log_frequency_all_values,
        all_Love_numbers,
    )


def Love_numbers_computing_subfunction(
    n: int,
    anelasticity_description: AnelasticityDescription,
    y_system_hyper_parameters: YSystemHyperParameters,
    run_hyper_parameters: RunHyperParameters,
    log_frequency_initial_values: ndarray[float],
    max_tol: float,
    decimals: int,
) -> tuple[ndarray[float], ndarray[complex]]:
    """
    Computes Love numbers for all frequencies, for a given degree.
    """

    # Defines a callable that computes Love numbers for an array of log10(frequency/ unit_frequency) values.
    Love_numbers_computing_loop_on_frequencies = lambda log_frequency_values: array(
        object=[
            Integration(
                anelasticity_description=anelasticity_description,
                log_frequency=log_frequency,
                use_long_term_anelasticity=run_hyper_parameters.use_long_term_anelasticity,
                use_short_term_anelasticity=run_hyper_parameters.use_short_term_anelasticity,
                use_bounded_attenuation_functions=run_hyper_parameters.use_bounded_attenuation_functions,
            ).y_system_integration(
                n=n,
                hyper_parameters=y_system_hyper_parameters,
            )
            for log_frequency in log_frequency_values
        ]
    )

    # Processes for frequencies. Adaptive step for precise curvature.
    log_frequency_values, Love_numbers = precise_curvature(
        x_initial_values=log_frequency_initial_values,
        f=Love_numbers_computing_loop_on_frequencies,
        max_tol=max_tol,
        decimals=decimals,
    )

    return log_frequency_values, Love_numbers
