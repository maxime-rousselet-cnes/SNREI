from multiprocessing import Pool
from pathlib import Path

from numpy import Inf, array, concatenate, ndarray, round, unique

from ..abstract_computing import interpolate_all, precise_curvature
from ..classes import (
    AnelasticityDescription,
    Integration,
    Result,
    YSystemHyperParameters,
)
from ..database import save_base_model


def save_frequencies(log_frequency_values: ndarray[float], frequency_unit: float, path: Path) -> None:
    """
    Maps back log unitless frequencies to (Hz) and save to (.JSON) file.
    """
    save_base_model(obj=10.0**log_frequency_values * frequency_unit, name="frequencies", path=path)


def anelastic_Love_numbers_computing(
    max_tol: float,
    decimals: int,
    y_system_hyper_parameters: YSystemHyperParameters,
    use_long_term_anelasticity: bool,
    use_short_term_anelasticity: bool,
    use_bounded_attenuation_functions: bool,
    degrees: list[int],
    log_frequency_initial_values: ndarray[float],
    anelasticity_description: AnelasticityDescription,
    result_subpath: Path,
    save_result_per_degree: bool,
) -> tuple[ndarray, ndarray]:
    """
    Performs Love numbers computing (n, frequency) with given anelasticity description and hyper-parameters.
    Returns log frequency array and Love numbers as an array.
    """
    # Initializes a Callable as a global variable to parallelize.
    global anelastic_Love_number_computing_per_degree

    def anelastic_Love_number_computing_per_degree(n: int) -> tuple[ndarray[float], ndarray[complex]]:
        """
        To multiprocess. Returns log frequency array and Love numbers array for a given degree.
        """
        return anelastic_Love_number_computing_per_degree_function(
            n=n,
            anelasticity_description=anelasticity_description,
            y_system_hyper_parameters=y_system_hyper_parameters,
            use_long_term_anelasticity=use_long_term_anelasticity,
            use_short_term_anelasticity=use_short_term_anelasticity,
            use_bounded_attenuation_functions=use_bounded_attenuation_functions,
            log_frequency_initial_values=log_frequency_initial_values,
            max_tol=max_tol,
            decimals=decimals,
            degree_path=result_subpath.joinpath("per_degree").joinpath(str(n)),
            save_result_per_degree=save_result_per_degree,
        )

    with Pool() as p:  # Processes for degrees.
        anelastic_Love_numbers_tuples: list[tuple[ndarray[float], ndarray[complex]]] = p.map(
            func=anelastic_Love_number_computing_per_degree, iterable=degrees
        )

    # Interpolates in frequency for all degrees.
    log_frequency_values_per_degree = [
        round(a=anelastic_Love_numbers_tuple[0], decimals=decimals)
        for anelastic_Love_numbers_tuple in anelastic_Love_numbers_tuples
    ]
    Love_numbers = [anelastic_Love_numbers_tuple[1] for anelastic_Love_numbers_tuple in anelastic_Love_numbers_tuples]
    log_frequency_all_values = unique(concatenate(log_frequency_values_per_degree))
    all_Love_numbers = interpolate_all(
        x_values_per_component=log_frequency_values_per_degree,
        function_values=Love_numbers,
        x_shared_values=log_frequency_all_values,
    )

    # Returns id for comparison purposes.
    return (
        log_frequency_all_values,
        all_Love_numbers,
    )


def anelastic_Love_number_computing_per_degree_function(
    n: int,
    anelasticity_description: AnelasticityDescription,
    y_system_hyper_parameters: YSystemHyperParameters,
    use_long_term_anelasticity: bool,
    use_short_term_anelasticity: bool,
    use_bounded_attenuation_functions: bool,
    log_frequency_initial_values: ndarray[float],
    max_tol: float,
    decimals: int,
    degree_path: Path,
    save_result_per_degree: bool,
) -> tuple[ndarray[float], ndarray[complex]]:
    """
    Computes Love numbers for all frequencies, for a given degree.
    """

    # Defines a callable that computes Love numbers for an array of log10(frequency/ unit_frequency) values.
    Love_number_computing_parallel = lambda log_frequency_values: array(
        [
            Integration(
                anelasticity_description=anelasticity_description,
                log_frequency=log_frequency,
                use_long_term_anelasticity=use_long_term_anelasticity,
                use_short_term_anelasticity=use_short_term_anelasticity,
                use_bounded_attenuation_functions=use_bounded_attenuation_functions,
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
        f=Love_number_computing_parallel,
        max_tol=max_tol,
        decimals=decimals,
    )

    # Saves single degree results.
    if save_result_per_degree:
        save_frequencies(
            log_frequency_values=log_frequency_values, frequency_unit=anelasticity_description.frequency_unit, path=degree_path
        )
        Love_numbers_result = Result(hyper_parameters=y_system_hyper_parameters)
        Love_numbers_result.update_values_from_array(result_array=Love_numbers, degrees=[n])
        Love_numbers_result.save(name="anelastic_Love_numbers", path=degree_path)

    return log_frequency_values, Love_numbers


def elastic_Love_numbers_computing(
    y_system_hyper_parameters: YSystemHyperParameters,
    degrees: list[int],
    anelasticity_description: AnelasticityDescription,
) -> ndarray:
    """
    Performs Love numbers computing (n) for elastic case with given anelasticity description and hyper-parameters.
    """
    global elastic_Love_number_computing_per_degree

    def elastic_Love_number_computing_per_degree(n: int) -> list[ndarray]:
        """
        To multiprocess. Returns Love numbers array for a given degree.
        """
        return [
            Integration(
                anelasticity_description=anelasticity_description,
                log_frequency=Inf,
                use_long_term_anelasticity=False,
                use_short_term_anelasticity=False,
                use_bounded_attenuation_functions=False,
            ).y_system_integration(
                n=n,
                hyper_parameters=y_system_hyper_parameters,
            )
        ]

    with Pool() as p:  # Processes for degrees.
        return array(p.map(func=elastic_Love_number_computing_per_degree, iterable=degrees))
