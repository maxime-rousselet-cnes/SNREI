from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from numpy import Inf, array, concatenate, linspace, log10, ndarray, round, unique

from .abstract_computing import interpolate_all, precise_curvature
from .classes import (
    Integration,
    LoveNumbersHyperParameters,
    RealDescription,
    Result,
    YSystemHyperParameters,
    load_Love_numbers_hyper_parameters,
    real_description_from_parameters,
)
from .database import generate_degrees_list, save_base_model
from .paths import results_path

BOOLEANS = [False, True]
SAMPLINGS = {"low": 10, "mid": 100, "high": 1000}


def elastic_Love_numbers_computing(
    y_system_hyper_parameters: YSystemHyperParameters,
    degrees: list[int],
    real_description: RealDescription,
) -> ndarray:
    """
    Performs Love numbers computing (n) for elastic case with given real description and hyper-parameters.
    """
    global elastic_Love_number_computing_per_degree

    def elastic_Love_number_computing_per_degree(n: int) -> list[ndarray]:
        """
        To multiprocess.
        """
        return [
            Integration(
                real_description=real_description,
                log_frequency=Inf,
                use_attenuation=False,
                use_anelasticity=False,
                bounded_attenuation_functions=False,
            ).y_system_integration(
                n=n,
                hyper_parameters=y_system_hyper_parameters,
            )
        ]

    with Pool() as p:  # Processes for degrees.
        return array(p.map(func=elastic_Love_number_computing_per_degree, iterable=degrees))


def save_frequencies(log_frequency_values: ndarray[float], frequency_unit: float, path: Path) -> None:
    """
    Maps back log unitless frequencies to (Hz) and save to (.JSON) file.
    """
    save_base_model(obj=10.0**log_frequency_values * frequency_unit, name="frequencies", path=path)


def anelastic_Love_number_computing_per_degree_function(
    n: int,
    real_description: RealDescription,
    y_system_hyper_parameters: YSystemHyperParameters,
    use_anelasticity: bool,
    use_attenuation: bool,
    bounded_attenuation_functions: bool,
    log_frequency_initial_values: ndarray[float],
    max_tol: float,
    decimals: int,
    result_per_degree_path: Path,
) -> tuple[ndarray[float], ndarray[complex]]:
    """
    Computes Love numbers for all frequencies, for a given degree.
    """

    # Defines a callable that computes Love numbers for an array of log10(frequency/ unit_frequency) values.
    Love_number_computing_parallel = lambda log_frequency_values: array(
        [
            Integration(
                real_description=real_description,
                log_frequency=log_frequency,
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
                bounded_attenuation_functions=bounded_attenuation_functions,
            ).y_system_integration(
                n=n,
                hyper_parameters=y_system_hyper_parameters,
            )
            for log_frequency in log_frequency_values
        ]
    )

    # Processes for frequencies. Adaptative step for precise curvature.
    log_frequency_values, Love_numbers = precise_curvature(
        x_initial_values=log_frequency_initial_values,
        f=Love_number_computing_parallel,
        max_tol=max_tol,
        decimals=decimals,
    )

    # Saves single degree results.
    path_for_degree = result_per_degree_path.joinpath(str(n))
    save_frequencies(
        log_frequency_values=log_frequency_values, frequency_unit=real_description.frequency_unit, path=path_for_degree
    )
    save_base_model(
        obj={"real": Love_numbers.real, "imag": Love_numbers.imag},
        name="Love_numbers",
        path=path_for_degree,
    )

    return log_frequency_values, Love_numbers


def Love_numbers_computing(
    max_tol: float,
    decimals: int,
    y_system_hyper_parameters: YSystemHyperParameters,
    use_anelasticity: bool,
    use_attenuation: bool,
    bounded_attenuation_functions: bool,
    degrees: list[int],
    log_frequency_initial_values: ndarray[float],
    real_description: RealDescription,
    runs_path: Path,
    run_id: str,
) -> tuple[Path, ndarray, ndarray]:
    """
    Performs Love numbers computing (n, frequency) with given real description and hyper-parameters.
    """
    # Initializes the run.
    run_path = runs_path.joinpath(run_id)
    result_per_degree_path = run_path.joinpath("per_degree")

    # Anelastic case.
    global anelastic_Love_number_computing_per_degree

    def anelastic_Love_number_computing_per_degree(n: int) -> tuple[ndarray[float], ndarray[complex]]:
        """
        To multiprocess.
        """
        return anelastic_Love_number_computing_per_degree_function(
            n=n,
            real_description=real_description,
            y_system_hyper_parameters=y_system_hyper_parameters,
            use_anelasticity=use_anelasticity,
            use_attenuation=use_attenuation,
            bounded_attenuation_functions=bounded_attenuation_functions,
            log_frequency_initial_values=log_frequency_initial_values,
            max_tol=max_tol,
            decimals=decimals,
            result_per_degree_path=result_per_degree_path,
        )

    with Pool() as p:  # Processes for degrees.
        anelastic_Love_numbers: list[tuple[ndarray[float], ndarray[complex]]] = p.map(
            func=anelastic_Love_number_computing_per_degree, iterable=degrees
        )

    # Interpolates in frequency for all degrees.
    log_frequency_values_per_degree = [
        round(a=anelastic_Love_numbers_tuple[0], decimals=decimals) for anelastic_Love_numbers_tuple in anelastic_Love_numbers
    ]
    Love_numbers = [anelastic_Love_numbers_tuple[1] for anelastic_Love_numbers_tuple in anelastic_Love_numbers]
    log_frequency_all_values = unique(concatenate(log_frequency_values_per_degree))
    all_Love_numbers = interpolate_all(
        x_values_per_component=log_frequency_values_per_degree,
        function_values=Love_numbers,
        x_shared_values=log_frequency_all_values,
    )

    # Returns id for comparison purposes.
    return (
        run_path,
        log_frequency_all_values,
        all_Love_numbers,
    )


def gets_run_id(use_anelasticity: bool, bounded_attenuation_functions: bool, use_attenuation: bool) -> str:
    """
    Generates an ID for a run using its hyper parameters.
    """
    return "_".join(
        (
            "anelasticity" if use_anelasticity else "",
            "bouded" if bounded_attenuation_functions else "",
            "attenuation" if use_attenuation else "",
        )
    )


def generate_log_frequency_initial_values(
    frequency_min: float, frequency_max: float, n_frequency_0: int, frequency_unit: float
) -> ndarray:
    """
    Generates an array of logarithm-spaced frequency values.
    """
    return linspace(
        start=log10(frequency_min / frequency_unit),
        stop=log10(frequency_max / frequency_unit),
        num=n_frequency_0,
    )


def Love_numbers_from_models_to_result(
    real_description_id: Optional[str],
    run_id: Optional[str] = None,
    load_description: Optional[bool] = None,
    elasticity_model_from_name: Optional[str] = None,
    anelasticity_model_from_name: Optional[str] = None,
    attenuation_model_from_name: Optional[str] = None,
    Love_numbers_hyper_parameters: Optional[LoveNumbersHyperParameters] = None,
    save: bool = True,
) -> None:
    """
    Loads models/descriptions, hyper parameters.
    Compute elastic and anelastic Love numbers.
    Save Results in (.JSON) files.
    """
    # Loads hyper parameters.
    if not Love_numbers_hyper_parameters:
        Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()

    # Generates an ID for the run.
    if not run_id:
        run_id = gets_run_id(
            use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
            bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
            use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
        )

    # Eventually stops.
    if Love_numbers_hyper_parameters.bounded_attenuation_functions and not Love_numbers_hyper_parameters.use_attenuation:
        print("Skip description", real_description_id, "- run ", run_id, "because of incompatible attenuation options.")
        return

    # Loads/buils the planet's description.
    real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=real_description_id,
        load_description=load_description,
        elasticity_model_from_name=elasticity_model_from_name,
        anelasticity_model_from_name=anelasticity_model_from_name,
        attenuation_model_from_name=attenuation_model_from_name,
        save=save,
    )

    # Generates degrees.
    degrees = generate_degrees_list(
        degree_thresholds=Love_numbers_hyper_parameters.degree_thresholds,
        degree_steps=Love_numbers_hyper_parameters.degree_steps,
    )

    # Generates frequencies.
    log_frequency_initial_values = generate_log_frequency_initial_values(
        frequency_min=Love_numbers_hyper_parameters.frequency_min,
        frequency_max=Love_numbers_hyper_parameters.frequency_max,
        n_frequency_0=Love_numbers_hyper_parameters.n_frequency_0,
        frequency_unit=real_description.frequency_unit,
    )

    # Computes all Love numbers.
    results_for_description_path = results_path.joinpath(real_description.id)
    run_path, log_frequency_values, anelastic_Love_numbers = Love_numbers_computing(
        max_tol=Love_numbers_hyper_parameters.max_tol,
        decimals=Love_numbers_hyper_parameters.decimals,
        y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
        use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
        use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
        bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
        degrees=degrees,
        log_frequency_initial_values=log_frequency_initial_values,
        real_description=real_description,
        runs_path=results_for_description_path.joinpath("runs"),
        run_id=run_id,
    )

    # Builds result structures and saves to (.JSON) files.
    # Elastic.
    elastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
    elastic_result.update_values_from_array(
        result_array=elastic_Love_numbers_computing(
            y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
            degrees=degrees,
            real_description=real_description,
        ),
        degrees=degrees,
    )
    elastic_result.save(name="elastic_Love_numbers", path=results_for_description_path)
    # Anelastic.
    anelastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
    anelastic_result.update_values_from_array(result_array=anelastic_Love_numbers, degrees=degrees)
    anelastic_result.save(name="anelastic_Love_numbers", path=run_path)
    # Frequencies.
    save_frequencies(log_frequency_values=log_frequency_values, frequency_unit=real_description.frequency_unit, path=run_path)
    # Save degrees.
    save_base_model(obj=degrees, name="degrees", path=results_for_description_path)

    # Prints status.
    print("Finished description", real_description_id, "- run", run_id)
