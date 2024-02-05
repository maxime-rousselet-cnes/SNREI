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
)
from .constants import Earth_radius
from .database import (
    generate_degrees_list,
    generate_id,
    load_base_model,
    save_base_model,
)
from .paths import parameters_path, results_path


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
                log_omega=Inf,
                use_attenuation=False,
                use_anelasticity=False,
            ).y_system_integration(
                n=n,
                hyper_parameters=y_system_hyper_parameters,
            )
        ]

    with Pool() as p:  # Processes for degrees.
        return array(p.map(func=elastic_Love_number_computing_per_degree, iterable=degrees))


def Love_numbers_computing(
    max_tol: float,
    decimals: int,
    y_system_hyper_parameters: YSystemHyperParameters,
    use_anelasticity: bool,
    use_attenuation: bool,
    degrees: list[int],
    log_omega_initial_values: ndarray[float],
    real_description: RealDescription,
    runs_path: Path,
    id: Optional[str] = None,
) -> tuple[Path, ndarray, ndarray]:
    """
    Performs Love numbers computing (n, omega) with given real description and hyper-parameters.
    """
    # Initializes the run.
    id_run = id if id else generate_id()
    run_path = runs_path.joinpath(id_run)
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
            log_omega_initial_values=log_omega_initial_values,
            max_tol=max_tol,
            decimals=decimals,
            result_per_degree_path=result_per_degree_path,
        )

    with Pool() as p:  # Processes for degrees.
        anelastic_Love_numbers: list[tuple[ndarray[float], ndarray[complex]]] = p.map(
            func=anelastic_Love_number_computing_per_degree, iterable=degrees
        )

    # Interpolates in frequency for all degrees.
    log_omega_values_per_degree = [
        round(a=anelastic_Love_numbers_tuple[0], decimals=decimals) for anelastic_Love_numbers_tuple in anelastic_Love_numbers
    ]
    Love_numbers = [anelastic_Love_numbers_tuple[1] for anelastic_Love_numbers_tuple in anelastic_Love_numbers]
    log_omega_all_values = unique(concatenate(log_omega_values_per_degree))
    all_Love_numbers = interpolate_all(
        x_values_per_component=log_omega_values_per_degree, function_values=Love_numbers, x_shared_values=log_omega_all_values
    )

    # Returns id for comparison purposes.
    return (
        run_path,
        log_omega_all_values,
        all_Love_numbers,
    )


def anelastic_Love_number_computing_per_degree_function(
    n: int,
    real_description: RealDescription,
    y_system_hyper_parameters: YSystemHyperParameters,
    use_anelasticity: bool,
    use_attenuation: bool,
    log_omega_initial_values: ndarray[float],
    max_tol: float,
    decimals: int,
    result_per_degree_path: Path,
) -> tuple[ndarray[float], ndarray[complex]]:
    """
    Computes Love numbers for all frequencies, for a given degree.
    """

    # Defines a callable that computes Love numbers for an array of log10(omega) values.
    Love_number_computing_parallel = lambda log_omega_values: array(
        [
            Integration(
                real_description=real_description,
                log_omega=log_omega,
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
            ).y_system_integration(
                n=n,
                hyper_parameters=y_system_hyper_parameters,
            )
            for log_omega in log_omega_values
        ]
    )

    # Processes for frequencies. Adaptative step for precise curvature.
    log_omega_values, Love_numbers = precise_curvature(
        x_initial_values=log_omega_initial_values,
        f=Love_number_computing_parallel,
        max_tol=max_tol,
        decimals=decimals,
    )

    # Saves single degree results.
    path_for_degree = result_per_degree_path.joinpath(str(n))
    save_base_model(obj=log_omega_values, name="frequencies", path=path_for_degree)
    save_base_model(
        obj={"real": Love_numbers.real, "imag": Love_numbers.imag},
        name="Love_numbers",
        path=path_for_degree,
    )

    return log_omega_values, Love_numbers


def Love_numbers_from_models_to_result() -> str:
    """
    Loads models/descriptions, hyper parameters.
    Compute elastic and anelastic Love numbers.
    Save Results in (.JSON) files.
    """
    # Loads hyper parameters.
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name="Love_numbers_hyper_parameters", path=parameters_path, base_model_type=LoveNumbersHyperParameters
    )
    Love_numbers_hyper_parameters.load()

    # Loads/buils the planet's description.
    real_description_parameters = Love_numbers_hyper_parameters.real_description_parameters
    real_description = RealDescription(
        below_ICB_layers=real_description_parameters.below_ICB_layers,
        below_CMB_layers=real_description_parameters.below_CMB_layers,
        splines_degree=real_description_parameters.splines_degree,
        radius_unit=real_description_parameters.radius_unit if real_description_parameters.radius_unit else Earth_radius,
        real_crust=real_description_parameters.real_crust,
        n_splines_base=real_description_parameters.n_splines_base,
        profile_precision=real_description_parameters.profile_precision,
        radius=real_description_parameters.radius if real_description_parameters.radius else Earth_radius,
        load_description=False,
    )

    # Generates degrees.
    degrees = generate_degrees_list(
        degree_thresholds=Love_numbers_hyper_parameters.degree_thresholds,
        degree_steps=Love_numbers_hyper_parameters.degree_steps,
    )

    # Generates frequencies.
    log_omega_initial_values = generate_log_omega_initial_values(
        omega_min=Love_numbers_hyper_parameters.omega_min,
        omega_max=Love_numbers_hyper_parameters.omega_max,
        n_omega_0=Love_numbers_hyper_parameters.n_omega_0,
        frequency_unit=real_description.frequency_unit,
    )

    # Computes all Love numbers.
    results_for_description_path = results_path.joinpath(real_description.id)
    run_path, log_omega_values, anelastic_Love_numbers = Love_numbers_computing(
        max_tol=Love_numbers_hyper_parameters.max_tol,
        decimals=Love_numbers_hyper_parameters.decimals,
        y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
        use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
        use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
        degrees=degrees,
        log_omega_initial_values=log_omega_initial_values,
        real_description=real_description,
        runs_path=results_for_description_path.joinpath("runs"),
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
    save_base_model(obj=10.0**log_omega_values * real_description.frequency_unit, name="frequencies", path=run_path)

    # returns id.
    return run_path.name


def generate_log_omega_initial_values(omega_min: float, omega_max: float, n_omega_0: int, frequency_unit: float) -> ndarray:
    """
    Generates an array of logarithm-spaced frequency values.
    """
    return linspace(
        start=log10(omega_min / frequency_unit),
        stop=log10(omega_max / frequency_unit),
        num=n_omega_0,
    )
