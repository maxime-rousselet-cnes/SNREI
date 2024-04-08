from typing import Optional

from numpy import linspace, log10, ndarray

from ..classes import (
    OPTIONS,
    AnelasticityDescription,
    LoveNumbersHyperParameters,
    ModelPart,
    Result,
    RunHyperParameters,
    load_Love_numbers_hyper_parameters,
    results_path,
)
from ..database import generate_degrees_list, get_run_folder_name, save_base_model
from .run import (
    anelastic_Love_numbers_computing,
    elastic_Love_numbers_computing,
    save_frequencies,
)


def generate_log_frequency_initial_values(
    frequency_min: float, frequency_max: float, n_frequency_0: int, frequency_unit: float
) -> ndarray[float]:
    """
    Generates an array of log-spaced frequency values.
    """
    return linspace(
        start=log10(frequency_min / frequency_unit),
        stop=log10(frequency_max / frequency_unit),
        num=n_frequency_0,
    )


def Love_numbers_from_models_for_options(
    forced_anelasticity_description_id: Optional[str] = None,
    overwrite_descriptions: bool = False,
    part_names: dict[ModelPart, Optional[str]] = {model_part: None for model_part in ModelPart},
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
    do_elastic_case: bool = True,
) -> str:
    """
    Loads models/descriptions, hyper parameters.
    Compute elastic and anelastic Love numbers.
    Save Results in (.JSON) files.
    Returns the anelasticity description id.
    """

    # Loads/buils the planet's description.
    anelasticity_description = AnelasticityDescription(
        anelasticity_description_parameters=Love_numbers_hyper_parameters.anelasticity_description_parameters,
        load_description=True,
        id=forced_anelasticity_description_id,
        save=True,
        overwrite_descriptions=overwrite_descriptions,
        elasticity_name=part_names[ModelPart.elasticity],
        long_term_anelasticity_name=part_names[ModelPart.long_term_anelasticity],
        short_term_anelasticity_name=part_names[ModelPart.short_term_anelasticity],
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
        frequency_unit=anelasticity_description.frequency_unit,
    )

    # Prints status.
    print("Description: " + anelasticity_description.id + ":")

    # Loops on run options.
    for option in options:

        # Generates an ID for the run.
        run_id = option.run_id()
        result_subpath = results_path.joinpath(
            get_run_folder_name(anelasticity_description_id=anelasticity_description.id, run_id=run_id)
        )

        # Computes all Love numbers.
        log_frequency_values, anelastic_Love_numbers = anelastic_Love_numbers_computing(
            max_tol=Love_numbers_hyper_parameters.max_tol,
            decimals=Love_numbers_hyper_parameters.decimals,
            y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
            use_long_term_anelasticity=Love_numbers_hyper_parameters.run_hyper_parameters.use_long_term_anelasticity,
            use_short_term_anelasticity=Love_numbers_hyper_parameters.run_hyper_parameters.use_short_term_anelasticity,
            use_bounded_attenuation_functions=Love_numbers_hyper_parameters.run_hyper_parameters.use_bounded_attenuation_functions,
            degrees=degrees,
            log_frequency_initial_values=log_frequency_initial_values,
            anelasticity_description=anelasticity_description,
            result_subpath=result_subpath,
        )

        # Saves.
        anelastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
        anelastic_result.update_values_from_array(result_array=anelastic_Love_numbers, degrees=degrees)
        anelastic_result.save(name="anelastic_Love_numbers", path=result_subpath)
        save_frequencies(
            log_frequency_values=log_frequency_values,
            frequency_unit=anelasticity_description.frequency_unit,
            path=result_subpath,
        )
        # Load bar.
        print("----Run: " + run_id + ": Done.")

    # Saves elastic results.
    if do_elastic_case:
        elastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
        elastic_result.update_values_from_array(
            result_array=elastic_Love_numbers_computing(
                y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
                degrees=degrees,
                anelasticity_description=anelasticity_description,
            ),
            degrees=degrees,
        )
        elastic_result.save(name="elastic_Love_numbers", path=result_subpath.parent.parent)
        save_base_model(obj=degrees, name="degrees", path=result_subpath.parent.parent)

    return anelasticity_description.id
