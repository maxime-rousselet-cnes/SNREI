from pathlib import Path
from typing import Optional

from ..classes import LoveNumbersHyperParameters, ModelPart, RunHyperParameters
from ..constants import OPTIONS


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
    # Generates an ID for the run.
    if not run_id:
        run_id = gets_run_id(
            use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
            bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
            use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
        )

    # Eventually stops.
    if Love_numbers_hyper_parameters.bounded_attenuation_functions and not Love_numbers_hyper_parameters.use_attenuation:
        print("Skip description", anelasticity_description_id, "- run ", run_id, "because of incompatible attenuation options.")
        return

    # Loads/buils the planet's description.
    anelasticity_description = anelasticity_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        anelasticity_description_id=anelasticity_description_id,
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
        frequency_unit=anelasticity_description.frequency_unit,
    )

    # Computes all Love numbers.
    results_for_description_path = results_path.joinpath(anelasticity_description.id)
    run_path, log_frequency_values, anelastic_Love_numbers = Love_numbers_computing(
        max_tol=Love_numbers_hyper_parameters.max_tol,
        decimals=Love_numbers_hyper_parameters.decimals,
        y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
        use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
        use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
        bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
        degrees=degrees,
        log_frequency_initial_values=log_frequency_initial_values,
        anelasticity_description=anelasticity_description,
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
            anelasticity_description=anelasticity_description,
        ),
        degrees=degrees,
    )
    elastic_result.save(name="elastic_Love_numbers", path=results_for_description_path)
    # Anelastic.
    anelastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
    anelastic_result.update_values_from_array(result_array=anelastic_Love_numbers, degrees=degrees)
    anelastic_result.save(name="anelastic_Love_numbers", path=run_path)
    # Frequencies.
    save_frequencies(
        log_frequency_values=log_frequency_values, frequency_unit=anelasticity_description.frequency_unit, path=run_path
    )
    # Save degrees.
    save_base_model(obj=degrees, name="degrees", path=results_for_description_path)

    # Prints status.
    print("Finished description", anelasticity_description_id, "- run", run_id)
