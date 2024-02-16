import argparse
from itertools import product
from typing import Optional

from utils import (
    Earth_radius,
    Love_numbers_computing,
    LoveNumbersHyperParameters,
    RealDescription,
    Result,
    elastic_Love_numbers_computing,
    generate_degrees_list,
    generate_log_frequency_initial_values,
    load_base_model,
    parameters_path,
    real_descriptions_path,
    results_path,
    save_base_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--load_description", action="store_true", help="Option to tell if the description should be loaded")

args = parser.parse_args()


def Love_number_comparative(
    real_description_id: Optional[str],
    load_description: Optional[bool],
    anelasticity_model_from_name: Optional[str] = None,
    Love_numbers_hyper_parameters: Optional[LoveNumbersHyperParameters] = None,
) -> str:
    """
    Computes anelastic Love numbers with and without anelasticity and with and without attenuation.
    """
    # Loads hyper parameters.
    if not Love_numbers_hyper_parameters:
        Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
            name="Love_numbers_hyper_parameters", path=parameters_path, base_model_type=LoveNumbersHyperParameters
        )
        Love_numbers_hyper_parameters.load()

    # Loads/buils the planet's description.
    real_description_parameters = Love_numbers_hyper_parameters.real_description_parameters
    real_description = RealDescription(
        id=real_description_id,
        below_ICB_layers=real_description_parameters.below_ICB_layers,
        below_CMB_layers=real_description_parameters.below_CMB_layers,
        splines_degree=real_description_parameters.splines_degree,
        radius_unit=real_description_parameters.radius_unit if real_description_parameters.radius_unit else Earth_radius,
        real_crust=real_description_parameters.real_crust,
        n_splines_base=real_description_parameters.n_splines_base,
        profile_precision=real_description_parameters.profile_precision,
        radius=real_description_parameters.radius if real_description_parameters.radius else Earth_radius,
        load_description=load_description,
        anelasticity_model_from_name=anelasticity_model_from_name,
    )
    if load_description:
        real_description.load(path=real_descriptions_path)

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

    results_for_description_path = results_path.joinpath(real_description.id)
    # Loops on boolean hyper parameters.
    for use_anelasticity, use_attenuation in product([False, True], [True, False]):
        if use_anelasticity or use_attenuation:
            # Computes Love numbers.
            run_path, log_frequency_values, anelastic_Love_numbers = Love_numbers_computing(
                max_tol=Love_numbers_hyper_parameters.max_tol,
                decimals=Love_numbers_hyper_parameters.decimals,
                y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
                bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
                degrees=degrees,
                log_frequency_initial_values=log_frequency_initial_values,
                real_description=real_description,
                runs_path=results_for_description_path.joinpath("runs"),
                id="anelasticity_" + str(use_anelasticity) + "__attenuation_" + str(use_attenuation),
            )
            # Builds result structures and saves to (.JSON) files.
            # Anelastic.
            anelastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
            anelastic_result.update_values_from_array(result_array=anelastic_Love_numbers, degrees=degrees)
            anelastic_result.save(name="anelastic_Love_numbers", path=run_path)
            # Frequencies.
            save_base_model(obj=10.0**log_frequency_values * real_description.frequency_unit, name="frequencies", path=run_path)

    # Elastic.
    elastic_Love_numbers = elastic_Love_numbers_computing(
        y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
        degrees=degrees,
        real_description=real_description,
    )
    elastic_result = Result(hyper_parameters=Love_numbers_hyper_parameters)
    elastic_result.update_values_from_array(result_array=elastic_Love_numbers, degrees=degrees)
    elastic_result.save(name="elastic_Love_numbers", path=results_for_description_path)
    # Save degrees.
    save_base_model(obj=degrees, name="degrees", path=results_for_description_path)

    return real_description_id


if __name__ == "__main__":
    print(Love_number_comparative(real_description_id=args.real_description_id, load_description=args.load_description))
