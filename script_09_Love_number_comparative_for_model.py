from itertools import product

from script_01_Love_number_computation import Love_numbers_from_models_to_result
from utils import (
    LoveNumbersHyperParameters,
    load_base_model,
    parameters_path,
    save_base_model,
)


def Love_number_comparative_for_sampling() -> None:
    """
    Computes anelastic Love numbers with and without low viscosity Asthenosphere model and with and without bounded attenuation
    functions.
    """
    # Loads hyper parameters.
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name="Love_numbers_hyper_parameters", path=parameters_path, base_model_type=LoveNumbersHyperParameters
    )
    Love_numbers_hyper_parameters.use_anelasticity = True

    # Test model.
    Love_numbers_hyper_parameters.bounded_attenuation_functions = False
    Love_numbers_hyper_parameters.use_attenuation = True
    save_base_model(obj=Love_numbers_hyper_parameters, name="Love_numbers_hyper_parameters", path=parameters_path)
    Love_numbers_from_models_to_result(
        real_description_id="test",
        run_id="anelasticity_True__attenuation_True",
        load_description=False,
        anelasticity_model_from_name="test",
    )


if __name__ == "__main__":
    Love_number_comparative_for_sampling()
