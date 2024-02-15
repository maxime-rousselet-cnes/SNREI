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

    for use_Asthenosphere_model, bounded_attenuation_functions in product([True, False], [False, True]):
        Love_numbers_hyper_parameters.bounded_attenuation_functions = bounded_attenuation_functions
        save_base_model(obj=Love_numbers_hyper_parameters, name="Love_numbers_hyper_parameters", path=parameters_path)
        Love_numbers_from_models_to_result(
            real_description_id="with"
            + ("" if use_Asthenosphere_model else "out")
            + "_Asthenosphere_model"
            + "_with"
            + ("" if bounded_attenuation_functions else "out")
            + "_bounded_attenuation_functions",
            run_id="anelasticity_True__attenuation_True",
            load_description=False,
            anelasticity_model_from_name="test-low-viscosity-Asthenosphere" if use_Asthenosphere_model else "test",
        )


if __name__ == "__main__":
    Love_number_comparative_for_sampling()
