from enum import Enum
from pathlib import Path


class ModelPart(Enum):
    """
    Available model parts.
    """

    elasticity = "elasticity"
    long_term_anelasticity = "long_term_anelasticity"
    short_term_anelasticity = "short_term_anelasticity"


data_path = Path("data")
GRACE_data_path = data_path.joinpath("GRACE")

GMSL_data_path = data_path.joinpath("GMSL_data")
masks_data_path = data_path.joinpath("masks")
GRACE_trends_data_path = data_path.joinpath("trends_GRACE")

models_base_path = data_path.joinpath("models")
models_path: dict[ModelPart, Path] = {
    model_part: models_base_path.joinpath(model_part.name) for model_part in ModelPart
}

parameters_path = data_path.joinpath("parameters")

descriptions_base_path = data_path.joinpath("descriptions")
descriptions_path: dict[ModelPart, Path] = {
    model_part: descriptions_base_path.joinpath(model_part.name) for model_part in ModelPart
}
anelasticity_descriptions_path = data_path.joinpath("descriptions").joinpath("anelasticity_descriptions")

results_path = data_path.joinpath("results")
tables_path = results_path.joinpath("tables")

dates_path = results_path.joinpath("dates")
frequencies_path = results_path.joinpath("frequencies")
Love_numbers_path = results_path.joinpath("Love_numbers")
elastic_load_signals_path = results_path.joinpath("elastic_load_signals")
elastic_load_signal_trends_path = results_path.joinpath("elastic_load_signal_trends")
harmonic_load_signal_trends_path = results_path.joinpath("harmonic_load_signal_trends")
harmonic_geoid_trends_path = results_path.joinpath("harmonic_geoid_trends")
harmonic_radial_displacement_trends_path = results_path.joinpath("harmonic_radial_displacement_trends")
harmonic_load_signal_trends_before_degree_one_replacement_path = results_path.joinpath(
    "harmonic_load_signal_trends_before_degree_one_replacement"
)
harmonic_residual_trends_path = results_path.joinpath("harmonic_residual_trends")
base_format_load_signal_trends_path = results_path.joinpath("base_format_load_signal_trends")

figures_path = data_path.joinpath("figures")
