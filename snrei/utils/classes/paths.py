from enum import Enum
from pathlib import Path


class ModelPart(Enum):
    """
    Available model parts.
    """

    elasticity = "elasticity"
    long_term_anelasticity = "long_term_anelasticity"
    short_term_anelasticity = "short_term_anelasticity"


# General paths.
data_path = Path("data")
results_path = data_path.joinpath("results")
tables_path = results_path.joinpath("tables")
figures_path = data_path.joinpath("figures")

# Input data.
GRACE_data_path = data_path.joinpath("GRACE")
GMSL_data_path = data_path.joinpath("GMSL_data")
pole_data_path = data_path.joinpath("pole_data")
masks_data_path = data_path.joinpath("masks")

# Models.
models_base_path = data_path.joinpath("models")
models_path: dict[ModelPart, Path] = {model_part: models_base_path.joinpath(model_part.name) for model_part in ModelPart}

# Parameters.
parameters_path = data_path.joinpath("parameters")

# Descriptions.
descriptions_base_path = data_path.joinpath("descriptions")
descriptions_path: dict[ModelPart, Path] = {model_part: descriptions_base_path.joinpath(model_part.name) for model_part in ModelPart}
anelasticity_descriptions_path = data_path.joinpath("descriptions").joinpath("anelasticity_descriptions")


# Elastic load signal.
dates_path = results_path.joinpath("dates")
frequencies_path = results_path.joinpath("frequencies")
elastic_load_signal_trends_path = results_path.joinpath("elastic_load_signal_trends")

# Love numbers
Love_numbers_path = results_path.joinpath("Love_numbers")

# Degree one inversion components.
harmonic_geoid_trends_path = results_path.joinpath("harmonic_geoid_trends")
harmonic_radial_displacement_trends_path = results_path.joinpath("harmonic_radial_displacement_trends")
harmonic_residual_trends_path = results_path.joinpath("harmonic_residual_trends")

# Load signals.
harmonic_load_signal_trends_path = results_path.joinpath("harmonic_load_signal_trends")
base_format_load_signal_trends_path = results_path.joinpath("base_format_load_signal_trends")
