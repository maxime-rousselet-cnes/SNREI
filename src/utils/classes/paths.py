from pathlib import Path

from .model import ModelPart

data_path = Path("data")

GMSL_data_path = data_path.joinpath("GMSL_data")
data_masks_path = data_path.joinpath("masks")
data_trends_GRACE_path = data_path.joinpath("trends_GRACE")

models_base_path = data_path.joinpath("models")
models_path: dict[ModelPart, Path] = {model_part: models_base_path.joinpath(model_part.name) for model_part in ModelPart}

parameters_path = data_path.joinpath("parameters")

descriptions_base_path = data_path.joinpath("descriptions")
descriptions_path: dict[ModelPart, Path] = {
    model_part: descriptions_base_path.joinpath(model_part.name) for model_part in ModelPart
}
anelasticity_descriptions_path = data_path.joinpath("descriptions").joinpath("anelasticity_descriptions")

results_path = data_path.absolute().parent.parent.joinpath("data/results")

figures_path = data_path.joinpath("figures")
