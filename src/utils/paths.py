from pathlib import Path

from .classes import ModelPart

data_path = Path("data")

models_path: dict[ModelPart, Path] = {
    model_part: data_path.joinpath("models").joinpath(model_part.name) for model_part in ModelPart
}

parameters_path = data_path.joinpath("parameters")

descriptions_path: dict[ModelPart, Path] = {
    model_part: data_path.joinpath("descriptions").joinpath(model_part.name) for model_part in ModelPart
}
real_descriptions_path = data_path.joinpath("descriptions").joinpath("real_descriptions")

results_path = data_path.joinpath("results")

figures_path = data_path.joinpath("figures")
