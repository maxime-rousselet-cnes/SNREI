from pathlib import Path

data_path = Path("data")

models_path = data_path.joinpath("models")
elasticity_models_path = models_path.joinpath("elasticity")
anelasticity_models_path = models_path.joinpath("anelasticity")
attenuation_models_path = models_path.joinpath("attenuation")

parameters_path = data_path.joinpath("parameters")

descriptions_path = data_path.joinpath("descriptions")
elasticity_descriptions_path = descriptions_path.joinpath("elasticity_descriptions")
anelasticity_descriptions_path = descriptions_path.joinpath("anelasticity_descriptions")
attenuation_descriptions_path = descriptions_path.joinpath("attenuation_descriptions")
real_descriptions_path = descriptions_path.joinpath("real_descriptions")


results_path = data_path.joinpath("results")
