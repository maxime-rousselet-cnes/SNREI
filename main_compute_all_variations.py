from pathlib import Path

from src import (
    Love_numbers_for_options_for_models_for_parameters,
    ModelPart,
    clear_subs,
    load_signal_for_options_for_models_for_parameters_for_elastic_load_signals,
)

clear_subs()

anelasticity_description_ids, model_filenames = Love_numbers_for_options_for_models_for_parameters(
    elasticity_model_names=["PREM"],
    long_term_anelasticity_model_names=["Caron", "Lambeck_2017", "Lau_2016", "VM5a", "VM7"],
    short_term_anelasticity_model_names=[
        "Benjamin_Q_PAR3P",
        "Benjamin_Q_PREM",
        "Benjamin_Q_QL6",
        "Benjamin_Q_QM1",
        "Benjamin_Q_Resovsky",
    ],
    parameters={ModelPart.long_term_anelasticity: {"eta_m": {"D''": [[1e17]], "ASTHENOSPHERE": [[3e19]]}}},
    symlinks=False,
)
load_result_folders: list[Path] = load_signal_for_options_for_models_for_parameters_for_elastic_load_signals(
    anelasticity_description_ids=anelasticity_description_ids,
    model_filenames=model_filenames,
    load_signal_hyper_parameter_variations={
        "case": ["mean", "lower", "upper", "best", "worst"],
        "little_isostatic_adjustment": [False, True],
        "opposite_load_on_continents": [False, True],
    },
    symlinks=True,
)
