from pathlib import Path

from src import (
    Love_numbers_for_options_for_models_for_parameters,
    ModelPart,
    RunHyperParameters,
    clear_subs,
    load_signal_for_options_for_models_for_parameters_for_elastic_load_signals,
)

# TODO: 3 cases, verify symlinks out.
options = [
    RunHyperParameters(
        use_long_term_anelasticity=True, use_short_term_anelasticity=True, use_bounded_attenuation_functions=True
    ),
    RunHyperParameters(
        use_long_term_anelasticity=True, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False
    ),
    RunHyperParameters(
        use_long_term_anelasticity=False, use_short_term_anelasticity=True, use_bounded_attenuation_functions=True
    ),
]
clear_subs()

# 16h00.

anelasticity_description_ids, model_filenames = Love_numbers_for_options_for_models_for_parameters(
    elasticity_model_names=["PREM"],
    long_term_anelasticity_model_names=["VM7"],  # ["Caron", "Lambeck_2017", "Lau_2016", "VM5a", "VM7"],
    short_term_anelasticity_model_names=[
        "Benjamin_Q_Resovsky",
    ],  # ["Benjamin_Q_PAR3P", "Benjamin_Q_PREM", "Benjamin_Q_QL6", "Benjamin_Q_QM1", "Benjamin_Q_Resovsky"],
    parameters={
        ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}}
    },  #  {"eta_m": {"D''": [[1e17]], "ASTHENOSPHERE": [[3e19]]}}}
    symlinks=True,
    options=options,
)
load_result_folders: list[Path] = load_signal_for_options_for_models_for_parameters_for_elastic_load_signals(
    anelasticity_description_ids=anelasticity_description_ids,
    model_filenames=model_filenames,
    load_signal_hyper_parameter_variations={
        "case": ["mean"],
        "little_isostatic_adjustment": [False],
        "opposite_load_on_continents": [False, True],
    },
    symlinks=True,
    options=options,
)
