from pathlib import Path

from src import (
    Love_numbers_for_options_for_models_for_parameters,
    ModelPart,
    RunHyperParameters,
    load_signal_for_options_for_models_for_parameters_for_elastic_load_signals,
)

options = [
    RunHyperParameters(
        use_long_term_anelasticity=True,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=True,
    ),
    RunHyperParameters(
        use_long_term_anelasticity=False,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=True,
    ),
    RunHyperParameters(
        use_long_term_anelasticity=True,
        use_short_term_anelasticity=False,
        use_bounded_attenuation_functions=False,
    ),
]

anelasticity_description_ids, model_filenames = (
    Love_numbers_for_options_for_models_for_parameters(
        elasticity_model_names=["PREM"],
        long_term_anelasticity_model_names=["Mao_Zhong"],
        short_term_anelasticity_model_names=[
            "Benjamin_Q_Resovsky",
        ],
        parameters={
            ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
            ModelPart.short_term_anelasticity: {
                "asymptotic_mu_ratio": {"MANTLE": [[0.1]]}
            },
        },
    )
)
"""
load_result_folders: list[Path] = load_signal_for_options_for_models_for_parameters_for_elastic_load_signals(
    anelasticity_description_ids=anelasticity_description_ids,
    model_filenames=model_filenames,
    load_signal_hyper_parameter_variations={
        "case": ["lower", "mean", "upper"],
        "little_isostatic_adjustment": [False],
        "opposite_load_on_continents": [True],
    },
    symlinks=True,
    options=options,
)
"""
