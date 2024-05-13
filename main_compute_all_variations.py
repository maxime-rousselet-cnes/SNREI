from pathlib import Path

from src import (
    ModelPart,
    RunHyperParameters,
    create_all_model_variations,
    load_signal_for_options_for_models_for_parameters_for_elastic_load_signals,
)

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

model_filenames = create_all_model_variations(
    elasticity_model_names=["PREM"],
    long_term_anelasticity_model_names=["VM7", "Lambeck_2017", "Caron", "Lau_2016", "VM5a"],
    short_term_anelasticity_model_names=[
        "Benjamin_Q_Resovsky",
        "Benjamin_Q_PAR3P",
        "Benjamin_Q_PREM",
        "Benjamin_Q_QL6",
        "Benjamin_Q_QM1",
    ],
    parameters={
        ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
        ModelPart.short_term_anelasticity: {"asymptotic_mu_ratio": {"MANTLE": [[0.1], [0.2]]}},
    },
    create=False,
)

load_result_folders: list[Path] = load_signal_for_options_for_models_for_parameters_for_elastic_load_signals(
    model_filenames=model_filenames,
    load_signal_hyper_parameter_variations={
        "case": ["mean", "upper", "lower"],
        "little_isostatic_adjustment": [True, False],
        "opposite_load_on_continents": [False, True],
    },
    symlinks=True,
    options=options,
)
