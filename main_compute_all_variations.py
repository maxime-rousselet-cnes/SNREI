from snrei import ModelPart, RunHyperParameters, clear_subs, compute_load_signal_trends_for_anelastic_Earth_models

clear_subs()

compute_load_signal_trends_for_anelastic_Earth_models(
    elasticity_model_names=["PREM"],
    long_term_anelasticity_model_names=["VM7"],
    short_term_anelasticity_model_names=[
        "Benjamin_Q_Resovsky",
    ],
    rheological_parameters={
        ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
        # ModelPart.short_term_anelasticity: {"asymptotic_mu_ratio": {"MANTLE": [[0.2]]}},
    },
    load_signal_parameters={
        "case": ["mean"],
        "LIA": [False],
        "opposite_load_on_continents": [False],
        # "pixels_to_coast": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # "leakage_correction_iterations": [1, 0, 2],
        "erode_high_signal_zones": [False, True],
    },
    options=[
        RunHyperParameters(
            use_long_term_anelasticity=True,
            use_short_term_anelasticity=True,
            use_bounded_attenuation_functions=True,
        ),
    ],
)
