from snrei import ModelPart, RunHyperParameters, clear_subs, compute_load_signal_trends_for_anelastic_Earth_models

if __name__ == "__main__":
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
            "leakage_correction_iterations": [0, 1, 2],
            "buffer_distance": [50, 100, 200, 300, 500, 700, 1000],
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
