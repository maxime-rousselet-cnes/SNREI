from snrei import ModelPart, RunHyperParameters, clear_subs, compute_load_signal_trends_for_anelastic_Earth_models

if __name__ == "__main__":
    clear_subs()
    compute_load_signal_trends_for_anelastic_Earth_models(
        elasticity_model_names=["PREM"],
        long_term_anelasticity_model_names=["VM7", "VM5a"],  # ["VM7"],
        short_term_anelasticity_model_names=["Benjamin_Q_Resovsky", "Benjamin_Q_PAR3P"],  # ["Benjamin_Q_Resovsky"],
        rheological_parameters={
            ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
            ModelPart.short_term_anelasticity: {"asymptotic_mu_ratio": {"MANTLE": [[0.2, 0.1]]}},
        },
        load_signal_parameters={
            "case": ["mean", "lower", "upper"],
            "LIA": [False, True],
            "opposite_load_on_continents": [False, True],
            "load_spatial_behaviour_file": ["COST-G", "CSR"],
        },
        options=[
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
        ],
    )
