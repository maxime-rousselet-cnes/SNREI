from snrei import ModelPart, RunHyperParameters, clear_subs, compute_load_signal_trends_for_anelastic_Earth_models

if __name__ == "__main__":
    clear_subs()
    compute_load_signal_trends_for_anelastic_Earth_models(
        elasticity_model_names=["PREM"],
        long_term_anelasticity_model_names=["VM7", "VM5a"],
        short_term_anelasticity_model_names=["Benjamin_Q_Resovsky", "Benjamin_Q_PAR3P"],
        rheological_parameters={
            ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
            ModelPart.short_term_anelasticity: {"asymptotic_mu_ratio": {"MANTLE": [[0.1], [0.2]]}},
        },
        load_signal_parameters={
            "case": ["mean", "lower", "upper"],
            "LIA": [True, False],
            "opposite_load_on_continents": [True, False],
            "load_spatial_behaviour_file": ["TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv", "CSR"],
        },
        options=[
            RunHyperParameters(
                use_long_term_anelasticity=True,
                use_short_term_anelasticity=True,
                use_bounded_attenuation_functions=True,
            ),
            RunHyperParameters(
                use_long_term_anelasticity=True,
                use_short_term_anelasticity=False,
                use_bounded_attenuation_functions=False,
            ),
            RunHyperParameters(
                use_long_term_anelasticity=False,
                use_short_term_anelasticity=True,
                use_bounded_attenuation_functions=True,
            ),
        ],
    )
