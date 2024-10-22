from numpy import arange

from snrei import ModelPart, RunHyperParameters, clear_subs, compute_load_signal_trends_for_anelastic_Earth_models

if __name__ == "__main__":
    # clear_subs()
    compute_load_signal_trends_for_anelastic_Earth_models(
        elasticity_model_names=["PREM"],
        long_term_anelasticity_model_names=["VM7"],  # , "VM5a", "Mao_Zhong", "Lau_2016", "Lambeck_2017", "Caron"],
        short_term_anelasticity_model_names=["Benjamin_Q_Resovsky"],  # , "Benjamin_Q_PAR3P", "Benjamin_Q_PREM", "Benjamin_Q_QL6", "Benjamin_Q_QM1"],
        rheological_parameters={
            # ModelPart.long_term_anelasticity: {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
            # ModelPart.short_term_anelasticity: {"asymptotic_mu_ratio": {"MANTLE": [[0.1], [0.2]]}},
        },
        load_signal_parameters={
            "pole_secular_term_trend_start_date": [1900, 1977],
            "pole_secular_term_trend_end_date": [1978, 2000, 2010],
            # "load_history_case": ["mean", "lower", "upper"],
            # "pole_case": ["mean", "lower", "upper"],
            # "LIA": [False, True],
            # "opposite_load_on_continents": [False, True],
            # "load_spatial_behaviour_file": ["TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"],
            # "buffer_distance": arange(0, 1050, 50),
        },
        options=[
            RunHyperParameters(
                use_long_term_anelasticity=True,
                use_short_term_anelasticity=True,
                use_bounded_attenuation_functions=True,
            ),
        ],
    )
