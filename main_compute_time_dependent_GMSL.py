from pathlib import Path

from snrei import (
    RunHyperParameters,
    compute_anelastic_harmonic_load_per_description_per_options,
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

load_result_folder = compute_anelastic_harmonic_load_per_description_per_options(
    anelasticity_description_ids=[
        "PREM_____VM7____eta_m__ASTHENOSPHERE__3e+19_____Benjamin_Q_Resovsky____asymptotic_mu_ratio__MANTLE__0.15"
    ],
    options=options,
    do_elastic=True,
    src_directory=None,
)
