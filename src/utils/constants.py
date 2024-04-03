from itertools import product

from .classes import RunHyperParameters

# Universal Gravitationnal constant (m^3.kg^-1.s^-2).
G = 6.67430e-11

# Earth mean radius (m).
EARTH_RADIUS = 6.371e6

# s.y^-1
SECONDS_PER_YEAR = 365.25 * 86400

# List of all possible boolean triplets except (False, False, False).
BOOLEANS = [True, False]
OPTIONS: list[RunHyperParameters] = [
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
    RunHyperParameters(
        use_long_term_anelasticity=False,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=False,
    ),
    RunHyperParameters(
        use_long_term_anelasticity=True,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=False,
    ),
]
