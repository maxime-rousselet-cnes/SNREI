from typing import Optional

from numpy import array

from .hyper_parameters import RunHyperParameters
from .paths import ModelPart

# Universal Gravitationnal constant (m^3.kg^-1.s^-2).
G = 6.67430e-11

# Earth mean radius (m).
EARTH_RADIUS = 6.371e6

# Ratio betwenn surface water density and mean Earth density.
DENSITY_RATIO = 997.0 / 5513.0
# s.y^-1
SECONDS_PER_YEAR = 365.25 * 86400

# For integration.
INITIAL_Y_VECTOR = array(
    object=[
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)

# List of all possible boolean triplets.
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
    RunHyperParameters(
        use_long_term_anelasticity=False,
        use_short_term_anelasticity=False,
        use_bounded_attenuation_functions=False,
    ),
]

# Elastic.
ELASTIC_RUN_HYPER_PARAMETERS = RunHyperParameters(
    use_long_term_anelasticity=False,
    use_short_term_anelasticity=False,
    use_bounded_attenuation_functions=False,
)

# Default hyper parameters.
DEFAULT_MODELS: dict[Optional[ModelPart], Optional[str]] = {
    ModelPart.elasticity: "PREM",
    ModelPart.long_term_anelasticity: "uniform",
    ModelPart.short_term_anelasticity: "uniform",
    None: None,
}
DEFAULT_SPLINE_NUMBER = 10

# Other low level parameters.
ASYMPTOTIC_MU_RATIO_DECIMALS = 5

# (cm/yr) -> (mm/yr).
GRACE_DATA_UNIT_FACTOR = 10
