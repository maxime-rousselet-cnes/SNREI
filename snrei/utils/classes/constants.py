from typing import Optional

from numpy import array, pi
from pydantic import BaseModel

from .hyper_parameters import RunHyperParameters
from .paths import ModelPart

ARC_SECOND_TO_RADIANS = 2 * pi / (60 * 60 * 360)
MILLI_ARC_SECOND_TO_RADIANS = ARC_SECOND_TO_RADIANS / 1000

# Earth mean radius (m).
EARTH_RADIUS = 6.371e6

# Ratio betwenn surface water density and mean Earth density.
DENSITY_RATIO = 997.0 / 5513.0

OMEGA = 2.0 * pi / 86164.0  # (Rad.s^-1)
MEAN_G = 9.8  # (m.s^-2).
PHI_CONSTANT = OMEGA**2 * EARTH_RADIUS / MEAN_G / 15**0.5  # (Unitless).
MEAN_POLE_COEFFICIENTS = {
    "IERS_2018_update": {"m_1": [55.0, 1.677], "m_2": [320.5, 3.460]},  # (mas/yr^index).
}
MEAN_POLE_T_0 = {
    "IERS_2018_update": 2000.0,  # (yr).
}
METERS_TO_MILLIMETERS = 1000
STOKES_TO_EWH_CONSTANT = 5.0 / 3.0 / DENSITY_RATIO * EARTH_RADIUS * METERS_TO_MILLIMETERS

M_1_GIA_TREND = 0.62  # (mas/yr).
M_2_GIA_TREND = -3.48  # (mas/yr).
K_2_BASE = 0.3077 + 0.0036j
K_2_PRIME_BASE = 0.30523

M_1_TREND = MEAN_POLE_COEFFICIENTS["IERS_2018_update"]["m_1"][1]
M_2_TREND = MEAN_POLE_COEFFICIENTS["IERS_2018_update"]["m_2"][1]

C_2_1_PT_SE_ELASTIC_CORRECTION = (
    -STOKES_TO_EWH_CONSTANT
    / (K_2_PRIME_BASE + 1.0)
    * PHI_CONSTANT
    * MILLI_ARC_SECOND_TO_RADIANS
    * (K_2_BASE.real * M_1_TREND + K_2_BASE.imag * M_2_TREND)
)
S_2_1_PT_SE_ELASTIC_CORRECTION = (
    STOKES_TO_EWH_CONSTANT
    / (K_2_PRIME_BASE + 1.0)
    * PHI_CONSTANT
    * MILLI_ARC_SECOND_TO_RADIANS
    * (K_2_BASE.real * M_2_TREND - K_2_BASE.imag * M_1_TREND)
)

# Universal Gravitationnal constant (m^3.kg^-1.s^-2).
G = 6.67430e-11

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
LAYER_DECIMALS = 5

# (cm/yr) -> (mm/yr).
GRACE_DATA_UNIT_FACTOR = 10


# Lakes and islands to remove from ocean label.
class Rectangle(BaseModel):
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


RECTANGLES: dict[str, Rectangle] = {
    "Lake Superior": Rectangle(
        min_latitude=41.375963,
        max_latitude=50.582521,
        min_longitude=-93.748270,
        max_longitude=-75.225322,
    ),
    "Lake Victoria": Rectangle(
        min_latitude=-2.809322,
        max_latitude=0.836983,
        min_longitude=31.207700,
        max_longitude=34.530942,
    ),
    "Caspian Sea": Rectangle(
        min_latitude=35.569650,
        max_latitude=47.844035,
        min_longitude=44.303403,
        max_longitude=60.937192,
    ),
    "Svalbard": Rectangle(
        min_latitude=76.515316,
        max_latitude=80.801823,
        min_longitude=5.435317,
        max_longitude=35.933364,
    ),
    "Southern Georgia": Rectangle(
        min_latitude=-57.496798,
        max_latitude=-52.844035,
        min_longitude=-40.681460,
        max_longitude=-32.937192,
    ),
    "Kerguelen": Rectangle(
        min_latitude=-53.262545,
        max_latitude=-47.572174,
        min_longitude=64.599279,
        max_longitude=72.572817,
    ),
    "Arkhanglesk": Rectangle(
        min_latitude=79.447297,
        max_latitude=82.165567,
        min_longitude=43.404067,
        max_longitude=70.925551,
    ),
    "Krasnoiarsk": Rectangle(
        min_latitude=77.162848,
        max_latitude=81.586588,
        min_longitude=85.326874,
        max_longitude=112.397186,
    ),
}
