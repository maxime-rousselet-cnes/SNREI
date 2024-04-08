from ...utils import (
    BoundaryCondition,
    Direction,
    Result,
    RunHyperParameters,
    frequencies_to_periods,
    interpolate_Love_numbers,
)


def get_single_float_Love_number(
    real_desciption_id: str,
    option: RunHyperParameters = RunHyperParameters(
        use_long_term_anelasticity=False, use_short_term_anelasticity=True, use_bounded_attenuation_functions=False
    ),
    degree: int = 2,
    direction: Direction = Direction.potential,
    boundary_condition: BoundaryCondition = BoundaryCondition.potential,
    period: float = 18.6,  # (y).
) -> float:
    """
    gets the wanted Love number and interpolates it at the wanted period.
    """
    result: Result = interpolate_Love_numbers(
        real_desciption_id=real_desciption_id,
        target_frequencies=frequencies_to_periods(frequencies=[period]),  # (y) -> (Hz).
        option=option,
        degrees=[degree],
        directions=[direction],
        boundary_conditions=[boundary_condition],
    )[0]
    return result.values[direction][boundary_condition]
