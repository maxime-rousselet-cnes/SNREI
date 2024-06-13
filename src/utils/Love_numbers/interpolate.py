from typing import Callable

from numpy import array, concatenate, flip, ndarray
from scipy import interpolate

from ...functions import build_hermitian
from ..classes import BoundaryCondition, Direction, LoveNumbersHyperParameters, Result
from ..data import load_Love_number_result


def interpolate_Love_numbers(
    anelasticity_description_id: str,
    target_frequencies: ndarray[float],  # (Hz).
    target_degrees: ndarray[float],  # (Hz).
    Love_number_hyper_parameters: LoveNumbersHyperParameters,
    directions: list[Direction] = [
        Direction.radial,
        Direction.tangential,
        Direction.potential,
    ],
    boundary_conditions: list[BoundaryCondition] = [
        BoundaryCondition.load,
        BoundaryCondition.shear,
        BoundaryCondition.potential,
    ],
) -> Result:
    """
    Gets the wanted anelastic Love numbers from the wanted description, with wanted options and interpolates them to the wanted
    frequencies.
    """

    # Initializes.
    anelastic_Love_numbers: Result = load_Love_number_result(
        Love_number_hyper_parameters=Love_number_hyper_parameters,
        anelasticity_description_id=anelasticity_description_id,
    )
    source_frequencies = anelastic_Love_numbers.axes["frequencies"]
    source_degrees = anelastic_Love_numbers.axes["degrees"]

    # Interpolates Love numbers on signal frequencies as hermitian signal.
    symmetric_source_frequencies = concatenate(
        (-flip(m=source_frequencies), source_frequencies)
    )
    return Result(
        values={
            direction: {
                boundary_condition: array(
                    object=[
                        interpolate.interp2d(
                            x=source_degrees,
                            y=symmetric_source_frequencies,
                            z=array(
                                object=[
                                    build_hermitian(
                                        signal=(
                                            1.0
                                            if direction == Direction.potential
                                            else 0.0
                                        )
                                        + (
                                            anelastic_Love_numbers.values[direction][
                                                boundary_condition
                                            ][degree_index]
                                            / (
                                                1.0
                                                if direction == Direction.radial
                                                else degree
                                            )
                                        )
                                    )
                                    for degree_index, degree in enumerate(
                                        source_degrees
                                    )
                                ]
                            ),
                            kind="linear",
                        )(x=target_degrees, y=target_frequencies)
                    ]
                )
                for boundary_condition in boundary_conditions
            }
            for direction in directions
        }
    )
