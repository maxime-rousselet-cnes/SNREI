from numpy import array, concatenate, flip, ndarray
from scipy import interpolate

from ...functions import build_hermitian, interpolate_array
from ..classes import BoundaryCondition, Direction, LoveNumbersHyperParameters, Result
from ..data import load_Love_numbers_result


def interpolate_Love_numbers(
    anelasticity_description_id: str,
    target_frequencies: ndarray[float],  # (Hz).
    target_degrees: ndarray[float],  # (Hz).
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
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
    Love_numbers: Result = load_Love_numbers_result(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        anelasticity_description_id=anelasticity_description_id,
    )
    source_frequencies = Love_numbers.axes["frequencies"].real
    source_degrees = Love_numbers.axes["degrees"].real

    # Interpolates Love numbers on signal frequencies as hermitian signal.
    symmetric_source_frequencies = concatenate(
        (-flip(m=source_frequencies), source_frequencies)
    )

    if len(source_frequencies) == 1:
        return Result(
            values={
                direction: {
                    boundary_condition: interpolate.interp1d(
                        x=source_degrees,
                        y=(1.0 if direction == Direction.potential else 0.0)
                        + (
                            Love_numbers.values[direction][boundary_condition].real.T[0]
                            / (1.0 if direction == Direction.radial else source_degrees)
                        ),
                        kind="linear",
                    )(x=target_degrees)
                    for boundary_condition in boundary_conditions
                }
                for direction in directions
            },
            axes={"degrees": target_degrees, "frequencies": source_frequencies},
        )
    else:
        hermitian_Love_numbers: dict[Direction, dict[BoundaryCondition, ndarray]] = {
            direction: {
                boundary_condition: array(
                    object=[
                        build_hermitian(
                            signal=(1.0 if direction == Direction.potential else 0.0)
                            + (
                                Love_numbers.values[direction][boundary_condition][
                                    degree_index
                                ]
                                / (1.0 if direction == Direction.radial else degree)
                            )
                        )
                        for degree_index, degree in enumerate(source_degrees)
                    ]
                )
                for boundary_condition in boundary_conditions
            }
            for direction in directions
        }
        return Result(
            values={
                direction: {
                    boundary_condition: interpolate.interp2d(
                        x=symmetric_source_frequencies,
                        y=source_degrees,
                        z=hermitian_Love_numbers[direction][boundary_condition].real,
                        kind="linear",
                    )(x=target_degrees, y=target_frequencies).T
                    + 1.0j
                    * interpolate.interp2d(
                        x=symmetric_source_frequencies,
                        y=source_degrees,
                        z=hermitian_Love_numbers[direction][boundary_condition].imag,
                        kind="linear",
                    )(x=target_degrees, y=target_frequencies).T
                    for boundary_condition in boundary_conditions
                }
                for direction in directions
            },
            axes={"degrees": target_degrees, "frequencies": target_frequencies},
        )
