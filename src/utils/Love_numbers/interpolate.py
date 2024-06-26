from numpy import Inf, array, concatenate, expand_dims, flip, meshgrid, ndarray
from scipy import interpolate

from ...functions import build_hermitian
from ..classes import ELASTIC_RUN_HYPER_PARAMETERS, BoundaryCondition, Direction, LoveNumbersHyperParameters, Result
from ..data import load_Love_numbers_result


def interpolate_elastic_Love_numbers(
    anelasticity_description_id: str,
    target_degrees: ndarray[float],
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
    Gets the wanted elastic Love numbers from the wanted description, with wanted options and interpolates them to the wanted
    degrees.
    """

    # Initializes.
    Love_numbers_hyper_parameters.run_hyper_parameters = ELASTIC_RUN_HYPER_PARAMETERS
    Love_numbers: Result = load_Love_numbers_result(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        anelasticity_description_id=anelasticity_description_id,
    )
    source_degrees = Love_numbers.axes["degrees"].real

    return Result(
        values={
            direction: {
                boundary_condition: expand_dims(
                    a=interpolate.interp1d(
                        x=source_degrees,
                        y=(1.0 if direction == Direction.potential else 0.0)
                        + (
                            Love_numbers.values[direction][boundary_condition].real.T[0]
                            / (1.0 if direction == Direction.radial else source_degrees)
                        ),
                        kind="linear",
                    )(x=target_degrees),
                    axis=1,
                )  # Shape (n_degrees, 1).
                for boundary_condition in boundary_conditions
            }
            for direction in directions
        },
        axes={"degrees": target_degrees, "frequencies": array(object=[Inf])},
    )


def interpolate_anelastic_Love_numbers(
    anelasticity_description_id: str,
    target_frequencies: ndarray[float],  # (Hz).
    target_degrees: ndarray[float],
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
    degrees and frequencies.
    """

    # Initializes.
    Love_numbers: Result = load_Love_numbers_result(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        anelasticity_description_id=anelasticity_description_id,
    )
    source_frequencies = Love_numbers.axes["frequencies"].real
    source_degrees = Love_numbers.axes["degrees"].real

    # Interpolates Love numbers on signal frequencies as hermitian signal.
    symmetric_source_frequencies = concatenate((-flip(m=source_frequencies), source_frequencies))

    hermitian_Love_numbers: dict[Direction, dict[BoundaryCondition, ndarray]] = {
        direction: {
            boundary_condition: array(
                object=[
                    build_hermitian(
                        signal=(1.0 if direction == Direction.potential else 0.0)
                        + (
                            Love_numbers.values[direction][boundary_condition][degree_index]
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

    target_grid_degrees, target_grid_frequencies = meshgrid(target_degrees, target_frequencies)

    print(
        interpolate.RectBivariateSpline(
            x=source_degrees.real,
            y=symmetric_source_frequencies.real,
            z=hermitian_Love_numbers[Direction.potential][BoundaryCondition.load].real.T,
        )
        .ev(xi=target_grid_degrees, yi=target_grid_frequencies)
        .T.shape
    )
    return Result(
        values={
            direction: {
                boundary_condition: interpolate.RectBivariateSpline(
                    x=source_degrees.real,
                    y=symmetric_source_frequencies.real,
                    z=hermitian_Love_numbers[direction][boundary_condition].real.T,
                )
                .ev(xi=target_grid_degrees, yi=target_grid_frequencies)
                .T
                + 1.0j
                * interpolate.RectBivariateSpline(
                    x=source_degrees.real,
                    y=symmetric_source_frequencies.real,
                    z=hermitian_Love_numbers[direction][boundary_condition].imag.T,
                )
                .ev(xi=target_grid_degrees, yi=target_grid_frequencies)
                .T
                for boundary_condition in boundary_conditions
            }
            for direction in directions
        },
        axes={"degrees": target_degrees, "frequencies": target_frequencies},
    )
