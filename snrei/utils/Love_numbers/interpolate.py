from numpy import inf, arange, array, errstate, expand_dims, log, meshgrid, nan_to_num, ndarray, sign
from scipy import interpolate

from ..classes import ELASTIC_RUN_HYPER_PARAMETERS, BoundaryCondition, Direction, LoveNumbersHyperParameters, Result
from ..data import load_Love_numbers_result


def interpolate_elastic_Love_numbers(
    anelasticity_description_id: str,
    n_max: int,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
    directions: list[Direction] = [
        Direction.radial,
        Direction.potential,
    ],
    boundary_conditions: list[BoundaryCondition] = [
        BoundaryCondition.load,
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
    target_degrees = arange(n_max) + 1  # Does not include n = 0.
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
        axes={"degrees": target_degrees, "frequencies": array(object=[inf])},
    )


def interpolate_anelastic_Love_numbers(
    anelasticity_description_id: str,
    n_max: int,
    target_frequencies: ndarray[float],  # (yr^-1).
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
    directions: list[Direction] = [
        Direction.radial,
        Direction.potential,
    ],
    boundary_conditions: list[BoundaryCondition] = [
        BoundaryCondition.load,
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
    target_degrees = arange(n_max) + 1  # Does not include n = 0.

    # Interpolates Love numbers on signal positive frequencies.
    abs_target_frequencies = abs(target_frequencies)
    min_positive_frequency = min(source_frequencies)
    with errstate(invalid="ignore", divide="ignore"):
        target_absolute_frequencies = nan_to_num(x=log(abs_target_frequencies / min_positive_frequency), nan=0.0)

    target_grid_degrees, target_grid_absolute_frequencies = meshgrid(
        target_degrees, target_absolute_frequencies, indexing="ij"
    )

    result = Result(
        values={
            direction: {
                boundary_condition: (
                    interpolate.RectBivariateSpline(
                        x=source_degrees,
                        y=log(source_frequencies / min_positive_frequency),
                        z=Love_numbers.values[direction][boundary_condition].real,
                    ).ev(xi=target_grid_degrees, yi=target_grid_absolute_frequencies)
                    + 1.0j
                    # To get hermitian signal.
                    * expand_dims(a=sign(target_frequencies), axis=0)
                    * interpolate.RectBivariateSpline(
                        x=source_degrees,
                        y=log(source_frequencies / min_positive_frequency),
                        z=Love_numbers.values[direction][boundary_condition].imag,
                    ).ev(xi=target_grid_degrees, yi=target_grid_absolute_frequencies)
                )
                # Because (h_n, n * l_n and n * k_n are stored.)
                / (1.0 if direction == Direction.radial else expand_dims(a=target_degrees, axis=1))
                # Zero value for frequency == 0 yr^-1 => zero mean value.
                * expand_dims(a=abs_target_frequencies >= min_positive_frequency, axis=0)
                # 1 + k for potential.
                + (1.0 if direction == Direction.potential else 0.0)
                for boundary_condition in boundary_conditions
            }
            for direction in directions
        },
        axes={"degrees": target_degrees, "frequencies": target_frequencies},
    )

    return result
