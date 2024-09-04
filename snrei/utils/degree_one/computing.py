from multiprocessing import Pool

from numpy import array, concatenate, multiply, ndarray, ones, sqrt, zeros
from scipy.linalg import lstsq

from ...functions import make_grid, surface_ponderation
from ..classes import DENSITY_RATIO, BoundaryCondition, Direction, Result


def frequencial_harmonic_component(
    anelastic_frequencial_harmonic_load_signal: ndarray[complex],
    Love_numbers: Result,
    direction: Direction,
    boundary_condition: BoundaryCondition,
):
    """
    Optimally performs the multiplication Love_number_term(n, omega) * frequencial_harmonic_load(C/S, n, m, omega).
    """

    n_frequencies = len(Love_numbers.axes["frequencies"])
    return multiply(
        anelastic_frequencial_harmonic_load_signal.transpose((0, 2, 3, 1)),
        (
            DENSITY_RATIO
            * concatenate(
                (  # Adds a line of zero values for degree zero.
                    (ones(shape=(1, n_frequencies)) if direction == Direction.potential else zeros(shape=(1, n_frequencies))),
                    multiply(
                        Love_numbers.values[direction][boundary_condition].T,
                        3 / (2 * Love_numbers.axes["degrees"] + 1),
                    ).T,
                )
            )
        ).T,
    ).transpose((0, 3, 1, 2))


def degree_one_inversion(
    anelastic_frequencial_harmonic_load_signal: ndarray[complex],
    Love_numbers: Result,
    ocean_land_buffered_mask: ndarray[float],
    latitudes: ndarray[float],
    longitudes: ndarray[float],
) -> tuple[ndarray[complex], ndarray[complex], ndarray[complex], ndarray[complex]]:

    n_frequencies = anelastic_frequencial_harmonic_load_signal.shape[-1]
    degree_one = zeros(
        shape=(2, 2, n_frequencies),
        dtype=complex,
    )
    scale_factor = zeros(shape=(n_frequencies), dtype=complex)
    ocean_mask_indices = ocean_mask.flatten().astype(dtype=bool)
    least_square_weights = sqrt(surface_ponderation(mask=ocean_mask, latitudes=latitudes).flatten()[ocean_mask_indices])

    frequencial_harmonic_geoid = frequencial_harmonic_component(
        anelastic_frequencial_harmonic_load_signal=anelastic_frequencial_harmonic_load_signal,
        Love_numbers=Love_numbers,
        direction=Direction.potential,
        boundary_condition=BoundaryCondition.load,
    )
    frequencial_harmonic_radial_displacement = frequencial_harmonic_component(
        anelastic_frequencial_harmonic_load_signal=anelastic_frequencial_harmonic_load_signal,
        Love_numbers=Love_numbers,
        direction=Direction.radial,
        boundary_condition=BoundaryCondition.load,
    )
    right_hand_side_terms = frequencial_harmonic_geoid - frequencial_harmonic_radial_displacement - anelastic_frequencial_harmonic_load_signal

    P_1_0: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]), latitudes=latitudes, longitudes=longitudes
    )
    P_1_1_C: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]), latitudes=latitudes, longitudes=longitudes
    )
    P_1_1_S: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]), latitudes=latitudes, longitudes=longitudes
    )

    degree_one_potential_Love_numbers = Love_numbers.values[Direction.potential][BoundaryCondition.load][0]
    degree_one_radial_Love_numbers = Love_numbers.values[Direction.radial][BoundaryCondition.load][0]
    degree_one_Love_numbers_term = 1 - DENSITY_RATIO * (degree_one_potential_Love_numbers - degree_one_radial_Love_numbers)
    degree_one_polynomials = array(
        object=[
            P_1_0.flatten()[ocean_mask_indices],
            P_1_1_C.flatten()[ocean_mask_indices],
            P_1_1_S.flatten()[ocean_mask_indices],
        ]
    )
    constant_column = [[1.0] * sum(ocean_mask_indices)]

    if len(Love_numbers.axes["frequencies"]) == 1:
        degree_one_Love_numbers_term = [degree_one_Love_numbers_term] * n_frequencies
        degree_one_radial_Love_numbers = [degree_one_radial_Love_numbers] * n_frequencies
        degree_one_potential_Love_numbers = [degree_one_potential_Love_numbers] * n_frequencies

    # Prepare arguments for multiprocessing
    pool_args = [
        (
            frequencial_index,
            degree_one_Love_number_term,
            harmonic_right_hand_side,
            degree_one_potential_Love_number,
            degree_one_radial_Love_number,
            degree_one_polynomials,
            constant_column,
            least_square_weights,
            ocean_mask_indices,
            latitudes,
            longitudes,
        )
        for frequencial_index, (
            degree_one_Love_number_term,
            harmonic_right_hand_side,
            degree_one_potential_Love_number,
            degree_one_radial_Love_number,
        ) in enumerate(
            zip(
                degree_one_Love_numbers_term,
                right_hand_side_terms.transpose((3, 0, 1, 2)),
                degree_one_potential_Love_numbers,
                degree_one_radial_Love_numbers,
            )
        )
    ]

    # Use multiprocessing to execute solve_degree_one_inversion in parallel
    with Pool() as p:  # Adjust the number of processes as needed
        results = p.starmap(solve_degree_one_inversion, pool_args)

    # Unpack the results
    for result in results:
        (
            frequencial_index,
            degree_one_coefficients,
            scale_factor_value,
            frequencial_harmonic_geoid_value,
            frequencial_harmonic_radial_displacement_value,
        ) = result

        degree_one[:, :, frequencial_index] = degree_one_coefficients
        scale_factor[frequencial_index] = scale_factor_value
        frequencial_harmonic_geoid[:, 1, :2, frequencial_index] = frequencial_harmonic_geoid_value
        frequencial_harmonic_radial_displacement[:, 1, :2, frequencial_index] = frequencial_harmonic_radial_displacement_value

    return (
        degree_one,
        scale_factor,
        frequencial_harmonic_geoid,
        frequencial_harmonic_radial_displacement,
    )


def solve_degree_one_inversion(
    frequencial_index: int,
    degree_one_Love_number_term: ndarray[complex],
    harmonic_right_hand_side: ndarray[complex],
    degree_one_potential_Love_number: ndarray[complex],
    degree_one_radial_Love_number: ndarray[complex],
    degree_one_polynomials: ndarray[complex],
    constant_column: ndarray[complex],
    least_square_weights: ndarray[float],
    ocean_mask_indices: ndarray[float],
    latitudes: ndarray[float],
    longitudes: ndarray[float],
):
    # Left-Hand Side.
    left_hand_side = concatenate(
        (degree_one_Love_number_term * degree_one_polynomials, constant_column),
    ).T

    # Right-Hand Side.
    harmonic_right_hand_side[:, :2, :] = 0.0
    spatial_right_hand_side: ndarray = make_grid(
        harmonics=harmonic_right_hand_side.real, latitudes=latitudes, longitudes=longitudes
    ) + 1.0j * make_grid(harmonics=harmonic_right_hand_side.imag, latitudes=latitudes, longitudes=longitudes)
    right_hand_side = spatial_right_hand_side.flatten()[ocean_mask_indices]

    # Inversion.
    solution_vector, _, _, _ = lstsq(
        a=multiply(least_square_weights, left_hand_side.T).T,
        b=least_square_weights * right_hand_side,
    )

    # Updates.
    degree_one_coefficients = array(
        object=[
            [solution_vector[0], solution_vector[1]],
            [0.0, solution_vector[2]],
        ]
    )
    scale_factor_value = solution_vector[3]
    frequencial_harmonic_geoid_value = DENSITY_RATIO * degree_one_potential_Love_number * degree_one_coefficients
    frequencial_harmonic_radial_displacement_value = DENSITY_RATIO * degree_one_radial_Love_number * degree_one_coefficients

    return (
        frequencial_index,
        degree_one_coefficients,
        scale_factor_value,
        frequencial_harmonic_geoid_value,
        frequencial_harmonic_radial_displacement_value,
    )
