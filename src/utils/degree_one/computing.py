from numpy import array, concatenate, multiply, ndarray, ones, sqrt, zeros
from pyshtools.expand import MakeGridDH
from scipy.linalg import lstsq

from ...functions import surface_ponderation
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
                (  # Adds a line of one values for degree zero.
                    ones(shape=(1, n_frequencies)),
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
    anelastic_hermitian_Love_numbers: Result,
    ocean_mask: ndarray[float],
) -> tuple[ndarray[complex], ndarray[complex], ndarray[complex], ndarray[complex]]:
    """
    Re-estimates degree 1 coefficients by inversion.
    Input is 4-D array: C/S, n, m, frequency (mm).
    Returns:
        - Degree one coefficients as frequencial signals (2, 2, n_frequencies).
        - anelastic frequencial scale factor (D) (n_frequencies).
        - anelastic frequential-harmonic geoid height (G) (2, n_max, n_max, n_frequencies).
        - anelastic frequential-harmonic radial discplacement (R) (2, n_max, n_max, n_frequencies).

    """

    # Initializes.
    n_frequencies = anelastic_frequencial_harmonic_load_signal.shape[-1]
    degree_one = zeros(
        shape=(2, 2, n_frequencies),
        dtype=complex,
    )
    scale_factor = zeros(shape=(n_frequencies), dtype=complex)
    ocean_mask_indices = ocean_mask.flatten().astype(dtype=bool)
    least_square_weights = sqrt(surface_ponderation(mask=ocean_mask).flatten()[ocean_mask_indices])

    # Computes components for degrees >= 2.
    frequencial_harmonic_geoid = frequencial_harmonic_component(
        anelastic_frequencial_harmonic_load_signal=anelastic_frequencial_harmonic_load_signal,
        Love_numbers=anelastic_hermitian_Love_numbers,
        direction=Direction.potential,
        boundary_condition=BoundaryCondition.load,
    )
    frequencial_harmonic_radial_displacement = frequencial_harmonic_component(
        anelastic_frequencial_harmonic_load_signal=anelastic_frequencial_harmonic_load_signal,
        Love_numbers=anelastic_hermitian_Love_numbers,
        direction=Direction.radial,
        boundary_condition=BoundaryCondition.load,
    )
    # Right-Hand Side terms that includes 1 + k'_n - h'_n.
    right_hand_side_terms = (
        frequencial_harmonic_geoid
        - frequencial_harmonic_radial_displacement
        - anelastic_frequencial_harmonic_load_signal
    )

    # Builds the matrix rows using pyshtools to ensure polynomial normalization is coherent with RHS.
    P_1_0: ndarray = MakeGridDH(
        array(object=[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        sampling=2,
        lmax=anelastic_frequencial_harmonic_load_signal.shape[1] - 1,
    )
    P_1_1_C: ndarray = MakeGridDH(
        array(object=[[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        sampling=2,
        lmax=anelastic_frequencial_harmonic_load_signal.shape[1] - 1,
    )
    P_1_1_S: ndarray = MakeGridDH(
        array(object=[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
        sampling=2,
        lmax=anelastic_frequencial_harmonic_load_signal.shape[1] - 1,
    )

    # Preprocesses degree one polynomials and constants. Index 0 for degree 1.
    degree_one_potential_Love_numbers = anelastic_hermitian_Love_numbers.values[Direction.potential][
        BoundaryCondition.load
    ][0]
    degree_one_radial_Love_numbers = anelastic_hermitian_Love_numbers.values[Direction.radial][BoundaryCondition.load][
        0
    ]
    degree_one_Love_numbers_term = 1 - DENSITY_RATIO * (
        degree_one_potential_Love_numbers - degree_one_radial_Love_numbers
    )
    degree_one_polynomials = array(
        object=[
            P_1_0.flatten()[ocean_mask_indices],
            P_1_1_C.flatten()[ocean_mask_indices],
            P_1_1_S.flatten()[ocean_mask_indices],
        ]
    )
    constant_column = [[1.0] * sum(ocean_mask_indices)]

    # Handles elstic case.
    if len(anelastic_hermitian_Love_numbers.axes["frequencies"]) == 1:
        degree_one_Love_numbers_term = [degree_one_Love_numbers_term] * n_frequencies
        degree_one_radial_Love_numbers = [degree_one_radial_Love_numbers] * n_frequencies
        degree_one_potential_Love_numbers = [degree_one_potential_Love_numbers] * n_frequencies

    # Solves a system per frequency.
    degree_one_Love_number_term: complex
    harmonic_right_hand_side: ndarray[complex]
    for frequencial_index, (
        degree_one_Love_number_term,
        harmonic_right_hand_side,
        degree_one_potential_Love_number,
        degree_one_radial_Love_number,
    ) in enumerate(
        zip(
            degree_one_Love_numbers_term,  # Handles elastic case.
            right_hand_side_terms.transpose((3, 0, 1, 2)),
            degree_one_potential_Love_numbers,  # Handles elastic case.
            degree_one_radial_Love_numbers,  # Handles elastic case.
        )
    ):

        # Left-Hand Side.
        left_hand_side = concatenate(
            (degree_one_Love_number_term * degree_one_polynomials, constant_column),
        ).T

        # Right-Hand Side.
        harmonic_right_hand_side[:, :2, :] = 0.0
        spatial_right_hand_side: ndarray = MakeGridDH(harmonic_right_hand_side.real, sampling=2) + 1.0j * MakeGridDH(
            harmonic_right_hand_side.imag, sampling=2
        )
        right_hand_side = spatial_right_hand_side.flatten()[ocean_mask_indices]

        # Inversion.
        solution_vector, _, _, _ = lstsq(
            a=multiply(least_square_weights, left_hand_side.T).T,
            b=least_square_weights * right_hand_side,
        )

        # Updates.
        degree_one[:, :, frequencial_index] = array(
            object=[
                [solution_vector[0], solution_vector[1]],
                [0.0, solution_vector[2]],
            ]
        )
        scale_factor[frequencial_index] = solution_vector[3]
        frequencial_harmonic_geoid[:, 1, :2, frequencial_index] = (
            DENSITY_RATIO * degree_one_potential_Love_number * degree_one[:, :, frequencial_index]
        )
        frequencial_harmonic_radial_displacement[:, 1, :2, frequencial_index] = (
            DENSITY_RATIO * degree_one_radial_Love_number * degree_one[:, :, frequencial_index]
        )

    return (
        degree_one,
        scale_factor,
        frequencial_harmonic_geoid,
        frequencial_harmonic_radial_displacement,
    )
