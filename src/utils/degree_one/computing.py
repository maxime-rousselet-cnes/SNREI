from numpy import array, concatenate, multiply, ndarray, sqrt, zeros
from pyshtools.expand import MakeGridDH
from scipy.linalg import lstsq

from ...functions import surface_ponderation
from ..classes import DENSITY_RATIO, BoundaryCondition, Direction, Result


def degree_one_inversion(
    anelastic_frequencial_harmonic_load_signal: ndarray[complex],
    anelastic_hermitian_Love_numbers: Result,
    ocean_mask: ndarray[float],
) -> ndarray[complex]:
    """
    Re-estimates degree 1 coefficients by inversion.
    Returns degree_one coefficients as frequencial signals.
    Input is 4-D array: C/S, n, m, frequency (mm).
    """

    # Initializes.
    n_frequencies = anelastic_frequencial_harmonic_load_signal.shape[-1]
    degree_one = zeros(
        shape=(2, 2, n_frequencies),
        dtype=complex,
    )
    ocean_mask_indices = ocean_mask.flatten().astype(dtype=bool)
    least_square_weights = sqrt(
        surface_ponderation(mask=ocean_mask).flatten()[ocean_mask_indices]
    )

    # Right-Hand Side terms that includes 1 + k'_n - h'_n.
    right_hand_side_terms = multiply(
        anelastic_frequencial_harmonic_load_signal.transpose((0, 2, 3, 1)),
        (
            DENSITY_RATIO
            * multiply(
                (
                    anelastic_hermitian_Love_numbers.values[Direction.potential][
                        BoundaryCondition.load
                    ]
                    - anelastic_hermitian_Love_numbers.values[Direction.radial][
                        BoundaryCondition.load
                    ]
                ).T,
                3 / (2 * anelastic_hermitian_Love_numbers.axes["degrees"] + 1),
            ).T
            - 1.0
        ).T,
    ).transpose((0, 3, 1, 2))

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

    # Preprocesses degree one polynomials and constants.
    degree_one_Love_numbers = 1 - DENSITY_RATIO * (
        anelastic_hermitian_Love_numbers.values[Direction.potential][
            BoundaryCondition.load
        ][
            0
        ]  # Index 0 for degree 1.
        - anelastic_hermitian_Love_numbers.values[Direction.radial][
            BoundaryCondition.load
        ][
            0
        ]  # Index 0 for degree 1.
    )
    degree_one_polynomials = array(
        object=[
            P_1_0.flatten()[ocean_mask_indices],
            P_1_1_C.flatten()[ocean_mask_indices],
            P_1_1_S.flatten()[ocean_mask_indices],
        ]
    )
    constant_column = [[1.0] * sum(ocean_mask_indices)]

    # Solves a system per frequency.
    degree_one_Love_number: complex
    harmonic_right_hand_side: ndarray[complex]
    for frequencial_index, (
        degree_one_Love_number,
        harmonic_right_hand_side,
    ) in enumerate(
        zip(
            (
                degree_one_Love_numbers
                if len(anelastic_hermitian_Love_numbers.axes["frequencies"]) != 1
                else [degree_one_Love_numbers] * n_frequencies
            ),  # Handles elastic case.
            right_hand_side_terms.transpose((3, 0, 1, 2)),
        )
    ):

        # Left-Hand Side.
        left_hand_side = concatenate(
            (degree_one_Love_number * degree_one_polynomials, constant_column),
        ).T

        # Right-Hand Side.
        harmonic_right_hand_side[:, :2, :] = 0.0
        spatial_right_hand_side: ndarray = MakeGridDH(
            harmonic_right_hand_side.real, sampling=2
        ) + 1.0j * MakeGridDH(harmonic_right_hand_side.imag, sampling=2)
        right_hand_side = spatial_right_hand_side.flatten()[ocean_mask_indices]

        # Inversion.
        solution_vector, _, _, _ = lstsq(
            a=multiply(least_square_weights, left_hand_side.T).T,
            b=least_square_weights * right_hand_side,
        )
        degree_one[:, :, frequencial_index] = array(
            object=[
                [solution_vector[0], solution_vector[1]],
                [0.0, solution_vector[2]],
            ]
        )

    return degree_one
