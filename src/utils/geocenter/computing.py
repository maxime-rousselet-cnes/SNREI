from numpy import array, multiply, ndarray, tensordot, zeros
from pyshtools.expand import MakeGridDH
from scipy.linalg import solve

from ..classes import DENSITY_RATIO, BoundaryCondition, Direction, Result


def geocenter_inversion(
    anelastic_frequencial_harmonic_load_signal: ndarray[complex],
    anelastic_hermitian_Love_numbers: Result,
    ocean_mask: ndarray[float],
) -> ndarray[complex]:
    """
    Re-estimates degree 1 coefficients by inversion.
    Returns geocenter coefficients as frequencial signals.
    Input is 4-D array: C/S, n, m, frequency (mm).
    """

    # Initializes.
    degree_one = zeros(
        shape=[2, 1] + anelastic_frequencial_harmonic_load_signal.shape[2:]
    )
    ocean_mask_indices = ocean_mask.flatten().astype(dtype=bool)

    # RHS includes 1 + k'_n - h'_n.
    right_hand_side_terms = multiply(
        anelastic_frequencial_harmonic_load_signal.transpose((0, 2, 3, 1)),
        (
            3
            * DENSITY_RATIO
            * multiply(
                (
                    anelastic_hermitian_Love_numbers.values[Direction.potential][
                        BoundaryCondition.load
                    ]
                    - anelastic_hermitian_Love_numbers.values[Direction.radial][
                        BoundaryCondition.load
                    ]
                ).T,
                1 / (2 * anelastic_hermitian_Love_numbers.axes["degrees"] + 1),
            ).T
            - 1.0
        ).T,
    ).transpose((0, 3, 1, 2))

    # Builds the matrix rows using pyshtools to ensure polynomial normalization is coherent with RHS.
    P_1_0: ndarray = MakeGridDH(
        array(object=[[[0.0, 0.0], [1.0, 0.0]], [0.0, 0.0], [0.0, 0.0]]),
        sampling=2,
        lmax=anelastic_frequencial_harmonic_load_signal.shape[1] - 1,
    )
    P_1_1_C: ndarray = MakeGridDH(
        array(object=[[[0.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [0.0, 0.0]]),
        sampling=2,
        lmax=anelastic_frequencial_harmonic_load_signal.shape[1] - 1,
    )
    P_1_1_S: ndarray = MakeGridDH(
        array(object=[[[0.0, 0.0], [0.0, 0.0]], [0.0, 1.0], [0.0, 0.0]]),
        sampling=2,
        lmax=anelastic_frequencial_harmonic_load_signal.shape[1] - 1,
    )
    left_hand_sides = tensordot(
        a=1
        - DENSITY_RATIO
        * (
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
        ),
        b=array(
            object=[
                P_1_0.flatten()[ocean_mask_indices],
                P_1_1_C.flatten()[ocean_mask_indices],
                P_1_1_S.flatten()[ocean_mask_indices],
                [1.0] * ocean_mask_indices,
            ]
        ).T,
        axes=0,
    )

    # Solves a system per frequency.
    for frequencial_index, (left_hand_side, harmonic_right_hand_side) in enumerate(
        zip(left_hand_sides, right_hand_side_terms.transpose((3, 0, 1, 2)))
    ):
        harmonic_right_hand_side[:, :2, :] = 0.0
        spatial_right_hand_side: ndarray = MakeGridDH(
            harmonic_right_hand_side, sampling=2
        )
        right_hand_side = spatial_right_hand_side.flatten()[ocean_mask_indices]
        solution_vector = solve(a=left_hand_side, b=right_hand_side)
        degree_one[:, :, :2, frequencial_index] = array(
            object=[
                [[0.0, 0.0], [solution_vector[0], solution_vector[1]]],
                [[0.0, 0.0], [0.0, solution_vector[2]]],
            ]
        )

    return degree_one
