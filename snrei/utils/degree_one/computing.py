from multiprocessing import Pool
from typing import Optional

from numpy import array, concatenate, expand_dims, matmul, meshgrid, multiply, ndarray, ones, zeros
from scipy.linalg import lstsq

from ...functions import make_grid, surface_ponderation
from ..classes import DENSITY_RATIO, BoundaryCondition, Direction, Result
from ..data import map_sampling


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
                (  # Adds a line of zero values for degree zero. TODO?
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
    anelastic_harmonic_load_signal_trends: ndarray[float],
    Love_numbers: Result,
    ocean_land_mask: ndarray[float],
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    n_max: int,
    signal_threshold: float,
    compute_residuals: bool,
    invert_for_J2: bool,
):
    """"""

    # Initializes output shapes.
    n_frequencies = anelastic_frequencial_harmonic_load_signal.shape[-1]
    scale_factor = zeros(shape=(n_frequencies), dtype=complex)
    frequencial_harmonic_residuals = zeros(shape=anelastic_frequencial_harmonic_load_signal.shape)

    # Sets a unitless one value on variables to invert.
    anelastic_frequencial_harmonic_load_signal[0, 1, 0, :] = ones(shape=(n_frequencies), dtype=complex)
    anelastic_frequencial_harmonic_load_signal[0, 1, 1, :] = ones(shape=(n_frequencies), dtype=complex)
    anelastic_frequencial_harmonic_load_signal[1, 1, 1, :] = ones(shape=(n_frequencies), dtype=complex)
    if invert_for_J2:
        anelastic_frequencial_harmonic_load_signal[0, 2, 0, :] = ones(shape=(n_frequencies), dtype=complex)

    # Gets relevant signal pixel indices.
    mask = ocean_land_mask * (abs(make_grid(harmonics=anelastic_harmonic_load_signal_trends, n_max=n_max)) < signal_threshold)
    ocean_mask_indices = mask.flatten().astype(dtype=bool)

    # Ponderates by latitude.
    least_square_weights = surface_ponderation(mask=mask, latitudes=latitudes).flatten()[ocean_mask_indices] ** 0.5

    # Builds a coordinates mesh for residuals
    if compute_residuals:
        latitude_indices = range(len(latitudes))
        longitude_indices = range(len(longitudes))
        latitude_indices_mesh, longitude_indices_mesh = meshgrid(latitude_indices, longitude_indices, indexing="ij")
        latitude_indices_mesh_ocean = latitude_indices_mesh.flatten()[ocean_mask_indices]
        longitude_indices_mesh_ocean = longitude_indices_mesh.flatten()[ocean_mask_indices]

    # Creates all equation terms.
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

    # Creates separated spherical harmonic terms for low degrees in spatial domain.
    P_1_0: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        n_max=n_max,
    )
    P_1_1_C: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        n_max=n_max,
    )
    P_1_1_S: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
        n_max=n_max,
    )
    P_0 = make_grid(
        harmonics=array(object=[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        n_max=n_max,
    )
    P_2_0: ndarray = make_grid(
        harmonics=array(object=[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        n_max=n_max,
    )
    low_degrees_polynomials = array(
        object=[
            P_1_0.flatten()[ocean_mask_indices],
            P_1_1_C.flatten()[ocean_mask_indices],
            P_1_1_S.flatten()[ocean_mask_indices],
            P_0.flatten()[ocean_mask_indices],
            P_2_0.flatten()[ocean_mask_indices],
        ]
    )

    # Prepares the arguments for multiprocessing.
    pool_args = [
        (
            frequencial_index,
            harmonic_right_hand_side,
            low_degrees_polynomials,
            least_square_weights,
            ocean_mask_indices,
            n_max,
            latitude_indices_mesh_ocean if compute_residuals else None,
            longitude_indices_mesh_ocean if compute_residuals else None,
            compute_residuals,
            invert_for_J2,
        )
        for frequencial_index, harmonic_right_hand_side in enumerate(right_hand_side_terms.transpose((3, 0, 1, 2)))
    ]

    # Uses multiprocessing to execute solve_degree_one_inversion in parallel.
    with Pool() as p:
        results = p.starmap(solve_degree_one_inversion, pool_args)

    # Unpacks the results.
    for result in results:
        (
            frequencial_index,
            C_1_0,
            C_1_1,
            S_1_1,
            D,
            J2,
            frequencial_harmonic_residual_value,
        ) = result

        # Uopdates degree one and eventually J2
        anelastic_frequencial_harmonic_load_signal[0, 2, 0, frequencial_index] = (
            J2 if invert_for_J2 else anelastic_frequencial_harmonic_load_signal[0, 2, 0, frequencial_index]
        )
        anelastic_frequencial_harmonic_load_signal[:, 1, :2, frequencial_index] = array(
            object=[
                [C_1_0, C_1_1],
                [0.0, S_1_1],
            ],
            dtype=complex,
        )
        frequencial_harmonic_geoid[:, 1, :2, frequencial_index] = (
            DENSITY_RATIO
            * Love_numbers.values[Direction.potential][BoundaryCondition.load][0][
                frequencial_index if len(Love_numbers.axes["frequencies"]) > 1 else 0
            ]
            * anelastic_frequencial_harmonic_load_signal[:, 1, :2, frequencial_index]
        )
        frequencial_harmonic_radial_displacement[:, 1, :2, frequencial_index] = (
            DENSITY_RATIO
            * Love_numbers.values[Direction.radial][BoundaryCondition.load][0][frequencial_index if len(Love_numbers.axes["frequencies"]) > 1 else 0]
            * anelastic_frequencial_harmonic_load_signal[:, 1, :2, frequencial_index]
        )
        scale_factor[frequencial_index] = D
        if compute_residuals:
            frequencial_harmonic_residuals[:, :, :, frequencial_index] = frequencial_harmonic_residual_value

    return (
        anelastic_frequencial_harmonic_load_signal,
        scale_factor,
        frequencial_harmonic_geoid,
        frequencial_harmonic_radial_displacement,
        frequencial_harmonic_residuals,
    )


def solve_degree_one_inversion(
    frequencial_index: int,
    harmonic_right_hand_side: ndarray[complex],
    low_degrees_polynomials: ndarray[complex],
    least_square_weights: ndarray[float],
    ocean_mask_indices: ndarray[float],
    n_max: int,
    latitude_indices_mesh_ocean: Optional[ndarray[int]],
    longitude_indices_mesh_ocean: Optional[ndarray[int]],
    compute_residuals: bool,
    invert_for_J2: bool,
):
    """"""
    # Left-hand Side.
    left_hand_side = (
        expand_dims(a=least_square_weights, axis=0)
        * low_degrees_polynomials[: 5 if invert_for_J2 else 4]
        * expand_dims(
            [-harmonic_right_hand_side[0, 1, 0], -harmonic_right_hand_side[0, 1, 1], -harmonic_right_hand_side[1, 1, 1], 1]
            + (
                [
                    -harmonic_right_hand_side[0, 2, 0],
                ]
                if invert_for_J2
                else []
            ),
            axis=-1,
        )
    ).T

    # Updates the right-hand Side.
    harmonic_right_hand_side[:, :2, :] = 0.0
    if invert_for_J2:
        harmonic_right_hand_side[0, 2, 0] = 0.0
    spatial_right_hand_side: ndarray[complex] = make_grid(
        harmonics=harmonic_right_hand_side.real,
        n_max=n_max,
    ) + 1.0j * make_grid(
        harmonics=harmonic_right_hand_side.imag,
        n_max=n_max,
    )
    right_hand_side = expand_dims(a=least_square_weights * spatial_right_hand_side.flatten()[ocean_mask_indices], axis=-1)

    # Inversion.
    solution_vector, _, _, _ = lstsq(
        a=left_hand_side,
        b=right_hand_side,
    )
    if compute_residuals:
        residual_map = zeros(shape=spatial_right_hand_side.shape)
        residuals = matmul(left_hand_side, solution_vector) - right_hand_side
        residual_map[latitude_indices_mesh_ocean, longitude_indices_mesh_ocean] = residuals.flatten()

    return (
        frequencial_index,
        solution_vector[0, 0],
        solution_vector[1, 0],
        solution_vector[2, 0],
        solution_vector[3, 0],
        0.0 if not invert_for_J2 else solution_vector[4, 0],
        None if not compute_residuals else map_sampling(map=residual_map, n_max=n_max, harmonic_domain=True)[0],
    )
