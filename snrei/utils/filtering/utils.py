from multiprocessing import Pool, cpu_count

from numpy import arange, array, meshgrid, ndarray, zeros
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH

from ..data import map_sampling


def collection_SH_from_map(spatial_load_signal: ndarray[float], n_max: int) -> GRACE_collection_SH:  # (2 * (n_max + 1), 4 * (n_max + 1)) - shaped.
    """"""
    collection = GRACE_collection_SH()
    collection.unit = "EWH [mm]"
    degrees = arange(n_max + 1)
    fake_frequencies_mesh, degrees_mesh, _ = meshgrid([0], degrees, degrees, indexing="ij")
    harmonic_load_signal = map_sampling(map=array(spatial_load_signal), n_max=n_max, harmonic_domain=True)[0]
    c = harmonic_load_signal[0]
    s = harmonic_load_signal[1]
    data = array([fake_frequencies_mesh.flatten(), degrees_mesh.flatten(), degrees_mesh.flatten(), c.flatten(), s.flatten()]).T
    collection.data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection.data[column] = collection.data[column].astype(int)
    collection.data = collection.data.set_index(["date", "degree", "order"]).sort_index()
    return collection


def map_from_collection_SH(collection: GRACE_collection_SH, n_max: int) -> ndarray[float]:  # (2 * (n_max + 1), 4 * (n_max + 1)) - shaped.S
    """"""
    return MakeGridDH(collection.data.to_numpy().T.reshape((2, 1, n_max + 1, n_max + 1)).transpose((1, 0, 2, 3))[0], sampling=2, lmax=n_max)


def leakage_correction_iterations_function(
    harmonic_load_signal: ndarray[complex],  # (2, n_max + 1, n_max + 1) - shaped.
    right_hand_side: ndarray[complex],  # (2, n_max + 1, n_max + 1) - shaped.
    ocean_mask: ndarray[float],  # (2 * (n_max + 1), 4 * (n_max + 1)) - shaped.
    n_max: int,
    ddk_filter_level: int,
    iterations: int,
) -> ndarray[complex]:

    # Gets the input in spatial domain.
    spatial_load_signal: ndarray[complex] = MakeGridDH(harmonic_load_signal.real, sampling=2, lmax=n_max) + 1.0j * MakeGridDH(
        harmonic_load_signal.imag, sampling=2, lmax=n_max
    )

    # Oceanic true level.
    Ocean_true_level: ndarray[complex] = (
        MakeGridDH(right_hand_side.real, sampling=2, lmax=n_max) + 1.0j * MakeGridDH(right_hand_side.imag, sampling=2, lmax=n_max)
    ) * ocean_mask

    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(iterations):

        # Leakage input.
        EWH_2_prime: ndarray[complex] = Ocean_true_level + spatial_load_signal * (1.0 - ocean_mask)

        # Computes continental leakage on oceans.
        EWH_2_second: ndarray[complex] = map_from_collection_SH(
            collection=apply_DDK_filter(collection_SH_from_map(spatial_load_signal=EWH_2_prime.real, n_max=n_max), ddk_filter_level=ddk_filter_level),
            n_max=n_max,
        ) + 1.0j * map_from_collection_SH(
            collection=apply_DDK_filter(collection_SH_from_map(spatial_load_signal=EWH_2_prime.imag, n_max=n_max), ddk_filter_level=ddk_filter_level),
            n_max=n_max,
        )

        # Applies correction.
        differential_term: ndarray[complex] = EWH_2_second - EWH_2_prime
        spatial_load_signal = spatial_load_signal + differential_term * (1.0 - ocean_mask) - differential_term * ocean_mask

    # Gets the result back in spherical harmonics domain.
    return (
        map_sampling(map=spatial_load_signal.real, n_max=n_max, harmonic_domain=True)[0]
        + 1.0j * map_sampling(map=spatial_load_signal.imag, n_max=n_max, harmonic_domain=True)[0]
    )


def parallel_leakage_correction_iteration(args):
    """
    A helper function for parallel processing.
    """
    (i_frequency, frequencial_harmonic_load_signal_initial, frequencial_right_hand_side, ocean_mask, n_max, ddk_filter_level, iterations) = args

    corrected_signal = leakage_correction_iterations_function(
        harmonic_load_signal=frequencial_harmonic_load_signal_initial[:, :, :, i_frequency],
        right_hand_side=frequencial_right_hand_side[:, :, :, i_frequency],
        ocean_mask=ocean_mask,
        n_max=n_max,
        ddk_filter_level=ddk_filter_level,
        iterations=iterations,
    )
    return (i_frequency, corrected_signal)


def leakage_correction(
    frequencial_harmonic_load_signal_initial: ndarray[complex],
    frequencial_scale_factor: ndarray[complex],
    frequencial_harmonic_geoid: ndarray[complex],
    frequencial_harmonic_radial_displacement: ndarray[complex],
    ocean_mask: ndarray[float],
    iterations: int,
    ddk_filter_level: int,
    n_max: int,
) -> ndarray[complex]:
    """
    Performs a correction for continental data leak on oceans and ocean data leak on continents.
    Iterates on frequencies and asked iterations for leakage correction.
    Returns data corrected from leakage.
    """

    # Prepares arrays.
    frequencial_right_hand_side = frequencial_harmonic_geoid - frequencial_harmonic_radial_displacement - frequencial_scale_factor
    corrected_from_leakage_frequencial_harmonic_load_signal = zeros(shape=frequencial_harmonic_load_signal_initial.shape, dtype=complex)

    # Prepare arguments for parallel processing
    args = [
        (i_frequency, frequencial_harmonic_load_signal_initial, frequencial_right_hand_side, ocean_mask, n_max, ddk_filter_level, iterations)
        for i_frequency in range(len(frequencial_scale_factor))
    ]

    # Use multiprocessing to parallelize the loop over frequencies
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(parallel_leakage_correction_iteration, args)

    # Gather the results
    for i_frequency, corrected_signal in results:
        corrected_from_leakage_frequencial_harmonic_load_signal[:, :, :, i_frequency] = corrected_signal

    return corrected_from_leakage_frequencial_harmonic_load_signal
