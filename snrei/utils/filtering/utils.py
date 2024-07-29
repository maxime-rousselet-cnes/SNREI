from multiprocessing import Pool

from numpy import arange, array, meshgrid, ndarray, zeros
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH

from ..data import map_sampling


def collection_SH_from_map(spatial_load_signal: ndarray[float], n_max: int) -> GRACE_collection_SH:
    """"""
    collection = GRACE_collection_SH()
    collection.unit = "EWH [mm]"
    degrees = arange(n_max + 1)
    FREQUENCIES, DEGREES, ORDERS = meshgrid([0], degrees, degrees, indexing="ij")
    harmonic_load_signal = map_sampling(map=spatial_load_signal, n_max=n_max, harmonic_domain=True)[0]
    data = array(
        [
            FREQUENCIES.flatten(),
            DEGREES.flatten(),
            ORDERS.flatten(),
            harmonic_load_signal[0].flatten(),
            harmonic_load_signal[1].flatten(),
        ]
    ).T
    collection.data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection.data[column] = collection.data[column].astype(int)
    collection.data = collection.data.set_index(["date", "degree", "order"]).sort_index()
    return collection


def map_from_collection_SH(collection: GRACE_collection_SH) -> ndarray[float]:
    """"""
    n_max = max(collection.data.index.get_level_values("degree"))
    return MakeGridDH(
        collection.data.to_numpy().T.reshape((2, n_max + 1, n_max + 1)),
        sampling=2,
        lmax=n_max,
    )


def leakage_correction_iteration_step(
    harmonic_geoid: ndarray[complex],
    harmonic_radial_displacement: ndarray[complex],
    scale_factor: complex,
    spatial_load_signal: ndarray[complex],
    n_max: int,
    ocean_mask: ndarray[float],
    ddk_filter_level: int,
) -> ndarray[complex]:
    """
    Subfunction for parallel processing.
    """

    # Leakage input.
    EWH_2_prime: ndarray[complex] = (
        MakeGridDH(
            harmonic_geoid.real - harmonic_radial_displacement.real - scale_factor.real,
            sampling=2,
            lmax=n_max,
        )
        + 1.0j
        * MakeGridDH(
            harmonic_geoid.imag - harmonic_radial_displacement.imag - scale_factor.imag,
            sampling=2,
            lmax=n_max,
        )
    ) * ocean_mask + spatial_load_signal * (1.0 - ocean_mask)

    # Computes continental leakage on oceans.
    EWH_2_second = map_from_collection_SH(
        collection=apply_DDK_filter(
            collection_SH_from_map(spatial_load_signal=EWH_2_prime.real, n_max=n_max),
            ddk_filter_level=ddk_filter_level,
        )
    ) + 1.0j * map_from_collection_SH(
        collection=apply_DDK_filter(
            collection_SH_from_map(spatial_load_signal=EWH_2_prime.imag, n_max=n_max),
            ddk_filter_level=ddk_filter_level,
        )
    )

    # Applies correction.
    differential_term = EWH_2_second - EWH_2_prime
    return spatial_load_signal + differential_term * (1.0 - ocean_mask) - differential_term * ocean_mask


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
    """

    n_frequencies = frequencial_harmonic_load_signal_initial.shape[-1]

    # Gets the input in spatial domain.
    frequencial_spatial_load_signal = array(
        object=[
            MakeGridDH(frequencial_harmonic_load_signal_initial[:, :, :, i_frequency].real, sampling=2, lmax=n_max)
            + 1.0j
            * MakeGridDH(frequencial_harmonic_load_signal_initial[:, :, :, i_frequency].imag, sampling=2, lmax=n_max)
            for i_frequency in range(n_frequencies)
        ],
        dtype=complex,
    )

    # Initializes a Callable as a global variable to parallelize.
    global leakage_correction_parallel_processing

    def leakage_correction_parallel_processing(
        arrays: tuple[ndarray[complex], ndarray[complex], complex, ndarray[complex]]
    ) -> ndarray[complex]:
        return leakage_correction_iteration_step(
            harmonic_geoid=arrays[0],
            harmonic_radial_displacement=arrays[1],
            scale_factor=arrays[2],
            spatial_load_signal=arrays[3],
            n_max=n_max,
            ocean_mask=ocean_mask,
            ddk_filter_level=ddk_filter_level,
        )

    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(iterations):

        """
        with Pool() as p:  # Processes for frequencies.
            frequencial_spatial_load_signal = array(
                object=p.map(
                    func=leakage_correction_parallel_processing,
                    iterable=[
                        (
                            frequencial_harmonic_geoid[:, :, :, i_frequency],
                            frequencial_harmonic_radial_displacement[:, :, :, i_frequency],
                            frequencial_scale_factor[i_frequency],
                            frequencial_spatial_load_signal[i_frequency],
                        )
                        for i_frequency in range(n_frequencies)
                    ],
                ),
                dtype=complex,
            )
        """
        frequencial_spatial_load_signal = array(
            object=[
                leakage_correction_parallel_processing(
                    (
                        frequencial_harmonic_geoid[:, :, :, i_frequency],
                        frequencial_harmonic_radial_displacement[:, :, :, i_frequency],
                        frequencial_scale_factor[i_frequency],
                        frequencial_spatial_load_signal[i_frequency],
                    )
                )
                for i_frequency in range(n_frequencies)
            ],
            dtype=complex,
        )

    # Gets the result back in spherical harmonics domain.
    return array(
        object=[
            map_sampling(map=spatial_load_signal.real, n_max=n_max, harmonic_domain=True)[0]
            + 1.0j * map_sampling(map=spatial_load_signal.imag, n_max=n_max, harmonic_domain=True)[0]
            for spatial_load_signal in frequencial_spatial_load_signal
        ],
        dtype=complex,
    ).transpose((1, 2, 3, 0))
