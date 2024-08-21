from os import cpu_count

from dask import visualize
from dask.array import map_blocks
from dask.array.core import Array, from_array
from dask.config import set
from numpy import arange, array, ceil, meshgrid, ndarray, zeros
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH

from ..data import map_sampling


def collection_SH_from_map(
    frequencial_spatial_load_signal: ndarray[float],
    n_max: int,
    n_frequencies_chunk: int,
) -> GRACE_collection_SH:
    """"""
    collection = GRACE_collection_SH()
    collection.unit = "EWH [mm]"
    degrees = arange(n_max + 1)
    FREQUENCIES, DEGREES, _ = meshgrid(
        arange(n_frequencies_chunk), degrees, degrees, indexing="ij"
    )
    frequencial_harmonic_load_signal = array(
        object=[
            map_sampling(
                map=array(spatial_load_signal), n_max=n_max, harmonic_domain=True
            )[0]
            for spatial_load_signal in frequencial_spatial_load_signal
        ]
    )
    data = array(
        [
            FREQUENCIES.flatten(),
            DEGREES.flatten(),
            DEGREES.flatten(),
            frequencial_harmonic_load_signal[:, 0].flatten(),
            frequencial_harmonic_load_signal[:, 1].flatten(),
        ]
    ).T
    collection.data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection.data[column] = collection.data[column].astype(int)
    collection.data = collection.data.set_index(
        ["date", "degree", "order"]
    ).sort_index()
    return collection


def map_from_collection_SH(
    collection: GRACE_collection_SH, n_max: int, n_frequencies_chunk: int
) -> ndarray[float]:
    """"""
    return from_array(
        (
            zeros(
                shape=(n_frequencies_chunk, 2 * (n_max + 1), 4 * (n_max + 1)),
                dtype=complex,
            )
            if len(collection.data) == (n_max + 1) ** 2
            else [
                MakeGridDH(harmonic_load_signal, sampling=2, lmax=n_max)
                for harmonic_load_signal in collection.data.to_numpy()
                .T.reshape((2, n_frequencies_chunk, n_max + 1, n_max + 1))
                .transpose((1, 0, 2, 3))
            ]
        ),
        chunks=(n_frequencies_chunk, -1, -1),
    )


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

    # Prepares Dask arrays.
    set(scheduler="single-threaded")
    chunk_size = int(ceil(len(frequencial_scale_factor) / cpu_count()))
    frequencial_harmonic_load_signal = from_array(
        frequencial_harmonic_load_signal_initial.transpose((3, 0, 1, 2)),
        chunks=(chunk_size, 2, n_max + 1, n_max + 1),
    )
    right_hand_side = (
        frequencial_harmonic_geoid
        - frequencial_harmonic_radial_displacement
        - frequencial_scale_factor
    )
    RHS_chuncks = from_array(
        right_hand_side.transpose((3, 0, 1, 2)),
        chunks=(chunk_size, 2, n_max + 1, n_max + 1),
    )

    # Defines parralel computing function.
    def leakage_correction_iterations_function(
        frequencial_harmonic_load_signal: Array,
        RHS_chuncks: Array,
    ) -> Array:

        n_frequencies_chunk = len(RHS_chuncks)

        # Gets the input in spatial domain.
        frequencial_spatial_load_signal = from_array(
            (
                zeros(
                    shape=(n_frequencies_chunk, 2 * (n_max + 1), 4 * (n_max + 1)),
                    dtype=complex,
                )
                if frequencial_harmonic_load_signal.shape[1] != 2
                else [
                    (
                        MakeGridDH(
                            frequencial_harmonic_load_signal[i_frequency].real,
                            sampling=2,
                            lmax=n_max,
                        )
                        + 1.0j
                        * MakeGridDH(
                            frequencial_harmonic_load_signal[i_frequency].imag,
                            sampling=2,
                            lmax=n_max,
                        )
                    )
                    for i_frequency in range(n_frequencies_chunk)
                ]
            ),
            chunks=(n_frequencies_chunk, -1, -1),
        )

        # Oceanic true level.
        Ocean_true_level = from_array(
            (
                zeros(
                    shape=(n_frequencies_chunk, 2 * (n_max + 1), 4 * (n_max + 1)),
                    dtype=complex,
                )
                if frequencial_harmonic_load_signal.shape[1] != 2
                else [
                    (
                        MakeGridDH(
                            RHS_chuncks[i_frequency].real,
                            sampling=2,
                            lmax=n_max,
                        )
                        + 1.0j
                        * MakeGridDH(
                            RHS_chuncks[i_frequency].imag,
                            sampling=2,
                            lmax=n_max,
                        )
                        * ocean_mask
                    )
                    for i_frequency in range(n_frequencies_chunk)
                ]
            ),
            chunks=(n_frequencies_chunk, -1, -1),
        )

        # Iterates a leakage correction procedure as many times as asked for.
        for _ in range(iterations):

            # Leakage input.
            EWH_2_prime = Ocean_true_level + frequencial_spatial_load_signal * array(
                object=[(1.0 - ocean_mask)]
            )

            # Computes continental leakage on oceans.
            EWH_2_second = map_from_collection_SH(
                collection=apply_DDK_filter(
                    collection_SH_from_map(
                        frequencial_spatial_load_signal=EWH_2_prime.real,
                        n_max=n_max,
                        n_frequencies_chunk=n_frequencies_chunk,
                    ),
                    ddk_filter_level=ddk_filter_level,
                ),
                n_max=n_max,
                n_frequencies_chunk=n_frequencies_chunk,
            ) + 1.0j * map_from_collection_SH(
                collection=apply_DDK_filter(
                    collection_SH_from_map(
                        frequencial_spatial_load_signal=EWH_2_prime.imag,
                        n_max=n_max,
                        n_frequencies_chunk=n_frequencies_chunk,
                    ),
                    ddk_filter_level=ddk_filter_level,
                ),
                n_max=n_max,
                n_frequencies_chunk=n_frequencies_chunk,
            )

            # Applies correction.
            differential_term = EWH_2_second - EWH_2_prime
            frequencial_spatial_load_signal = (
                frequencial_spatial_load_signal
                + differential_term * array(object=[(1.0 - ocean_mask)])
                - differential_term * array(object=[ocean_mask])
            )

        # Gets the result back in spherical harmonics domain.
        return from_array(
            [
                map_sampling(
                    map=array(spatial_load_signal).real,
                    n_max=n_max,
                    harmonic_domain=True,
                )[0]
                + 1.0j
                * map_sampling(
                    map=array(spatial_load_signal).imag,
                    n_max=n_max,
                    harmonic_domain=True,
                )[0]
                for spatial_load_signal in frequencial_spatial_load_signal
            ],
            chunks=(n_frequencies_chunk, 2, n_max + 1, n_max + 1),
        )

    return array(
        map_blocks(
            leakage_correction_iterations_function,
            frequencial_harmonic_load_signal,
            RHS_chuncks,
            dtype=Array,
        ).compute()
    ).transpose((1, 2, 3, 0))
