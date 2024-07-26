from numpy import arange, array, meshgrid, ndarray, zeros
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH

from ..data import map_sampling


def collection_SH_from_map(frequencial_spatial_load_signal: ndarray[float], n_max: int) -> GRACE_collection_SH:
    """"""
    collection = GRACE_collection_SH()
    collection.unit = "EWH [mm]"
    degrees = arange(n_max + 1)
    FREQUENCIES, DEGREES, ORDERS = meshgrid(
        arange(len(frequencial_spatial_load_signal)), degrees, degrees, indexing="ij"
    )
    frequencial_harmonic_load_signal = array(
        object=[
            map_sampling(map=spatial_load_signal, n_max=n_max, harmonic_domain=True)[0]
            for spatial_load_signal in frequencial_spatial_load_signal
        ]
    )
    data = array(
        [
            FREQUENCIES.flatten(),
            DEGREES.flatten(),
            ORDERS.flatten(),
            frequencial_harmonic_load_signal[:, 0].transpose((2, 0, 1)).flatten(),
            frequencial_harmonic_load_signal[:, 1].transpose((2, 0, 1)).flatten(),
        ]
    ).T
    collection.data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection.data[column] = collection.data[column].astype(int)
    collection.data = collection.data.set_index(["date", "degree", "order"]).sort_index()
    return collection


def map_from_collection_SH(collection: GRACE_collection_SH) -> ndarray[float]:
    """"""
    n_rows = len(collection.data)
    n_max = max(collection.data.index.get_level_values("degree"))
    return array(
        object=[
            MakeGridDH(harmonic_load_signal, sampling=2, lmax=n_max)
            for harmonic_load_signal in collection.data.to_numpy()
            .T.reshape((2, n_rows // (n_max + 1) ** 2, n_max + 1, n_max + 1))
            .transpose((1, 0, 2, 3))
        ]
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

    # Spatial domain.
    frequencial_spatial_load_signal = array(
        object=[
            MakeGridDH(harmonic_load_signal.real, sampling=2, lmax=n_max)
            + 1.0j * MakeGridDH(harmonic_load_signal.imag, sampling=2, lmax=n_max)
            for harmonic_load_signal in frequencial_harmonic_load_signal_initial.transpose((3, 0, 1, 2))
        ],
        dtype=complex,
    )
    EWH_2_prime = zeros(shape=frequencial_spatial_load_signal.shape, dtype=complex)

    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(iterations):

        # Generates initial signal
        for i_frequency, spatial_load_signal in enumerate(frequencial_spatial_load_signal):
            EWH_2_prime[i_frequency] = (
                MakeGridDH(
                    frequencial_harmonic_geoid[:, :, :, i_frequency].real
                    - frequencial_harmonic_radial_displacement[:, :, :, i_frequency].real
                    - frequencial_scale_factor.real[i_frequency],
                    sampling=2,
                    lmax=n_max,
                )
                + 1.0j
                * MakeGridDH(
                    frequencial_harmonic_geoid[:, :, :, i_frequency].imag
                    - frequencial_harmonic_radial_displacement[:, :, :, i_frequency].imag
                    - frequencial_scale_factor.imag[i_frequency],
                    sampling=2,
                    lmax=n_max,
                )
            ) * ocean_mask + spatial_load_signal * (1.0 - ocean_mask)

        # Computes continental leakage on oceans.
        EWH_2_second = map_from_collection_SH(
            collection=apply_DDK_filter(
                collection_SH_from_map(frequencial_spatial_load_signal=EWH_2_prime.real, n_max=n_max),
                ddk_filter_level=ddk_filter_level,
            )
        ) + 1.0j * map_from_collection_SH(
            collection=apply_DDK_filter(
                collection_SH_from_map(frequencial_spatial_load_signal=EWH_2_prime.imag, n_max=n_max),
                ddk_filter_level=ddk_filter_level,
            )
        )

        # Applies correction.
        differential_term = EWH_2_second - EWH_2_prime
        frequencial_spatial_load_signal = differential_term * array(
            object=[1.0 - ocean_mask]
        ) - differential_term * array(object=[ocean_mask])

    return array(
        object=[
            map_sampling(map=spatial_load_signal.real, n_max=n_max, harmonic_domain=True)[0]
            + 1.0j * map_sampling(map=spatial_load_signal.imag, n_max=n_max, harmonic_domain=True)[0]
            for spatial_load_signal in frequencial_spatial_load_signal
        ],
        dtype=complex,
    ).transpose((1, 2, 3, 0))
