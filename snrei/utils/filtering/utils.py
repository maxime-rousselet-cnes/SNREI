import gc
from copy import deepcopy

from numpy import arange, array, meshgrid, ndarray, zeros
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH
from tqdm import tqdm

from ..data import map_sampling


def collection_SH_from_SH(frequencial_harmonic_load_signal: ndarray[float]) -> GRACE_collection_SH:
    """"""
    collection = GRACE_collection_SH()
    collection.unit = "EWH [mm]"
    degrees = arange(frequencial_harmonic_load_signal.shape[1])
    FREQUENCIES, DEGREES, ORDERS = meshgrid(
        arange(frequencial_harmonic_load_signal.shape[3]), degrees, degrees, indexing="ij"
    )
    data = array(
        [
            FREQUENCIES.flatten(),
            DEGREES.flatten(),
            ORDERS.flatten(),
            frequencial_harmonic_load_signal[0, :, :].transpose((2, 0, 1)).flatten(),
            frequencial_harmonic_load_signal[1, :, :].transpose((2, 0, 1)).flatten(),
        ]
    ).T
    collection.data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection.data[column] = collection.data[column].astype(int)
    collection.data = collection.data.set_index(["date", "degree", "order"]).sort_index()
    return collection


def SH_from_collection_SH(collection: GRACE_collection_SH) -> ndarray[float]:
    """"""
    n_rows = len(collection.data)
    n_max = max(collection.data.index.get_level_values("degree"))
    return (
        collection.data.to_numpy()
        .T.reshape((2, n_rows // (n_max + 1) ** 2, n_max + 1, n_max + 1))
        .transpose((0, 2, 3, 1))
    )


def leakage_correction(
    frequencial_harmonic_load_signal_initial: ndarray[complex],
    frequencial_scale_factor: ndarray[complex],
    frequencial_harmonic_geoid: ndarray[complex],
    frequencial_harmonic_radial_displacement: ndarray[complex],
    ocean_mask: ndarray[float],
    iterations: int,
    ddk_filter_level: int,
) -> ndarray[float]:
    """
    Performs a correction for continental data leak on oceans and ocean data leak on continents.
    Iterates on frequencies and asked iterations for leakage correction.
    """

    frequencial_harmonic_load_signal = deepcopy(frequencial_harmonic_load_signal_initial)
    frequencial_harmonic_ocean_true_level = compute_frequencial_harmonic_ocean_true_level(
        frequencial_scale_factor=frequencial_scale_factor.real,
        frequencial_harmonic_geoid=frequencial_harmonic_geoid.real,
        frequencial_harmonic_radial_displacement=frequencial_harmonic_radial_displacement.real,
        ocean_mask=ocean_mask,
    ) + 1.0j * compute_frequencial_harmonic_ocean_true_level(
        frequencial_scale_factor=frequencial_scale_factor.imag,
        frequencial_harmonic_geoid=frequencial_harmonic_geoid.imag,
        frequencial_harmonic_radial_displacement=frequencial_harmonic_radial_displacement.imag,
        ocean_mask=ocean_mask,
    )

    # Iterates a leakage correction procedure as many times as asked for.
    for _ in tqdm(
        range(iterations),
        desc="            Leakage correction iterations",
        position=3,
        total=iterations,
        leave=False,
    ):

        # Creates a known signal.
        EWH_2_prime = (
            frequencial_harmonic_ocean_true_level
            + mask_only(frequencial_harmonic_load_signal=frequencial_harmonic_load_signal.real, mask=1.0 - ocean_mask)
            + 1.0j
            * mask_only(frequencial_harmonic_load_signal=frequencial_harmonic_load_signal.imag, mask=1.0 - ocean_mask)
        )

        # Computes continental leakage on oceans.
        EWH_2_second = SH_from_collection_SH(
            collection=apply_DDK_filter(
                collection_SH_from_SH(
                    frequencial_harmonic_load_signal=EWH_2_prime.real,
                ),
                ddk_filter_level=ddk_filter_level,
            )
        ) + 1.0j * SH_from_collection_SH(
            collection=apply_DDK_filter(
                collection_SH_from_SH(
                    frequencial_harmonic_load_signal=EWH_2_prime.imag,
                ),
                ddk_filter_level=ddk_filter_level,
            )
        )

        # Applies correction.
        differential_term = EWH_2_second - EWH_2_prime
        frequencial_harmonic_load_signal += (
            mask_only(frequencial_harmonic_load_signal=differential_term.real, mask=1.0 - ocean_mask)
            + 1.0j * mask_only(frequencial_harmonic_load_signal=differential_term.imag, mask=1.0 - ocean_mask)
            + mask_only(frequencial_harmonic_load_signal=differential_term.real, mask=ocean_mask)
            + 1.0j * mask_only(frequencial_harmonic_load_signal=differential_term.imag, mask=ocean_mask)
        )

        # Garbage collection for high RAM use in this loop.
        gc.collect()

    return frequencial_harmonic_load_signal


def mask_only(frequencial_harmonic_load_signal: ndarray[complex], mask: ndarray[float]) -> ndarray[complex]:
    """ """
    n_max = frequencial_harmonic_load_signal.shape[1] - 1
    for i_frequency, harmonic_load_signal in enumerate(frequencial_harmonic_load_signal.transpose((3, 0, 1, 2))):
        frequencial_harmonic_load_signal[:, :, :, i_frequency] = map_sampling(
            map=MakeGridDH(harmonic_load_signal, sampling=2, lmax=n_max) * mask,
            n_max=n_max,
            harmonic_domain=True,
        )[0]
    return frequencial_harmonic_load_signal


def compute_frequencial_harmonic_ocean_true_level(
    frequencial_scale_factor: ndarray[complex],
    frequencial_harmonic_geoid: ndarray[complex],
    frequencial_harmonic_radial_displacement: ndarray[complex],
    ocean_mask: ndarray[float],
) -> ndarray[complex]:
    """"""
    n_max = frequencial_harmonic_geoid.shape[1] - 1
    frequencial_harmonic_load_signal = zeros(shape=frequencial_harmonic_geoid.shape)
    for i_frequency in range(len(frequencial_scale_factor)):
        frequencial_harmonic_load_signal[:, :, :, i_frequency] = map_sampling(
            map=(
                MakeGridDH(frequencial_harmonic_geoid[:, :, :, i_frequency], sampling=2, lmax=n_max)
                - MakeGridDH(frequencial_harmonic_radial_displacement[:, :, :, i_frequency], sampling=2, lmax=n_max)
                - frequencial_scale_factor[i_frequency]
            )
            * ocean_mask,
            n_max=n_max,
            harmonic_domain=True,
        )[0]
    return frequencial_harmonic_load_signal
