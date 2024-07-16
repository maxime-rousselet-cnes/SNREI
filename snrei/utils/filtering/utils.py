from numpy import arange, array, meshgrid, ndarray
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH

from ...functions import mean_on_mask
from ..data import map_sampling


def collection_SH_from_SH(
    signal_frequencies: ndarray[float], frequencial_harmonic_load_signal: ndarray[float]
) -> GRACE_collection_SH:
    """"""
    collection = GRACE_collection_SH()
    collection.unit = "EWH [mm]"
    degrees = arange(frequencial_harmonic_load_signal.shape[1])
    FREQUENCIES, DEGREES, ORDERS = meshgrid(signal_frequencies, degrees, degrees, indexing="ij")
    data = array(
        [
            FREQUENCIES.flatten(),
            DEGREES.flatten(),
            ORDERS.flatten(),
            frequencial_harmonic_load_signal[0, :, :].transpose((2, 0, 1)).flatten(),
            frequencial_harmonic_load_signal[1, :, :].transpose((2, 0, 1)).flatten(),
        ]
    ).T
    collection.data = DataFrame(
        DataFrame(data, columns=["date", "degree", "order", "C", "S"])
        .set_index(["date", "degree", "order"])
        .sort_index()
    )
    return collection


def leakage_correction(
    signal_frequencies: ndarray[float],
    frequencial_harmonic_load_signal: ndarray[complex],
    ocean_mask: ndarray[float],
    iterations: int,
    ddk_filter_level: int,
) -> ndarray[complex]:
    """
    Performs a correction for continental data leak on oceans and ocean data leak on continents.
    Iterates on frequencies and asked iterations for leakage correction.
    """

    harmonic_load_signal: ndarray[complex]
    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(iterations):

        # Iterates on frequencies.
        for i_frequency, harmonic_load_signal in enumerate(frequencial_harmonic_load_signal.transpose((3, 0, 1, 2))):
            frequencial_harmonic_load_signal[:, :, :, i_frequency] = set_mean_value_on_oceans(
                harmonic_load_signal=harmonic_load_signal, ocean_mask=ocean_mask
            )

        frequencial_harmonic_load_signal_collection_real = apply_DDK_filter(
            collection_SH_from_SH(
                signal_frequencies=signal_frequencies,
                frequencial_harmonic_load_signal=frequencial_harmonic_load_signal.real,
            ),
            ddk_filter_level=ddk_filter_level,
        )

        frequencial_harmonic_load_signal_collection_imag = apply_DDK_filter(
            collection_SH_from_SH(
                signal_frequencies=signal_frequencies,
                frequencial_harmonic_load_signal=frequencial_harmonic_load_signal.imag,
            ),
            ddk_filter_level=ddk_filter_level,
        )

        # TODO.

        harmonic_load_signal -= single_leakage_correction(
            harmonic_load_signal=harmonic_load_signal.real, ocean_mask=ocean_mask
        ) + single_leakage_correction(harmonic_load_signal=harmonic_load_signal.imag, ocean_mask=ocean_mask)

        frequencial_harmonic_load_signal[:, :, :, i_frequency] = harmonic_load_signal

    return frequencial_harmonic_load_signal


def set_mean_value_on_oceans(harmonic_load_signal: ndarray[float], ocean_mask: ndarray[float]) -> ndarray[float]:
    """ """
    n_max = len(harmonic_load_signal[0]) - 1
    grid = MakeGridDH(harmonic_load_signal, sampling=2, lmax=n_max)
    ocean_mean = mean_on_mask(mask=ocean_mask, grid=grid)
    print(ocean_mean)  # TODO: remove.
    return map_sampling(
        map=grid * (1.0 - ocean_mask) + ocean_mean * ocean_mask,
        n_max=n_max,
        harmonic_domain=True,
    )[0]
