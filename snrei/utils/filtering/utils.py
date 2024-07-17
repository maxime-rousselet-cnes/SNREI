import gc

from numpy import arange, array, meshgrid, ndarray
from pandas import DataFrame
from pyGFOToolbox import GRACE_collection_SH
from pyGFOToolbox.processing.filter import apply_DDK_filter
from pyshtools.expand import MakeGridDH
from tqdm import tqdm

from ...functions import mean_on_mask
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
    frequencial_harmonic_load_signal: ndarray[float],
    ocean_mask: ndarray[float],
    iterations: int,
    ddk_filter_level: int,
) -> ndarray[float]:
    """
    Performs a correction for continental data leak on oceans and ocean data leak on continents.
    Iterates on frequencies and asked iterations for leakage correction.
    """

    harmonic_load_signal: ndarray[complex]
    # Iterates a leakage correction procedure as many times as asked for.
    for iteration in tqdm(range(iterations), desc="            Leakage correction iterations", position=3):

        # Iterates on frequencies to set mean value on oceans.
        for i_frequency, harmonic_load_signal in enumerate(frequencial_harmonic_load_signal.transpose((3, 0, 1, 2))):
            frequencial_harmonic_load_signal[:, :, :, i_frequency], ocean_mean = set_mean_value_on_oceans(
                harmonic_load_signal=harmonic_load_signal, ocean_mask=ocean_mask
            )
        print("\033[{0};{1}f{2}".format(7 + iteration, 1, str(ocean_mean)))

        # Computes continental leakage on oceans.
        frequencial_harmonic_load_signal_collection = apply_DDK_filter(
            collection_SH_from_SH(
                frequencial_harmonic_load_signal=(frequencial_harmonic_load_signal),
            ),
            ddk_filter_level=ddk_filter_level,
        )

        gc.collect()

        # Substract continental leakage on oceans.
        frequencial_harmonic_load_signal -= (
            SH_from_collection_SH(collection=frequencial_harmonic_load_signal_collection)
            - frequencial_harmonic_load_signal
        )

    return frequencial_harmonic_load_signal


def set_mean_value_on_oceans(
    harmonic_load_signal: ndarray[float], ocean_mask: ndarray[float]
) -> tuple[ndarray[float], float]:
    """ """
    n_max = len(harmonic_load_signal[0]) - 1
    grid = MakeGridDH(harmonic_load_signal, sampling=2, lmax=n_max)
    ocean_mean = mean_on_mask(mask=ocean_mask, grid=grid)
    return (
        map_sampling(
            map=grid * (1.0 - ocean_mask) + ocean_mean * ocean_mask,
            n_max=n_max,
            harmonic_domain=True,
        )[0],
        ocean_mean,
    )
