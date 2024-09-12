from multiprocessing import Pool

from geopandas import GeoDataFrame
from numpy import abs, arange, array, inf, meshgrid, ndarray
from pandas import DataFrame
from pyGFOToolbox.processing.filter.filter_ddk import _pool_apply_DDK_filter

from ...functions import make_grid, mean_on_mask
from ..data import map_sampling


def collection_SH_data_from_map(spatial_load_signal: ndarray[float], n_max: int) -> DataFrame:
    """"""
    degrees = arange(n_max + 1)
    fake_frequencies_mesh, degrees_mesh, orders_mesh = meshgrid([0], degrees, degrees, indexing="ij")
    harmonic_load_signal = map_sampling(map=spatial_load_signal, n_max=n_max, harmonic_domain=True)[0]
    c: ndarray[float] = harmonic_load_signal[0]
    s: ndarray[float] = harmonic_load_signal[1]
    data = array([fake_frequencies_mesh.flatten(), degrees_mesh.flatten(), orders_mesh.flatten(), c.flatten(), s.flatten()]).T
    collection_data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection_data[column] = collection_data[column].astype(int)
    collection_data = collection_data.set_index(["date", "degree", "order"]).sort_index()
    return collection_data


def map_from_collection_SH_data(
    collection_data: DataFrame,
    n_max: int,
) -> ndarray[float]:  # (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) - shaped.
    """"""
    a = array(object=[coeffs.reshape((n_max + 1, n_max + 1)) for coeffs in collection_data.to_numpy().T])
    return make_grid(harmonics=a, n_max=n_max)


def leakage_correction(
    harmonic_load_signal: ndarray[float],  # (2, n_max + 1, n_max + 1) - shaped.
    ocean_land_mask: ndarray[float],  # (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) - shaped.
    latitudes: ndarray[float],
    n_max: int,
    ddk_filter_level: int,
    iterations: int,
    signal_threshold: float,
    mean_signal_threshold: float,
) -> ndarray[complex]:
    """"""
    # Gets the input in spatial domain.
    spatial_load_signal: ndarray[complex] = make_grid(harmonics=harmonic_load_signal, n_max=n_max)

    # Oceanic true level.
    ocean_true_level: float = mean_on_mask(
        mask=ocean_land_mask,
        latitudes=latitudes,
        n_max=n_max,
        grid=spatial_load_signal,
        signal_threshold=mean_signal_threshold,
    )
    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(iterations):
        mask_non_oceanic_signal = ocean_land_mask * (abs(spatial_load_signal) > signal_threshold) + (1 - ocean_land_mask)

        # Leakage input.
        EWH_2_prime: ndarray[float] = ocean_true_level * (1 - mask_non_oceanic_signal) + spatial_load_signal * mask_non_oceanic_signal
        EWH_2_third: ndarray[float] = ocean_true_level * (1 - mask_non_oceanic_signal) + spatial_load_signal * (1 - ocean_land_mask)

        # Computes continental leakage on oceans.
        EWH_2_second: ndarray[complex] = map_from_collection_SH_data(
            collection_data=_pool_apply_DDK_filter(
                grace_monthly_sh=collection_SH_data_from_map(
                    spatial_load_signal=EWH_2_prime,
                    n_max=n_max,
                ),
                ddk_filter_level=ddk_filter_level,
            ),
            n_max=n_max,
        )
        # Applies correction.
        differential_term: ndarray[float] = EWH_2_second - EWH_2_third
        spatial_load_signal += differential_term * (1 - ocean_land_mask) - differential_term * ocean_land_mask

    # Gets the result back in spherical harmonics domain.
    return map_sampling(map=spatial_load_signal.real, n_max=n_max, harmonic_domain=True)[0]
