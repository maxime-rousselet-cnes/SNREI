from multiprocessing import Pool

from geopandas import GeoDataFrame
from numpy import arange, array, meshgrid, ndarray
from pandas import DataFrame
from pyGFOToolbox.processing.filter.filter_ddk import _pool_apply_DDK_filter

from ...functions import geopandas_oceanic_mean, make_grid
from ..data import map_sampling


def collection_SH_data_from_map(spatial_load_signal: ndarray[float], n_max: int) -> DataFrame:  # (2 * (n_max + 1), 4 * (n_max + 1)) - shaped.
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


def map_from_collection_SH_data(collection_data: DataFrame, n_max: int) -> ndarray[float]:  # (2 * (n_max + 1), 4 * (n_max + 1)) - shaped.
    """"""
    a = array(object=[coeffs.reshape((n_max + 1, n_max + 1)) for coeffs in collection_data.to_numpy().T])
    return make_grid(harmonics=a)


def leakage_correction_iterations_function(
    harmonic_load_signal: ndarray[complex],  # (2, n_max + 1, n_max + 1) - shaped.
    right_hand_side: ndarray[complex],  # (2, n_max + 1, n_max + 1) - shaped.
    continents_buffered_reprojected: GeoDataFrame,
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    ocean_mask: ndarray[float],  # (2 * (n_max + 1), 4 * (n_max + 1)) - shaped.
    n_max: int,
    ddk_filter_level: int,
    iterations: int,
) -> ndarray[complex]:

    # Gets the input in spatial domain.
    spatial_load_signal: ndarray[complex] = make_grid(harmonics=harmonic_load_signal.real) + 1.0j * make_grid(harmonics=harmonic_load_signal.imag)

    # Oceanic true level.
    ocean_true_level: complex = (  # TODO: replace ocean mean by right hand side (i.e. (1 + k' - h') * EWH)
        # make_grid(harmonics=right_hand_side.real)
        # + 1.0j * make_grid(harmonics=right_hand_side.imag)
        geopandas_oceanic_mean(continents=continents_buffered_reprojected, latitudes=latitudes, longitudes=longitudes, grid=spatial_load_signal.real)
        + 1.0j
        * geopandas_oceanic_mean(
            continents=continents_buffered_reprojected, latitudes=latitudes, longitudes=longitudes, grid=spatial_load_signal.imag
        )
    )

    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(iterations):

        # Leakage input.
        EWH_2_prime: ndarray[complex] = map_sampling(map=ocean_true_level * ocean_mask + spatial_load_signal * (1.0 - ocean_mask), n_max=n_max)[0]

        # Computes continental leakage on oceans.
        EWH_2_second: ndarray[complex] = map_from_collection_SH_data(
            collection_data=_pool_apply_DDK_filter(
                grace_monthly_sh=collection_SH_data_from_map(spatial_load_signal=EWH_2_prime.real, n_max=n_max),
                ddk_filter_level=ddk_filter_level,
            ),
            n_max=n_max,
        ) + 1.0j * map_from_collection_SH_data(
            collection_data=_pool_apply_DDK_filter(
                grace_monthly_sh=collection_SH_data_from_map(spatial_load_signal=EWH_2_prime.imag, n_max=n_max),
                ddk_filter_level=ddk_filter_level,
            ),
            n_max=n_max,
        )

        # Applies correction.
        differential_term: ndarray[complex] = EWH_2_second - EWH_2_prime
        spatial_load_signal += differential_term * (1.0 - ocean_mask) - differential_term * ocean_mask

    # Gets the result back in spherical harmonics domain.
    return (
        map_sampling(map=spatial_load_signal.real, n_max=n_max, harmonic_domain=True)[0]
        + 1.0j * map_sampling(map=spatial_load_signal.imag, n_max=n_max, harmonic_domain=True)[0]
    )


def leakage_correction(  # TODO.
    frequencial_harmonic_load_signal_initial: ndarray[complex],
    frequencial_scale_factor: ndarray[complex],
    frequencial_harmonic_geoid: ndarray[complex],
    frequencial_harmonic_radial_displacement: ndarray[complex],
    ocean_mask: ndarray[float],
    continents_buffered_reprojected: GeoDataFrame,
    latitudes: ndarray[float],
    longitudes: ndarray[float],
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

    # Prepares arguments for parallel processing.
    args = [
        (
            frequencial_harmonic_load_signal_initial[:, :, :, i_frequency],
            frequencial_right_hand_side[:, :, :, i_frequency],
            continents_buffered_reprojected,
            latitudes,
            longitudes,
            ocean_mask,
            n_max,
            ddk_filter_level,
            iterations,
        )
        for i_frequency in range(len(frequencial_scale_factor))
    ]

    # Uses multiprocessing to parallelize the loop over frequencies.
    with Pool() as p:
        return array(object=p.starmap(leakage_correction_iterations_function, args)).transpose((1, 2, 3, 0))
