from pathlib import Path

import netCDF4
from numpy import array, expand_dims, flip, ndarray, prod, round, unique
from pandas import read_csv
from pyshtools.expand import MakeGridDH, SHExpandDH

from ..paths import data_Frederikse_path, data_masks_path, data_trends_GRACE_path


def extract_temporal_load_signal(
    path: Path = data_Frederikse_path, name: str = "global_basin_timeseries.csv"
) -> tuple[ndarray[float], ndarray[float]]:
    """
    Opens Frederikse et al.'s file and formats its data. Mean load in equivalent water height with respect to time.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(name), sep=",")
    dates = df["Unnamed: 0"].values
    # Formats.
    sum_values = array(
        object=[float(item.split(",")[0] + "." + item.split(",")[1]) for item in df["Sum of contributors [mean]"].values],
        dtype=float,
    )
    steric_values = array(
        object=[float(item.split(",")[0] + "." + item.split(",")[1]) for item in df["Steric [mean]"].values],
        dtype=float,
    )
    barystatic = sum_values - steric_values
    return dates, barystatic - barystatic[0]


def erase_area(
    map: ndarray[float],
    lat: ndarray[float],
    lon: ndarray[float],
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
):
    """
    Erases a rectangle area on a (latitude, longitude) map: sets zero values.
    """
    return map * (
        1
        - expand_dims(a=flip(m=(lat > min_latitude) * (lat < max_latitude), axis=0), axis=1)
        * expand_dims(a=(lon > (min_longitude % 360)) * (lon < (max_longitude % 360)), axis=0)
    )


def extract_mask_nc(path: Path = data_masks_path, name: str = "IMERG_land_sea_mask.nc") -> ndarray:
    """
    Opens NASA's nc file for land/sea mask and formats its data.
    """
    # Gets raw data.
    ds = netCDF4.Dataset(path.joinpath(name))
    map: ndarray[float] = flip(m=ds.variables["landseamask"], axis=0).data
    lat = array(object=[latitude for latitude in ds.variables["lat"]])
    lon = array(object=[longitude for longitude in ds.variables["lon"]])
    map /= max(map.flatten())  # Normalizes (land = 0, ocean = 1).

    # Rejects big lakes from ocean label.
    # Lake Superior.
    map = erase_area(
        map=map,
        lat=lat,
        lon=lon,
        min_latitude=41.375963,
        max_latitude=50.582521,
        min_longitude=-93.748270,
        max_longitude=-75.225322,
    )
    # Lake Victoria.
    map = erase_area(
        map=map,
        lat=lat,
        lon=lon,
        min_latitude=-2.809322,
        max_latitude=0.836983,
        min_longitude=31.207700,
        max_longitude=34.530942,
    )
    # Caspian Sea.
    map = erase_area(
        map=map,
        lat=lat,
        lon=lon,
        min_latitude=35.569650,
        max_latitude=47.844035,
        min_longitude=44.303403,
        max_longitude=60.937192,
    )

    return map


def extract_trends_GRACE(
    path: Path = data_trends_GRACE_path, name: str = "GRACE(-FO)_MSSA_corrected_for_leakage_2003_2022.xyz"
) -> ndarray:
    """
    Opens and formats GRACE trends CSV datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(name), skiprows=3 if name.split(".")[-1] == "xyz" else 11, sep=";")
    return flip(m=[[value for value in df["EWH"][df["lat"] == lat]] for lat in unique(ar=df["lat"])], axis=0)[1:, 1:]


def extract_mask_csv(path: Path = data_masks_path, name: str = "ocean_mask_buffer_coast_300km_eq_removed_0_360.csv") -> ndarray:
    """
    Opens and formats ocean mask CSV datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(name), sep=",")
    return flip(m=[[value for value in df["mask"][df["lat"] == lat]] for lat in unique(ar=df["lat"])], axis=0)[1:, 1:]


def map_normalizing(
    map: ndarray,
) -> ndarray:
    """
    Sets global mean as zero and max as one by homothety.
    """
    n_t = prod(map.shape)
    sum_map = sum(map.flatten())
    max_map = max(map.flatten())
    return map / (max_map - sum_map / n_t) + sum_map / (sum_map - max_map * n_t)


def map_sampling(map: ndarray[float], n_max: int, harmonic_domain: bool = False) -> ndarray[float]:
    """
    Redefined a (latitude, longitude) map definition. Eventually returns it in harmonic domain.
    """
    harmonics = SHExpandDH(
        map,
        sampling=2,
        lmax_calc=min(n_max, (len(map) - 1) // 2),
    )
    return (
        harmonics
        if harmonic_domain
        else MakeGridDH(
            harmonics,
            sampling=2,
        )
    )


def get_ocean_mask(name: str, n_max: int) -> ndarray[float]:
    """
    Gets the wanted ocean mask and adjusts it.
    """
    if name is None:
        return 1.0
    else:
        if name.split(".")[-1] == "csv":
            ocean_mask = extract_mask_csv(name=name)
        else:
            ocean_mask = extract_mask_nc()
        return round(
            a=map_sampling(
                map=ocean_mask,
                n_max=n_max,
            )
        )
