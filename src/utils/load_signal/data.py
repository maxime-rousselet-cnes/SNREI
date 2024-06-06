from pathlib import Path
from typing import Optional

import netCDF4
from cv2 import erode
from numpy import (
    array,
    cos,
    expand_dims,
    flip,
    linspace,
    ndarray,
    ones,
    pi,
    prod,
    round,
    unique,
)
from pandas import read_csv
from pyshtools.expand import MakeGridDH, SHExpandDH

from ..classes import GMSL_data_path, LoadSignalHyperParameters, data_masks_path

COLUMNS = ["lower", "mean", "upper"]


def extract_temporal_load_signal(
    name: str,
    path: Path = GMSL_data_path,
    filename: str = "Frederikse/global_basin_timeseries.csv",
    zero: bool = True,
) -> tuple[ndarray[float], ndarray[float]]:
    """
    Opens Frederikse et al.'s file and formats its data. Mean load in equivalent water height with respect to time.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(filename), sep=",")
    signal_dates = df["Unnamed: 0"].values
    # Formats. Barystatic = Sum - Steric.
    barystatic: dict[str, ndarray[float]] = {
        column: array(
            object=[
                float(item.split(",")[0] + "." + item.split(",")[1])
                for item in df["Sum of contributors [" + column + "]"].values
            ],
            dtype=float,
        )
        - array(
            object=[float(item.split(",")[0] + "." + item.split(",")[1]) for item in df["Steric [" + column + "]"].values],
            dtype=float,
        )
        for column in COLUMNS
    }
    # For worst/best case, build the maximum/minimum slope barystatic curb.
    if name == "best" or name == "worst":
        if name == "best":
            start, end = "upper", "lower"
        elif name == "worst":
            start, end = "lower", "upper"
        a = (barystatic[start][-1] - barystatic[end][0]) / (barystatic["mean"][-1] - barystatic["mean"][0])
        b = barystatic[end][0] - a * barystatic["mean"][0]
        barystatic[name] = a * barystatic["mean"] + b
    return signal_dates, barystatic[name] - (barystatic[name][0] if zero else 0)


def surface_ponderation(
    territorial_mask: ndarray[float],
) -> ndarray[float]:
    """
    Gets the surface of a (latitude * longitude) array.
    """
    return territorial_mask * expand_dims(a=cos(linspace(start=-pi / 2, stop=pi / 2, num=len(territorial_mask))), axis=1)


def territorial_mean(
    territorial_mask: ndarray[float], harmonics: Optional[ndarray[float]] = None, grid: Optional[ndarray[float]] = None
) -> float:
    """
    Computes mean value over a given surface. Uses a given mask.
    """
    if grid is None:
        grid: ndarray[float] = MakeGridDH(harmonics, sampling=2)
    surface = surface_ponderation(territorial_mask=territorial_mask)
    weighted_values = grid * surface
    return round(a=sum(weighted_values.flatten()) / sum(surface.flatten()), decimals=5)


def load_subpath(path: Path, load_signal_hyper_parameters: LoadSignalHyperParameters) -> Path:
    """
    Creates a subpath for results depending on load function.
    """
    return (
        path.joinpath("load")
        .joinpath(load_signal_hyper_parameters.load_signal)
        .joinpath(load_signal_hyper_parameters.ocean_load.split("/")[0])
        .joinpath(load_signal_hyper_parameters.case)
        .joinpath("with" + ("" if load_signal_hyper_parameters.little_isostatic_adjustment else "out") + "_LIA")
        .joinpath(
            "with"
            + ("" if load_signal_hyper_parameters.opposite_load_on_continents else "out")
            + "_opposite_load_on_continents"
        )
    )


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


def extract_mask_nc(path: Path = data_masks_path, name: str = "IMERG_land_sea_mask.nc", pixels_to_coast: int = 10) -> ndarray:
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
    # Dilates continents (100km).
    map = erode(map, ones(shape=(3, 3)), iterations=pixels_to_coast)
    return map


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


def map_sampling(map: ndarray[float], n_max: int, harmonic_domain: bool = False) -> tuple[ndarray[float], int]:
    """
    Redefined a (latitude, longitude) map definition. Eventually returns it in harmonic domain.
    """
    n_max = min(n_max, (len(map) - 1) // 2)
    harmonics = SHExpandDH(
        map,
        sampling=2,
        lmax_calc=n_max,
    )
    return (
        (harmonics if harmonic_domain else MakeGridDH(harmonics, sampling=2, lmax=n_max)),
        n_max,
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
            )[0]
        )
