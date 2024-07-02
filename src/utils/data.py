from csv import DictWriter
from functools import reduce
from pathlib import Path
from typing import Optional

import netCDF4
from cv2 import erode
from numpy import argsort, array, expand_dims, flip, ndarray, ones, round, unique, zeros
from pandas import read_csv
from pyshtools.expand import MakeGridDH, SHExpandDH

from .classes import (
    GRACE_DATA_UNIT_FACTOR,
    GMSL_data_path,
    GRACE_data_path,
    Love_numbers_path,
    LoveNumbersHyperParameters,
    Result,
    frequencies_path,
    masks_data_path,
    tables_path,
)
from .database import save_base_model

COLUMNS = ["lower", "mean", "upper"]


def extract_temporal_load_signal(
    name: str,
    path: Path = GMSL_data_path,
    filename: str = "Frederikse/global_basin_timeseries.csv",
    zero_at_origin: bool = True,
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
            object=[
                float(item.split(",")[0] + "." + item.split(",")[1]) for item in df["Steric [" + column + "]"].values
            ],
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

    return signal_dates, barystatic[name] - (barystatic[name][0] if zero_at_origin else 0)


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
    Used to erase big lakes from sea/land masks.
    """
    return map * (
        1
        - expand_dims(a=flip(m=(lat > min_latitude) * (lat < max_latitude), axis=0), axis=1)
        * expand_dims(a=(lon > (min_longitude % 360)) * (lon < (max_longitude % 360)), axis=0)
    )


def extract_mask_nc(
    path: Path = masks_data_path,
    name: str = "IMERG_land_sea_mask.nc",
    pixels_to_coast: int = 10,
) -> ndarray:
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


def extract_mask_csv(
    path: Path = masks_data_path,
    name: str = "ocean_mask_buffer_coast_300km_eq_removed_0_360.csv",
) -> ndarray:
    """
    Opens and formats ocean mask CSV datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(name), sep=",")
    return flip(
        m=[[value for value in df["mask"][df["lat"] == lat]] for lat in unique(ar=df["lat"])],
        axis=0,
    )


def redefine_n_max(n_max: int, map: Optional[ndarray] = None, harmonics: Optional[ndarray] = None) -> int:
    """
    Gets maximal number of degees, limited by map length.
    """
    if map is None:
        return min(n_max, len(harmonics[0, :, 0]) - 1)
    else:
        return min(n_max, len(map) // 2 - 1)


def map_sampling(map: ndarray[float], n_max: int, harmonic_domain: bool = False) -> tuple[ndarray[float], int]:
    """
    Redefined a (latitude, longitude) map definition. Eventually returns it in harmonic domain.
    Returns maximal degree as second output.
    """
    n_max = redefine_n_max(n_max=n_max, map=map)
    harmonics = SHExpandDH(
        map,
        sampling=2,
        lmax_calc=n_max,
    )
    return (
        (harmonics if harmonic_domain else MakeGridDH(harmonics, sampling=2, lmax=n_max)),
        n_max,
    )


def get_ocean_mask(name: str, n_max: int, pixels_to_coast: int = 10) -> ndarray[float] | float:
    """
    Gets the wanted ocean mask and adjusts it.
    """
    if name is None:
        return 1.0
    else:
        if name.split(".")[-1] == "csv":
            ocean_mask = extract_mask_csv(name=name)
        else:
            ocean_mask = extract_mask_nc(name=name, pixels_to_coast=pixels_to_coast)
        return round(
            a=map_sampling(
                map=ocean_mask,
                n_max=n_max,
            )[0]
        )


def extract_GRACE_data(name: str, path: Path = GRACE_data_path, skiprows: int = 11) -> ndarray:
    """
    Opens and formats GRACE (.xyz) datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(name), skiprows=skiprows, sep=";")
    # Converts to array.
    return (
        GRACE_DATA_UNIT_FACTOR
        * flip(
            m=[[value for value in df["EWH"][df["lat"] == lat]] for lat in unique(ar=df["lat"])],
            axis=0,
        )[:-1, :-1],
        flip(m=unique(df["lat"]), axis=0)[:-1],
        unique(df["lon"])[:-1],
    )


def format_GRACE_name_to_date(name: str) -> float:
    """
    Transforms GRACE level 3 solution filename into date
    """
    year_and_mounth = name.replace("GRACE_MSSA_", "").replace(".xyz", "").split("_")
    year = float(year_and_mounth[0])
    mounth = float(year_and_mounth[1])
    return year + (mounth - 1.0) / 12


def extract_all_GRACE_data(
    path: Path = GRACE_data_path, solution_name: str = "MSSA"
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Opens and formats all GRACE mounthly solutions.
    """

    # Counts files.
    filepaths = list(path.joinpath(solution_name).glob("*xyz"))

    # Gets dimensions with first file.
    map, lat, lon = extract_GRACE_data(name=filepaths[0].name, path=path.joinpath(solution_name), skiprows=11)
    all_GRACE_data = zeros(shape=(len(filepaths), len(lat), len(lon)))
    all_GRACE_data[0] = map
    times = len(filepaths) * [format_GRACE_name_to_date(filepaths[0].name)]

    # Concatenates.
    for index, filepath in enumerate(filepaths[1:]):
        all_GRACE_data[index + 1, :, :] = extract_GRACE_data(
            name=filepath.name, path=path.joinpath(solution_name), skiprows=11
        )[0]
        times[index + 1] = format_GRACE_name_to_date(filepath.name)

    # Sorts by time.
    indices = argsort(times)

    return (
        array(object=[all_GRACE_data[index, :, :] for index in indices]),
        lat,
        lon,
        array(object=[times[index] for index in indices]),
    )


def add_result_to_table(table_name: str, result_caracteristics: dict[str, str | bool | float]) -> None:
    """
    Adds a line to the wanted result table with a result informations and filename.
    """

    table_filepath = tables_path.joinpath(table_name + ".csv")
    tables_path.mkdir(exist_ok=True, parents=True)
    write = not table_filepath.exists()

    # Adds a line to the table (whether it exists or not)..
    with open(table_filepath, "a+", newline="") as file:
        writer = DictWriter(file, result_caracteristics.keys())
        if write:
            writer.writeheader()
        writer.writerow(result_caracteristics)


def save_map(
    map: ndarray[float],
    lat: ndarray[float],
    lon: ndarray[float],
    path: Path,
    filename: str,
    result_name: str = "EWH",
) -> None:
    """
    Saves a static result map in (.CSV) file.
    """
    filepath = path.joinpath(filename + ".csv")
    with open(filepath, "w", newline="") as file:
        writer = DictWriter(file, ["lat", "lon", result_name])
        writer.writeheader()
        for latitude, map_row in zip(lat, map):
            for longitude, value in zip(lon, map_row):
                writer.writerow({"lat": latitude, "lon": longitude, result_name: value})


def find_results(table_name: str, result_caracteristics: dict[str, str | bool | float]) -> list[str]:
    """
    Filters a result table by result identifier characteristics.
    """

    # Gets result informations.
    df = read_csv(filepath_or_buffer=tables_path.joinpath(table_name + ".csv"), sep=",")
    return df[
        reduce(
            lambda a, b: a & b,
            [df[key] == value for key, value in result_caracteristics.items()],
        )
    ]["ID"]


def generate_new_id(path: Path) -> str:
    """
    Accumulates the number of already existing result files.
    """
    return str(len(list(path.glob("*"))))


def load_Love_numbers_result(
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
    anelasticity_description_id: str,
) -> Result:
    """
    Loads Love number results given the parameters used to produce them.
    First, it has a look on the Love number result sum up table, where the file ID can be found.
    Then, it loads the result file as a 'Result' instance.
    """

    # Filters for parameters.
    file_ids: str = find_results(
        table_name="Love_numbers",
        result_caracteristics={
            "anelasticity_description_id": anelasticity_description_id,
            "max_tol": Love_numbers_hyper_parameters.max_tol,
            "decimals": Love_numbers_hyper_parameters.decimals,
        }
        | Love_numbers_hyper_parameters.run_hyper_parameters.__dict__
        | {
            key: value
            for key, value in Love_numbers_hyper_parameters.y_system_hyper_parameters.__dict__.items()
            if type(value) is bool
        },
    )

    # Loads result.
    Love_numbers = Result()
    Love_numbers.load(name=str(list(file_ids)[0]), path=Love_numbers_path)

    return Love_numbers
