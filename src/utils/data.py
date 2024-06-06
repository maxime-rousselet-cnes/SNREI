from csv import DictWriter
from pathlib import Path

from numpy import argsort, array, flip, ndarray, unique, zeros
from pandas import read_csv

from .classes import (
    GRACE_DATA_UNIT_FACTOR,
    GRACE_data_path,
    frequencies_path,
    tables_path,
)
from .database import save_base_model


def extract_GRACE_data(name: str, path: Path = GRACE_data_path, skiprows: int = 11) -> ndarray:
    """
    Opens and formats GRACE (.xyz) datafile.
    """
    # Gets raw data.
    df = read_csv(filepath_or_buffer=path.joinpath(name), skiprows=skiprows, sep=";")
    # Converts to array.
    return (
        GRACE_DATA_UNIT_FACTOR
        * flip(m=[[value for value in df["EWH"][df["lat"] == lat]] for lat in unique(ar=df["lat"])], axis=0),
        flip(m=unique(df["lat"]), axis=0),
        unique(df["lon"]),
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

    # Adds a line to the table (whether it exists or not)..
    with open(table_filepath, "a+", newline="") as file:
        writer = DictWriter(file, result_caracteristics.keys())
        if not table_filepath.exists():
            writer.writeheader()
        writer.writerow(result_caracteristics)


def save_frequencies(log_frequency_values: ndarray[float], frequency_unit: float) -> None:
    """
    Maps back log unitless frequencies to (Hz) and save to (.JSON) file.
    """
    save_base_model(obj=10.0**log_frequency_values * frequency_unit, name="frequencies", path=frequencies_path)


def save_map(
    map: ndarray[float], lat: ndarray[float], lon: ndarray[float], path: Path, filename: str, result_name: str
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
