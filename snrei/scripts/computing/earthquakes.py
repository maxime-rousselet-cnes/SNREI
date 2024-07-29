from pathlib import Path

from numpy import array

from ...functions import signal_trend
from ...utils import GRACE_data_path, extract_all_GRACE_data, remove_earthquakes, save_map, signal_trend


def process_for_earthquakes(
    path: Path = GRACE_data_path,
    solution_name: str = "MSSA",
    save_path: Path = GRACE_data_path,
    result_filename: str = "TREND_MSSA_PROCESSED_FOR_EARTHQUAKES",
    result_name: str = "EWH",
) -> None:
    """
    Removes Earthquakes effects from GRACE level-3 data. Saves the resulting trends.
    """

    # Extracts level-3 data.
    time_dependent_maps, lat, lon, times = extract_all_GRACE_data(path=path, solution_name=solution_name)

    # Gets all earthquakes parameters and remove their effects from level-3 data.
    _, corrected_time_dependent_maps = remove_earthquakes(
        time_dependent_maps=time_dependent_maps,
        times=times,
        lat=lat,
        lon=lon,
    )

    # Computes trends.
    processed_data = array(
        object=[
            [signal_trend(trend_dates=times, signal=time_serie)[0] for time_serie in time_series]
            for time_series in corrected_time_dependent_maps.transpose([1, 2, 0])
        ],
    )

    # Saves the trends.
    save_map(map=processed_data, lat=lat, lon=lon, path=save_path, filename=result_filename, result_name=result_name)
