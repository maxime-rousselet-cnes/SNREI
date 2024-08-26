from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi

from ...utils import tables_path
from ...utils.classes import EARTH_RADIUS


def plot_ocean_mean_vs_pixels(file_name: str = "harmonic_load_signal_trends", folder_path: Path = tables_path) -> None:
    """
    Plots `ocean_mean_step_3` with respect to `pixels_to_coast` for each combination of
    `leakage_correction_iterations`, `erode_high_signal_zones`, and `Love_numbers_ID`.

    Parameters:
    - file_name (str): The name of the CSV file without the ".csv" extension.
    - folder_path (Path): The path to the folder containing the CSV file.

    Returns:
    - None
    """
    # Construct the full file path
    file_path = folder_path / f"{file_name}.csv"

    # Load the data
    df = pd.read_csv(file_path)

    # Set up the unique values for subplots
    erode_high_signal_zones_values = df["erode_high_signal_zones"].unique()
    Love_numbers_ID_values = df["Love_numbers_ID"].unique()

    # Create the figure and axes
    fig, axes = plt.subplots(len(erode_high_signal_zones_values), len(Love_numbers_ID_values), figsize=(12, 8), sharex=True, sharey=True)

    # Ensure axes is always a 2D array, even if there's only one subplot
    if len(erode_high_signal_zones_values) == 1:
        axes = [axes]
    if len(Love_numbers_ID_values) == 1:
        axes = [[ax] for ax in axes]

    # Plotting loop
    for i, erode_value in enumerate(erode_high_signal_zones_values):
        for j, love_value in enumerate(Love_numbers_ID_values):
            ax = axes[i][j]
            subset = df[(df["erode_high_signal_zones"] == erode_value) & (df["Love_numbers_ID"] == love_value)]

            # Plot a line for each `leakage_correction_iterations`
            for iteration in subset["leakage_correction_iterations"].unique():
                data = subset[subset["leakage_correction_iterations"] == iteration]
                ax.plot(EARTH_RADIUS * pi / 180 * data["pixels_to_coast"], data["ocean_mean_step_3"], label=f"Iteration {iteration}")

            # Set titles and labels
            ax.set_title(f"Erode: {erode_value}, Love: {love_value}")
            ax.set_xlabel("Distance to Coast (km)")
            ax.set_ylabel("Ocean Mean Step 3")

    # Add legends and adjust layout
    for ax_row in axes:
        for ax in ax_row:
            ax.legend()
    plt.tight_layout()
    plt.show()
