from cartopy.crs import Robinson
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show, tight_layout

from ...utils import extract_GRACE_data, extract_temporal_load_signal, load_load_signal_hyper_parameters
from .utils import REFERENCE_RED, natural_projection


def generate_figure_1(name: str = "TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"):
    """
    2024's article.
    """

    # Gets data.
    load_signal_hyper_parameters = load_load_signal_hyper_parameters()
    dates, mean_curb = extract_temporal_load_signal(
        name="mean",
        filename=load_signal_hyper_parameters.load_history,
    )
    _, lower_bound = extract_temporal_load_signal(
        name="lower",
        filename=load_signal_hyper_parameters.load_history,
    )
    _, upper_bound = extract_temporal_load_signal(
        name="upper",
        filename=load_signal_hyper_parameters.load_history,
    )

    map, latitudes, longitudes = extract_GRACE_data(name=name)

    # Creates subplots.
    fig = figure(figsize=(8.0, 8.0))
    ax1: GeoAxes = fig.add_subplot(2, 1, 1)
    ax2: GeoAxes = fig.add_subplot(2, 1, 2, projection=Robinson(central_longitude=0))

    # Panel A.
    ax1.plot(dates, mean_curb, color=REFERENCE_RED)
    ax1.fill_between(dates, lower_bound, upper_bound, color="grey", alpha=0.3)
    ax1.tick_params(axis="both", which="both", length=6, direction="inout")
    ax1.set_xlabel(xlabel="date (year)")
    ax1.set_ylabel(
        ylabel="Barystatic mean sea level\n(mm)",
    )

    # Panel B.
    contour = natural_projection(ax=ax2, saturation_threshold=50.0, map=map, latitudes=latitudes, longitudes=longitudes)

    # Add "A" and "B" labels in the top-left corners of each subplot inside boxes.
    for ax, panel in zip([ax1, ax2], ["A", "B"]):
        ax.text(
            -0.1,
            0.95 if panel == "A" else 0.99,
            panel,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

    # Adds legends.
    cbar = fig.colorbar(contour, ax=ax2, orientation="vertical", shrink=0.9, extend="both")
    cbar.set_label(label="GRACE/-FO Equivalent Water Height\ntrends 2002-2022 (mm/yr)")
    ax1.legend()
    ax2.legend()
    # tight_layout()
    show()
