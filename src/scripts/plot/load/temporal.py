import matplotlib.pyplot as plt
from numpy import ndarray, real
from scipy.fft import ifft

from ....utils import (
    OPTIONS,
    LoadSignalHyperParameters,
    RunHyperParameters,
    figures_path,
    get_run_folder_name,
    get_trend_dates,
    load_base_model,
    load_load_signal_hyper_parameters,
    results_path,
    signal_trend,
)


def plot_anelastic_induced_load_per_degree_per_description_per_options(
    real_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
    degrees_to_plot: list[int] = [2, 3, 4, 10, 20],
) -> None:
    """
    Generates figures showing the anelastic induced load signal trend per degree for given descriptions and options:
        - Gets already computed anelastic induced load signals per degree.
        - Computes trends.
        - Generates and saves figures in the specified subfolder.
    """
    # Loops on descriptions.
    for real_description_id in real_description_ids:
        # Loops on options.
        for run_hyper_parameters in options:
            load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters
            # Gets already computed anelastic induced load signal per degree.
            run_folder_name = (
                get_run_folder_name(real_description_id=real_description_id, run_id=run_hyper_parameters.run_id())
                + "/load/"
                + load_signal_hyper_parameters.ocean_load_Frederikse
            )
            result_subpath = results_path.joinpath(run_folder_name)
            dates = load_base_model(name="dates", path=result_subpath)
            elastic_load_signal_trend = load_base_model(name="elastic_load_signal_trend", path=result_subpath)
            elastic_load_signal = load_base_model(name="elastic_load_signal", path=result_subpath)
            anelastic_frequencial_load_signal_per_degree: dict[int, dict[str, ndarray[float]]] = {
                degree: load_base_model(
                    name=str(degree), path=result_subpath.joinpath("anelastic_induced_frequencial_load_per_degree")
                )
                for degree in degrees_to_plot
            }
            trend_indices, trend_dates = get_trend_dates(dates=dates, load_signal_hyper_parameters=load_signal_hyper_parameters)
            # Computes trends.
            anelastic_temporal_load_signal_per_degree = {
                degree: real(ifft(x=anelastic_frequencial_load_signal)) * elastic_load_signal_trend
                for degree, anelastic_frequencial_load_signal in anelastic_frequencial_load_signal_per_degree.items()
            }
            anelastic_load_signal_trend_per_degree = {
                degree: signal_trend(trend_dates=trend_dates, signal=anelastic_temporal_load_signal[trend_indices])[0]
                for degree, anelastic_temporal_load_signal in anelastic_temporal_load_signal_per_degree.items()
            }

            # Saves the figures.
            figure_subpath = figures_path.joinpath(run_folder_name)
            figure_subpath.mkdir(parents=True, exist_ok=True)

            # Whole signal.
            plt.figure(figsize=(16, 9))
            plt.plot(dates, elastic_load_signal, label="elastic")
            for degree in degrees_to_plot:
                plt.plot(
                    dates,
                    anelastic_temporal_load_signal_per_degree[degree],
                    label="degree " + str(degree),
                )
            plt.legend()
            plt.xlabel("time (y)")
            plt.grid()
            plt.title("anelastic induced load signal per degree")
            plt.legend()
            plt.savefig(figure_subpath.joinpath("anelastic_induced_load_signal_per_degree.png"))
            plt.clf()

            # Trend since first_year_for_trend.
            plt.figure(figsize=(16, 9))
            plt.plot(
                trend_dates,
                elastic_load_signal[trend_indices],
                label="elastic : trend = " + str(round(number=elastic_load_signal_trend, ndigits=5)) + "(mm/y)",
            )
            for degree in degrees_to_plot:
                plt.plot(
                    trend_dates,
                    anelastic_temporal_load_signal_per_degree[degree][trend_indices],
                    label=("degree " + str(degree))
                    + " : trend difference with elastic = "
                    + str(round(number=anelastic_load_signal_trend_per_degree[degree], ndigits=5))
                    + "(mm/y)",
                )
            plt.legend()
            plt.xlabel("time (y)")
            plt.grid()
            plt.title(
                "anelastic induced load signal per degree: trend since "
                + str(load_signal_hyper_parameters.first_year_for_trend)
            )
            plt.legend()
            plt.savefig(
                figure_subpath.joinpath(
                    "anelastic_induced_load_signal_per_degree_trend_since"
                    + str(load_signal_hyper_parameters.first_year_for_trend)
                    + ".png"
                )
            )
            plt.clf()
