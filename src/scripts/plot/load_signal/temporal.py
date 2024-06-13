import matplotlib.pyplot as plt
from numpy import array, ndarray, real
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
    load_subpath,
    results_path,
    signal_trend,
)


def plot_anelastic_load_per_degree_per_description_per_options(
    anelasticity_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
    degrees_to_plot: list[int] = [2, 3, 4, 10, 20],
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Generates figures showing the anelastic induced load signal trend per degree for given descriptions and options:
        - Gets already computed anelastic induced load signals per degree.
        - Computes trends.
        - Generates and saves figures in the specified subfolder.
    """
    # Loops on descriptions.
    for anelasticity_description_id in anelasticity_description_ids:
        # Loops on options.
        for run_hyper_parameters in options:
            load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters
            # Gets already computed anelastic induced load signal per degree.
            run_folder_name = get_run_folder_name(
                anelasticity_description_id=anelasticity_description_id,
                run_id=run_hyper_parameters.run_id(),
            )
            result_subpath = load_subpath(
                path=results_path.joinpath(run_folder_name),
                load_signal_hyper_parameters=load_signal_hyper_parameters,
            )
            signal_dates = load_base_model(name="signal_dates", path=result_subpath)
            elastic_load_signal_trend = load_base_model(
                name="elastic_load_signal_trend", path=result_subpath
            )
            frequencial_elastic_normalized_load_signal = load_base_model(
                name="frequencial_elastic_normalized_load_signal", path=result_subpath
            )
            elastic_load_signal = (
                real(
                    ifft(
                        x=array(
                            object=frequencial_elastic_normalized_load_signal["real"]
                            + 1.0j
                            * array(
                                object=frequencial_elastic_normalized_load_signal[
                                    "imag"
                                ]
                            )
                        )
                    )
                )
                * elastic_load_signal_trend
            )
            frequencial_anelastic_load_signal_per_degree: dict[
                int, dict[str, ndarray[float]]
            ] = {
                degree: load_base_model(
                    name=str(degree),
                    path=result_subpath.joinpath(
                        "anelastic_frequencial_load_signal_per_degree"
                    ),
                )
                for degree in degrees_to_plot
            }
            trend_indices, trend_dates = get_trend_dates(
                signal_dates=signal_dates,
                load_signal_hyper_parameters=load_signal_hyper_parameters,
            )
            # Computes trends.
            anelastic_temporal_load_signal_per_degree = {
                degree: real(
                    ifft(
                        x=array(object=anelastic_frequencial_load_signal["real"])
                        + 1.0j * array(object=anelastic_frequencial_load_signal["imag"])
                    )
                )
                * elastic_load_signal_trend
                for degree, anelastic_frequencial_load_signal in frequencial_anelastic_load_signal_per_degree.items()
            }
            anelastic_load_signal_trend_per_degree: dict[int, float] = {
                degree: signal_trend(
                    trend_dates=trend_dates,
                    signal=anelastic_temporal_load_signal[trend_indices],
                )[0]
                for degree, anelastic_temporal_load_signal in anelastic_temporal_load_signal_per_degree.items()
            }

            # Saves the figures.
            figure_subpath = load_subpath(
                path=figures_path.joinpath(run_folder_name),
                load_signal_hyper_parameters=load_signal_hyper_parameters,
            )
            figure_subpath.mkdir(parents=True, exist_ok=True)

            # Whole signal.
            plt.figure(figsize=figsize)
            plt.plot(signal_dates, elastic_load_signal, label="elastic")
            for degree in degrees_to_plot:
                plt.plot(
                    signal_dates,
                    anelastic_temporal_load_signal_per_degree[degree],
                    label="degree " + str(degree),
                )
            plt.legend()
            plt.xlabel("time (yr)")
            plt.grid()
            plt.title("anelastic induced load signal per degree")
            plt.legend()
            plt.savefig(figure_subpath.joinpath("anelastic_load_signal_per_degree.png"))
            plt.clf()

            # Trend since first_year_for_trend.
            plt.figure(figsize=figsize)
            plt.plot(
                trend_dates,
                elastic_load_signal[trend_indices]
                - elastic_load_signal[trend_indices][0],
                label="elastic : trend = "
                + str(round(number=elastic_load_signal_trend, ndigits=5))
                + "(mm/y)",
            )
            for degree in degrees_to_plot:
                plt.plot(
                    trend_dates,
                    anelastic_temporal_load_signal_per_degree[degree][trend_indices]
                    - anelastic_temporal_load_signal_per_degree[degree][trend_indices][
                        0
                    ],
                    label=("degree " + str(degree))
                    + " : trend = "
                    + str(
                        round(
                            number=anelastic_load_signal_trend_per_degree[degree],
                            ndigits=5,
                        )
                    )
                    + "(mm/y)",
                )
            plt.legend()
            plt.xlabel("time (yr)")
            plt.grid()
            plt.title(
                "anelastic induced load signal per degree: trend since "
                + str(load_signal_hyper_parameters.first_year_for_trend)
            )
            plt.legend()
            plt.savefig(
                figure_subpath.joinpath(
                    "anelastic_load_signal_per_degree_trend_since"
                    + str(load_signal_hyper_parameters.first_year_for_trend)
                    + ".png"
                )
            )
            plt.clf()
