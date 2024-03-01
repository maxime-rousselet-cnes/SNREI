import matplotlib.pyplot as plt
from numpy import linspace
from scipy.fft import fft, fftfreq, fftshift

from .classes import SignalHyperParameters
from .database import load_base_model
from .paths import figures_path, parameters_path
from .signal import (
    build_load_signal,
    extract_land_sea_weights,
    extract_ocean_load_data,
    spatial_viscoelastic_load_signal,
    viscoelastic_load_signal_from_result_to_result,
)


def single_viscoelastic_load_signal(
    real_description_id: str,
    figure_subpath_string: str,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
    degrees_to_plot: list[int] = [1, 2, 3, 5, 10, 20],
    last_year: int = 2018,
    last_years_for_trend: int = 15,
) -> None:
    """
    Computes viscoelastic induced signal using already computed Love numbers and save it as a Result instance in (.JSON) file.
    """
    # Builds frequencial signal.
    dates, time_domain_signal, time_step = build_load_signal(
        ocean_loads=extract_ocean_load_data() if signal_hyper_parameters.signal == "ocean_load" else (),  # TODO.
        signal_hyper_parameters=signal_hyper_parameters,
    )  # (y).
    # Gets signal's Fourier tranform.
    frequencial_domain_signal = fft(x=time_domain_signal)
    frequencies = fftfreq(n=len(frequencial_domain_signal), d=time_step)

    # Gets Love numbers, computes viscoelastic induced signal and saves.
    _, result, degrees, trend_dates, trends, mean_trends = viscoelastic_load_signal_from_result_to_result(
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        frequencies=frequencies,
        frequencial_domain_signal=frequencial_domain_signal,
        dates=dates,
        last_year=last_year,
        last_years_for_trend=last_years_for_trend,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Extended signal.
    plt.figure(figsize=(16, 10))
    plt.plot(dates, time_domain_signal)
    plt.title(label="signal")
    plt.xlabel("time (y)")
    plt.ylabel("(mm)")
    plt.grid()
    plt.savefig(figure_subpath.parent.joinpath("extended_load_signal.png"))
    plt.show(block=False)

    # Fourier transform.
    _, plots = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plots[0].plot(
        fftshift(x=frequencies),
        fftshift(x=frequencial_domain_signal.real),
    )
    plots[0].set_title("signal fft")
    plots[1].plot(
        fftshift(x=frequencies),
        fftshift(x=frequencial_domain_signal.imag),
    )
    plots[0].grid()
    plots[1].grid()
    plt.xlabel("(y^-1)")
    plt.savefig(figure_subpath.parent.joinpath("fft_load_signal.png"))
    plt.show(block=False)

    # Results.
    degrees_indices = [list(degrees).index(degree) for degree in degrees_to_plot]
    plt.figure(figsize=(16, 10))
    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
        plt.plot(
            dates,
            result[i_degree].real,
            label="elastic" if i_degree == 0 else "degree " + str(degree),
        )
    plt.legend()
    plt.xlabel("time (y)")
    plt.grid()
    plt.title("viscoelastic induced load")
    plt.legend()
    plt.savefig(figure_subpath.joinpath("viscoelastic_induced_load_signal.png"))
    plt.show(block=False)

    # Trend since last_year - last_years_for_trend.
    plt.figure(figsize=(16, 10))
    for i_degree, degree in zip(degrees_indices, degrees_to_plot):
        plt.plot(
            trend_dates,
            trends[i_degree],
            label=("elastic" if i_degree == 0 else "degree " + str(degree))
            + " - trend = "
            + str(mean_trends[i_degree])
            + "(mm/y)",
        )
    plt.legend()
    plt.xlabel("time (y)")
    plt.grid()
    plt.title("viscoelastic induced load - trend since " + str(last_year - last_years_for_trend))
    plt.legend()
    plt.savefig(figure_subpath.joinpath("viscoelastic_induced_load_signal_trend.png"))
    plt.show()


def single_spatial_viscoelastic_load_signal(
    real_description_id: str,
    figure_subpath_string: str,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
    last_year: int = 2018,
    last_years_for_trend: int = 15,
) -> None:
    """
    Computes viscoelastic induced signal using already computed Love numbers and save it as a Result instance in (.JSON) file.
    Applies spacial dependency to the mean oceanic load signal in harmonic domain, saves harmonic and spatial datas.
    May save corresponding figures in specified subfolder.
    """
    # Builds frequencial signal.
    dates, time_domain_signal, time_step = build_load_signal(
        ocean_loads=extract_ocean_load_data() if signal_hyper_parameters.signal == "ocean_load" else (),  # TODO.
        signal_hyper_parameters=signal_hyper_parameters,
    )  # (y).
    # Gets signal's Fourier tranform.
    frequencial_domain_signal = fft(x=time_domain_signal)
    frequencies = fftfreq(n=len(frequencial_domain_signal), d=time_step)
    # Gets weights map.
    harmonic_weights = (
        extract_land_sea_weights(n_max=signal_hyper_parameters.n_max) if signal_hyper_parameters.weights_map == "mask" else ()
    )

    # Builds spatial signal.
    spatial_result = spatial_viscoelastic_load_signal(
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        frequencies=frequencies,
        frequencial_domain_signal=frequencial_domain_signal,
        dates=dates,
        harmonic_weights=harmonic_weights,
        last_year=last_year,
        last_years_for_trend=last_years_for_trend,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Plots.
    min_spatial_result = min([min(row) for row in spatial_result])
    max_spatial_result = max([max(row) for row in spatial_result])
    boundaries = linspace(start=min_spatial_result, stop=max_spatial_result, num=10)
    plt.colorbar(
        plt.imshow(spatial_result),
        boundaries=boundaries,
    )
    plt.title("viscoelastic induced load - trend since " + str(last_year - last_years_for_trend))
    plt.savefig(figure_subpath.joinpath(signal_hyper_parameters.weights_map + "_" + signal_hyper_parameters.signal + ".png"))
    plt.show()
