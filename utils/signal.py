from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
from matplotlib import cm, colors
from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    conjugate,
    expand_dims,
    flip,
    linspace,
    log2,
    ndarray,
    real,
    round,
    sum,
    where,
    zeros,
)
from pyshtools.expand import MakeGridDH, SHExpandDH
from scipy import interpolate
from scipy.fft import fft, fftfreq, fftshift, ifft

from .classes import BoundaryCondition, Direction, Result, SignalHyperParameters
from .constants import SECONDS_PER_YEAR
from .database import load_base_model, save_base_model
from .Love_numbers import gets_run_id
from .paths import data_path, figures_path, parameters_path, results_path


def extract_ocean_charge_data(path: Path = data_path) -> dict[str, list[float]]:
    """
    Opens Frederik et al. file and formats its data.
    """
    with open(path.joinpath("datafrederik")) as file:
        lines = array([[float(value) for value in line.rstrip().split("\t")] for line in list(file)[:-1]])
    return {"dates": lines[:, 0], "barystatic": lines[:, 1] - min(lines[:, 1])}


def extract_land_sea_weights(n_max: int, path: Path = data_path) -> ndarray:
    """
    Opens NASA's nc file for land/sea mask and formats its data.
    """
    # Gets raw data.
    ds = netCDF4.Dataset(path.joinpath("IMERG_land_sea_mask.nc"))
    land_sea_mask = flip(ds.variables["landseamask"], axis=0).data
    # Sets mean as zero and max as one by homothety.
    sum_land_sea_mask = sum(sum(land_sea_mask))
    max_land_sea_mask = max([max(row) for row in land_sea_mask])
    n_t = len(ds.variables["lon"]) * len(ds.variables["lat"])
    land_sea_weights = n_t * land_sea_mask / (n_t * max_land_sea_mask - sum_land_sea_mask) + sum_land_sea_mask / (
        sum_land_sea_mask - max_land_sea_mask * n_t
    )
    # Gets harmionics.
    return SHExpandDH(land_sea_weights, sampling=2, lmax_calc=min(n_max, (len(land_sea_weights) - 1) // 2))


def build_signal(
    ocean_charges: dict[str, list[float]],
    signal_hyper_parameters: SignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], float]:
    """
    Builds an artificial signal history that has mean value, antisymetry and no Gibbs effect.
    """
    # Creates cubic spline for antisymetry.
    mean_slope = ocean_charges["barystatic"][-1] / signal_hyper_parameters.spline_time
    spline = lambda T: mean_slope / (2.0 * signal_hyper_parameters.spline_time**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    # Builds signal history / Creates a constant step at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(signal_hyper_parameters.zero_duration)),
            ocean_charges["barystatic"],
            spline(T=arange(start=-signal_hyper_parameters.spline_time, stop=0)),
        )
    )
    # Applies antisymetry.
    extended_time_serie = concatenate((extended_time_serie_past, [0], -flip(m=extended_time_serie_past)))
    # Deduces dates axis.
    n_extended_signal = len(extended_time_serie)
    extended_dates = arange(stop=n_extended_signal) - n_extended_signal // 2
    # Interpolates at sufficient sampling for no Gibbs effect.
    n_log_min_no_Gibbs = round(ceil(log2(n_extended_signal)))
    half_signal_period = max(extended_dates)
    n_signal = int(2 ** (n_log_min_no_Gibbs + signal_hyper_parameters.anti_Gibbs_effect_factor))
    dates = linspace(-half_signal_period, stop=half_signal_period, num=n_signal)
    signal_splines = interpolate.splrep(x=extended_dates, y=extended_time_serie, k=3)
    return (
        dates,
        interpolate.splev(x=dates, tck=signal_splines),  # Signal.
        2.0 * half_signal_period / n_signal,  # Time step.
    )


def builds_hermitian(signal: ndarray[complex]) -> ndarray[complex]:
    """
    For a given signal defined for positive values, builds the extended signal from it that has hermitian symetry.
    """
    return concatenate((conjugate(flip(m=signal)), signal))


def viscoelastic_signal_from_result_to_result(
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    frequencies: ndarray[float],  # (y^-1).
    frequencial_domain_signal: ndarray[complex],
    dates: ndarray[float],
    last_year: int,
    last_years_for_trend: int,
) -> tuple[Path, ndarray[complex], ndarray[int], ndarray[float], ndarray[float], ndarray[float]]:
    """
    Gets already computed Love numbers, computes viscoelastic induced signal and save it as a Result instance in (.JSON) file.
    """
    # Gets Love numbers.
    base_path = results_path.joinpath(real_description_id)
    path = base_path.joinpath("runs").joinpath(
        gets_run_id(
            use_anelasticity=signal_hyper_parameters.use_anelasticity,
            bounded_attenuation_functions=signal_hyper_parameters.bounded_attenuation_functions,
            use_attenuation=signal_hyper_parameters.use_attenuation,
        )
    )
    elastic_Love_numbers, anelastic_Love_numbers = Result(), Result()
    elastic_Love_numbers.load(name="elastic_Love_numbers", path=base_path)
    degrees = array(load_base_model(name="degrees", path=base_path))
    anelastic_Love_numbers.load(name="anelastic_Love_numbers", path=path)
    Love_number_frequencies = SECONDS_PER_YEAR * array(load_base_model("frequencies", path=path))  # (y^-1).

    # Saves dates time serie.
    save_base_model(obj=dates, name="signal_times", path=path)

    # Interpolates Love numbers on signal positive frequencies.
    viscoelastic_factors = [
        interpolate.interp1d(
            x=concatenate((-flip(m=Love_number_frequencies), Love_number_frequencies)),
            y=builds_hermitian(signal=(1.0 + (elastic_k_for_degree[0] / degree)) / (1.0 + (k_for_degree / degree))),
            kind="linear",
        )(
            x=frequencies,
        )
        for k_for_degree, elastic_k_for_degree, degree in zip(
            anelastic_Love_numbers.values[Direction.potential][BoundaryCondition.load],
            elastic_Love_numbers.values[Direction.potential][BoundaryCondition.load],
            degrees,
        )
    ]

    # Computes viscoelastic induced signal.
    result = array(
        object=[
            real(ifft(x=frequencial_domain_signal * viscoelastic_factor_for_degree))
            for viscoelastic_factor_for_degree in viscoelastic_factors
        ],
        dtype=float,
    )

    # Saves as a Result instance.
    viscoelastic_induced_signal = Result(
        hyper_parameters=signal_hyper_parameters,
        values={
            Direction.potential: {
                BoundaryCondition.load: result,
            }
        },
    )
    viscoelastic_induced_signal.save(name="viscoelastic_induced_signal", path=path)

    # Gets last years trends.
    shift_dates = dates + signal_hyper_parameters.spline_time + last_year
    trend_indices = where((shift_dates <= last_year) * (shift_dates >= last_year - last_years_for_trend))[0]
    trends = result[:, trend_indices].real - expand_dims(a=result[:, trend_indices[0]].real, axis=1)

    return path, result, degrees, shift_dates[trend_indices], trends, trends[:, -1] / last_years_for_trend


def spatial_viscoelastic_signal(
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    frequencies: ndarray[float],  # (y^-1).
    frequencial_domain_signal: ndarray[complex],
    dates: ndarray[float],
    harmonic_weights: ndarray,
    last_year: int,
    last_years_for_trend: int,
) -> ndarray:
    """
    Computes the spatially dependent viscoelastic induced load and saves it in a (.JSON) file.
    """

    # Gets Love numbers, computes viscoelastic induced signal and saves.
    path, _, degrees, _, _, mean_trends = viscoelastic_signal_from_result_to_result(
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        frequencies=frequencies,
        frequencial_domain_signal=frequencial_domain_signal,
        dates=dates,
        last_year=last_year,
        last_years_for_trend=last_years_for_trend,
    )

    # Interpolates in degrees, for each time step.
    all_degrees = arange(stop=len(harmonic_weights[0]))
    # Gets trend value.
    degree_splines = interpolate.splrep(x=degrees, y=mean_trends, k=3)
    interpolated_mean_trends = array(
        object=[expand_dims(a=interpolate.splev(x=all_degrees, tck=degree_splines, ext=0.0), axis=1)], dtype=float
    )
    # Computes result.
    harmonic_result = interpolated_mean_trends * harmonic_weights

    # Saves in (.JSON) file.
    save_base_model(obj=harmonic_result, name="harmonic_viscoelastic_induced_load", path=path)

    return MakeGridDH(harmonic_result, sampling=2)


def single_viscoelastic_signal(
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
    dates, time_domain_signal, time_step = build_signal(
        ocean_charges=extract_ocean_charge_data() if signal_hyper_parameters.signal == "ocean_charge" else (),  # TODO.
        signal_hyper_parameters=signal_hyper_parameters,
    )  # (y).
    # Gets signal's Fourier tranform.
    frequencial_domain_signal = fft(x=time_domain_signal)
    frequencies = fftfreq(n=len(frequencial_domain_signal), d=time_step)

    # Gets Love numbers, computes viscoelastic induced signal and saves.
    _, result, degrees, trend_dates, trends, mean_trends = viscoelastic_signal_from_result_to_result(
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


def single_spatial_viscoelastic_signal(
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
    dates, time_domain_signal, time_step = build_signal(
        ocean_charges=extract_ocean_charge_data() if signal_hyper_parameters.signal == "ocean_charge" else (),  # TODO.
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
    spatial_result = spatial_viscoelastic_signal(
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
