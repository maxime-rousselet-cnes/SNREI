from pathlib import Path

import matplotlib.pyplot as plt
from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    conjugate,
    flip,
    linspace,
    log2,
    ndarray,
    where,
    zeros,
)
from numpy.fft import fft
from scipy import interpolate
from scipy.fftpack import fft, fftfreq, fftshift, ifft

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


def build_signal(
    ocean_charges: dict[str, list[float]],
    spline_time: int = 50,  # (y).
    zero_duration: int = 350,  # (y).
    anti_Gibbs_effect_factor: int = 1,
) -> tuple[ndarray[float], ndarray[float], float, int]:
    """
    Builds an artificial signal history that has mean value, antisymetry and no Gibbs effect.
    """
    # Creates cubic spline for antisymetry.
    mean_slope = ocean_charges["barystatic"][-1] / spline_time
    spline = lambda T: mean_slope / (2.0 * spline_time**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    # Builds signal history / Creates a constant step at zero value.
    Barystatic_extended_time_serie_past = concatenate(
        (zeros(shape=(zero_duration)), ocean_charges["barystatic"], spline(T=arange(start=-spline_time, stop=0)))
    )
    # Applies antisymetry.
    Barystatic_extended_time_serie = concatenate(
        (Barystatic_extended_time_serie_past, [0], -flip(m=Barystatic_extended_time_serie_past))
    )
    # Deduces dates axis.
    n_ocean_charge = len(Barystatic_extended_time_serie)
    dates_extended_time_serie = arange(stop=n_ocean_charge) - n_ocean_charge // 2
    # Interpolates at 2 * n minimal value for no Gibbs effect.
    n_log_min_no_Gibbs_for_ocean_charge = round(ceil(log2(n_ocean_charge)))
    half_signal_period = max(dates_extended_time_serie)
    n_interpolated_time_serie = 2 ** (n_log_min_no_Gibbs_for_ocean_charge + anti_Gibbs_effect_factor)
    dates_interpolated_time_serie = linspace(start=-half_signal_period, stop=half_signal_period, num=n_interpolated_time_serie)
    Barystatic_splines = interpolate.splrep(x=dates_extended_time_serie, y=Barystatic_extended_time_serie, k=3)
    Barystatic_interpolated_time_serie = interpolate.splev(x=dates_interpolated_time_serie, tck=Barystatic_splines)
    return (
        dates_interpolated_time_serie,
        Barystatic_interpolated_time_serie,
        2.0 * half_signal_period / (n_interpolated_time_serie - 1),
        n_interpolated_time_serie,
    )


def direct_fft(n_time_serie: int, dt: float, time_domain_signal: ndarray[float]) -> tuple[ndarray[float], ndarray[complex]]:
    """
    Computes the fast Fourier transform of a signal on time domain.
    Returns the frequencies array and the fast Fourier transform of the signal
    """
    return (
        fftfreq(n=n_time_serie, d=dt),  # (y^-1).
        fft(time_domain_signal) * 2.0 / n_time_serie,
    )


def inverse_fft(n_time_serie: float, frequency_domain_function: ndarray[complex]) -> ndarray[complex]:
    """
    Computes the inverse Fourier transform of a signal on frequency domain.
    """
    return n_time_serie * ifft(x=frequency_domain_function) / 2.0


def viscoelastic_signal_from_result_to_result(
    real_description_id: str,
    signal_hyper_parameters: SignalHyperParameters,
    signal_frequencies: ndarray[float],  # (y^-1).
    frequencial_signal: ndarray[complex],
    n_time_serie: float,
    dates_time_serie: ndarray[float],
) -> list[ndarray[complex]]:
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
    save_base_model(obj=dates_time_serie, name="signal_times", path=path)

    positive_freq_indices = where(signal_frequencies >= 0)[0]
    negative_freq_indices = where(signal_frequencies < 0)[0]
    print(len(negative_freq_indices), len(positive_freq_indices))
    print(negative_freq_indices, positive_freq_indices)
    # Interpolates Love numbers on signal positive frequencies.
    viscoelastic_factor_positive_frequencies = [
        interpolate.interp1d(
            x=Love_number_frequencies,
            y=(1.0 + (elastic_k_for_degree[0] / degree)) / (1.0 + (k_for_degree / degree)),
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )(
            x=signal_frequencies[positive_freq_indices]  # Interpolates on positive frequencies only.
        )
        for k_for_degree, elastic_k_for_degree, degree in zip(
            anelastic_Love_numbers.values[Direction.potential][BoundaryCondition.potential],
            elastic_Love_numbers.values[Direction.potential][BoundaryCondition.potential],
            degrees,
        )
    ]
    viscoelastic_factor = []  # Applies anti-symtery for real output.
    for viscoelastic_factor_positive_frequencies_for_degree in viscoelastic_factor_positive_frequencies:
        viscoelastic_factor_for_degree = zeros(shape=(n_time_serie), dtype=complex)
        viscoelastic_factor_for_degree[negative_freq_indices] = conjugate(viscoelastic_factor_positive_frequencies_for_degree)
        viscoelastic_factor_for_degree[positive_freq_indices] = viscoelastic_factor_positive_frequencies_for_degree
        viscoelastic_factor += [viscoelastic_factor_for_degree]

    # Computes viscoelastic induced signal.
    result = [
        inverse_fft(n_time_serie=n_time_serie, frequency_domain_function=frequencial_signal * viscoelastic_factor_for_degree)
        for viscoelastic_factor_for_degree in viscoelastic_factor
    ]
    # Saves as a Result instance.
    viscoelastic_induced_signal = Result(
        hyper_parameters=signal_hyper_parameters,
        values={
            Direction.potential: {
                BoundaryCondition.potential: array(
                    object=result,
                    dtype=complex,
                )
            }
        },
    )
    viscoelastic_induced_signal.save(name="viscoelastic_induced_signal", path=path)

    return result


def single_viscoelastic_signal(
    real_description_id: str,
    figure_subpath_string: str,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
) -> None:
    """
    Computes viscoelastic induced signal using already computed Love numbers and save it as a Result instance in (.JSON) file.
    """
    # Builds frequencial signal.
    dates_time_serie, time_domain_signal, dt, n_time_serie = build_signal(
        ocean_charges=extract_ocean_charge_data(),
        spline_time=signal_hyper_parameters.spline_time,
        zero_duration=signal_hyper_parameters.zero_duration,
        anti_Gibbs_effect_factor=signal_hyper_parameters.anti_Gibbs_effect_factor,
    )  # (y).
    signal_frequencies, frequencial_signal = direct_fft(
        n_time_serie=n_time_serie, dt=dt, time_domain_signal=time_domain_signal
    )  # (y^-1).
    # Gets rid of numerical errors in Fourier transform real part for real odd signal.
    frequencial_signal = 0.5 * (frequencial_signal - conjugate(frequencial_signal))

    # Gets Love numbers, computes viscoelastic induced signal and saves.
    result = viscoelastic_signal_from_result_to_result(
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        signal_frequencies=signal_frequencies,
        frequencial_signal=frequencial_signal,
        n_time_serie=n_time_serie,
        dates_time_serie=dates_time_serie,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Extended signal.
    plt.figure(figsize=(16, 10))
    plt.plot(dates_time_serie, time_domain_signal)
    plt.title(label="signal")
    plt.grid()
    plt.savefig(figure_subpath.joinpath("extended_load_signal.png"))
    plt.show()

    # Fourier transform.
    _, plots = plt.subplots(2, 1, figsize=(16, 10))
    plots[0].plot(
        fftshift(x=signal_frequencies),
        fftshift(x=frequencial_signal.real),
    )
    plots[0].set_title("signal fft")
    plots[1].plot(
        fftshift(x=signal_frequencies),
        fftshift(x=frequencial_signal.imag),
    )
    plots[0].grid()
    plots[1].grid()
    plt.savefig(figure_subpath.joinpath("fft_load_signal.png"))
    plt.show()

    # Results.
    _, plots = plt.subplots(2, 1, figsize=(16, 10))
    for i_degree in range(4):
        plots[0].plot(
            dates_time_serie,
            result[i_degree].real,
            label="elastic" if i_degree == 0 else "degree " + str(i_degree + 1),
        )
        plots[1].plot(
            dates_time_serie,
            result[i_degree].imag,
            label="elastic" if i_degree == 0 else "degree " + str(i_degree + 1),
        )
    plots[0].legend()
    plots[0].grid()
    plt.title(label="viscoelastic induced load")
    plots[1].legend()
    plots[1].grid()
    plt.savefig(figure_subpath.joinpath("viscoelastic_induced_load_signal.png"))
    plt.show()
