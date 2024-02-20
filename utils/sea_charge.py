from pathlib import Path

from numpy import arange, array, ceil, concatenate, flip, linspace, log2, ndarray, zeros
from scipy import interpolate
from scipy.fftpack import fft, fftshift, ifft

from .paths import data_path


def extract_sea_level_data(path: Path = data_path) -> dict[str, list[float]]:
    """
    Opens and formats Frederik et al. file.
    """
    with open(path.joinpath("datafrederik")) as file:
        lines = array([[float(value) for value in line.rstrip().split("\t")] for line in list(file)[:-1]])
    return {"dates": lines[:, 0], "Barystatic": lines[:, 1]}


def build_sea_level_signal(
    sea_levels: dict[str, list[float]],
    spline_time: int = 50,  # (y).
    zero_duration: int = 350,  # (y).
    anti_Gibbs_effect_factor: int = 1,
) -> tuple[ndarray[float], ndarray[float]]:
    """
    Builds an artificial sea level signal history that has mean value, antisymetry and no Gibbs effect.
    """
    # Creates cubic spline for antisymetry.
    mean_slope = sea_levels["barystatic"][-1] / spline_time
    spline = lambda T: mean_slope / (2.0 * spline_time**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    # Creates a constant step at zero value.
    # Builds signal history.
    Barystatic_extended_time_serie_past = concatenate(
        (zeros(shape=(zero_duration)), sea_levels["barystatic"], spline(T=arange(start=-spline_time, stop=0)))
    )
    # Applies antisymetry.
    Barystatic_extended_time_serie = concatenate(
        (Barystatic_extended_time_serie_past, [0], -flip(m=Barystatic_extended_time_serie_past))
    )
    # Deduces dates axis.
    n_sea_level = len(Barystatic_extended_time_serie)
    dates_extended_time_serie = arange(stop=n_sea_level) - n_sea_level // 2
    # Interpolates at 2 * n minimal value for no Gibbs effect.
    n_log_min_no_Gibbs_for_sea_level = round(ceil(log2(n_sea_level)))
    half_signal_period = max(dates_extended_time_serie)
    n_interpolated_time_serie = 2 ** (n_log_min_no_Gibbs_for_sea_level + anti_Gibbs_effect_factor)
    dates_interpolated_time_serie = linspace(start=-half_signal_period, stop=half_signal_period, num=n_interpolated_time_serie)
    Barystatic_splines = interpolate.splrep(x=dates_extended_time_serie, y=Barystatic_extended_time_serie, k=3)
    Barystatic_interpolated_time_serie = interpolate.splev(x=dates_interpolated_time_serie, tck=Barystatic_splines)
    return dates_interpolated_time_serie, Barystatic_interpolated_time_serie


def direct_fft(
    n_time_serie: int, half_signal_period: float, time_domain_function: ndarray[float]
) -> tuple[ndarray[float], ndarray[complex], float]:
    """
    Computes the fast Fourier transform of a function on time domain.
    Returns:
        - symetric frequencies array
        - fast Fourier transform
        - normalisation coefficient needed for inverse transform.
    """
    norm_coefficient = n_time_serie / 2.0
    max_frequency = n_time_serie / (4.0 * half_signal_period)
    return (
        linspace(start=-max_frequency, stop=max_frequency, num=n_time_serie),
        fftshift(fft(time_domain_function)) / norm_coefficient,
        norm_coefficient,
    )


def inverse_fft(norm_coef: float, frequency_domain_function: ndarray[complex]) -> ndarray[complex]:
    """
    Computes the inverse Fourier transform of a signal on frequency domain.
    """
    return norm_coef * ifft(fftshift(x=frequency_domain_function))
