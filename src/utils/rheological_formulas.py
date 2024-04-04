from numpy import (
    array,
    convolve,
    cumsum,
    diff,
    errstate,
    exp,
    log,
    nan_to_num,
    ndarray,
    pi,
    zeros,
)
from scipy import integrate

from .constants import SECONDS_PER_YEAR


def frequencies_to_periods(
    frequencies: ndarray | list[float],
) -> ndarray:
    """
    Converts tab from (Hz) to (y). Works also from (y) to (Hz).
    """
    return (1.0 / SECONDS_PER_YEAR) / array(frequencies)


def mu_0_computing(rho_0: ndarray, Vs: ndarray) -> ndarray:
    """
    Computes real elastic modulus mu given density rho_0 and S wave speed Vs.
    """
    return rho_0 * Vs**2


def lambda_0_computing(rho_0: ndarray, Vp: ndarray, mu_0: ndarray) -> ndarray:
    """
    Computes real elastic modulus lambda given density rho_0, P wave speed Vp and real elastic modulus mu.
    """
    return rho_0 * Vp**2 - 2 * mu_0


def g_0_computing(x: ndarray, piG: float, rho_0: ndarray, g_0_inf: float, x_inf: float, profile_precision: int) -> ndarray:
    """
    Integrates the internal mass GM to get gravitational acceleration g.
    """
    # Trapezoidal rule integral method for GM = integral(rho_0 G dV).
    GdV_spherical = 4.0 / 3.0 * piG * diff(x**3)
    mean_rho = convolve(a=rho_0, v=[0.5, 0.5])[1:-1]
    dGM_0 = zeros(shape=(profile_precision))
    dGM_0[0] = g_0_inf * x_inf**2
    dGM_0[1:] = mean_rho * GdV_spherical
    GM_0 = cumsum(dGM_0)
    g_0 = zeros(shape=(profile_precision))
    g_0[0] = g_0_inf
    g_0[1:] = GM_0[1:] / x[1:] ** 2  # To avoid division by 0 for first point.
    return g_0


def mu_k_computing(mu_K1: ndarray, c: ndarray, mu_0: ndarray) -> ndarray:
    """
    Computes Kelvin's equivalent elastic modulus given the parameters mu_K1, c, and real elastic modulus mu_0.
    """
    return mu_K1 + c * mu_0


def omega_cut_computing(mu: ndarray, eta: ndarray) -> ndarray[complex]:
    """
    Computes cut frequency value given the real elastic modulus mu and viscosity eta. Handles infinite viscosities.
    """
    with errstate(divide="ignore"):
        return nan_to_num(x=mu / eta, nan=0.0)


def build_cutting_omegas(variables: dict[str, ndarray[complex]]) -> dict[str, ndarray[complex]]:
    """
    Builds a dictionary containing cut frequencies.
    """
    return {
        "omega_cut_m": omega_cut_computing(
            mu=variables["mu"],
            eta=variables["eta_m"],
        ),
        "omega_cut_k": omega_cut_computing(
            mu=variables["mu_k"],
            eta=variables["eta_k"],
        ),
        "omega_cut_b": omega_cut_computing(
            mu=variables["mu"],
            eta=variables["eta_k"],
        ),
    }


def m_prime_computing(omega_cut_m: ndarray[complex], omega_j: complex) -> ndarray[complex]:
    """
    Computes m_prime transfert function value given the Maxwell's cut pulsation omega_cut_m, and pulsation value omega.
    """
    return omega_cut_m / (omega_cut_m + omega_j)


def b_computing(omega_cut_m: ndarray, omega_cut_k: ndarray, omega_cut_b: ndarray, omega_j: complex) -> ndarray[complex]:
    """
    Computes b transfert function value given the Maxwell's, Kelvin's and Burgers cut frequencies omega_cut_m, omega_cut_k and
    omega_cut_b and pulsation value omega.
    """
    return (omega_j * omega_cut_b) / ((omega_j + omega_cut_k) * (omega_j + omega_cut_m))


def lambda_computing(
    mu_complex: ndarray[complex],
    lambda_complex: ndarray[complex],
    m_prime: ndarray[complex],
    b: ndarray[complex],
) -> ndarray[complex]:
    """
    Computes complex analog lambda values, given the complex elastic moduli mu and lambda and m_prime and b transfert function
    values at pulsation value omega.
    """
    return lambda_complex + (2.0 / 3.0) * mu_complex * (m_prime + b) / (1 + b)


def mu_computing(
    mu_complex: ndarray[complex],
    m_prime: ndarray[complex],
    b: ndarray[complex],
) -> ndarray[complex]:
    """
    Computes complex analog mu values, given the complex elastic modulus mu and m_prime and b transfert function values at
    pulsation value omega.
    """
    return mu_complex * (1 - m_prime) / (1 + b)


def f_attenuation_computing(
    omega_m_tab: ndarray,
    tau_M_tab: ndarray,
    alpha_tab: ndarray,
    omega: float,
    frequency: float,
    frequency_unit: float,
    use_bounded_attenuation_functions: bool,
) -> ndarray[complex]:
    """
    computes the attenuation function f using parameters omega_m and alpha.
    omega_m is a unitless frequency.
    frequency is a unitless frequency.
    omega is a unitless pulsation.
    """
    if use_bounded_attenuation_functions:
        tau = lambda tau_log: exp(tau_log)
        Y = lambda tau_log, alpha: (tau(tau_log=tau_log) ** alpha)
        denom = lambda tau_log, omega: (1.0 + (omega * tau(tau_log=tau_log) * 1.0j))
        integrand = lambda tau_log, omega, alpha: Y(tau_log=tau_log, alpha=alpha) / denom(tau_log=tau_log, omega=omega)
        with errstate(invalid="ignore", divide="ignore"):
            return array(
                [
                    (
                        0.0
                        if omega_m <= 0.0 or tau_M <= 0.0
                        else -integrate.quad(
                            func=integrand,
                            a=log(1.0 / omega_m),
                            b=log(tau_M),
                            args=(omega, alpha),
                            complex_func=True,
                        )[0]
                    )
                    for alpha, omega_m, tau_M in zip(alpha_tab, omega_m_tab, tau_M_tab)
                ]
            )
    else:
        high_frequency_domain: ndarray[bool] = frequency >= omega_m_tab
        omega_0 = 1.0 / frequency_unit  # (Unitless frequency).
        with errstate(invalid="ignore", divide="ignore"):
            return nan_to_num(  # Alpha or omega_m may be equal to 0.0, meaning no attenuation should be taken into account.
                x=((2.0 / pi) * log(frequency / omega_0) + 1.0j) * high_frequency_domain
                + (
                    (2.0 / pi) * (log(omega_m_tab / omega_0) + (1 / alpha_tab) * (1 - (omega_m_tab / frequency) ** alpha_tab))
                    + (omega_m_tab / frequency) ** alpha_tab * 1.0j
                )
                * (1 - high_frequency_domain),
                nan=0.0,
            )


def delta_mu_computing(mu_0: ndarray, Qmu: ndarray, f: ndarray[complex]) -> ndarray[complex]:
    """
    Computes the first order frequency dependent variation from elasticity delta_mu at frequency value frequency, given the real
    elastic modulus mu_0, the elasticicty's quality factor Qmu and attenuation function f.
    """
    with errstate(invalid="ignore", divide="ignore"):
        return nan_to_num(  # Qmu may be equal to infinity, meaning no attenuation should be taken into account.
            x=(mu_0 / Qmu) * f,
            nan=0.0,
        )


def find_tau_M(omega_m: float, alpha: float, asymptotic_mu_ratio: float, Q_mu: float) -> float:
    """
    Uses asymptotic equation to find tau_M such as
    """
    with errstate(invalid="ignore"):
        return (
            0.0
            if round(number=asymptotic_mu_ratio, ndigits=8) == 1.0 or alpha == 0.0
            else (alpha * (1.0 - asymptotic_mu_ratio) * Q_mu + omega_m ** (-alpha)) ** (1.0 / alpha)
        )
