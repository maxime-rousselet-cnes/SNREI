from numpy import convolve, cumsum, diff, errstate, log, nan_to_num, ndarray, pi, zeros


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


def m_prime_computing(omega_cut_m: ndarray, omega_j: complex) -> ndarray[complex]:
    """
    Computes m_prime transfert function value given the Maxwell's cut frequency omega_cut_m, and frequency value omega.
    """
    return omega_cut_m / (omega_cut_m + omega_j)


def b_computing(omega_cut_m: ndarray, omega_cut_k: ndarray, omega_cut_b: ndarray, omega_j: complex) -> ndarray[complex]:
    """
    Computes b transfert function value given the Maxwell's, Kelvin's and Burgers cut frequencies omega_cut_m, omega_cut_k and
    omega_cut_b and frequency value omega.
    """
    return (omega_j * omega_cut_b) / ((omega_j + omega_cut_k) * (omega_j + omega_cut_m))


def lambda_computing(
    mu_0: ndarray[complex],
    lambda_0: ndarray[complex],
    m_prime: ndarray[complex],
    b: ndarray[complex],
) -> ndarray[complex]:
    """
    Computes complex analog lambda values, given the real elastic moduli mu and lambda and m_prime and b transfert function
    values at frequency value omega.
    """
    return lambda_0 + (2.0 / 3.0) * mu_0 * (m_prime + b) / (1 + b)


def mu_computing(
    mu_0: ndarray[complex],
    m_prime: ndarray[complex],
    b: ndarray[complex],
) -> ndarray[complex]:
    """
    Computes complex analog mu values, given the real elastic modulus mu and m_prime and b transfert function values at
    frequency value omega.
    """
    return mu_0 * (1 - m_prime) / (1 + b)


def delta_mu_computing(
    mu_0: ndarray, Qmu: ndarray, omega_m: ndarray, alpha: ndarray, omega: float, omega_unit: float
) -> ndarray[complex]:
    """
    Computes the first order frequency dependent variation from elasticity delta_mu at frequency value omega, given the real
    elastic modulus mu_0, the elasticicty's quality factor Qmu and generalized attenuation parameters omega_m and alpha.
    The omega_m and omega parameters are unitless.
    """
    omega_0 = 1.0 / omega_unit  # (Unitless).
    high_frequency_domain: ndarray[bool] = omega >= omega_m
    with errstate(invalid="ignore", divide="ignore"):
        return nan_to_num(  # Alpha or omega_m may equal 0.0, meaning no attenuation should be taken into account.
            x=(mu_0 / Qmu)
            * (
                ((2.0 / pi) * log(omega / omega_0) + 1.0j) * high_frequency_domain
                + (
                    (2.0 / pi) * (log(omega_m / omega_0) + (1 / alpha) * (1 - (omega_m / omega) ** alpha))
                    + (omega_m / omega) ** alpha * 1.0j
                )
                * (1 - high_frequency_domain)
            ),
            nan=0.0,
        )
