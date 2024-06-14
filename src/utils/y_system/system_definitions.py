from math import factorial

from numpy import Inf, array, dot, lib, ndarray, sqrt, zeros

from ..classes import DescriptionLayer


def solid_system(
    x: float,
    Y: ndarray,
    n: int,
    layer: DescriptionLayer,
    piG: float,
    omega: float,
    dynamic_term: bool,
    inhomogeneity_gradients: bool,
) -> ndarray:
    """
    Performs a single iteration of the gravito-anelastic solid differential system dY/dr = A Y.
    """

    # Interpolate Parameters at Current Radius
    lndi = (
        layer.evaluate(x=x, variable="lambda_real")
        + layer.evaluate(x=x, variable="lambda_imag") * 1.0j
    )
    mndi = (
        layer.evaluate(x=x, variable="mu_real")
        + layer.evaluate(x=x, variable="mu_imag") * 1.0j
    )
    rndi = layer.evaluate(x=x, variable="rho_0")
    gndi = layer.evaluate(x=x, variable="g_0")

    # add dynamics terms (in w square)
    # change the high frequencies behavior of LN
    dyn_term = -rndi * omega**2.0 if dynamic_term and omega != Inf else 0.0
    if inhomogeneity_gradients:
        lndi_prime = (
            layer.evaluate(x=x, variable="lambda_real", derivative_order=1)
            + layer.evaluate(x=x, variable="lambda_imag", derivative_order=1) * 1.0j
        )
        mndi_prime = (
            layer.evaluate(x=x, variable="mu_real", derivative_order=1)
            + layer.evaluate(x=x, variable="mu_imag", derivative_order=1) * 1.0j
        )
        # TODO: discuss on inhomogeneity consequences.

    n1 = n * (n + 1.0)
    bndi = 1.0 / (lndi + 2.0 * mndi)
    dndi = 2.0 * mndi * (3.0 * lndi + 2.0 * mndi) * bndi
    endi = 4.0 * n1 * mndi * (lndi + mndi) * bndi - 2.0 * mndi

    # Builds A Matrix (where dY/dr = A Y).
    # See Smylie (2013).
    A = zeros(shape=(6, 6), dtype=complex)

    A[0, 0] = -2.0 * lndi * bndi / x
    A[0, 1] = bndi
    A[0, 2] = n1 * lndi * bndi / x

    A[1, 0] = (-4.0 * gndi * rndi / x) + (2.0 * dndi / (x**2)) + dyn_term
    A[1, 1] = -4.0 * mndi * bndi / x
    A[1, 2] = n1 * (rndi * gndi / x - dndi / (x**2))
    A[1, 3] = n1 / x

    A[1, 5] = -rndi

    A[2, 0] = -1.0 / x

    A[2, 2] = 1.0 / x
    A[2, 3] = 1.0 / mndi

    A[3, 0] = rndi * gndi / x - dndi / (x**2)
    A[3, 1] = -lndi * bndi / x
    A[3, 2] = endi / (x**2) + dyn_term
    A[3, 3] = -3.0 / x
    A[3, 4] = -rndi / x

    A[4, 0] = 4.0 * piG * rndi

    A[4, 5] = 1.0

    A[5, 2] = -4.0 * piG * rndi * n1 / x

    A[5, 4] = n1 / (x**2)
    A[5, 5] = -2.0 / x

    return dot(A, Y)


def fluid_system(
    x: float,
    Y: ndarray,
    n: int,
    fluid_layer: DescriptionLayer,
    piG: float,
) -> ndarray:
    """
    Performs a single iteration of the gravito-anelastic fluid differential system dY/dr = B Y.
    """

    # Interpolate Parameters at Current Radius
    rndi = fluid_layer.evaluate(x=x, variable="rho_0")
    gndi = fluid_layer.evaluate(x=x, variable="g_0")

    n1 = n * (n + 1.0)

    # Smylie (2013) Eq.9.42 & 9.43.
    B = zeros(shape=(2, 2), dtype=complex)

    B[0, 0] = 4.0 * piG * rndi / gndi
    B[0, 1] = 1.0

    B[1, 0] = (n1 / x**2) - 16.0 * piG * rndi / (gndi * x)
    B[1, 1] = (-2.0 / x) - (4.0 * piG * rndi / gndi)

    return dot(B, Y)


def solid_to_fluid(
    Y1: ndarray,
    Y2: ndarray,
    Y3: ndarray,
    x: float,
    first_fluid_layer: DescriptionLayer,
    piG: float,
) -> ndarray:
    """
    Transforms the solid system of equation into the fluid system of equation at an interface.
    """
    # Interpolate Parameters at Current Radius
    rndi = first_fluid_layer.evaluate(x=x, variable="rho_0")
    gndi = first_fluid_layer.evaluate(x=x, variable="g_0")

    k_13 = Y1[3] / Y3[3]
    k_23 = Y2[3] / Y3[3]
    k_num = (
        gndi * (Y1[0] + Y3[0] * k_13)
        - (Y1[4] + Y3[4] * k_13)
        + (1.0 / rndi) * (Y1[1] + Y3[1] * k_13)
    )
    k_den = (
        gndi * (Y2[0] + Y3[0] * k_23)
        - (Y2[4] + Y3[4] * k_23)
        + (1.0 / rndi) * (Y2[1] + Y3[1] * k_23)
    )
    kk = k_num / k_den

    sol2 = Y1[1] + kk * Y2[1] + (k_13 + kk * k_23) * Y3[1]
    sol5 = Y1[4] + kk * Y2[4] + (k_13 + kk * k_23) * Y3[4]
    sol6 = Y1[5] + kk * Y2[5] + (k_13 + kk * k_23) * Y3[5]

    Yf1 = array(object=[sol5, sol6 + (4.0 * piG / gndi) * sol2], dtype=complex)
    return Yf1


def fluid_to_solid(
    Yf1: ndarray, x: float, last_fluid_layer: DescriptionLayer, piG: float
) -> tuple[ndarray, ndarray, ndarray]:
    # Interpolate Parameters at Current Radius
    rndi = last_fluid_layer.evaluate(x=x, variable="rho_0")
    gndi = last_fluid_layer.evaluate(x=x, variable="g_0")

    Y1 = array(
        object=[1.0, gndi * rndi, 0.0, 0.0, 0.0, -4.0 * piG * rndi], dtype=complex
    )

    Y2 = array(object=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex)

    Y3 = array(object=[Yf1[0] / gndi, 0.0, 0.0, 0.0, Yf1[0], Yf1[1]], dtype=complex)
    return Y1, Y2, Y3


def solid_homogeneous_system(x: float, n: int, layer: DescriptionLayer, piG: float):
    """
    Computes analytical solution to homogeneous sphere system of radius x, with n_layer-th layer elastic rheology.
    """
    # Interpolates Parameters at Current Radius.
    lndi = (
        layer.evaluate(x=x, variable="lambda_real")
        + layer.evaluate(x=x, variable="lambda_imag") * 1.0j
    )
    mndi = (
        layer.evaluate(x=x, variable="mu_real")
        + layer.evaluate(x=x, variable="mu_imag") * 1.0j
    )
    rndi = layer.evaluate(x=x, variable="rho_0")
    vsndi = layer.evaluate(x=x, variable="Vs")
    vpndi = layer.evaluate(x=x, variable="Vp")

    wnd = 0.0

    # Computes Additional Non-Dimensional Parameters.
    n1 = n * (n + 1.0)
    gamma = 4.0 * piG * rndi / 3.0

    # k May be Imaginary.
    ksq1 = 0.5 * (
        (((wnd**2) + 4.0 * gamma) / (vpndi**2))
        + ((wnd**2) / (vsndi**2))
        - sqrt(
            ((wnd**2) / (vsndi**2) - ((wnd**2) + 4.0 * gamma) / (vpndi**2)) ** 2
            + (4.0 * n1 * (gamma**2) / ((vsndi**2) * (vpndi**2)))
        )
    )
    ksq2 = 0.5 * (
        (((wnd**2) + 4.0 * gamma) / (vpndi**2))
        + ((wnd**2) / (vsndi**2))
        + sqrt(
            ((wnd**2) / (vsndi**2) - ((wnd**2) + 4.0 * gamma) / (vpndi**2)) ** 2
            + (4.0 * n1 * (gamma**2) / ((vsndi**2) * (vpndi**2)))
        )
    )
    # From Takeuchi & Saito (1972), Eq. 99: Factored for Numerical Stability.
    f1 = (
        (1.0 / gamma)
        * (vsndi * lib.scimath.sqrt(ksq1) + wnd)
        * (vsndi * lib.scimath.sqrt(ksq1) - wnd)
    )
    f2 = (
        (1.0 / gamma)
        * (vsndi * lib.scimath.sqrt(ksq2) + wnd)
        * (vsndi * lib.scimath.sqrt(ksq2) - wnd)
    )

    # Imaginary Part is Effectively Zero -- Gets Rid of It.
    f1 = f1.real
    f2 = f2.real
    h1 = f1 - (n + 1.0)
    h2 = f2 - (n + 1.0)

    # x May be Imaginary -- Only Even Powers will be Used Later.
    x1 = lib.scimath.sqrt(ksq1) * x
    x2 = lib.scimath.sqrt(ksq2) * x

    # Computes the squares.
    x1sqr = x1 * x1
    x1sqr = x1sqr.real
    x2sqr = x2 * x2
    x2sqr = x2sqr.real

    # Computes the Bessel functions using expansion formulas (Takeuchi & Saito 1972, Eq. 102).
    phi1_n = (
        1.0
        - x1sqr / (2.0 * (2.0 * n + 3.0))
        + (x1sqr**2) / (4.0 * (2.0 * n + 3.0) * (2.0 * n + 5.0) * 2.0)
        - (x1sqr**3)
        / (
            factorial(3)
            * (2.0**3)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        + (x1sqr**4)
        / (
            factorial(4)
            * (2.0**4)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        - (x1sqr**5)
        / (
            factorial(5)
            * (2.0**5)
            * (2.0 * n + 11.0)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        + (x1sqr**6)
        / (
            factorial(6)
            * (2.0**6)
            * (2.0 * n + 13.0)
            * (2.0 * n + 11.0)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        - (x1sqr**7)
        / (
            factorial(7)
            * (2.0**7)
            * (2.0 * n + 15.0)
            * (2.0 * n + 13.0)
            * (2.0 * n + 11.0)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
    )
    psi1_n = (1.0 - phi1_n) * ((2.0 * (2.0 * n + 3.0)) / x1sqr)
    phi2_n = (
        1.0
        - x2sqr / (2.0 * (2.0 * n + 3.0))
        + (x2sqr**2) / (4.0 * (2.0 * n + 3.0) * (2.0 * n + 5.0) * 2.0)
        - (x2sqr**3)
        / (
            factorial(3)
            * (2.0**3)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        + (x2sqr**4)
        / (
            factorial(4)
            * (2.0**4)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        - (x2sqr**5)
        / (
            factorial(5)
            * (2.0**5)
            * (2.0 * n + 11.0)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        + (x2sqr**6)
        / (
            factorial(6)
            * (2.0**6)
            * (2.0 * n + 13.0)
            * (2.0 * n + 11.0)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
        - (x2sqr**7)
        / (
            factorial(7)
            * (2.0**7)
            * (2.0 * n + 15.0)
            * (2.0 * n + 13.0)
            * (2.0 * n + 11.0)
            * (2.0 * n + 9.0)
            * (2.0 * n + 7.0)
            * (2.0 * n + 5.0)
            * (2.0 * n + 3.0)
        )
    )
    psi2_n = (1.0 - phi2_n) * ((2.0 * (2.0 * n + 3.0)) / x2sqr)

    phi1_np1 = (
        1.0
        - x1sqr / (2.0 * (2.0 * (n + 1.0) + 3.0))
        + (x1sqr**2) / (4.0 * (2.0 * (n + 1.0) + 3.0) * (2.0 * (n + 1.0) + 5.0) * 2.0)
        - (x1sqr**3)
        / (
            factorial(3)
            * (2.0**3)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        + (x1sqr**4)
        / (
            factorial(4)
            * (2.0**4)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        - (x1sqr**5)
        / (
            factorial(5)
            * (2.0**5)
            * (2.0 * (n + 1.0) + 11.0)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        + (x1sqr**6)
        / (
            factorial(6)
            * (2.0**6)
            * (2.0 * (n + 1.0) + 13.0)
            * (2.0 * (n + 1.0) + 11.0)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        - (x1sqr**7)
        / (
            factorial(7)
            * (2.0**7)
            * (2.0 * (n + 1.0) + 15.0)
            * (2.0 * (n + 1.0) + 13.0)
            * (2.0 * (n + 1.0) + 11.0)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
    )
    phi2_np1 = (
        1.0
        - x2sqr / (2.0 * (2.0 * (n + 1.0) + 3.0))
        + (x2sqr**2) / (4.0 * (2.0 * (n + 1.0) + 3.0) * (2.0 * (n + 1.0) + 5.0) * 2.0)
        - (x2sqr**3)
        / (
            factorial(3)
            * (2.0**3)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        + (x2sqr**4)
        / (
            factorial(4)
            * (2.0**4)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        - (x2sqr**5)
        / (
            factorial(5)
            * (2.0**5)
            * (2.0 * (n + 1.0) + 11.0)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        + (x2sqr**6)
        / (
            factorial(6)
            * (2.0**6)
            * (2.0 * (n + 1.0) + 13.0)
            * (2.0 * (n + 1.0) + 11.0)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
        - (x2sqr**7)
        / (
            factorial(7)
            * (2.0**7)
            * (2.0 * (n + 1.0) + 15.0)
            * (2.0 * (n + 1.0) + 13.0)
            * (2.0 * (n + 1.0) + 11.0)
            * (2.0 * (n + 1.0) + 9.0)
            * (2.0 * (n + 1.0) + 7.0)
            * (2.0 * (n + 1.0) + 5.0)
            * (2.0 * (n + 1.0) + 3.0)
        )
    )

    # FIRST SOLUTION
    Y11 = -(x ** (n + 1.0) / (2.0 * n + 3.0)) * (0.5 * n * h1 * psi1_n + f1 * phi1_np1)
    Y21 = -(lndi + 2.0 * mndi) * (x**n) * f1 * phi1_n + (
        mndi * (x**n) / (2.0 * n + 3.0)
    ) * (-(n * (n - 1.0)) * h1 * psi1_n + 2.0 * (2.0 * f1 + n1) * phi1_np1)
    Y31 = -(x ** (n + 1.0) / (2.0 * n + 3.0)) * (0.5 * h1 * psi1_n - phi1_np1)
    Y41 = (
        mndi
        * (x**n)
        * (
            phi1_n
            - (1.0 / (2.0 * n + 3.0))
            * ((n - 1.0) * h1 * psi1_n + 2.0 * (f1 + 1.0) * phi1_np1)
        )
    )
    Y51 = (x ** (n + 2.0)) * (
        ((vpndi**2) * f1 - (n + 1.0) * (vsndi**2)) / (x**2)
        - ((3.0 * gamma * f1) / (2.0 * (2.0 * n + 3.0))) * psi1_n
    )
    Y61 = (2.0 * n + 1.0) * (x ** (n + 1.0)) * (
        ((vpndi**2) * f1 - (n + 1.0) * (vsndi**2)) / (x**2)
        - ((3.0 * gamma * f1) / (2.0 * (2.0 * n + 3.0))) * psi1_n
    ) + ((3.0 * n * gamma * h1 * (x ** (n + 1.0))) / (2.0 * (2.0 * n + 3.0))) * psi1_n

    # SECOND SOLUTION
    Y12 = -(x ** (n + 1.0) / (2.0 * n + 3.0)) * (0.5 * n * h2 * psi2_n + f2 * phi2_np1)
    Y22 = -(lndi + 2.0 * mndi) * (x**n) * f2 * phi2_n + (
        mndi * (x**n) / (2.0 * n + 3.0)
    ) * (-(n * (n - 1.0)) * h2 * psi2_n + 2.0 * (2.0 * f2 + n1) * phi2_np1)
    Y32 = -(x ** (n + 1.0) / (2.0 * n + 3.0)) * (0.5 * h2 * psi2_n - phi2_np1)
    Y42 = (
        mndi
        * (x**n)
        * (
            phi2_n
            - (1.0 / (2.0 * n + 3.0))
            * ((n - 1.0) * h2 * psi2_n + 2.0 * (f2 + 1.0) * phi2_np1)
        )
    )
    Y52 = (x ** (n + 2.0)) * (
        ((vpndi**2) * f2 - (n + 1.0) * (vsndi**2)) / (x**2)
        - ((3.0 * gamma * f2) / (2.0 * (2.0 * n + 3.0))) * psi2_n
    )
    Y62 = (2.0 * n + 1.0) * (x ** (n + 1.0)) * (
        ((vpndi**2) * f2 - (n + 1.0) * (vsndi**2)) / (x**2)
        - ((3.0 * gamma * f2) / (2.0 * (2.0 * n + 3.0))) * psi2_n
    ) + ((3.0 * n * gamma * h2 * (x ** (n + 1.0))) / (2.0 * (2.0 * n + 3.0))) * psi2_n

    # THIRD SOLUTION
    Y13 = n * (x ** (n - 1.0))
    Y23 = 2.0 * mndi * n * (n - 1.0) * (x ** (n - 2.0))
    Y33 = x ** (n - 1.0)
    Y43 = 2.0 * mndi * (n - 1.0) * (x ** (n - 2.0))
    Y53 = (n * gamma - (wnd**2)) * (x**n)
    Y63 = (2.0 * n + 1.0) * (
        (n * gamma - (wnd**2)) * (x ** (n - 1.0))
    ) - 3.0 * n * gamma * (x ** (n - 1.0))

    # CONVERT TAKEUCHI & SAITO Y CONVENTION BACK TO SMYLIE CONVENTION
    Y61 = Y61 - ((n + 1.0) / x) * Y51
    Y62 = Y62 - ((n + 1.0) / x) * Y52
    Y63 = Y63 - ((n + 1.0) / x) * Y53

    # Return Y-Variable Starting Solutions
    return array(
        object=[
            [Y11, Y21, Y31, Y41, Y51, Y61],
            [Y12, Y22, Y32, Y42, Y52, Y62],
            [Y13, Y23, Y33, Y43, Y53, Y63],
        ]
    )
