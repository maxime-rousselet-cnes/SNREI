from numpy import array, dot, linalg, ndarray


def load_surface_solution(
    n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, piG: float
) -> tuple[float, float, float]:
    """
    Returns load Love numbers from the Y_i system solution at Earth surface.
    """
    # Forms the outer surface vector describing load.
    dmat = array(
        [
            [
                -((2.0 * n + 1.0) * (g_0_surface**2) / (4.0 * piG)),
                0.0,
                (2.0 * n + 1.0) * g_0_surface,
            ]
        ]
    )

    # Forms the G matrix from integrated solutions.
    Gmat = array(
        [
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [
                (Y1s[5] + (n + 1.0) * Y1s[4]),
                (Y2s[5] + (n + 1.0) * Y2s[4]),
                (Y3s[5] + (n + 1.0) * Y3s[4]),
            ],
        ]
    )

    # Solve the system.
    mvec = linalg.solve(Gmat, dmat.T).flatten()

    # Computes solutions.
    Y1sol = dot(array([Y1s[0], Y2s[0], Y3s[0]]), mvec).flatten()[0]
    Y3sol = dot(array([Y1s[2], Y2s[2], Y3s[2]]), mvec).flatten()[0]
    Y5sol = dot(array([Y1s[4], Y2s[4], Y3s[4]]), mvec).flatten()[0]

    # Computes load Love numbers.
    h_load = Y1sol
    l_load = Y3sol
    k_load = Y5sol / g_0_surface - 1.0

    # Adjust degree 1 to ensure that the potential field outside the Earth vanishes (e.g. Merriam 1985).
    if n == 1:
        h_load = h_load - k_load
        l_load = l_load - k_load
        k_load = k_load - k_load

    return h_load, l_load, k_load


def shear_surface_solution(
    n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, piG: float
) -> tuple[float, float, float]:
    """
    Returns shear Love numbers from the Y_i system solution at Earth surface.
    """

    # Degree 1 is not well defined.
    if n == 1:
        return 0.0, 0.0, 0.0

    # Forms the outer surface vector describing shear. See Okubo & Saito (1983), Saito (1978).
    dmat = array(
        [[0.0, ((2.0 * n + 1.0) * (g_0_surface**2)) / ((4.0 * piG * n) * (n + 1)), 0.0]]
    )

    # Forms the G matrix from integrated solutions.
    Gmat = array(
        [
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [
                (Y1s[5] + (n + 1.0) * Y1s[4]),
                (Y2s[5] + (n + 1.0) * Y2s[4]),
                (Y3s[5] + (n + 1.0) * Y3s[4]),
            ],
        ]
    )

    # Solves the system.
    mvec = linalg.solve(Gmat, dmat.T)

    # Computes solutions.
    Y1sol = dot(array([[Y1s[0], Y2s[0], Y3s[0]]]), mvec).flatten()[0]
    Y3sol = dot(array([[Y1s[2], Y2s[2], Y3s[2]]]), mvec).flatten()[0]
    Y5sol = dot(array([[Y1s[4], Y2s[4], Y3s[4]]]), mvec).flatten()[0]

    # Computes shear Love numbers.
    h_shr = Y1sol
    l_shr = Y3sol
    k_shr = Y5sol / g_0_surface

    return h_shr, l_shr, k_shr


def potential_surface_solution(
    n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float
) -> tuple[float, float, float]:
    """
    Returns potential Love numbers from the Y_i system solution at Earth surface.
    """

    # Degree 1 is not well defined.
    if n == 1:
        return 0.0, 0.0, 0.0

    # Forms the outer surface vector describing external potential.
    dmat = array([[0.0, 0.0, (2.0 * n + 1) * g_0_surface]])

    # Forms the G matrix from integrated solutions.
    Gmat = array(
        [
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [
                (Y1s[5] + (n + 1.0) * Y1s[4]),
                (Y2s[5] + (n + 1.0) * Y2s[4]),
                (Y3s[5] + (n + 1.0) * Y3s[4]),
            ],
        ]
    )

    # Solves the system.
    mvec = linalg.solve(Gmat, dmat.T)

    # Computes solutions.
    Y1sol = dot(array([[Y1s[0], Y2s[0], Y3s[0]]]), mvec).flatten()[0]
    Y3sol = dot(array([[Y1s[2], Y2s[2], Y3s[2]]]), mvec).flatten()[0]
    Y5sol = dot(array([[Y1s[4], Y2s[4], Y3s[4]]]), mvec).flatten()[0]

    # Computes potential Love Numbers.
    h_pot = Y1sol
    l_pot = Y3sol
    k_pot = Y5sol / g_0_surface - 1.0

    return h_pot, l_pot, k_pot
