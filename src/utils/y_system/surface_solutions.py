from numpy import array, asarray, dot, linalg, ndarray


def load_surface_solution(n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, piG: float):
    """
    Defines Surface load Love numbers using the 3 independent integrated solutions.
    """
    # Forms the Outer Surface Vector Describing Load.
    dmat = array(
        object=[
            [
                -((2.0 * n + 1.0) * (g_0_surface**2) / (4.0 * piG)),
                0.0,
                (2.0 * n + 1.0) * g_0_surface,
            ]
        ]
    )

    # Form G Matrix From Integrated Solutions
    Gmat = array(
        object=[
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [(Y1s[5] + (n + 1.0) * Y1s[4]), (Y2s[5] + (n + 1.0) * Y2s[4]), (Y3s[5] + (n + 1.0) * Y3s[4])],
        ]
    )

    # Solve the System of Equations (NO Matrix Inversion, for Stability)
    mvec = linalg.solve(Gmat, dmat.T).flatten()

    # Compute Solutions
    Y1sol = dot(array(object=[Y1s[0], Y2s[0], Y3s[0]]), mvec)
    Y3sol = dot(array(object=[Y1s[2], Y2s[2], Y3s[2]]), mvec)
    Y5sol = dot(array(object=[Y1s[4], Y2s[4], Y3s[4]]), mvec)

    # Compute Load Love Numbers
    h_load = Y1sol
    l_load = Y3sol
    k_load = Y5sol / g_0_surface - 1.0

    # Adjusts degree-one Love numbers to ensure that the potential field
    # outside the Earth vanishes in the CM frame (e.g. Merriam 1985).
    if n == 1:
        h_load = h_load - k_load
        l_load = l_load - k_load
        k_load = 0.0
    # Returns Solutions.
    return h_load, l_load, k_load


def shear_surface_solution(n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, piG: float):
    """
    Defines Surface shear Love numbers using the 3 independent integrated solutions.
    """

    # Degree 1 is not well defined.
    if n == 1:
        return 0.0, 0.0, 0.0

    # Forms the Outer Surface Vector Describing Shear.
    # See Okubo & Saito (1983), Saito (1978).
    dmat = array(object=[[0.0, ((2.0 * n + 1.0) * (g_0_surface**2)) / ((4.0 * piG * n) * (n + 1)), 0.0]])

    # Form G Matrix From Integrated Solutions
    Gmat = array(
        object=[
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [(Y1s[5] + (n + 1.0) * Y1s[4]), (Y2s[5] + (n + 1.0) * Y2s[4]), (Y3s[5] + (n + 1.0) * Y3s[4])],
        ]
    )

    # Solve the System of Equations (NO Matrix Inversion, for Stability)
    mvec = linalg.solve(Gmat, dmat.T)

    # Compute Solutions
    Y1sol = dot(array(object=[[Y1s[0], Y2s[0], Y3s[0]]]), mvec)
    Y3sol = dot(array(object=[[Y1s[2], Y2s[2], Y3s[2]]]), mvec)
    Y5sol = dot(array(object=[[Y1s[4], Y2s[4], Y3s[4]]]), mvec)

    # Compute Shear Love Numbers
    h_shr = Y1sol
    l_shr = Y3sol
    k_shr = Y5sol / g_0_surface

    return h_shr, l_shr, k_shr


def potential_surface_solution(n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float):
    """
    Defines Surface shear Love numbers using the 3 independent integrated solutions.
    """

    # Degree 1 is not well defined.
    if n == 1:
        return 0.0, 0.0, 0.0

    # Forms the Outer Surface Vector Describing External Potential.
    dmat = array(object=[[0.0, 0.0, (2.0 * n + 1) * g_0_surface]])

    # Form G Matrix From Integrated Solutions
    Gmat = array(
        object=[
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [(Y1s[5] + (n + 1.0) * Y1s[4]), (Y2s[5] + (n + 1.0) * Y2s[4]), (Y3s[5] + (n + 1.0) * Y3s[4])],
        ]
    )

    # Solve the System of Equations (NO Matrix Inversion, for Stability)
    mvec = linalg.solve(Gmat, dmat.T)

    # Compute Solutions
    Y1sol = dot(array(object=[[Y1s[0], Y2s[0], Y3s[0]]]), mvec)
    Y3sol = dot(array(object=[[Y1s[2], Y2s[2], Y3s[2]]]), mvec)
    Y5sol = dot(array(object=[[Y1s[4], Y2s[4], Y3s[4]]]), mvec)

    # Compute Potential Love Numbers
    h_pot = Y1sol
    l_pot = Y3sol
    k_pot = Y5sol / g_0_surface - 1.0

    return h_pot, l_pot, k_pot
