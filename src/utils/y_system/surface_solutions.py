from numpy import array, asarray, dot, linalg, ndarray

# TODO: Vectorize and implement minimal computing policy.


def load_surface_solution(
    n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, piG: float, length_ratio: float
):
    # Form Outer Surface Vector Describing Load
    dmat = array(
        [
            [
                -((2.0 * n + 1.0) * (g_0_surface**2) / (4.0 * piG)),
                0.0,
                (2.0 * n + 1.0) * g_0_surface,
            ]
        ]
    )

    # Form G Matrix From Integrated Solutions
    Gmat = array(
        [
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [(Y1s[5] + (n + 1.0) * Y1s[4]), (Y2s[5] + (n + 1.0) * Y2s[4]), (Y3s[5] + (n + 1.0) * Y3s[4])],
        ]
    )

    # Solve the System of Equations (NO Matrix Inversion, for Stability)
    mvec = linalg.solve(Gmat, dmat.T).flatten()

    # Compute Solutions
    Y1sol = dot(array([Y1s[0], Y2s[0], Y3s[0]]), mvec)
    Y2sol = dot(array([Y1s[1], Y2s[1], Y3s[1]]), mvec)
    Y3sol = dot(array([Y1s[2], Y2s[2], Y3s[2]]), mvec)
    Y4sol = dot(array([Y1s[3], Y2s[3], Y3s[3]]), mvec)
    Y5sol = dot(array([Y1s[4], Y2s[4], Y3s[4]]), mvec)
    Y6sol = dot(array([Y1s[5], Y2s[5], Y3s[5]]), mvec)

    # Compute Load Love Numbers
    h_load = Y1sol * length_ratio
    l_load = Y3sol * length_ratio
    k_load = Y5sol * length_ratio / g_0_surface - 1.0

    # Adjust degree-one Love numbers to ensure that the potential field
    # outside the Earth vanishes in the CE frame (e.g. Merriam 1985)
    if n == 1:
        h_load = h_load - k_load
        l_load = l_load - k_load
        k_load = k_load - k_load
    # Return Solutions
    return Y1sol, Y2sol, Y3sol, Y4sol, Y5sol, Y6sol, mvec, h_load, l_load, k_load


def shear_surface_solution(
    n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, piG: float, length_ratio: float
):
    # Form Outer Surface Vector Describing Shear
    # See Okubo & Saito (1983), Saito (1978)
    dmat = array([[0.0, ((2.0 * n + 1.0) * (g_0_surface**2)) / ((4.0 * piG * n) * (n + 1)), 0.0]])

    # Form G Matrix From Integrated Solutions
    Gmat = array(
        [
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [(Y1s[5] + (n + 1.0) * Y1s[4]), (Y2s[5] + (n + 1.0) * Y2s[4]), (Y3s[5] + (n + 1.0) * Y3s[4])],
        ]
    )

    # Solve the System of Equations (NO Matrix Inversion, for Stability)
    mvec = linalg.solve(Gmat, dmat.T)

    # Compute Solutions
    Y1sol = dot(array([[Y1s[0], Y2s[0], Y3s[0]]]), mvec).flatten()[0]
    Y2sol = dot(array([[Y1s[1], Y2s[1], Y3s[1]]]), mvec).flatten()[0]
    Y3sol = dot(array([[Y1s[2], Y2s[2], Y3s[2]]]), mvec).flatten()[0]
    Y4sol = dot(array([[Y1s[3], Y2s[3], Y3s[3]]]), mvec).flatten()[0]
    Y5sol = dot(array([[Y1s[4], Y2s[4], Y3s[4]]]), mvec).flatten()[0]
    Y6sol = dot(array([[Y1s[5], Y2s[5], Y3s[5]]]), mvec).flatten()[0]

    # Compute Shear Love Numbers
    h_shr = Y1sol * length_ratio
    l_shr = Y3sol * length_ratio
    k_shr = Y5sol * length_ratio / g_0_surface

    # Degree1 is not well defined
    if n == 1:
        h_shr = asarray(0.0)
        l_shr = asarray(0.0)
        k_shr = asarray(0.0)

    return Y1sol, Y2sol, Y3sol, Y4sol, Y5sol, Y6sol, mvec, h_shr, l_shr, k_shr


def potential_surface_solution(n: int, Y1s: ndarray, Y2s: ndarray, Y3s: ndarray, g_0_surface: float, length_ratio: float):
    # Form Outer Surface Vector Describing External Potential
    dmat = array([[0.0, 0.0, (2.0 * n + 1) * g_0_surface]])

    # Form G Matrix From Integrated Solutions
    Gmat = array(
        [
            [Y1s[1], Y2s[1], Y3s[1]],
            [Y1s[3], Y2s[3], Y3s[3]],
            [(Y1s[5] + (n + 1.0) * Y1s[4]), (Y2s[5] + (n + 1.0) * Y2s[4]), (Y3s[5] + (n + 1.0) * Y3s[4])],
        ]
    )

    # Solve the System of Equations (NO Matrix Inversion, for Stability)
    mvec = linalg.solve(Gmat, dmat.T)

    # Compute Solutions
    Y1sol = dot(array([[Y1s[0], Y2s[0], Y3s[0]]]), mvec)
    Y2sol = dot(array([[Y1s[1], Y2s[1], Y3s[1]]]), mvec)
    Y3sol = dot(array([[Y1s[2], Y2s[2], Y3s[2]]]), mvec)
    Y4sol = dot(array([[Y1s[3], Y2s[3], Y3s[3]]]), mvec)
    Y5sol = dot(array([[Y1s[4], Y2s[4], Y3s[4]]]), mvec)
    Y6sol = dot(array([[Y1s[5], Y2s[5], Y3s[5]]]), mvec)

    # Compute Potential Love Numbers
    h_pot = Y1sol * length_ratio
    l_pot = Y3sol * length_ratio
    k_pot = Y5sol * length_ratio / g_0_surface - 1.0

    # Degree1 is not well defined
    if n == 1:
        h_pot = asarray(0.0)
        l_pot = asarray(0.0)
        k_pot = asarray(0.0)

    return Y1sol, Y2sol, Y3sol, Y4sol, Y5sol, Y6sol, mvec, h_pot, l_pot, k_pot
