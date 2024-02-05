from numpy import ndarray


class Spline(tuple[ndarray, ndarray, int]):
    """
    Quantities necessary to describe a polynomial spline over a real interval according to the scipy.interpolate.splrep formalism.
    """

    pass
