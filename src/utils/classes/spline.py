from numpy import ndarray


class Spline(tuple[ndarray | float, ndarray | float, int]):
    """
    Quantities necessary to describe a polynomial spline over a real interval according to the scipy.interpolate.splrep formalism.
    """

    pass
