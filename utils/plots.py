from numpy import array, ndarray

SECONDS_PER_YEAR = 365.25 * 86400
SYMBOLS_PER_DIRECTION = ["h", "l", "k"]
SYMBOLS_PER_BOUNDARY_CONDITION = ["'", "*", ""]


def frequencies_to_periods(
    frequencies: ndarray | list[float],
) -> ndarray:
    """
    Converts tab from (Hz) to (y). Works also from (y) to (Hz).
    """
    return (1.0 / SECONDS_PER_YEAR) / array(frequencies)
