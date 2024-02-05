from numpy import array, ndarray

SECONDS_PER_YEAR = 365.25 * 86400


def frequencies_to_periods(
    frequencies: ndarray | list[float],
) -> ndarray:
    """
    Converts tab from (Hz) to (y).
    """
    return (1.0 / SECONDS_PER_YEAR) / array(frequencies)
