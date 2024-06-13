from numpy import ndarray
from ..classes import Result


def geocenter_inversion(
    anelastic_frequencial_harmonic_load_signal: ndarray[complex],
    anelastic_hermitian_Love_numbers: Result,
    ocean_mask: ndarray[float],
) -> ndarray[complex]:
    """
    Re-estimates degree 1 coefficients by inversion.
    Returns geocenter coefficients as frequencial signals.
    """

    return anelastic_frequencial_harmonic_load_signal[:, 1, :, :]
