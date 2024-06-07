from typing import Callable, Optional

from numpy import (
    abs,
    argsort,
    array,
    concatenate,
    conjugate,
    cos,
    expand_dims,
    flip,
    linspace,
    log10,
    max,
    ndarray,
    newaxis,
    ones,
    pi,
    prod,
    round,
    setdiff1d,
    unique,
    vstack,
    zeros,
)
from numpy.linalg import pinv
from pyshtools.expand import MakeGridDH
from scipy import interpolate


def precise_curvature(
    x_initial_values: ndarray, f: Callable[[float], ndarray[complex]], max_tol: float, decimals: int
) -> tuple[ndarray, ndarray]:
    """
    Finds a sufficiently precise sampling of x axis for f function representation. The criteria takes curvature into account is
    the error between the 1st and second orders.
    """

    # Initializes.
    x_new_values = x_initial_values
    x_values = array(object=[], dtype=float)
    f_values = array(object=[], dtype=complex)

    # Loops while there are still added abscissas.
    while len(x_new_values) != 0:
        # Calls f for new values only.
        f_new_values = f(x_new_values)

        # Updates.
        x_values = concatenate((x_values, x_new_values))
        f_values = f_new_values if len(f_values) == 0 else concatenate((f_values, f_new_values))
        order = argsort(x_values)
        x_values = x_values[order]
        f_values = f_values[order]
        x_new_values = array(object=[], dtype=float)

        # Iterates on new sampling.
        for f_left, f_x, f_right, x_left, x, x_right in zip(
            f_values[:-2], f_values[1:-1], f_values[2:], x_values[:-2], x_values[1:-1], x_values[2:]
        ):
            # For maximal curvature: finds where the error is above maximum threshold parameter and adds median values.
            condition: ndarray = abs((f_right - f_left) / (x_right - x_left) * (x - x_left) + f_left - f_x) > max_tol * max(
                a=abs([f_left, f_x, f_right]), axis=0
            )
            if condition.any():
                # Updates sampling.
                x_new_values = concatenate((x_new_values, [(x + x_left) / 2.0, (x + x_right) / 2.0]))

        # Keeps only values that are not already taken into account.
        x_new_values = setdiff1d(ar1=unique(round(a=x_new_values, decimals=decimals)), ar2=x_values)

    return x_values, f_values


def interpolate_array(x_values: ndarray, y_values: ndarray, new_x_values: ndarray) -> ndarray:
    """
    1D-Interpolates the given data on its first axis, whatever its shape is.
    """

    # Flattens all other dimensions.
    shape = y_values.shape
    y_values.reshape((shape[0], -1))
    components = y_values.shape[1]

    # Initializes
    function_values = zeros(shape=(len(new_x_values), components), dtype=complex)

    # Loops on components
    for i_component, component in enumerate(y_values.transpose()):
        # Creates callable (linear).
        function = interpolate.interp1d(x=x_values, y=component, kind="linear")

        # Calls linear interpolation on new x values.
        function_values[:, i_component] = function(x=new_x_values)

    #  Converts back into initial other dimension shapes.
    function_values.reshape((len(new_x_values), *shape[1:]))
    return function_values


def interpolate_all(x_values_per_component: list[ndarray], function_values: list[ndarray], x_shared_values: ndarray) -> ndarray:
    """
    Interpolate several function values on shared abscissas.
    """
    return array(
        object=[
            interpolate_array(x_values=x_tab, y_values=function_values_tab, new_x_values=x_shared_values)
            for x_tab, function_values_tab in zip(x_values_per_component, function_values)
        ]
    )


def build_hermitian(signal: ndarray[complex]) -> ndarray[complex]:
    """
    For a given signal defined for positive values, builds the corresponding extended signal that has hermitian symetry.
    """
    return concatenate((conjugate(flip(m=signal)), signal))


def get_degrees_indices(degrees: list[int], degrees_to_plot: list[int]) -> list[int]:
    """
    Returns the indices of the wanted degrees in the list of degrees.
    """
    return [list(degrees).index(degree) for degree in degrees_to_plot]


def generate_log_frequency_initial_values(
    frequency_min: float, frequency_max: float, n_frequency_0: int, frequency_unit: float
) -> ndarray[float]:
    """
    Generates an array of log-spaced frequency values.
    """
    return linspace(
        start=log10(frequency_min / frequency_unit),
        stop=log10(frequency_max / frequency_unit),
        num=n_frequency_0,
    )


def signal_trend(trend_dates: ndarray[float], signal: ndarray[float]) -> tuple[float, float]:
    """
    Returns signal's trend: mean slope and additive constant during last years (LSE).
    """
    # Assemble matrix A.
    A = vstack(
        [
            trend_dates,
            ones(len(trend_dates)),
        ]
    ).T
    # Direct least square regression using pseudo-inverse.
    result: ndarray = pinv(A).dot(signal[:, newaxis])
    return result.flatten()  # Turn the signal into a column vector. (slope, additive_constant)


def map_normalizing(
    map: ndarray,
) -> ndarray:
    """
    Sets global mean as zero and max as one by homothety.
    """
    n_t = prod(map.shape)
    sum_map = sum(map.flatten())
    max_map = max(map.flatten())
    return map / (max_map - sum_map / n_t) + sum_map / (sum_map - max_map * n_t)


def surface_ponderation(
    mask: ndarray[float],
) -> ndarray[float]:
    """
    Gets the surface of a (latitude * longitude) array.
    """
    return mask * expand_dims(a=cos(linspace(start=-pi / 2, stop=pi / 2, num=len(mask))), axis=1)


def mean_on_mask(
    mask: ndarray[float], harmonics: Optional[ndarray[float]] = None, grid: Optional[ndarray[float]] = None
) -> float:
    """
    Computes mean value over a given surface. Uses a given mask.
    """
    if grid is None:
        grid: ndarray[float] = MakeGridDH(harmonics, sampling=2)
    surface = surface_ponderation(mask=mask)
    weighted_values = grid * surface
    return round(a=sum(weighted_values.flatten()) / sum(surface.flatten()), decimals=5)
