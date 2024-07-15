from numpy import concatenate, errstate, multiply, nan_to_num, ndarray, ones
from pyshtools.expand import MakeGridDH

from ..classes import DENSITY_RATIO, BoundaryCondition, Direction, Result
from ..data import map_sampling


def leakage_correction(
    frequencial_harmonic_load_signal: ndarray[complex],
    ocean_mask: ndarray[float],
    Love_numbers: Result,
    iterations: int,
) -> ndarray[complex]:
    """
    Performs a correction for continental data leak on oceans and ocean data leak on continents.
    Iterates on frequencies and asked iterations for leakage correction.
    """

    # Gets coefficients back in gravitationnal domain.
    frequencial_Stokes_coefficients = multiply(
        concatenate(
            (  # Adds a line of one values for degree zero.
                ones(shape=(1, len(Love_numbers.axes["frequencies"]))),
                DENSITY_RATIO
                * multiply(
                    Love_numbers.values[Direction.potential][BoundaryCondition.load].T,
                    3 / (2 * Love_numbers.axes["degrees"] + 1),
                ).T,
            ),
            axis=0,
        ).T,
        frequencial_harmonic_load_signal.transpose((0, 2, 3, 1)),
    ).transpose((0, 3, 1, 2))

    Stokes_coefficients: ndarray[complex]
    # Iterates on frequencies.
    for i_frequency, Stokes_coefficients in enumerate(frequencial_Stokes_coefficients.transpose((3, 0, 1, 2))):

        # Iterates a leakage correction procedure as many times as asked for.
        for _ in range(iterations):

            Stokes_coefficients -= single_leakage_correction(
                Stokes_coefficients=Stokes_coefficients.real, ocean_mask=ocean_mask
            ) + single_leakage_correction(Stokes_coefficients=Stokes_coefficients.imag, ocean_mask=ocean_mask)

        frequencial_Stokes_coefficients[:, :, :, i_frequency] = Stokes_coefficients

    # Gets coefficients back in EWH domain.
    with errstate(invalid="ignore", divide="ignore"):
        return multiply(
            concatenate(
                (  # Adds a line of one values for degree zero.
                    ones(shape=(1, len(Love_numbers.axes["frequencies"]))),
                    multiply(
                        nan_to_num(
                            x=1 / Love_numbers.values[Direction.potential][BoundaryCondition.load],
                            nan=0.0,
                        ).T,
                        (2 * Love_numbers.axes["degrees"] + 1) / 3,
                    ).T
                    / DENSITY_RATIO,
                ),
                axis=0,
            ).T,
            frequencial_Stokes_coefficients.transpose((0, 2, 3, 1)),
        ).transpose((0, 3, 1, 2))


def single_leakage_correction(Stokes_coefficients: ndarray[float], ocean_mask: ndarray[float]) -> ndarray[float]:
    """
    Performs a correction for continental data leak on oceans and ocean data leak on continents.
    Does not iterate.
    Returns the corrective term.
    """

    # TODO.
    filtering = lambda x: x

    n_max = len(Stokes_coefficients[0]) - 1
    grid = MakeGridDH(Stokes_coefficients, sampling=2, lmax=n_max)
    return map_sampling(
        map=(
            # Oceanic leakage on continents.
            MakeGridDH(
                filtering(x=map_sampling(map=grid * ocean_mask, n_max=n_max, harmonic_domain=True)[0]),
                sampling=2,
                lmax=n_max,
            )
            * (1.0 - ocean_mask)
            # Continental leakage on oceans.
            + MakeGridDH(
                filtering(x=map_sampling(map=grid * (1.0 - ocean_mask), n_max=n_max, harmonic_domain=True)[0]),
                sampling=2,
                lmax=n_max,
            )
            * ocean_mask
        ),
        n_max=n_max,
        harmonic_domain=True,
    )[0]
