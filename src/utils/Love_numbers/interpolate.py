from pathlib import Path
from typing import Callable, Optional

from numpy import concatenate, flip, ndarray
from scipy import interpolate

from ...functions import get_degrees_indices
from ..abstract_computing import build_hermitian
from ..classes import (
    BoundaryCondition,
    Direction,
    Result,
    RunHyperParameters,
    results_path,
)
from ..database import get_run_folder_name, load_base_model


def interpolate_Love_numbers(
    anelasticity_description_id: str,
    target_frequencies: ndarray[float],  # (Hz).
    option: RunHyperParameters,
    degrees: Optional[list[int]] = None,
    directions: list[Direction] = [Direction.radial, Direction.tangential, Direction.potential],
    boundary_conditions: list[BoundaryCondition] = [
        BoundaryCondition.load,
        BoundaryCondition.shear,
        BoundaryCondition.potential,
    ],
    function: Callable[[ndarray], ndarray] = lambda x: x,  # Element-wise transformation to apply eventually.
) -> tuple[Result, Result, ndarray[int], Path]:
    """
    Gets the wanted anelastic Love numbers from the wanted description, with wanted options and interpolates them to the wanted
    frequencies.
    """
    # Initializes.
    result_subpath = results_path.joinpath(
        get_run_folder_name(anelasticity_description_id=anelasticity_description_id, run_id=option.run_id())
    )
    anelastic_Love_numbers = Result()
    anelastic_Love_numbers.load(name="anelastic_Love_numbers", path=result_subpath)
    elastic_Love_numbers = Result()
    elastic_Love_numbers.load(name="elastic_Love_numbers", path=result_subpath.parent.parent)
    source_frequencies = load_base_model(name="frequencies", path=result_subpath)
    base_degrees = load_base_model(name="degrees", path=result_subpath.parent.parent)
    if degrees is None:
        degrees = base_degrees
    # Interpolates Love numbers on signal frequencies as hermitian signal.
    symmetric_source_frequencies = concatenate((-flip(m=source_frequencies), source_frequencies))
    return (
        Result(
            values={
                direction: {
                    boundary_condition: [
                        interpolate.interp1d(
                            x=symmetric_source_frequencies,
                            y=build_hermitian(
                                signal=function(
                                    (1.0 if direction == Direction.potential else 0.0)
                                    + (
                                        anelastic_Love_numbers.values[direction][boundary_condition][degree_index]
                                        / (1.0 if direction == Direction.radial else degree)
                                    )
                                )
                            ),
                            kind="linear",
                        )(x=target_frequencies)
                        for degree_index, degree in zip(
                            get_degrees_indices(degrees=base_degrees, degrees_to_plot=degrees), degrees
                        )
                    ]
                    for boundary_condition in boundary_conditions
                }
                for direction in directions
            }
        ),
        Result(
            values={
                direction: {
                    boundary_condition: function(
                        (1.0 if direction == Direction.potential else 0.0)
                        + (
                            elastic_Love_numbers.values[direction][boundary_condition]
                            / (1.0 if direction == Direction.radial else degrees)
                        )
                    )
                    for boundary_condition in boundary_conditions
                }
                for direction in directions
            }
        ),
        degrees,
        result_subpath,
    )
