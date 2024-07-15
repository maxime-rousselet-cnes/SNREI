from enum import Enum
from pathlib import Path
from typing import Optional

from numpy import Inf, array, expand_dims, ndarray, ones

from ..database import load_base_model, save_base_model
from .hyper_parameters import HyperParameters


class Direction(Enum):
    """
    Result keys for values dict.
    """

    radial = 0
    tangential = 1
    potential = 2


class BoundaryCondition(Enum):
    """
    Result keys for values sub-dict.
    """

    load = 0
    shear = 1
    potential = 2


class Values(dict[Direction, dict[BoundaryCondition, ndarray[complex]]]):
    """
    Defines some result values.
    """

    pass


class Result:
    """
    Describes computing results, such as Love numbers.
    """

    hyper_parameters: HyperParameters
    values: Values
    axes: dict[str, ndarray[complex]]

    def __init__(
        self,
        values: Optional[Values] = None,
        hyper_parameters: Optional[HyperParameters] = None,
        axes: Optional[dict[str, ndarray[complex]]] = None,
    ) -> None:
        self.hyper_parameters = hyper_parameters
        self.values = values
        self.axes = axes

    def update_values_from_array(
        self,
        result_array: ndarray,
    ) -> None:
        """
        Converts p-dimmensionnal array to 'values' field, p >=3.
        The axis in position -1 should count 9 components corresponding to every combination of Direction and BoundaryCondition.
        """

        result_shape = result_array.shape[:-1]
        non_radial_factor = (
            array(object=self.axes["degrees"], dtype=int)
            if len(result_shape) == 1
            else expand_dims(a=self.axes["degrees"], axis=1)
        )
        radial_factor = ones(shape=result_shape)

        self.values = {
            direction: {
                boundary_condition: (
                    result_array[:, i_direction + 3 * i_boundary_condition]
                    if len(result_shape) == 1
                    else result_array[:, :, i_direction + 3 * i_boundary_condition]
                )
                * (
                    radial_factor
                    if direction == Direction.radial
                    else non_radial_factor
                )
                for i_boundary_condition, boundary_condition in enumerate(
                    BoundaryCondition
                )
            }
            for i_direction, direction in enumerate(Direction)
        }

    def save(self, name: str, path: Path):
        """
        Saves the results in a (.JSON) file. Handles Enum classes. Converts complex values arrays to
        """

        save_base_model(
            obj={
                "hyper_parameters": self.hyper_parameters,
                "values": {
                    key.value: {
                        sub_key.value: (
                            {"real": sub_values}
                            if not isinstance(sub_values.flatten()[0], complex)
                            else {"real": sub_values.real, "imag": sub_values.imag}
                        )
                        for sub_key, sub_values in values.items()
                    }
                    for key, values in self.values.items()
                },
                "axes": {
                    axe_name: (
                        (
                            {"real": ["Inf"] if axe_values[0] == Inf else axe_values}
                            if not isinstance(axe_values.flatten()[0], complex)
                            else {"real": axe_values.real, "imag": axe_values.imag}
                        )
                    )
                    for axe_name, axe_values in self.axes.items()
                },
            },
            name=name,
            path=path,
        )

    def load(self, name: str, path: Path) -> None:
        """
        Loads a Result structure from (.JSON) file.
        """

        loaded_content = load_base_model(
            name=name,
            path=path,
        )

        self.hyper_parameters = HyperParameters(**loaded_content["hyper_parameters"])
        result_values: dict[str, dict[str, dict[str, list[float]]]] = loaded_content[
            "values"
        ]
        result_axes: dict[str, dict[str, list[float]]] = loaded_content["axes"]
        self.values = Values(
            {
                Direction(int(direction)): {
                    (BoundaryCondition(int(boundary_condition))): array(
                        object=sub_values["real"]
                    )
                    + (
                        0.0
                        if not ("imag" in sub_values.keys())
                        else array(object=sub_values["imag"])
                    )
                    * 1.0j
                    for boundary_condition, sub_values in values.items()
                }
                for direction, values in result_values.items()
            }
        )

        self.axes = {
            axe_name: array(
                object=[Inf] if "Inf" in axe_values["real"] else axe_values["real"]
            )
            + (
                0.0
                if not ("imag" in axe_values.keys())
                else array(object=axe_values["imag"])
            )
            * 1.0j
            for axe_name, axe_values in result_axes.items()
        }
