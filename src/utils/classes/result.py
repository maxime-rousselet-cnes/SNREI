from enum import Enum
from pathlib import Path
from typing import Optional

from numpy import array, ndarray, ones, transpose

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

    def __init__(
        self,
        values: Optional[Values] = None,
        hyper_parameters: Optional[HyperParameters] = None,
    ) -> None:
        self.hyper_parameters = hyper_parameters
        self.values = values

    def update_values_from_array(
        self,
        result_array: ndarray,
        degrees: list[int] | ndarray,
    ) -> None:
        """
        Converts p-dimmensionnal array to 'values' attribute, p >=3.
        The axis in position 2 should count 9 components corresponding to every combination of Direction and BoundaryCondition.
        """
        result_shape = result_array.shape[:2]
        non_radial_factor = transpose([degrees])
        radial_factor = ones(result_shape)
        self.values = {
            direction: {
                boundary_condition: result_array[:, :, i_direction + 3 * i_boundary_condition]
                * (radial_factor if direction == Direction.radial else non_radial_factor)
                for i_boundary_condition, boundary_condition in enumerate(BoundaryCondition)
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
                            sub_values
                            if not isinstance(sub_values.flatten()[0], complex)
                            else {"real": sub_values.real, "imag": sub_values.imag}
                        )
                        for sub_key, sub_values in values.items()
                    }
                    for key, values in self.values.items()
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
        result: dict[str, dict[str, dict[str, list[float]]]] = loaded_content["values"]
        self.hyper_parameters = (HyperParameters(**loaded_content["hyper_parameters"]),)
        self.values = {
            Direction.radial if direction == "0" else (Direction.tangential if direction == "1" else Direction.potential): {
                (
                    BoundaryCondition.load
                    if boundary_condition == "0"
                    else (BoundaryCondition.shear if boundary_condition == "1" else BoundaryCondition.potential)
                ): array(sub_values["real"])
                + (0.0 if not ("imag" in sub_values.keys()) else array(sub_values["imag"])) * 1.0j
                for boundary_condition, sub_values in values.items()
            }
            for direction, values in result.items()
        }
