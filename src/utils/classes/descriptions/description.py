from pathlib import Path
from typing import Optional

from numpy import Inf, array, ndarray

from ...database import load_base_model, save_base_model
from ...paths import descriptions_path, models_path
from ..description_layer import DescriptionLayer
from ..model import Model, ModelPart
from ..spline import Spline


class Description:
    """
    Defines preprocessed model polynomial splines, layer by layer.
    """

    # Proper attributes.
    id: str
    model_filename: Optional[str]
    model_part: Optional[ModelPart]

    # Attributes to memorize hyper parameters.
    radius_unit: float
    real_crust: bool
    spline_number: int

    # Actual model's description.
    description_layers: list[DescriptionLayer]

    def __init__(
        self,
        id: Optional[str] = None,
        model_filename: Optional[str] = None,
        model_part: Optional[ModelPart] = None,
        radius_unit: Optional[float] = None,
        real_crust: Optional[bool] = None,
        spline_number: Optional[int] = None,
    ) -> None:
        # Initializes IDs.
        self.id = id if id else model_filename
        self.model_filename = model_filename
        self.model_part = model_part
        # Updates attributes.
        self.radius_unit = radius_unit
        self.real_crust = real_crust
        self.spline_number = spline_number
        # Initializes description layers as empty.
        self.description_layers = []

    def build(self, overwrite_description: bool = True):
        """
        Builds description layers from model file parameters.
        """
        # Loads (.JSON) model's file.
        model: Model = load_base_model(
            name=self.model_filename,
            path=models_path[self.model_part],
            base_model_type=Model,
        )
        # Gets layers descriptions.
        self.description_layers = model.build_description_layers_list(
            radius_unit=self.radius_unit,
            spline_number=self.spline_number,
            real_crust=self.real_crust,
        )

    def load(self) -> None:
        """
        Loads a Description instance with correctly formatted attributes.
        """
        # Gets raw description.
        description_dict: dict = load_base_model(name=self.id, path=descriptions_path)
        # Formats attributes.
        for key, value in description_dict.items():
            setattr(self, key, value)
        # Formats layers.
        for i_layer, layer in enumerate(description_dict["description_layers"]):
            self.description_layers[i_layer] = DescriptionLayer(**layer)
            for variable_name, spline in layer["splines"].items():
                # Handles infinite values, as strings in files but as Inf float for computing.
                if not isinstance(spline[0], list) and spline[0] == "Inf":
                    self.description_layers[i_layer].splines[variable_name] = (Inf, Inf, 0)
                else:
                    spline: Spline = (
                        array(self.description_layers[i_layer].splines[variable_name][0]),
                        array(self.description_layers[i_layer].splines[variable_name][1]),
                        self.description_layers[i_layer].splines[variable_name][2],
                    )
                    # Formats every polynoimial spline as a scipy polynomial spline.
                    self.description_layers[i_layer].splines[variable_name] = spline
        # Formats variable array values.
        if "variable_values_per_layer" in description_dict.keys():
            layer_values_list: list[dict[str, list[float]]] = description_dict["variable_values_per_layer"]
            self.variable_values_per_layer: list[dict[str, ndarray]] = [
                {variable_name: array(values, dtype=float) for variable_name, values in layer_values.items()}
                for layer_values in layer_values_list
            ]

    def save(self, path: Path) -> None:
        """
        Saves the Description instance in a (.JSON) file.
        """
        # Converts Infinite values to strings.
        for i_layer, layer in enumerate(self.description_layers):
            for variable_name, spline in layer.splines.items():
                if not isinstance(spline[0], ndarray) and spline[0] == Inf:
                    self.description_layers[i_layer].splines[variable_name] = ("Inf", "Inf", 0)
        # Saves as basic type.
        save_base_model(obj=self.__dict__, name=self.id, path=path)
        # Converts back to numpy.Inf.
        for i_layer, layer in enumerate(self.description_layers):
            for variable_name, spline in layer.splines.items():
                if not isinstance(spline[0], ndarray) and spline[0] == "Inf":
                    self.description_layers[i_layer].splines[variable_name] = (Inf, Inf, 0)
