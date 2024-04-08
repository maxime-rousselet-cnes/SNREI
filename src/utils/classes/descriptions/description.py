from pathlib import Path
from typing import Optional

from numpy import Inf, array, ndarray

from ...database import load_base_model, save_base_model
from ..constants import DEFAULT_MODELS, DEFAULT_SPLINE_NUMBER, EARTH_RADIUS
from ..description_layer import DescriptionLayer, Spline
from ..model import Model, ModelPart
from ..paths import anelasticity_descriptions_path, descriptions_path, models_path


class Description:
    """
    Defines preprocessed model polynomial splines, layer by layer.
    """

    # Proper attributes.
    id: Optional[str]
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
        self.model_filename = DEFAULT_MODELS[model_part] if model_filename is None else model_filename
        self.model_part = model_part
        self.id = id if not (id is None) else self.model_filename
        # Updates fields.
        self.radius_unit = EARTH_RADIUS if radius_unit is None else radius_unit
        self.real_crust = False if real_crust is None else real_crust
        self.spline_number = DEFAULT_SPLINE_NUMBER if spline_number is None else spline_number
        # Initializes description layers as empty.
        self.description_layers = []

    def build(self, overwrite_description: bool = False, save: bool = True):
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
        # Eventually saves.
        if save:
            self.save(overwrite_description=overwrite_description)

    def load(self) -> None:
        """
        Loads a Description instance with correctly formatted fields.
        """
        # Gets raw description.
        description_dict: dict = load_base_model(name=self.id, path=descriptions_path)
        self.model_part = ModelPart(description_dict["model_part"])
        # Formats attributes.
        for key, value in description_dict.items():
            setattr(self, key, value)
        # Formats layers.
        for i_layer, layer in enumerate(description_dict["description_layers"]):
            self.description_layers[i_layer] = DescriptionLayer(**layer)
            splines: dict[str, tuple] = layer["splines"]
            for variable_name, spline in splines.items():
                # Handles infinite values, as strings in files but as Inf float for computing.
                if not isinstance(spline[0], list) and spline[0] == "Inf":
                    self.description_layers[i_layer].splines[variable_name] = (Inf, Inf, 0)
                else:
                    spline: Spline = (
                        array(self.description_layers[i_layer].splines[variable_name][0]),
                        array(self.description_layers[i_layer].splines[variable_name][1]),
                        self.description_layers[i_layer].splines[variable_name][2],
                    )
                    # Formats every polynomial spline as a scipy polynomial spline.
                    self.description_layers[i_layer].splines[variable_name] = spline

    def save(self, overwrite_description: bool = True) -> None:
        """
        Saves the Description instance in a (.JSON) file.
        """
        path = self.get_path()
        if not (path.joinpath("id" + ".json").is_file() and not overwrite_description):
            self_dict = self.__dict__
            self_dict["model_part"] = None if self.model_part is None else self.model_part.value
            layer: DescriptionLayer
            # Converts Infinite values to strings.
            for i_layer, layer in enumerate(self_dict["description_layers"]):
                splines: dict[str, tuple] = layer.splines
                for variable_name, spline in splines.items():
                    if not isinstance(spline[0], ndarray) and spline[0] == Inf:
                        description_layer: DescriptionLayer = self_dict["description_layers"][i_layer]
                        description_layer.splines[variable_name] = ("Inf", "Inf", 0)
                        self_dict["description_layers"][i_layer] = description_layer
            # Saves as basic type.
            save_base_model(
                obj=self_dict,
                name=self.id,
                path=path,
            )
            # Convert back if needed.
            for i_layer, layer in enumerate(self.description_layers):
                splines: dict[str, tuple] = layer.splines
                for variable_name, spline in splines.items():
                    if not isinstance(spline[0], ndarray) and spline[0] == "Inf":
                        self.description_layers[i_layer].splines[variable_name] = (Inf, Inf, 0)

    def get_path(self) -> Path:
        """
        Returns directory path to save the description.
        """
        return anelasticity_descriptions_path if self.model_part is None else descriptions_path[self.model_part]
