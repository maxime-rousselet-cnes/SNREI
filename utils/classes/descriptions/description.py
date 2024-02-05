from pathlib import Path
from typing import Optional

from numpy import array, ndarray

from ...database import generate_id, load_base_model, save_base_model
from ..description_layer import DescriptionLayer
from ..model import Model


class Description:
    """
    Defines model polynomial splines, layer by layer.
    """

    # Proper attributes.
    id: str
    model_filename: Optional[str]
    description_layers: list[DescriptionLayer]

    # Attributes to memorize hyper parameters.
    radius_unit: float
    real_crust: bool
    n_splines_base: int

    def __init__(
        self,
        models_path: Optional[Path] = None,
        radius_unit: Optional[float] = None,
        real_crust: Optional[bool] = None,
        n_splines_base: Optional[int] = None,
        id: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> None:
        # Base initialization.
        self.id = id if id else generate_id()
        self.model_filename = model_filename
        self.description_layers = []

        # Builds description from model file if necessary.
        if self.model_filename and n_splines_base and models_path:
            # Loads (.JSON) models file.
            model: Model = load_base_model(
                name=model_filename,
                path=models_path,
                base_model_type=Model,
            )
            # Gets layers descriptions.
            self.description_layers = model.build_description_layers_list(
                radius_unit=radius_unit,
                n_splines_base=n_splines_base,
                real_crust=real_crust,
            )

        # Updates attributes.
        self.radius_unit = radius_unit
        self.real_crust = real_crust
        self.n_splines_base = n_splines_base

    def load(self, path: Path) -> None:
        """
        Loads a Description instance with correctly formatted fields.
        """
        # Gets raw description.
        description_dict: dict = load_base_model(name=self.id, path=path)
        # Formats attributes.
        for key, value in description_dict.items():
            setattr(self, key, value)
        # Formats layers.
        for i_layer, layer in enumerate(description_dict["description_layers"]):
            self.description_layers[i_layer] = DescriptionLayer(**layer)
        if "variable_values_per_layer" in description_dict.keys():
            self.variable_values_per_layer: list[dict[str, ndarray]] = [
                {variable_name: array(values) for variable_name, values in layer_values.items()}
                for layer_values in description_dict["variable_values_per_layer"]
            ]

    def save(self, path: Path) -> None:
        """
        Saves the Description instance in a (.JSON) file.
        """
        save_base_model(obj=self.__dict__, name=self.id, path=path)
