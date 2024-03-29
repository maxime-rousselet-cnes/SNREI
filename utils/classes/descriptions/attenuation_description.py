from pathlib import Path
from typing import Optional

from ...paths import attenuation_descriptions_path, attenuation_models_path
from .description import Description


class AttenuationDescription(Description):
    """
    Includes attenuation part description layers (with units).
    """

    def __init__(
        self,
        radius_unit: float,
        real_crust: bool,
        n_splines_base: int,
        model_filename: str,
        models_path: Path = attenuation_models_path,
        id: Optional[str] = None,
        load_description: bool = True,
        save: bool = True,
    ) -> None:
        if load_description:
            super().__init__(id=id)
            return
        super().__init__(
            models_path=models_path,
            radius_unit=radius_unit,
            real_crust=real_crust,
            n_splines_base=n_splines_base,
            id=id,
            model_filename=model_filename,
        )
        if save:
            self.save(path=attenuation_descriptions_path)
