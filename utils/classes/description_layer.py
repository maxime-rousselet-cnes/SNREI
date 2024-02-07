from typing import Optional

from numpy import Inf, linspace, ndarray, ones, shape
from pydantic import BaseModel
from scipy import interpolate

from .spline import Spline


class DescriptionLayer(BaseModel):
    """
    Defines a layer of an Earth description, such as:
        - elasticity description
        - anelasticity description
        - attenuation description
        - complete description
    """

    name: Optional[str]
    x_inf: float
    x_sup: float
    splines: dict[str, Spline]

    def evaluate(self, x: ndarray | float, variable: str, derivative_order: int = 0) -> ndarray | float:
        """
        Evaluates some quantity polynomial spline over an array x.
        """
        if not isinstance(self.splines[variable][0], ndarray):  # Handles constant cases.
            return (Inf if self.splines[variable][0] == Inf else self.splines[variable][0]) * ones(  # Handles infinite cases.
                shape=(shape(x))
            )
        return interpolate.splev(x=x, tck=self.splines[variable], der=derivative_order)

    def x_profile(self, profile_precision: int) -> ndarray:
        """
        Builds an array of x values in the layer.
        """
        return linspace(start=self.x_inf, stop=self.x_sup, num=profile_precision)
