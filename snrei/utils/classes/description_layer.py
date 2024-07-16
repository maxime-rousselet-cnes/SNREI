from typing import Optional

from numpy import inf, linspace, ndarray, ones, shape
from pydantic import BaseModel
from scipy import interpolate


class DescriptionLayer(BaseModel):
    """
    Defines a layer of a description.
    """

    name: Optional[str]
    x_inf: float
    x_sup: float
    splines: dict[str, tuple[ndarray | float, ndarray | float, int]]

    class Config:
        arbitrary_types_allowed = True

    def evaluate(self, x: ndarray | float, variable: str, derivative_order: int = 0) -> ndarray | float:
        """
        Evaluates some quantity polynomial spline over an array x.
        """
        if not isinstance(self.splines[variable][0], ndarray):  # Handles constant cases.
            return (
                inf if self.splines[variable][0] == inf else self.splines[variable][0]
            ) * ones(  # Handles infinite cases.
                shape=(shape(x))
            )
        return interpolate.splev(x=x, tck=self.splines[variable], der=derivative_order)

    def x_profile(self, spline_number: int) -> ndarray:
        """
        Builds an array of x values in the layer.
        """
        return linspace(start=self.x_inf, stop=self.x_sup, num=spline_number)
