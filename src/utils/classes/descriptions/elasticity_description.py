from typing import Optional

from numpy import pi, sqrt
from scipy import interpolate

from ...rheological_formulas import g_0_computing, lambda_0_computing, mu_0_computing
from ..constants import G
from ..model import ModelPart
from .description import Description


class ElasticityDescription(Description):
    """
    Includes the integration constants and all elasticity part description layers (unitless).
    """

    # Proper fields.
    x_CMB: float
    period_unit: float
    density_unit: float
    speed_unit: float
    piG: float

    # Fields to memorize hyper parameters.
    below_ICB_layers: Optional[int]
    below_CMB_layers: Optional[int]
    spline_degree: Optional[int]

    def __init__(
        self,
        # Inherited parameters.
        id: Optional[str] = None,
        model_filename: Optional[str] = None,
        radius_unit: Optional[float] = None,
        real_crust: Optional[float] = None,
        spline_number: Optional[int] = None,
        # Proper parameters.
        below_ICB_layers: Optional[int] = None,
        below_CMB_layers: Optional[int] = None,
        spline_degree: Optional[int] = None,
    ) -> None:

        # Updates inherited fields.
        super().__init__(
            id=id,
            model_filename=model_filename,
            model_part=ModelPart.elasticity,
            radius_unit=radius_unit,
            real_crust=real_crust,
            spline_number=spline_number,
        )

        # Updates proper fields.
        self.below_ICB_layers = below_ICB_layers
        self.below_CMB_layers = below_CMB_layers
        self.spline_degree = spline_degree

    def find_fluid_layers(self) -> tuple[int, int]:
        """
        Counts the number of layers describing the Inner-Core and the Outer-Core.
        All Outer-Core layers should include "FLUID" in their name.
        """

        below_ICB_layers, below_CMB_layers = 0, 0

        # Iterates on layer names from Geocenter.
        for layer_name in [description_layer.name for description_layer in self.description_layers]:
            if "FLUID" in layer_name:
                below_CMB_layers += 1
            elif below_ICB_layers == below_CMB_layers:
                below_CMB_layers += 1
                below_ICB_layers += 1
            else:
                return below_ICB_layers, below_CMB_layers

        return below_ICB_layers, below_CMB_layers

    def build(self, overwrite_description: bool = True, save: bool = True):
        """
        Builds description layers from model file parameters and preprocesses elasticity variables.
        """

        # Initializes description layers from model.
        super().build(model_part=ModelPart.elasticity, save=False)

        # Updates basic fields.
        if self.below_ICB_layers is None or self.below_CMB_layers is None:
            below_ICB_layers, below_CMB_layers = self.find_fluid_layers()
            if self.below_CMB_layers is None:
                self.below_CMB_layers = below_CMB_layers
            if self.below_ICB_layers is None:
                self.below_ICB_layers = below_ICB_layers

        self.x_CMB = self.description_layers[below_CMB_layers].x_inf

        # Defines units.
        self.density_unit = self.description_layers[below_CMB_layers].evaluate(
            x=self.x_CMB, variable="rho_0"
        )  # := rho_0(CMB+) (kg.m^-3).
        self.period_unit = 1.0 / sqrt(self.density_unit * pi * G)  # (s).
        self.speed_unit = self.radius_unit / self.period_unit
        self.piG = 1.0  # By definition of this unit system.

        # Preprocesses unitless variables, including g_0, mu_0 and lambda_0.
        g_0_inf = 0.0  # g_0 at the bottom of the layer (unitless).
        for i_layer, description_layer in enumerate(self.description_layers):

            # Gets unitless variables.
            for variable_name, variable_unit in [
                ("Vs", self.speed_unit),
                ("Vp", self.speed_unit),
                ("rho_0", self.density_unit),
            ]:
                self.description_layers[i_layer].splines[variable_name] = (
                    description_layer.splines[variable_name][0],
                    description_layer.splines[variable_name][1] / variable_unit,
                    description_layer.splines[variable_name][2],
                )

            # Computes g_0, mu_0, lambda_0.
            x = description_layer.x_profile(spline_number=self.spline_number)
            rho_0 = description_layer.evaluate(x=x, variable="rho_0")
            Vs = description_layer.evaluate(x=x, variable="Vs")
            Vp = description_layer.evaluate(x=x, variable="Vp")
            mu_0 = mu_0_computing(rho_0=rho_0, Vs=Vs)
            lambda_0 = lambda_0_computing(rho_0=rho_0, Vp=Vp, mu_0=mu_0)
            g_0 = g_0_computing(
                x=x,
                piG=self.piG,
                rho_0=rho_0,
                g_0_inf=g_0_inf,
                x_inf=description_layer.x_inf,
                spline_number=self.spline_number,
            )

            # Updates unitless variables.
            g_0_inf = g_0[-1]
            self.description_layers[i_layer].splines["g_0"] = interpolate.splrep(x=x, y=g_0, k=self.spline_degree)
            self.description_layers[i_layer].splines["mu_0"] = interpolate.splrep(x=x, y=mu_0, k=self.spline_degree)
            self.description_layers[i_layer].splines["lambda_0"] = interpolate.splrep(x=x, y=lambda_0, k=self.spline_degree)

        # Eventually saves in (.JSON) file.
        if save:
            self.save(overwrite_description=overwrite_description)
