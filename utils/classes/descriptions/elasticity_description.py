from pathlib import Path
from typing import Optional

from numpy import linspace, pi, sqrt
from scipy import interpolate

from ...constants import G
from ...formulas import g_0_computing, lambda_0_computing, mu_0_computing
from ...paths import elasticity_descriptions_path, elasticity_models_path
from .description import Description


class ElasticityDescription(Description):
    """
    Includes the integration constants and all layer elasticity part description layers (unitless).
    """

    # Proper attributes.
    period_unit: float
    density_unit: float
    speed_unit: float
    piG: float

    # Attributes to memorize hyper parameters.
    below_ICB_layers: int
    below_CMB_layers: int
    profile_precision: int
    splines_degree: int

    def __init__(
        self,
        radius_unit: float,
        below_ICB_layers: Optional[int],
        below_CMB_layers: Optional[int],
        real_crust: bool,
        n_splines_base: int,
        profile_precision: int,
        splines_degree: int,
        model_filename: str,
        models_path: Path = elasticity_models_path,
        id: Optional[str] = None,
        load_description: bool = True,
    ) -> None:
        if load_description:
            super().__init__(id=id)
            return
        # Gets model polynomials and builds description layers.
        super().__init__(
            models_path=models_path,
            radius_unit=radius_unit,
            real_crust=real_crust,
            n_splines_base=n_splines_base,
            id=id,
            model_filename=model_filename,
        )
        # Saves basic parameters.
        if below_ICB_layers == None or below_CMB_layers == None:
            below_ICB_layers, below_CMB_layers = self.find_fluid_layers(
                [description_layer.name for description_layer in self.description_layers]
            )

        # Defines units.
        self.radius_unit = radius_unit  # (m).
        self.density_unit = self.description_layers[below_CMB_layers].evaluate(
            x=self.description_layers[below_CMB_layers].x_inf, variable="rho_0"
        )  # rho_e := rho_0(CMB+) (kg.m^-3).
        self.period_unit = 1.0 / sqrt(self.density_unit * pi * G)  # (s).
        self.speed_unit = self.radius_unit / self.period_unit
        self.piG = 1.0  # By definition.

        # Preprocesses unitless variables, including g_0, mu_0 and lambda_0.
        g_0_inf = 0.0  # g_0 at th  bottom of the layer (unitless).
        for i_layer, description_layer in enumerate(self.description_layers):
            # Gets unitless variables.
            for variable_name, variable_unit in [
                ("Vs", self.speed_unit),
                ("Vp", self.speed_unit),
                ("rho_0", self.density_unit),
                ("Qmu", 1.0),
            ]:
                self.description_layers[i_layer].splines[variable_name] = (
                    description_layer.splines[variable_name][0],
                    description_layer.splines[variable_name][1] / variable_unit,
                    description_layer.splines[variable_name][2],
                )

            # Computes g_0, mu_0, lambda_0.
            x = linspace(description_layer.x_inf, description_layer.x_sup, profile_precision)
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
                profile_precision=profile_precision,
            )
            # Updates unitless variables.
            g_0_inf = g_0[-1]
            self.description_layers[i_layer].splines["g_0"] = interpolate.splrep(x=x, y=g_0, k=splines_degree)
            self.description_layers[i_layer].splines["mu_0"] = interpolate.splrep(x=x, y=mu_0, k=splines_degree)
            self.description_layers[i_layer].splines["lambda_0"] = interpolate.splrep(x=x, y=lambda_0, k=splines_degree)

        # Updates attributes.
        self.below_ICB_layers = below_ICB_layers
        self.below_CMB_layers = below_CMB_layers
        self.profile_precision = profile_precision
        self.splines_degree = splines_degree

        # Saves in (.JSON) file.
        self.save(path=elasticity_descriptions_path)

    def find_fluid_layers(self, layer_names: list[str]) -> tuple[int, int]:
        """
        Counts the number of layers describing the Inner-Core and the Outer-Core.
        """
        below_ICB_layers, below_CMB_layers = 0, 0
        for layer_name in layer_names:
            if "FLUID" in layer_name:
                below_CMB_layers += 1
            elif below_ICB_layers == below_CMB_layers:
                below_CMB_layers += 1
                below_ICB_layers += 1
            else:
                return below_ICB_layers, below_CMB_layers
        return below_ICB_layers, below_CMB_layers
