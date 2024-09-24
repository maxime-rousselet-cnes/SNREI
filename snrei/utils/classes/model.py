from json import load
from pathlib import Path
from typing import Optional

from numpy import inf, linspace, ndarray, sum
from scipy import interpolate

from ..database import save_base_model
from .description_layer import DescriptionLayer
from .paths import ModelPart, models_path


class Model:
    """
    Describes physical quantities by polynomials depending on the unitless radius.
    Can be used to encode all different parts of some rheology.
    """

    # Names of the spherical layers.
    layer_names: list[Optional[str]]
    # Boundaries of the spherical layers.
    r_limits: list[float]
    # Name of the physical quantities.
    variable_names: list[str]
    # Constant values in the crust depending on 'real_crust' boolean. The keys are the variable names.
    crust_values: dict[str, Optional[float]]

    # Polynomials (depending on x := unitless r) of physical quantities describing the planetary model. The keys are the
    # variable names. They should include:
    #   - for elasticity part:
    #       - Vs: S wave velocity (m.s^-1).
    #       - Vp: P wave velocity (m.s^-1).
    #       - rho_0: Density (kg.m^-3).
    #   - for long term anelasticity part:
    #       - eta_m: Maxwell's viscosity (Pa.s).
    #       - eta_k: Kelvin's viscosity (Pa.s).
    #       - mu_K1: Kelvin's elasticity constant term (Pa).
    #       - c: Elasticities ratio, such as mu_K = c * mu_E + mu_K1 (Unitless).
    #   - for short term anelasticity part:
    #       - alpha: (Unitless).
    #       - omega_m: (Hz).
    #       - tau_M: (yr).
    #       - asymptotic_mu_ratio: Defines mu(omega -> 0.0) / mu_0 (Unitless).
    #       - Q_mu: Quality factor (unitless).
    polynomials: dict[str, list[list[float | str]]]

    def __init__(self, name: str, model_part: ModelPart):
        """
        Loads the model file while managing infinite values.
        """
        # Loads file.
        filepath = models_path[model_part].joinpath(name + ("" if ".json" in name else ".json"))
        with open(filepath, "r") as file:
            loaded_content = load(fp=file)
        # Gets attributes.
        self.layer_names = loaded_content["layer_names"]
        self.r_limits = loaded_content["r_limits"]
        self.variable_names = loaded_content["variable_names"]
        self.crust_values = loaded_content["crust_values"]
        self.polynomials = loaded_content["polynomials"]
        # Manages infinite
        for parameter, polynomials_per_layer in self.polynomials.items():
            for i_layer, polynomial in enumerate(polynomials_per_layer):
                if inf in polynomial:
                    self.polynomials[parameter][i_layer] = ["inf"]

    def save(self, name: str, path: Path):
        """
        Method to save in (.JSON) file.
        """
        save_base_model(obj=self.__dict__, name=name, path=path)

    def build_description_layers_list(self, radius_unit: float, spline_number: int, real_crust: bool) -> list[DescriptionLayer]:
        """
        Constructs the layers of an Earth description given model polynomials.
        """
        description_layers = []
        for r_inf, r_sup, layer_name, layer_polynomials in zip(
            self.r_limits[:-1],
            self.r_limits[1:],
            self.layer_names,
            [
                {variable_name: variable_polynomials[i] for variable_name, variable_polynomials in self.polynomials.items()}
                for i in range(len(self.layer_names))
            ],
        ):
            description_layers += [
                self.build_description_layer(
                    r_inf=r_inf,
                    r_sup=r_sup,
                    layer_name=layer_name,
                    radius_unit=radius_unit,
                    spline_number=spline_number,
                    layer_polynomials=layer_polynomials,
                    real_crust=real_crust,
                )
            ]
        return description_layers

    def build_description_layer(
        self,
        r_inf: float,
        r_sup: float,
        layer_name: Optional[str],
        radius_unit: float,
        spline_number: int,
        layer_polynomials: dict[str, list[float | str]],
        real_crust: bool,
    ) -> DescriptionLayer:
        """
        Constructs a layer of an Earth description given its model polynomials.
        """
        x = linspace(r_inf, r_sup, spline_number) / radius_unit
        return DescriptionLayer(
            name=layer_name,
            x_inf=x[0],
            x_sup=x[-1],
            splines={
                variable_name: self.create_spline(
                    x=x,
                    polynomial=polynomial,
                    layer_name=layer_name,
                    real_crust=real_crust,
                    crust_value=self.crust_values[variable_name],
                )
                for variable_name, polynomial in layer_polynomials.items()
            },
        )

    def create_spline(
        self,
        x: ndarray,
        polynomial: list[float | str],
        layer_name: Optional[str],
        real_crust: bool,
        crust_value: Optional[float],
    ) -> tuple[ndarray | float, ndarray | float, int]:
        """
        Creates a polynomial spline structure to approximate a given physical quantity.
        Infinite values and modified crust values are handled.
        """
        if "inf" in polynomial:
            return inf, inf, 0
        else:
            return interpolate.splrep(
                x=x,
                y=sum(
                    [
                        (crust_value if "CRUST_2" in layer_name and not real_crust and i == 0 and crust_value != "None" else coefficient) * x**i
                        for i, coefficient in enumerate(polynomial)
                    ],
                    axis=0,
                ),
                k=max(len(polynomial) - 1, 1),
            )
