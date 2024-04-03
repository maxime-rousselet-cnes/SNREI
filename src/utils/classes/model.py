from enum import Enum
from typing import Optional

from numpy import Inf, linspace, ndarray, sum
from pydantic import BaseModel
from scipy import interpolate

from .description_layer import DescriptionLayer
from .spline import Spline


class ModelPart(Enum):
    """
    Available model parts.
    """

    elasticity = "elasticity"
    long_term_anelasticity = "long_term_anelasticity"
    short_term_anelasticity = "short_term_anelasticity"


class Model(BaseModel):
    """
    Describes physical quantities by 3rd order polynomials depending on the unitless radius.
    Can be used to encode different parts of some rheology:
        - elasticity part
        - anelasticity part
        - attenuation part
    """

    # Names of the spherical layers.
    layer_names: list[Optional[str]]
    # Boundaries of the spherical layers.
    r_limits: list[float]
    # Name of the physical quantities.
    variable_names: list[str]
    # Constant values in the crust depending on 'real_crust' boolean. The keys are the variable names.
    crust_values: dict[str, Optional[float]]

    # 3rd order polynomials (depending on x := unitless r) of physical quantities describing the planetary model. The keys are
    # the variable names. They should include:
    #   - for elasticity part:
    #       - Vs: S wave velocity (m.s^-1).
    #       - Vp: P wave velocity (m.s^-1).
    #       - rho_0: Density (kg.m^-3).
    #       - Qmu: Quality factor (unitless).
    #   - for anelasticity part:
    #       - eta_m: Maxwell's viscosity (Pa.s).
    #       - eta_k: Kelvin's viscosity (Pa.s).
    #       - mu_K1: Kelvin's elasticity constant term (Pa).
    #       - c: Elasticities ratio, such as mu_K = c * mu_E + mu_K1 (Unitless).
    #   - for attenuation part:
    #       - alpha: (Unitless).
    #       - omega_m: (Hz).
    #       - tau_M: (y).
    #       - asymptotic_mu_ratio: Defines mu(omega->0.0) / mu_0 (Unitless).
    polynomials: dict[str, list[list[float | str]]]

    def build_description_layers_list(
        self, radius_unit: float, n_splines_base: int, real_crust: bool
    ) -> list[DescriptionLayer]:
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
                    n_splines_base=n_splines_base,
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
        n_splines_base: int,
        layer_polynomials: dict[str, list[float | str]],
        real_crust: bool,
    ) -> DescriptionLayer:
        """
        Constructs a layer of an Earth description given its model polynomials.
        """
        x = linspace(r_inf, r_sup, n_splines_base) / radius_unit
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
    ) -> Spline:
        """
        Creates a polynomial spline structure to approximate a given physical quantity.
        Infinite values and modified crust values are handled.
        """
        polynomial_degree = len(polynomial) - 1
        if Inf in polynomial:
            return Inf, Inf, 0
        else:
            return interpolate.splrep(
                x=x,
                y=sum(
                    [
                        (
                            crust_value
                            if layer_name == "CRUST 2" and not real_crust and i == 0 and crust_value != "None"
                            else coefficient
                        )
                        * x**i
                        for i, coefficient in enumerate(polynomial)
                    ],
                    axis=0,
                ),
                k=polynomial_degree,
            )
