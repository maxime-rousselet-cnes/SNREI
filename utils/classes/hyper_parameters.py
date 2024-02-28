from typing import Any, Optional

from pydantic import BaseModel

from ..database import load_base_model
from ..paths import parameters_path


class HyperParameters(BaseModel):
    """
    Abstract class.
    """

    pass


class YSystemHyperParameters(HyperParameters):
    """
    Describes the parameters needed for the Y_i system integration algorithm.
    """

    # Physical parameters.
    n_max_for_sub_CMB_integration: int  # Maximal degree for integration under the Core-Mantle Boundary.
    homogeneous_solution: bool  # Whether to use analytical solution for homogeneous sphere at r ~= 0 km or not.
    dynamic_term: bool  # Whether to use omega^2 terms or not.
    first_order_cross_terms: bool  # Wheter to use lambda_prime an mu_prime terms or not.
    minimal_radius: float  # (m).

    # Solver parameters.
    method: str  # Solver's numerical integration method.
    atol: float  # The solver keeps the local error estimates less than atol + rtol * abs(y).
    rtol: float  # The solver keeps the local error estimates less than atol + rtol * abs(y).
    t_eval: Optional[Any]


class RealDescriptionParameters(HyperParameters):
    """
    Describe the hyper parameters needed to build an Earth real description.
    """

    # Physical parameters.
    radius: Optional[float]  # Planet's radius.
    radius_unit: Optional[float]  # Length unit.
    real_crust: bool  # Whether to use 'real_crust' values or not.

    # Numerical parameters.
    n_splines_base: int  # Should be >= 1 + polynomials degree.
    profile_precision: int  # Should be >= 2.
    splines_degree: int
    # Number of layers under boundaries. If they are None: Automatic detection using elasticity model layer names.
    # Number of layers under the Inner-Core Boundary.
    below_ICB_layers: Optional[int]  # Should be > 0.
    # Number of total layers under the Mantle-Core Boundary.
    below_CMB_layers: Optional[int]  # Should be > below_ICB_layers.


class LoveNumbersHyperParameters(HyperParameters):
    """
    Describes the parameters needed for Love Numbers(n, frequency) computing algorithm.
    """

    # Adaptative step (on frequency) algorithm parameters.
    frequency_min: float  # log10(Low frequency limit (Hz)).
    frequency_max: float  # log10(High frequency limit (Hz)).
    n_frequency_0: int  # Minimal number of computed frequencies per degree.
    max_tol: float  # Maximal curvature criteria between orders 1 and 2.
    decimals: int  # Precision in log10(frequency / frequency_unit).

    # Lower level parameters. They can be file names.
    real_description_parameters: Optional[str] | RealDescriptionParameters  # To build an Earth Complete description.
    y_system_hyper_parameters: Optional[str] | YSystemHyperParameters  # For the Y_i system integration algorithm.
    degree_steps: Optional[str] | list[int]  # Love numbers are computed every degree_steps[i] between...
    degree_thresholds: Optional[str] | list[int]  # ... degree_thresholds[i] and degree_thresholds[i + 1].

    # Run parameters.
    use_anelasticity: bool  # Whether to use attenuation model or not.
    use_attenuation: bool  # Whether to use attenuation model or not.
    bounded_attenuation_functions: bool  # Whether to use bounded version of attenuation functions or not.

    def load(self) -> None:
        """
        Loads the lower level parameter fields from file names.
        """
        #  Parameters for the Y_i system integration algorithm.
        if not isinstance(self.y_system_hyper_parameters, YSystemHyperParameters):
            self.y_system_hyper_parameters = load_base_model(
                name=self.y_system_hyper_parameters if self.y_system_hyper_parameters else "Y_system_hyper_parameters",
                base_model_type=YSystemHyperParameters,
                path=parameters_path,
            )
        # Parameters to build an Earth Complete description.
        if not isinstance(self.real_description_parameters, RealDescriptionParameters):
            self.real_description_parameters = load_base_model(
                name=self.real_description_parameters if self.real_description_parameters else "real_description_parameters",
                base_model_type=RealDescriptionParameters,
                path=parameters_path,
            )
        # Parameters to build degrees list.
        if not isinstance(self.degree_steps, list):
            self.degree_steps = load_base_model(
                name=self.degree_steps if self.degree_steps else "degree_steps",
                path=parameters_path,
            )
        if not isinstance(self.degree_thresholds, list):
            self.degree_thresholds = load_base_model(
                name=self.degree_thresholds if self.degree_thresholds else "degree_thresholds",
                path=parameters_path,
            )


def load_Love_numbers_hyper_parameters() -> LoveNumbersHyperParameters:
    """
    Routine that gets Love numbers hyper parameters from (.JSON) file.
    """
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name="Love_numbers_hyper_parameters", path=parameters_path, base_model_type=LoveNumbersHyperParameters
    )
    Love_numbers_hyper_parameters.load()
    return Love_numbers_hyper_parameters


class SignalHyperParameters(HyperParameters):
    """
    Describes the parameters needed for to compute some viscoelastic induced signal using elastic induced signal and Love
    numbers.
    """

    # Parameters describing the extended signal.
    spline_time: int
    zero_duration: int
    anti_Gibbs_effect_factor: int
    signal: str

    # Parameters describing spacially-dependent signal.
    weights_map: str
    n_max: int

    # Run parameters.
    use_anelasticity: bool  # Whether to use attenuation model or not.
    use_attenuation: bool  # Whether to use attenuation model or not.
    bounded_attenuation_functions: bool  # Whether to use bounded version of attenuation functions or not.
