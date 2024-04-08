from typing import Any, Optional

from pydantic import BaseModel

from ..database import load_base_model
from .paths import parameters_path


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
    inhomogeneity_gradients: bool  # Wheter to use lambda_prime an mu_prime terms or not.
    minimal_radius: float  # (m).

    # Solver parameters.
    method: str  # Solver's numerical integration method.
    atol: float  # The solver keeps the local error estimates less than atol + rtol * abs(y).
    rtol: float  # The solver keeps the local error estimates less than atol + rtol * abs(y).
    t_eval: Optional[Any]


class AnelasticityDescriptionParameters(HyperParameters):
    """
    Describe the hyper parameters needed to build an anelasticity description.
    """

    # Physical parameters.
    radius_unit: Optional[float]  # Length unit.
    real_crust: bool  # Whether to use 'real_crust' values or not.

    # Numerical parameters.
    spline_number: int  # Should be >= max(2, 1 + polynomials degree).
    spline_degree: int  # Should be >= 0.
    # Number of layers under boundaries. If they are None: Automatic detection using elasticity model layer names.
    # Number of layers under the Inner-Core Boundary.
    below_ICB_layers: Optional[int]  # Should be >= 0.
    # Number of total layers under the Mantle-Core Boundary.
    below_CMB_layers: Optional[int]  # Should be > below_ICB_layers.


class RunHyperParameters(HyperParameters):
    """
    Describes a run's options.
    """

    use_long_term_anelasticity: bool  # Whether to use long term anelasticity model or not.
    use_short_term_anelasticity: bool  # Whether to use short term anelasticity model or not.
    use_bounded_attenuation_functions: bool  # Whether to use the bounded version of attenuation functions or not.

    def run_id(self):
        """
        Generates a run ID.
        """
        return "__".join(
            (
                "long_term_anelasticity" if self.use_long_term_anelasticity else "",
                "short_term_anelasticity" if self.use_short_term_anelasticity else "",
                "bounded_attenuation_functions" if self.use_bounded_attenuation_functions else "",
            )
        )


class LoveNumbersHyperParameters(HyperParameters):
    """
    Describes the parameters needed for Love Numbers(n, frequency) computing algorithm.
    """

    # Adaptive step (on frequency) algorithm parameters.
    frequency_min: float  # log10(Low frequency limit (Hz)).
    frequency_max: float  # log10(High frequency limit (Hz)).
    n_frequency_0: int  # Minimal number of computed frequencies per degree.
    max_tol: float  # Maximal curvature criteria between orders 1 and 2.
    decimals: int  # Precision in log10(frequency / frequency_unit).

    # Lower level parameters. They can be file names.
    anelasticity_description_parameters: (
        Optional[str] | AnelasticityDescriptionParameters
    )  # To build an Earth Complete description.
    y_system_hyper_parameters: Optional[str] | YSystemHyperParameters  # For the Y_i system integration algorithm.
    degree_steps: Optional[str] | list[int]  # Love numbers are computed every degree_steps[i] between...
    degree_thresholds: Optional[str] | list[int]  # ... degree_thresholds[i] and degree_thresholds[i + 1].
    run_hyper_parameters: Optional[str] | RunHyperParameters  # Run parameters.
    save_result_per_degree: bool  # Whether to save a result per degree or not.

    def load(self) -> None:
        """
        Loads the lower level parameter fields from file names.
        """
        #  Parameters for the run.
        if not isinstance(self.run_hyper_parameters, RunHyperParameters):
            self.run_hyper_parameters = load_base_model(
                name=self.run_hyper_parameters if self.run_hyper_parameters else "run_hyper_parameters",
                base_model_type=RunHyperParameters,
                path=parameters_path,
            )
        #  Parameters for the Y_i system integration algorithm.
        if not isinstance(self.y_system_hyper_parameters, YSystemHyperParameters):
            self.y_system_hyper_parameters = load_base_model(
                name=self.y_system_hyper_parameters if self.y_system_hyper_parameters else "Y_system_hyper_parameters",
                base_model_type=YSystemHyperParameters,
                path=parameters_path,
            )
        # Parameters to build an Earth Complete description.
        if not isinstance(self.anelasticity_description_parameters, AnelasticityDescriptionParameters):
            self.anelasticity_description_parameters = load_base_model(
                name=(
                    self.anelasticity_description_parameters
                    if self.anelasticity_description_parameters
                    else "anelasticity_description_parameters"
                ),
                base_model_type=AnelasticityDescriptionParameters,
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


def load_Love_numbers_hyper_parameters(name: Optional[str] = None) -> LoveNumbersHyperParameters:
    """
    Routine that gets Love numbers hyper parameters from (.JSON) file.
    """
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name=name if not name is None else "Love_numbers_hyper_parameters",
        path=parameters_path,
        base_model_type=LoveNumbersHyperParameters,
    )
    Love_numbers_hyper_parameters.load()
    return Love_numbers_hyper_parameters


class LoadSignalHyperParameters(HyperParameters):
    """
    Describes the parameters needed for to compute some anelastic induced signal using elastic induced signal and Love
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
    GRACE: Optional[str]
    ocean_mask: Optional[str]

    # Trend parameters.
    first_year_for_trend: int
    last_year_for_trend: int

    # Run parameters.
    run_hyper_parameters: Optional[str] | RunHyperParameters

    def load(self) -> None:
        """
        Loads the lower level parameter fields from file names.
        """
        #  Parameters for the run.
        if not isinstance(self.run_hyper_parameters, RunHyperParameters):
            self.y_system_hyper_parameters = load_base_model(
                name=self.run_hyper_parameters if self.run_hyper_parameters else "run_hyper_parameters",
                base_model_type=RunHyperParameters,
                path=parameters_path,
            )


def load_load_signal_hyper_parameters(name: Optional[str] = None) -> LoadSignalHyperParameters:
    """
    Routine that gets Love numbers hyper parameters from (.JSON) file.
    """
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_base_model(
        name=name if not name is None else "load_signal_hyper_parameters",
        path=parameters_path,
        base_model_type=LoadSignalHyperParameters,
    )
    load_signal_hyper_parameters.load()
    return load_signal_hyper_parameters
