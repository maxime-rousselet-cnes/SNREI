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
    atol: float  # The solver keeps the local error estimates less than atol + rtol * abs(yr).
    rtol: float  # The solver keeps the local error estimates less than atol + rtol * abs(yr).
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

    def string(self):
        return (
            "long-term and short-term anelasticities"
            if self.use_long_term_anelasticity and self.use_short_term_anelasticity
            else (
                "long-term anelasticity"
                if self.use_long_term_anelasticity
                else ("short-term anelasticity" if self.use_short_term_anelasticity else "pure elastic")
            )
        )


class LoveNumbersHyperParameters(HyperParameters):
    """
    Describes the parameters needed for Love Numbers(n, frequency) computing algorithm.
    """

    # Adaptive step (on frequency) algorithm parameters.
    period_min_year: float  # High frequency limit (yr).
    period_max_year: float  # Low frequency limit (yr).
    n_frequency_0: int  # Minimal number of computed frequencies per degree.
    max_tol: float  # Maximal curvature criteria between orders 1 and 2.
    decimals: int  # Precision in log10(frequency / frequency_unit).

    # Lower level parameters. They can be file names.
    anelasticity_description_parameters: Optional[str] | AnelasticityDescriptionParameters  # To build an Earth Complete description.
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
                name=(self.run_hyper_parameters if self.run_hyper_parameters else "run_hyper_parameters"),
                base_model_type=RunHyperParameters,
                path=parameters_path,
            )

        #  Parameters for the Y_i system integration algorithm.
        if not isinstance(self.y_system_hyper_parameters, YSystemHyperParameters):
            self.y_system_hyper_parameters = load_base_model(
                name=(self.y_system_hyper_parameters if self.y_system_hyper_parameters else "Y_system_hyper_parameters"),
                base_model_type=YSystemHyperParameters,
                path=parameters_path,
            )

        # Parameters to build an Earth Complete description.
        if not isinstance(self.anelasticity_description_parameters, AnelasticityDescriptionParameters):
            self.anelasticity_description_parameters = load_base_model(
                name=(
                    self.anelasticity_description_parameters if self.anelasticity_description_parameters else "anelasticity_description_parameters"
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
                name=(self.degree_thresholds if self.degree_thresholds else "degree_thresholds"),
                path=parameters_path,
            )


def load_Love_numbers_hyper_parameters(
    name: str = "Love_numbers_hyper_parameters",
) -> LoveNumbersHyperParameters:
    """
    Routine that gets Love numbers hyper parameters from (.JSON) file.
    """
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_base_model(
        name=name,
        path=parameters_path,
        base_model_type=LoveNumbersHyperParameters,
    )
    Love_numbers_hyper_parameters.load()
    return Love_numbers_hyper_parameters


class SaveParameters(HyperParameters):
    """
    Describes whether to save all the resulting load signals or not.
    """

    # Anelastic load signal computed after frequencial filtering by Love number fractions.
    step_1: bool
    # Anelastic load signal computed after degree one inversion.
    step_2: bool
    # Anelastic load signal computed after leakage corretion.
    step_3: bool
    # Three remaining components of degree one inversion equation: geoid height, radial displacement and residuals.
    inversion_components: bool


class LoadSignalHyperParameters(HyperParameters):
    """
    Describes the parameters needed for to compute some anelastic induced signal using elastic induced signal and Love
    numbers.
    """

    # Parameters describing the load signal.
    load_signal: str  # For now: "load_history" is managed only.
    pole_data: str  # (.csv) file path relative to data/pole_data.
    # Load history parameters.
    load_history: str  # (.csv) file path relative to data/GMSL_data.
    case: str  # Whether "lower", "mean" or "upper".
    load_history_start_date: int  # Usually 1900 for Frederikse GMSL data.
    spline_time_years: int  # Time for the anti-symmetrization spline process in years.
    initial_plateau_time_years: int  # Time of the zero-value plateau before the signal history (yr).
    anti_Gibbs_effect_factor: int  # Integer, minimum equal to 1 (unitless).
    # Little Isostatic Adjustment (LIA) parameters.
    LIA: bool  # Whethter to take LIA into account or not.
    LIA_end_date: int  # Usualy ~ 1400 (yr).
    LIA_time_years: int  # Usually ~ 100 (yr).
    LIA_amplitude_effect: float  # Usually ~ 0.25 (unitless).
    # Parameters describing spacially-dependent load signal.
    load_spatial_behaviour_data: str  # For now: "GRACE" is managed only.
    opposite_load_on_continents: bool
    n_max: int
    load_spatial_behaviour_file: Optional[str]  # (.csv) file path relative to data.
    polar_tide_correction: bool  # Whether to performs Wahr (2015) recommended polar tide correction.
    renormalize_recent_trend: bool  # Wether to rescale recent period trends on GRACE ocean mean trend.
    leakage_correction_iterations: int
    signal_threshold: float  # (mm).
    erode_high_signal_zones: bool  # Whether to consider as oceanic signal the high level signal areas or not during leakage correction.
    ddk_filter_level: int
    ocean_mask: Optional[str]
    continents: Optional[str]
    buffer_distance: float  # Buffer to coast (km).

    # Trend computing parameters.
    first_year_for_trend: int
    last_year_for_trend: int
    past_trend_error: float  # Maximal admitted error for past trend matching to data.

    # Load signal save parameters.
    save_parameters: dict[str, SaveParameters | str]


def load_load_signal_hyper_parameters(
    name: str = "load_signal_hyper_parameters",
) -> LoadSignalHyperParameters:
    """
    Routine that gets Love numbers hyper parameters from (.JSON) file.
    """
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_base_model(
        name=name,
        path=parameters_path,
        base_model_type=LoadSignalHyperParameters,
    )
    save_parameters: dict[str, SaveParameters] = {
        "harmonics": load_base_model(
            name=load_signal_hyper_parameters.save_parameters["harmonics"],
            path=parameters_path,
            base_model_type=SaveParameters,
        ),
        "base_formats": load_base_model(
            name=load_signal_hyper_parameters.save_parameters["base_formats"],
            path=parameters_path,
            base_model_type=SaveParameters,
        ),
    }
    load_signal_hyper_parameters.save_parameters = save_parameters
    load_signal_hyper_parameters.ddk_filter_level = 7 if "MSSA" in load_signal_hyper_parameters.load_spatial_behaviour_file else 5
    return load_signal_hyper_parameters
