from typing import Optional

from ...utils import (
    OPTIONS,
    BoundaryCondition,
    Direction,
    Love_numbers_for_options_for_models_for_parameters,
    LoveNumbersHyperParameters,
    Result,
    RunHyperParameters,
    frequencies_to_periods,
    interpolate_Love_numbers,
    load_Love_numbers_hyper_parameters,
)

# TODO: re-do.


def get_single_float_Love_number(
    anelasticity_desciption_id: str,
    option: RunHyperParameters = RunHyperParameters(
        use_long_term_anelasticity=False, use_short_term_anelasticity=True, use_bounded_attenuation_functions=False
    ),
    degree: int = 2,
    direction: Direction = Direction.potential,
    boundary_condition: BoundaryCondition = BoundaryCondition.potential,
    period: float = 18.6,  # (yr).
) -> float:
    """
    gets the wanted Love number and interpolates it at the wanted period.
    """
    result: Result = interpolate_Love_numbers(
        anelasticity_description_id=anelasticity_desciption_id,
        target_frequencies=frequencies_to_periods(frequencies=[period]),  # (yr) -> (Hz).
        option=option,
        degrees=[degree],
        directions=[direction],
        boundary_conditions=[boundary_condition],
    )[0]
    return result.values[direction][boundary_condition]


def Love_numbers_for_options(
    forced_anelasticity_description_id: Optional[str] = None,
    overwrite_descriptions: bool = False,
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_name: Optional[str] = None,
    options: list[RunHyperParameters] = OPTIONS,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
) -> str:
    """
    Computes anelastic Love numbers by iterating on options. A single triplet of models is used.
    Returns the anelasticity description ID.
    """
    return Love_numbers_for_options_for_models_for_parameters(
        forced_anelasticity_description_id=forced_anelasticity_description_id,
        overwrite_descriptions=overwrite_descriptions,
        elasticity_model_names=[elasticity_model_name],
        long_term_anelasticity_model_names=[long_term_anelasticity_model_name],
        short_term_anelasticity_model_names=[short_term_anelasticity_model_name],
        options=options,
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
    )[0]


def Love_numbers_single_run(
    forced_anelasticity_description_id: Optional[str] = None,
    overwrite_descriptions: bool = False,
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_name: Optional[str] = None,
    run_hyper_parameters: RunHyperParameters = RunHyperParameters(
        use_long_term_anelasticity=True, use_short_term_anelasticity=True, use_bounded_attenuation_functions=True
    ),
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
) -> str:
    """
    Computes anelastic Love numbers for given run options. A single triplet of models is used.
    Returns the anelasticity description ID.
    """
    return Love_numbers_for_options(
        forced_anelasticity_description_id=forced_anelasticity_description_id,
        overwrite_descriptions=overwrite_descriptions,
        elasticity_model_name=elasticity_model_name,
        long_term_anelasticity_model_name=long_term_anelasticity_model_name,
        short_term_anelasticity_model_name=short_term_anelasticity_model_name,
        options=[run_hyper_parameters],
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
    )


def Love_numbers_for_samplings(
    spline_number_sampling_options: list[int] = [10, 100, 1000],
    spline_degree_sampling_options: list[int] = [1, 2, 3],
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_name: Optional[str] = None,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
) -> list[str]:
    """
    Computes anelastic Love numbers by iterating on anelasticity description sampling parameters.
    Return the anelasticity description IDs.
    """
    anelasticity_description_ids = []
    for spline_number in spline_number_sampling_options:
        Love_numbers_hyper_parameters.anelasticity_description_parameters.spline_number = spline_number
        for spline_degree in spline_degree_sampling_options:
            Love_numbers_hyper_parameters.anelasticity_description_parameters.spline_degree = spline_degree
            anelasticity_description_ids += [
                Love_numbers_single_run(
                    forced_anelasticity_description_id="sampling_test__n_s_" + str(spline_number) + "_d_s_" + str(spline_degree),
                    overwrite_descriptions=True,
                    elasticity_model_name=elasticity_model_name,
                    long_term_anelasticity_model_name=long_term_anelasticity_model_name,
                    short_term_anelasticity_model_name=short_term_anelasticity_model_name,
                    Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                )
            ]
    return anelasticity_description_ids
