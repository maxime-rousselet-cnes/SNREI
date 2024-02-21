from itertools import product
from typing import Optional

from .classes import (
    LoveNumbersHyperParameters,
    Model,
    RealDescription,
    load_Love_numbers_hyper_parameters,
    real_description_from_parameters,
)
from .database import load_base_model, save_base_model
from .Love_numbers import Love_numbers_from_models_to_result, gets_run_id
from .paths import attenuation_models_path

BOOLEANS = [False, True]
SAMPLINGS = {"low": 10, "mid": 100, "high": 1000}


def Love_number_comparative_for_options(
    real_description_id: str,
    load_description: Optional[bool],
    elasticity_model_from_name: Optional[str] = None,
    anelasticity_model_from_name: Optional[str] = None,
    attenuation_model_from_name: Optional[str] = None,
    Love_numbers_hyper_parameters: Optional[LoveNumbersHyperParameters] = None,
) -> None:
    """
    Computes anelastic Love numbers by iterating on run options: uses long term anelasticity or attenuation or both,
    with/without bounded functions when it is possible.
    """
    # Loads hyper parameters.
    if not Love_numbers_hyper_parameters:
        Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()

    # Eventually builds description.
    real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=real_description_id,
        load_description=load_description,
        elasticity_model_from_name=elasticity_model_from_name,
        anelasticity_model_from_name=anelasticity_model_from_name,
        attenuation_model_from_name=attenuation_model_from_name,
    )

    # Loops on boolean options.
    for use_anelasticity, use_attenuation, bounded_attenuation_functions in product(BOOLEANS, BOOLEANS, BOOLEANS):
        if use_anelasticity or use_attenuation:
            Love_numbers_hyper_parameters.use_anelasticity = use_anelasticity
            Love_numbers_hyper_parameters.use_attenuation = use_attenuation
            Love_numbers_hyper_parameters.bounded_attenuation_functions = bounded_attenuation_functions
            Love_numbers_from_models_to_result(
                real_description_id=real_description.id,
                load_description=True,
                Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
            )


def Love_number_comparative_for_sampling(
    initial_real_description_id: str,
    load_initial_description: Optional[bool] = None,
    profile_precisions: dict[str, int] = SAMPLINGS,
    n_splines_bases: dict[str, int] = SAMPLINGS,
    # Sets boolean options to worst case in terms of variations with frequency.
    use_anelasticity: bool = True,
    use_attenuation: bool = True,
) -> None:
    """
    Computes anelastic Love numbers by iterating on description sampling parameters.
    """
    # Loads hyper parameters.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()

    # Eventually builds description.
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=initial_real_description_id,
        load_description=load_initial_description,
        save=False,
    )

    Love_numbers_hyper_parameters.use_anelasticity = use_anelasticity
    Love_numbers_hyper_parameters.use_attenuation = use_attenuation
    run_id = gets_run_id(
        use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
        bounded_attenuation_functions=Love_numbers_hyper_parameters.bounded_attenuation_functions,
        use_attenuation=Love_numbers_hyper_parameters.use_attenuation,
    )

    # Iterates on sampling parameters.
    for (profile_precision_name_part, profile_precision), (n_splines_base_name_part, n_splines_base) in product(
        profile_precisions.items(), n_splines_bases.items()
    ):
        Love_numbers_hyper_parameters.real_description_parameters.profile_precision = profile_precision
        Love_numbers_hyper_parameters.real_description_parameters.n_splines_base = n_splines_base
        Love_numbers_from_models_to_result(
            real_description_id=profile_precision_name_part + "_p_" + n_splines_base_name_part + "_ns",
            run_id=run_id,
            load_description=False,
            elasticity_model_from_name=initial_real_description.elasticity_model_name,
            anelasticity_model_from_name=initial_real_description.anelasticity_model_name,
            attenuation_model_from_name=initial_real_description.attenuation_model_name,
            Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        )


def id_from_model_names(
    id: str,
    real_description: RealDescription,
    elasticity_model_name: str,
    anelasticity_model_name: str,
    attenuation_model_name: str,
) -> str:
    """
    Generates an ID for a real description given the name of the used model files.
    """
    return (
        id
        if (elasticity_model_name == real_description.elasticity_model_name)
        and (anelasticity_model_name == real_description.anelasticity_model_name)
        and (attenuation_model_name == real_description.attenuation_model_name)
        else "_".join((elasticity_model_name, anelasticity_model_name, attenuation_model_name))
    )


def Love_number_comparative_for_models(
    initial_real_description_id: str,
    load_initial_description: Optional[bool] = None,
    elasticity_model_names: Optional[list[str]] = None,
    anelasticity_model_names: Optional[list[str]] = None,
    attenuation_model_names: Optional[list[str]] = None,
) -> None:
    """
    Computes anelastic Love numbers by iterating on:
        - run options: uses long term anelasticity or attenuation or both, with/without bounded functions when it is possible.
        - models: A real description is used per triplet of:
            - 'elasticity_model_name'
            - 'anelasticity_model_name'
            - 'attenuation_model_name'
    """
    # Loads hyper parameters.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()

    # Eventually builds description.
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=initial_real_description_id,
        load_description=load_initial_description,
        save=False,
    )

    # Builds dummy lists for unmodified models.
    if not elasticity_model_names:
        elasticity_model_names = [initial_real_description.elasticity_model_name]
    if not anelasticity_model_names:
        anelasticity_model_names = [initial_real_description.anelasticity_model_name]
    if not attenuation_model_names:
        attenuation_model_names = [initial_real_description.attenuation_model_name]

    # Loops on model files.
    for elasticity_model_name, anelasticity_model_name, attenuation_model_name in product(
        elasticity_model_names, anelasticity_model_names, attenuation_model_names
    ):
        # Loops on options.
        Love_number_comparative_for_options(
            real_description_id=id_from_model_names(
                id=initial_real_description_id,
                real_description=initial_real_description,
                elasticity_model_name=elasticity_model_name,
                anelasticity_model_name=anelasticity_model_name,
                attenuation_model_name=attenuation_model_name,
            ),
            load_description=False,
            elasticity_model_from_name=elasticity_model_name,
            anelasticity_model_from_name=anelasticity_model_name,
            attenuation_model_from_name=attenuation_model_name,
            Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        )


def gets_id_asymptotic_ratios(
    asymptotic_ratios_per_layer: list[int],
    real_description_id: str = "",
) -> str:
    """
    Generates an ID for a run using its asymptotic ratios per layer.
    """
    return real_description_id + "-".join([str(asymptotic_ratio) for asymptotic_ratio in asymptotic_ratios_per_layer])


def Love_number_comparative_for_asymptotic_ratio(
    initial_real_description_id: str,
    asymptotic_ratios: list[list[float]],
    load_initial_description: Optional[bool] = None,
    elasticity_model_names: Optional[list[str]] = None,
    anelasticity_model_names: Optional[list[str]] = None,
    attenuation_model_names: Optional[list[str]] = None,
) -> None:
    """
    Computes anelastic Love numbers by iterating on:
        - models: A real description is used per triplet of:
            - 'elasticity_model_name'
            - 'anelasticity_model_name'
            - 'attenuation_model_name'
        - asymptotic_ratios
    """
    # Loads hyper parameters.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    Love_numbers_hyper_parameters.use_attenuation = True
    Love_numbers_hyper_parameters.bounded_attenuation_functions = True

    # Eventually builds description.
    initial_real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=initial_real_description_id,
        load_description=load_initial_description,
        save=False,
    )

    # Builds dummy lists for unmodified models.
    if not elasticity_model_names:
        elasticity_model_names = [initial_real_description.elasticity_model_name]
    if not anelasticity_model_names:
        anelasticity_model_names = [initial_real_description.anelasticity_model_name]
    if not attenuation_model_names:
        attenuation_model_names = [initial_real_description.attenuation_model_name]

    # Loops on whether to use anelasticity or not.
    for use_anelasticity in BOOLEANS:
        Love_numbers_hyper_parameters.use_anelasticity = use_anelasticity
        # Loops on model files.
        for elasticity_model_name, anelasticity_model_name, attenuation_model_name in product(
            elasticity_model_names, anelasticity_model_names, attenuation_model_names
        ):
            attenuation_model: Model = load_base_model(
                name=attenuation_model_name, path=attenuation_models_path, base_model_type=Model
            )
            temp_name_attenuation_model = attenuation_model_name + "-variable-asymptotic_ratio"
            # Loops on asymptotic_ratio.
            for asymptotic_ratios_per_layer in asymptotic_ratios:
                for k_layer, asymptotic_ratio in enumerate(asymptotic_ratios_per_layer):
                    attenuation_model.polynomials["asymptotic_attenuation"][k_layer][0] = 1.0 - asymptotic_ratio
                save_base_model(obj=attenuation_model, name=temp_name_attenuation_model, path=attenuation_models_path)
                Love_numbers_from_models_to_result(
                    real_description_id=gets_id_asymptotic_ratios(
                        real_description_id=id_from_model_names(
                            id=initial_real_description_id,
                            real_description=initial_real_description,
                            elasticity_model_name=elasticity_model_name,
                            anelasticity_model_name=anelasticity_model_name,
                            attenuation_model_name=temp_name_attenuation_model,
                        ),
                        asymptotic_ratios_per_layer=asymptotic_ratios_per_layer,
                    ),
                    run_id=gets_run_id(
                        use_anelasticity=Love_numbers_hyper_parameters.use_anelasticity,
                        bounded_attenuation_functions=True,
                        use_attenuation=True,
                    ),
                    load_description=False,
                    elasticity_model_from_name=elasticity_model_name,
                    anelasticity_model_from_name=anelasticity_model_name,
                    attenuation_model_from_name=temp_name_attenuation_model,
                    Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                )
