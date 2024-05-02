from itertools import product
from typing import Optional

from ..classes import (
    OPTIONS,
    LoveNumbersHyperParameters,
    ModelPart,
    RunHyperParameters,
    load_Love_numbers_hyper_parameters,
)
from ..fors import (
    create_all_model_variations,
    create_symlinks_to_results,
    find_minimal_computing_options,
    minimal_computing_options,
)
from .single import Love_numbers_from_models_for_options


def Love_numbers_for_options_for_models_for_parameters(
    forced_anelasticity_description_id: Optional[str] = None,
    overwrite_descriptions: bool = False,
    elasticity_model_names: list[Optional[str]] = [None],
    long_term_anelasticity_model_names: list[Optional[str]] = [None],
    short_term_anelasticity_model_names: list[Optional[str]] = [None],
    parameters: dict[ModelPart, dict[str, dict[str, list[list[float]]]]] = {model_part: {} for model_part in ModelPart},
    options: list[RunHyperParameters] = OPTIONS,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
    symlinks: bool = True,
) -> tuple[list[str], dict[ModelPart, list[str]]]:
    """
    Computes anelastic Love numbers by iterating on:
        - models: An anelasticity description is used per triplet of:
            - 'elasticity_model_name'
            - 'anelasticity_model_name'
            - 'attenuation_model_name'
        - options
        - specified parameters, when the options allow it.
    Returns the list of anelasticity description IDs the list of model filename variations per model part.

    The 'parameters' input corresponds to parameters polynomial coefficients (list[float]) per model part, per
    parameter name, per layer name and per possibility. Here is an example that tests two possibilities for polynomials of
    mu_asymptotic_ratio in the lower mantle: {
        ModelPart.short_term_anelasticity: {
            "asymptotic_mu_ratio": {
                "LOWER_MANTLE": [
                    [
                        [0.2, 1e-2],
                    ],
                    [
                        [1.0],
                    ]
                ]
            }
        }
    }
    Here the first polynmial is 0.2 + 1e-2 X and the second one is constant at 1.0.
    """
    # Creates all model files variations.
    model_filenames = create_all_model_variations(
        elasticity_model_names=elasticity_model_names,
        long_term_anelasticity_model_names=long_term_anelasticity_model_names,
        short_term_anelasticity_model_names=short_term_anelasticity_model_names,
        parameters=parameters,
        create=True,
    )

    anelasticity_description_ids = []
    # Loops on all possible triplet of model files to launch runs.
    for elasticity_model_name, long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
        model_filenames[ModelPart.elasticity],
        model_filenames[ModelPart.long_term_anelasticity],
        model_filenames[ModelPart.short_term_anelasticity],
    ):
        # Finds minimal computing options.
        do_elastic_case, do_long_term_only_case, do_short_term_only_case = find_minimal_computing_options(
            long_term_anelasticity_model_name=long_term_anelasticity_model_name,
            short_term_anelasticity_model_name=short_term_anelasticity_model_name,
            model_filenames=model_filenames,
        )

        # Compute Love numbers for all considered options.
        anelasticity_description_ids += [
            Love_numbers_from_models_for_options(
                forced_anelasticity_description_id=forced_anelasticity_description_id,
                overwrite_descriptions=overwrite_descriptions,
                part_names={
                    ModelPart.elasticity: elasticity_model_name,
                    ModelPart.long_term_anelasticity: long_term_anelasticity_model_name,
                    ModelPart.short_term_anelasticity: short_term_anelasticity_model_name,
                },
                Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                options=minimal_computing_options(
                    options=options,
                    do_long_term_only_case=do_long_term_only_case,
                    do_short_term_only_case=do_short_term_only_case,
                ),
                do_elastic_case=do_elastic_case,
            )
        ]

    # Symlinks.
    if forced_anelasticity_description_id is None and symlinks:
        create_symlinks_to_results(model_filenames=model_filenames, options=options)

    return anelasticity_description_ids, model_filenames
