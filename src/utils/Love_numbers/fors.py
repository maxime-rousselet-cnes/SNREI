from itertools import product
from typing import Optional

from numpy import concatenate, ndarray

from ..classes import (
    BOOLEANS,
    OPTIONS,
    LoveNumbersHyperParameters,
    Model,
    ModelPart,
    RunHyperParameters,
    anelasticity_description_id_from_part_names,
    load_Love_numbers_hyper_parameters,
    models_path,
    results_path,
)
from ..database import load_base_model, save_base_model, symlink, symlinkfolder
from .single import Love_numbers_from_models_for_options


def create_model_variation(
    model_part: ModelPart,
    model_base_name: Optional[str],
    parameter_names: list[str],
    parameter_values_per_layer: list[list[list[float]]],
) -> str:
    """
    Gets an initial model file and creates a new version of it by modifying the specified parameters with specified polynomials
    per layer.
    """
    model: Model = load_base_model(name=model_base_name, path=models_path[model_part], base_model_type=Model)
    for parameter_name, parameter_values in zip(parameter_names, parameter_values_per_layer):
        model.polynomials[parameter_name] = parameter_values
    name = "___".join(
        [
            "__".join([parameter_name] + ["_".join((str(value) for value in values)) for values in parameter_values])
            for parameter_name, parameter_values in zip(parameter_names, parameter_values_per_layer)
        ]
    )
    save_base_model(
        obj=model,
        name=name,
        path=models_path[model_part].joinpath(model_base_name),
    )
    return name


def Love_numbers_for_options_for_models_for_parameters(
    forced_anelasticity_description_id: Optional[str] = None,
    overwrite_descriptions: bool = False,
    elasticity_model_names: list[Optional[str]] = [None],
    long_term_anelasticity_model_names: list[Optional[str]] = [None],
    short_term_anelasticity_model_names: list[Optional[str]] = [None],
    parameters: dict[ModelPart, dict[str, list[list[list[float]]]]] = {model_part: {} for model_part in ModelPart},
    options: list[RunHyperParameters] = OPTIONS,
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters(),
) -> list[str]:
    """
    Computes anelastic Love numbers by iterating on:
        - models: An anelasticity description is used per triplet of:
            - 'elasticity_model_name'
            - 'anelasticity_model_name'
            - 'attenuation_model_name'
        - options
        - specified parameters, when the options allow it.
    Returns the list of anelasticity description IDs.

    The 'parameters' input corresponds to parameters polynomial coefficients per layer (list[list[float]]) per model part, per
    parameter name and per possibility.Here is an example with two possibilites for a short term anelasticity model with 5
    layers: {
        ModelPart.short_term_anelasticity: {
            "asymptotic_mu_ratio": [
                [
                    [0.2], [0.2], [0.2], [0.2], [1.0]
                ],
                [
                    [1.0], [1.0], [1.0], [1.0], [1.0]
                ]
            ]
        }
    }
    """
    # Creates all model files variations.
    model_filenames: dict[ModelPart, list[str]] = {}
    for model_part, model_names in zip(
        ModelPart, [elasticity_model_names, long_term_anelasticity_model_names, short_term_anelasticity_model_names]
    ):
        if (not model_part in parameters.keys()) or (parameters[model_part] == {}):
            model_filenames[model_part] = model_names
        else:
            filename_variations: ndarray = concatenate(
                [
                    [
                        model_name
                        + "/"
                        + create_model_variation(
                            model_part=model_part,
                            model_base_name=model_name,
                            parameter_names=parameters[model_part].keys(),
                            parameter_values_per_layer=list(parameter_values),
                        )
                        for parameter_values in product(
                            *(
                                parameter_values_per_possibility
                                for _, parameter_values_per_possibility in parameters[model_part].items()
                            )
                        )
                    ]
                    for model_name in model_names
                ]
            )
            model_filenames[model_part] = filename_variations.tolist()

    anelasticity_description_ids = []
    # Loops on all possible triplet of model files to launch runs.
    for elasticity_model_name, long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
        model_filenames[ModelPart.elasticity],
        model_filenames[ModelPart.long_term_anelasticity],
        model_filenames[ModelPart.short_term_anelasticity],
    ):
        # Finds minimal computing options.
        do_elastic_case = (long_term_anelasticity_model_name == model_filenames[ModelPart.long_term_anelasticity][0]) and (
            short_term_anelasticity_model_name == model_filenames[ModelPart.short_term_anelasticity][0]
        )
        do_long_term_only_case = short_term_anelasticity_model_name == model_filenames[ModelPart.short_term_anelasticity][0]
        do_short_term_only_case = long_term_anelasticity_model_name == model_filenames[ModelPart.long_term_anelasticity][0]

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
                options=[
                    run_hyper_parameters
                    for run_hyper_parameters in options
                    if not (
                        not do_long_term_only_case
                        and (
                            run_hyper_parameters.use_long_term_anelasticity == True
                            and run_hyper_parameters.use_short_term_anelasticity == False
                        )
                    )
                    and not (
                        not do_short_term_only_case
                        and (
                            run_hyper_parameters.use_long_term_anelasticity == False
                            and run_hyper_parameters.use_short_term_anelasticity == True
                        )
                    )
                ],
                do_elastic_case=do_elastic_case,
            )
        ]

    # Symlinks.
    if forced_anelasticity_description_id is None:
        for elasticity_model_name, long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
            model_filenames[ModelPart.elasticity],
            model_filenames[ModelPart.long_term_anelasticity],
            model_filenames[ModelPart.short_term_anelasticity],
        ):
            # Finds minimal conmputing options.
            do_elastic_case = (long_term_anelasticity_model_name == model_filenames[ModelPart.long_term_anelasticity][0]) and (
                short_term_anelasticity_model_name == model_filenames[ModelPart.short_term_anelasticity][0]
            )
            do_long_term_only_case = short_term_anelasticity_model_name == model_filenames[ModelPart.short_term_anelasticity][0]
            do_short_term_only_case = long_term_anelasticity_model_name == model_filenames[ModelPart.long_term_anelasticity][0]
            # Eventually creates a symlink to equivalent model's result.
            anelasticity_description_result_path = results_path.joinpath(
                anelasticity_description_id_from_part_names(
                    elasticity_name=elasticity_model_name,
                    long_term_anelasticity_name=long_term_anelasticity_model_name,
                    short_term_anelasticity_name=short_term_anelasticity_model_name,
                )
            )
            # Creates a symlink to equivalent elastic model's result.
            if not do_elastic_case:
                src_path = results_path.joinpath(
                    anelasticity_description_id_from_part_names(
                        elasticity_name=elasticity_model_name,
                        long_term_anelasticity_name=model_filenames[ModelPart.long_term_anelasticity][0],
                        short_term_anelasticity_name=model_filenames[ModelPart.short_term_anelasticity][0],
                    )
                )
                symlink(
                    src=src_path.joinpath("elastic_Love_numbers.json").absolute(),
                    dst=anelasticity_description_result_path.joinpath("elastic_Love_numbers.json"),
                )
                symlink(
                    src=src_path.joinpath("degrees.json").absolute(),
                    dst=anelasticity_description_result_path.joinpath("degrees.json"),
                )
            # Creates a symlink to equivalent long term anelasticity model's results for long term anelasticity only run.
            if not do_long_term_only_case:
                run_id = RunHyperParameters(
                    use_long_term_anelasticity=True, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False
                ).run_id()
                src_path = (
                    results_path.joinpath(
                        anelasticity_description_id_from_part_names(
                            elasticity_name=elasticity_model_name,
                            long_term_anelasticity_name=long_term_anelasticity_model_name,
                            short_term_anelasticity_name=model_filenames[ModelPart.short_term_anelasticity][0],
                        )
                    )
                    .joinpath("runs")
                    .joinpath(run_id)
                )
                symlinkfolder(
                    src=src_path.absolute(),
                    dst=anelasticity_description_result_path.joinpath("runs").joinpath(run_id),
                )
            # Creates a symlink to equivalent short term anelasticity model's results for short term anelasticity only run.
            if not do_short_term_only_case:
                for use_bounded_attenuation_functions in BOOLEANS:
                    run_id = RunHyperParameters(
                        use_long_term_anelasticity=False,
                        use_short_term_anelasticity=True,
                        use_bounded_attenuation_functions=use_bounded_attenuation_functions,
                    ).run_id()
                    src_path = (
                        results_path.joinpath(
                            anelasticity_description_id_from_part_names(
                                elasticity_name=elasticity_model_name,
                                long_term_anelasticity_name=model_filenames[ModelPart.long_term_anelasticity][0],
                                short_term_anelasticity_name=short_term_anelasticity_model_name,
                            )
                        )
                        .joinpath("runs")
                        .joinpath(run_id)
                    )
                    symlinkfolder(
                        src=src_path.absolute(),
                        dst=anelasticity_description_result_path.joinpath("runs").joinpath(run_id),
                    )

    return anelasticity_description_ids
