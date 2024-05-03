from copy import deepcopy
from itertools import product
from typing import Any, Optional

from .classes import (
    BOOLEANS,
    LoadSignalHyperParameters,
    Model,
    ModelPart,
    RunHyperParameters,
    anelasticity_description_id_from_part_names,
    load_load_signal_hyper_parameters,
    models_path,
    results_path,
)
from .database import load_base_model, save_base_model, symlink, symlinkfolder


def create_model_variation(
    model_part: ModelPart,
    base_model: Model,
    base_model_name: Optional[str],
    parameter_values: list[tuple[str, str, list[float]]],
    create: bool = True,
) -> str:
    """
    Gets an initial model file and creates a new version of it by modifying the specified parameters with specified polynomials
    per layer.
    """
    if create:
        for parameter_name, layer_name_part, polynomial in parameter_values:
            for layer_name in base_model.layer_names:
                if layer_name_part in layer_name:
                    base_model.polynomials[parameter_name][base_model.layer_names.index(layer_name)] = polynomial
    name = "___".join(
        [
            "__".join([parameter_name, layer_name] + [str(value) for value in polynomial])
            for parameter_name, layer_name, polynomial in parameter_values
        ]
    )
    if create:
        base_model.save(
            name=name,
            path=models_path[model_part].joinpath(base_model_name),
        )
    return name


def sum_lists(lists: list[list]) -> list:
    """
    Concatenates lists.
    """
    concatenated_list = []
    for elt in lists:
        for sub_elt in elt:
            concatenated_list += sub_elt
    return concatenated_list


def create_all_model_variations(
    elasticity_model_names: list[Optional[str]] = [None],
    long_term_anelasticity_model_names: list[Optional[str]] = [None],
    short_term_anelasticity_model_names: list[Optional[str]] = [None],
    parameters: dict[ModelPart, dict[str, dict[str, list[list[float]]]]] = {model_part: {} for model_part in ModelPart},
    create: bool = True,
) -> dict[ModelPart, list[str]]:
    """
    Creates all possible variations of parameters for the wanted models and creates the corresponding files accordingly.
    Returns all their IDs.
    """
    model_filenames: dict[ModelPart, list[str]] = {}
    for model_part, model_names in zip(
        ModelPart, [elasticity_model_names, long_term_anelasticity_model_names, short_term_anelasticity_model_names]
    ):
        if (model_part in parameters.keys()) and (parameters[model_part] != {}):
            for model_name in model_names:
                model: Model = load_base_model(name=model_name, path=models_path[model_part], base_model_type=Model)
                # Adds all possible combinations.
                model_filenames[model_part] = list(
                    set(
                        [
                            model_name
                            + "/"
                            + create_model_variation(
                                model_part=model_part,
                                base_model=model,
                                base_model_name=model_name,
                                parameter_values=sum_lists(lists=parameter_values),
                                create=create,
                            )
                            for parameter_values in product(
                                *(
                                    product(
                                        *(
                                            product(
                                                (
                                                    (parameter_name, layer_name_part, parameter_values_possibility)
                                                    for parameter_values_possibility in [
                                                        model.polynomials[parameter_name][model.layer_names.index(layer_name)]
                                                        for layer_name in model.layer_names
                                                        if layer_name_part in layer_name
                                                    ]
                                                    + parameter_values_per_possibility  # Adds default values in the list of values to iterate on.
                                                )
                                            )
                                            for layer_name_part, parameter_values_per_possibility in parameter_values_per_layer.items()
                                        )
                                    )
                                    for parameter_name, parameter_values_per_layer in parameters[model_part].items()
                                )
                            )
                        ]
                    )
                )
        else:
            model_filenames[model_part] = model_names

    return model_filenames


def create_load_signal_hyper_parameter_variation(
    base_load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    parameter_values=list[list[tuple[str, Any]]],
) -> LoadSignalHyperParameters:
    """
    Creates a variation of load signal hyper parameters.
    """
    load_signal_hyper_parameters = deepcopy(base_load_signal_hyper_parameters)
    for parameter_tuple in parameter_values:
        setattr(load_signal_hyper_parameters, parameter_tuple[0][0], parameter_tuple[0][1])
    return load_signal_hyper_parameters


def create_all_load_signal_hyper_parameters_variations(
    base_load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    load_signal_hyper_parameter_variations: dict[str, list] = {},
) -> list[str]:
    """
    Creates all possible variations of load signal hyper parameters.
    """
    return [
        create_load_signal_hyper_parameter_variation(
            base_load_signal_hyper_parameters=base_load_signal_hyper_parameters,
            parameter_values=parameter_values,
        )
        for parameter_values in product(
            *(
                product(
                    (parameter_name, parameter_value_possibility)
                    for parameter_value_possibility in parameter_values_per_possibility
                )
                for parameter_name, parameter_values_per_possibility in load_signal_hyper_parameter_variations.items()
            )
        )
    ]


def find_minimal_computing_options(
    long_term_anelasticity_model_name: str, short_term_anelasticity_model_name: str, model_filenames: dict[ModelPart, list[str]]
) -> tuple[bool, bool, bool]:
    """
    Tells whether it is necessary to compute elastic case, long_term_only case or short_term_only case or to create symlink to
    equivalent model's results.
    """
    return (
        (long_term_anelasticity_model_name == model_filenames[ModelPart.long_term_anelasticity][0])
        and (short_term_anelasticity_model_name == model_filenames[ModelPart.short_term_anelasticity][0]),
        short_term_anelasticity_model_name == model_filenames[ModelPart.short_term_anelasticity][0],
        long_term_anelasticity_model_name == model_filenames[ModelPart.long_term_anelasticity][0],
    )


def minimal_computing_options(
    options: list[RunHyperParameters], do_long_term_only_case: bool, do_short_term_only_case: bool
) -> list[RunHyperParameters]:
    """
    Returns the list of options that are needed for computation.
    """
    return [
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
    ]


def create_symlinks_to_results(model_filenames: dict[ModelPart, list[str]], options: list[RunHyperParameters]) -> None:
    """
    Creates Symlinks for model's results to equivalent model's results. Considers the subfolders.
    """
    for elasticity_model_name, long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
        model_filenames[ModelPart.elasticity],
        model_filenames[ModelPart.long_term_anelasticity],
        model_filenames[ModelPart.short_term_anelasticity],
    ):
        # Finds minimal conmputing options.
        do_elastic_case, do_long_term_only_case, do_short_term_only_case = find_minimal_computing_options(
            long_term_anelasticity_model_name=long_term_anelasticity_model_name,
            short_term_anelasticity_model_name=short_term_anelasticity_model_name,
            model_filenames=model_filenames,
        )

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
            for filename in ["degrees", "elastic_Love_numbers"]:
                symlink(
                    src=src_path.joinpath(filename + ".json").absolute(),
                    dst=anelasticity_description_result_path.joinpath(filename + ".json"),
                )

        # Creates a symlink to equivalent long term anelasticity model's results for long term anelasticity only run.
        if not do_long_term_only_case:
            run_hyper_parameters = RunHyperParameters(
                use_long_term_anelasticity=True, use_short_term_anelasticity=False, use_bounded_attenuation_functions=False
            )
            if run_hyper_parameters in options:
                run_id = run_hyper_parameters.run_id()
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
                run_hyper_parameters = RunHyperParameters(
                    use_long_term_anelasticity=False,
                    use_short_term_anelasticity=True,
                    use_bounded_attenuation_functions=use_bounded_attenuation_functions,
                )
                if run_hyper_parameters in options:
                    run_id = run_hyper_parameters.run_id()
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
