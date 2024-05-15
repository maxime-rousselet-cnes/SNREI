from itertools import product
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..classes import (
    OPTIONS,
    LoadSignalHyperParameters,
    ModelPart,
    RunHyperParameters,
    anelasticity_description_id_from_part_names,
    load_load_signal_hyper_parameters,
)
from ..fors import (
    create_all_load_signal_hyper_parameters_variations,
    create_all_model_variations,
    create_symlinks_to_results,
)
from .single import compute_anelastic_induced_harmonic_load_per_description_per_options
from .trend import get_load_signal_harmonic_trends


def load_signal_for_options_for_models_for_parameters_for_elastic_load_signals(
    anelasticity_description_ids: Optional[list[str]] = None,
    model_filenames: Optional[dict[ModelPart, list[str]]] = None,
    elasticity_model_names: list[Optional[str]] = [None],
    long_term_anelasticity_model_names: list[Optional[str]] = [None],
    short_term_anelasticity_model_names: list[Optional[str]] = [None],
    parameters: dict[ModelPart, dict[str, dict[str, list[list[float]]]]] = {model_part: {} for model_part in ModelPart},
    options: list[RunHyperParameters] = OPTIONS,
    base_load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    load_signal_hyper_parameter_variations: dict[str, list] = {},
    symlinks: bool = True,
) -> tuple[list[Path], : dict[Path, dict[str, float]]]:
    """
    Computes anelastic load signal from Love numbers by iterating on:
        - models: An anelasticity description is used per triplet of:
            - 'elasticity_model_name'
            - 'anelasticity_model_name'
            - 'attenuation_model_name'
        - options
        - specified model parameters, when the options allow it.
        - specified elastic load signal hyper parameters.
    Returns the list of load signal results subfolders, starting with description's IDs.

    The user has to whether specify:
        - 'anelasticity_description_ids' and 'model_filenames'.
        - 'anelasticity_description_ids' alone. In this case, 'model_filenames' is reconstructed.
        - 'model_filenames' alone. In this case, 'anelasticity_description_ids' is reconstructed.
        - model part names and eventually parameter variations. In this case, the list of description ids and model filenames
        are reconstructed from scratch.

    The 'parameters' input corresponds to parameters polynomial coefficients (list[float]) per model part, per
    parameter name, per layer name and per possibility.
    See ..Love_numbers.fors.Love_numbers_for_options_for_models_for_parameters for an example.
    """

    # Takes care of the specified input.
    if model_filenames is None and anelasticity_description_ids is None:
        # Creates all model files variations.
        model_filenames = create_all_model_variations(
            elasticity_model_names=elasticity_model_names,
            long_term_anelasticity_model_names=long_term_anelasticity_model_names,
            short_term_anelasticity_model_names=short_term_anelasticity_model_names,
            parameters=parameters,
            create=False,
        )

    if model_filenames is None:
        # Reconstructs all model filenames from anelasticity description IDs.
        model_filenames = {model_part: [] for model_part in ModelPart}
        for anelasticity_description_id in anelasticity_description_ids:
            for model_part, model_name in zip(ModelPart, anelasticity_description_id.split("_____")):
                if not model_name.replace("____", "/") in model_filenames[model_part]:
                    model_filenames[model_part] += [model_name.replace("____", "/")]

    if anelasticity_description_ids is None:
        anelasticity_description_ids = []
        # Loops on all possible triplet of model files to reconstruct anelascticty description IDs.
        for elasticity_model_name, long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
            model_filenames[ModelPart.elasticity],
            model_filenames[ModelPart.long_term_anelasticity],
            model_filenames[ModelPart.short_term_anelasticity],
        ):
            anelasticity_description_ids += [
                anelasticity_description_id_from_part_names(
                    elasticity_name=elasticity_model_name,
                    long_term_anelasticity_name=long_term_anelasticity_model_name,
                    short_term_anelasticity_name=short_term_anelasticity_model_name,
                )
            ]

    # Builds the list of all possible load signal hyper parameters.
    if load_signal_hyper_parameter_variations == {}:
        load_signal_hyper_parameters_list = [base_load_signal_hyper_parameters]
    else:
        load_signal_hyper_parameters_list = create_all_load_signal_hyper_parameters_variations(
            base_load_signal_hyper_parameters=base_load_signal_hyper_parameters,
            load_signal_hyper_parameter_variations=load_signal_hyper_parameter_variations,
        )

    means_per_path: dict[Path, dict[str, float]] = {}
    # Iterates on load signal possibilities.
    for load_signal_hyper_parameters in load_signal_hyper_parameters_list:

        # Log status.
        print(
            "case=",
            load_signal_hyper_parameters.case,
            "LIA=",
            load_signal_hyper_parameters.little_isostatic_adjustment,
            "continents=",
            load_signal_hyper_parameters.opposite_load_on_continents,
        )

        # For the very first description:
        # Computes load signals for all considered options.
        load_result_folder_elastic = compute_anelastic_induced_harmonic_load_per_description_per_options(
            anelasticity_description_ids=[anelasticity_description_ids[0]],
            load_signal_hyper_parameters=load_signal_hyper_parameters,
            options=options,
            do_elastic=True,
            src_directory=None,
        )[
            0
        ]  # Memorizes result's path for elastic load values.
        # Loops on options. Loops only on the anelasticity descriptions that need to be taken into account by the current
        # option.
        for run_hyper_parameters in options:

            # Log status.
            print(
                "long_term=",
                run_hyper_parameters.use_long_term_anelasticity,
                "short_term=",
                run_hyper_parameters.use_short_term_anelasticity,
                "bounded=",
                run_hyper_parameters.use_bounded_attenuation_functions,
            )

            # For the very first description:
            # Compute mean trends.
            result_subpath_elastic, _, _, _, territorial_means, _ = get_load_signal_harmonic_trends(
                do_elastic=True,
                load_signal_hyper_parameters=load_signal_hyper_parameters,
                run_hyper_parameters=run_hyper_parameters,
                anelasticity_description_id=anelasticity_description_ids[0],
                src_diretory=None,
            )  # Memorizes result's path for elastic load values.
            means_per_path[result_subpath_elastic] = territorial_means

            # Compute load signals for all considered options.
            compute_anelastic_induced_harmonic_load_per_description_per_options(
                anelasticity_description_ids=anelasticity_description_ids[1:],
                load_signal_hyper_parameters=load_signal_hyper_parameters,
                options=[run_hyper_parameters],
                do_elastic=False,
                src_directory=load_result_folder_elastic,
            )

            # Compute mean trends.
            for anelasticity_description_id in tqdm(anelasticity_description_ids[1:]):
                result_subpath, _, _, _, territorial_means, _ = get_load_signal_harmonic_trends(
                    do_elastic=False,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    run_hyper_parameters=run_hyper_parameters,
                    anelasticity_description_id=anelasticity_description_id,
                    src_diretory=result_subpath_elastic,
                )
                means_per_path[result_subpath] = territorial_means

    return means_per_path
