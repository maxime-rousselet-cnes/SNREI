from itertools import product
from pathlib import Path
from typing import Optional

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
    find_minimal_computing_options,
    minimal_computing_options,
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
            for model_part, model_name in zip(ModelPart, anelasticity_description_id.split("____")):
                if not model_name in model_filenames[model_part]:
                    model_filenames[model_part] += [model_name]

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

    load_result_folders: list[Path] = []
    means_per_path: dict[Path, dict[str, float]] = {}
    anelasticity_description_id: str
    for anelasticity_description_id, load_signal_hyper_parameters in product(
        anelasticity_description_ids, load_signal_hyper_parameters_list
    ):
        # Finds minimal computing options.
        do_elastic, do_long_term_only_case, do_short_term_only_case = find_minimal_computing_options(
            long_term_anelasticity_model_name=anelasticity_description_id.split("____")[1],
            short_term_anelasticity_model_name=anelasticity_description_id.split("____")[2],
            model_filenames=model_filenames,
        )
        minimal_options = minimal_computing_options(
            options=options,
            do_long_term_only_case=do_long_term_only_case,
            do_short_term_only_case=do_short_term_only_case,
        )

        # Compute load signals for all considered options.
        load_result_folders += compute_anelastic_induced_harmonic_load_per_description_per_options(
            anelasticity_description_ids=[anelasticity_description_id],
            load_signal_hyper_parameters=load_signal_hyper_parameters,
            options=minimal_options,
            do_elastic=do_elastic,
            src_directory=None if do_elastic else load_result_folders[0],
        )

        # Compute mean trends.
        for run_hyper_parameters in minimal_options:
            result_subpath, _, _, _, territorial_means, _ = get_load_signal_harmonic_trends(
                do_elastic=do_elastic,
                load_signal_hyper_parameters=load_signal_hyper_parameters,
                run_hyper_parameters=run_hyper_parameters,
                anelasticity_description_id=anelasticity_description_id,
                src_diretory=None if do_elastic else result_subpath,
            )
            means_per_path[result_subpath] = territorial_means

    # Symlinks.
    create_symlinks_to_results(model_filenames=model_filenames, options=options)

    return load_result_folders, means_per_path
