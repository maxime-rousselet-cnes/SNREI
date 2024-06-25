from itertools import product

from numpy import array, ndarray
from pyshtools import SHCoeffs

from ...functions import mean_on_mask
from ...utils import (
    ELASTIC_RUN_HYPER_PARAMETERS,
    AnelasticityDescription,
    LoadSignalHyperParameters,
    Love_numbers_computing,
    Love_numbers_path,
    LoveNumbersHyperParameters,
    ModelPart,
    Result,
    RunHyperParameters,
    add_result_to_table,
    anelastic_frequencial_harmonic_load_signal_computing,
    build_frequencial_harmonic_elastic_load_signal,
    compute_harmonic_signal_trends,
    compute_signal_trend,
    create_all_load_signal_hyper_parameters_variations,
    create_all_model_variations,
    dates_path,
    degree_one_inversion,
    elastic_load_signal_trends_path,
    elastic_load_signals_path,
    elastic_Love_numbers_computing,
    find_minimal_computing_options,
    frequencies_path,
    generate_degrees_list,
    generate_log_frequency_initial_values,
    generate_new_id,
    get_ocean_mask,
    harmonic_geoid_trends_path,
    harmonic_load_signal_trends_before_degree_one_replacement_path,
    harmonic_load_signal_trends_path,
    harmonic_radial_displacement_trends_path,
    harmonic_residual_trends_path,
    load_base_model,
    load_complex_array_from_binary,
    load_Love_numbers_hyper_parameters,
    map_sampling,
    parameters_path,
    redefine_n_max,
    save_base_model,
    save_complex_array_to_binary,
)


def compute_load_signal_trends_for_anelastic_Earth_models(
    elasticity_model_names: list[str],
    long_term_anelasticity_model_names: list[str],
    short_term_anelasticity_model_names: list[str],
    rheological_parameters: dict[ModelPart, dict[str, dict[str, list[list[float]]]]],
    load_signal_parameters: dict[str, list[str | bool]],
    options: list[RunHyperParameters],
    print_status: bool = True,
    save_components: bool = True,
) -> None:
    """
    Computes the load signal trends estimated with anelastic Earth hypothesis for several rheological models and load history
    models. This function:
        - I - Computes load signals with elastic Earth hypothesis for all load history models.
        - II - Computes Love numbers for all rheological models.
            - III - Computes load signals with anelastic Earth hypothesis for all Love numbers and load signals with elastic Earth
        hypothesis. Re-estimates degree 1 coefficients by inversion on oceans and computes trends.
    """

    # Initialization.

    # Builds all possible rheological model combinations.
    rheological_model_variations = create_all_model_variations(
        elasticity_model_names=elasticity_model_names,
        long_term_anelasticity_model_names=long_term_anelasticity_model_names,
        short_term_anelasticity_model_names=short_term_anelasticity_model_names,
        parameters=rheological_parameters,
    )

    # Builds all possible load history model combinations.
    load_signal_hyper_parameter_variations = create_all_load_signal_hyper_parameters_variations(
        load_signal_parameters=load_signal_parameters
    )

    # Gets hyper parameters.
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters = load_Love_numbers_hyper_parameters()
    degrees = generate_degrees_list(
        degree_thresholds=load_base_model(name="degree_thresholds", path=parameters_path),
        degree_steps=load_base_model(name="degree_steps", path=parameters_path),
    )

    # Defines a reference model.
    reference_model_filenames: dict[ModelPart, str] = {}
    for model_part, model_variations in rheological_model_variations.items():
        model_variations.sort()
        reference_model_filenames[model_part] = model_variations[0]

    # I - Loops on all load history models to produce elastic load signals.
    elastic_load_signal_datas: dict[
        str,
        tuple[LoadSignalHyperParameters, ndarray],  # ID and ocean mask.
    ] = {}
    for load_signal_hyper_parameters in load_signal_hyper_parameter_variations:

        # Builds the signal.
        (
            dates,
            frequencies,
            frequencial_harmonic_elastic_load_signal,
        ) = build_frequencial_harmonic_elastic_load_signal(load_signal_hyper_parameters=load_signal_hyper_parameters)
        ocean_mask = get_ocean_mask(
            name=load_signal_hyper_parameters.ocean_mask,
            n_max=redefine_n_max(
                n_max=load_signal_hyper_parameters.n_max,
                harmonics=frequencial_harmonic_elastic_load_signal,
            ),
        )

        # Saves dates, frequencies, the load signal and its trends.
        elastic_load_signal_id = generate_new_id(path=elastic_load_signals_path)
        save_base_model(obj=dates, name=elastic_load_signal_id, path=dates_path)
        save_base_model(obj=frequencies, name=elastic_load_signal_id, path=frequencies_path)
        save_complex_array_to_binary(
            input_array=frequencial_harmonic_elastic_load_signal,
            name=elastic_load_signal_id,
            path=elastic_load_signals_path,
        )
        harmonic_elastic_load_signal_trends = compute_harmonic_signal_trends(
            signal_dates=dates,
            load_signal_hyper_parameters=load_signal_hyper_parameters,
            frequencial_harmonic_signal=frequencial_harmonic_elastic_load_signal,
        )
        save_complex_array_to_binary(
            input_array=harmonic_elastic_load_signal_trends,
            name=elastic_load_signal_id,
            path=elastic_load_signal_trends_path,
        )
        ocean_mean = mean_on_mask(mask=ocean_mask, harmonics=harmonic_elastic_load_signal_trends)
        add_result_to_table(
            table_name="elastic_load_signals",
            result_caracteristics={
                "ID": elastic_load_signal_id,
                "ocean_mean": ocean_mean,
            }
            | load_signal_hyper_parameters.__dict__,
        )
        elastic_load_signal_datas[elastic_load_signal_id] = (
            load_signal_hyper_parameters,
            ocean_mask,
        )

    # II - Loops on rheological models to produce Love numbers.
    for (
        elasticity_model_name,
        long_term_anelasticity_model_name,
        short_term_anelasticity_model_name,
    ) in product(
        rheological_model_variations[ModelPart.elasticity],
        rheological_model_variations[ModelPart.long_term_anelasticity],
        rheological_model_variations[ModelPart.short_term_anelasticity],
    ):

        options_to_compute = find_minimal_computing_options(
            options=options,
            long_term_anelasticity_model_name=long_term_anelasticity_model_name,
            short_term_anelasticity_model_name=short_term_anelasticity_model_name,
            reference_model_filenames=reference_model_filenames,
        )

        # Creates the rheological description instance.
        anelasticity_description = AnelasticityDescription(
            elasticity_name=elasticity_model_name,
            long_term_anelasticity_name=long_term_anelasticity_model_name,
            short_term_anelasticity_name=short_term_anelasticity_model_name,
        )

        # If the option needs to be computed for this rheological description.
        for run_hyper_parameters in options_to_compute:

            Love_numbers_hyper_parameters.run_hyper_parameters = run_hyper_parameters

            # Computes Love numbers.
            log_frequencies, Love_numbers_array = Love_numbers_computing(
                max_tol=Love_numbers_hyper_parameters.max_tol,
                decimals=Love_numbers_hyper_parameters.decimals,
                y_system_hyper_parameters=Love_numbers_hyper_parameters.y_system_hyper_parameters,
                run_hyper_parameters=run_hyper_parameters,
                degrees=degrees,
                log_frequency_initial_values=generate_log_frequency_initial_values(
                    period_min_year=Love_numbers_hyper_parameters.period_min_year,
                    period_max_year=Love_numbers_hyper_parameters.period_max_year,
                    n_frequency_0=Love_numbers_hyper_parameters.n_frequency_0,
                    frequency_unit=anelasticity_description.frequency_unit,
                ),
                anelasticity_description=anelasticity_description,
            )

            # Saves.
            Love_numbers_result = Result(
                hyper_parameters=Love_numbers_hyper_parameters,
                axes={
                    "degrees": array(object=degrees),
                    "frequencies": anelasticity_description.frequency_unit * 10.0**log_frequencies,
                },
            )
            Love_numbers_result.update_values_from_array(
                result_array=Love_numbers_array,
            )
            Love_numbers_id = generate_new_id(path=Love_numbers_path)
            Love_numbers_result.save(name=Love_numbers_id, path=Love_numbers_path)
            add_result_to_table(
                table_name="Love_numbers",
                result_caracteristics={
                    "ID": Love_numbers_id,
                    "elasticity_model": elasticity_model_name,
                    "long_term_anelasticity_model": long_term_anelasticity_model_name,
                    "short_term_anelasticity_model": short_term_anelasticity_model_name,
                    "anelasticity_description_id": anelasticity_description.id,
                    "max_tol": Love_numbers_hyper_parameters.max_tol,
                    "decimals": Love_numbers_hyper_parameters.decimals,
                }
                | {
                    key: value
                    for key, value in Love_numbers_hyper_parameters.y_system_hyper_parameters.__dict__.items()
                    if type(value) is bool
                }
                | run_hyper_parameters.__dict__,
            )

            # III - Loops on elastic load signals to compute anelastic load signals.
            for elastic_load_signal_id, (
                load_signal_hyper_parameters,
                ocean_mask,
            ) in elastic_load_signal_datas.items():

                # Loads needed datas.
                elastic_frequencial_harmonic_load_signal = load_complex_array_from_binary(
                    name=elastic_load_signal_id, path=elastic_load_signals_path
                )
                dates = array(object=load_base_model(name=elastic_load_signal_id, path=dates_path))
                frequencies = array(object=load_base_model(name=elastic_load_signal_id, path=frequencies_path))
                harmonic_load_signal_id = generate_new_id(path=harmonic_load_signal_trends_path)

                if run_hyper_parameters == ELASTIC_RUN_HYPER_PARAMETERS:
                    # Memorizes elastic Love numbers for other runs.
                    elastic_Love_numbers = elastic_Love_numbers_computing(
                        anelasticity_description_id=anelasticity_description.id,
                        n_max=load_signal_hyper_parameters.n_max,
                        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                    )
                    frequencial_harmonic_load_signal = elastic_frequencial_harmonic_load_signal
                    Love_numbers = elastic_Love_numbers
                else:
                    # Computes anelastic load signal.
                    (
                        frequencial_harmonic_load_signal,
                        Love_numbers,
                    ) = anelastic_frequencial_harmonic_load_signal_computing(
                        anelasticity_description_id=anelasticity_description.id,
                        n_max=load_signal_hyper_parameters.n_max,
                        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                        signal_frequencies=frequencies,
                        frequencial_elastic_normalized_load_signal=elastic_frequencial_harmonic_load_signal,
                        elastic_Love_numbers=elastic_Love_numbers,
                    )

                if print_status or save_components:
                    harmonic_load_signal_trends_before_degree_one_replacement = compute_harmonic_signal_trends(
                        signal_dates=dates,
                        load_signal_hyper_parameters=load_signal_hyper_parameters,
                        frequencial_harmonic_signal=frequencial_harmonic_load_signal,
                    )

                if print_status:

                    print("description id:", anelasticity_description.id)
                    print("run:", run_hyper_parameters)
                    print(
                        "load mean before degree one replacement:",
                        mean_on_mask(
                            mask=ocean_mask,
                            harmonics=harmonic_load_signal_trends_before_degree_one_replacement,
                        ),
                    )

                if save_components:
                    save_complex_array_to_binary(
                        input_array=harmonic_load_signal_trends_before_degree_one_replacement,
                        name=harmonic_load_signal_id,
                        path=harmonic_load_signal_trends_before_degree_one_replacement_path,
                    )

                # Derives degree one correction.
                (
                    frequencial_harmonic_load_signal[:, 1, :2, :],
                    frequencial_scale_factor,
                    frequencial_harmonic_geoid,
                    frequencial_harmonic_radial_displacement,
                ) = degree_one_inversion(
                    anelastic_frequencial_harmonic_load_signal=frequencial_harmonic_load_signal,
                    anelastic_hermitian_Love_numbers=Love_numbers,
                    ocean_mask=ocean_mask,
                )

                # Computes trends.
                harmonic_load_signal_trends = compute_harmonic_signal_trends(
                    signal_dates=dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal,
                )
                scale_factor_component = compute_signal_trend(
                    signal_dates=dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    input_signal=frequencial_scale_factor,
                )
                harmonic_geoid_trends = compute_harmonic_signal_trends(
                    signal_dates=dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_geoid,
                )
                harmonic_radial_displacement_trends = compute_harmonic_signal_trends(
                    signal_dates=dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_radial_displacement,
                )

                ocean_mean = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_load_signal_trends,
                )
                if print_status:
                    print(
                        "load mean after degree one replacement:",
                        ocean_mean,
                    )
                    print()

                # Saves.
                ocean_mean_geoid_component = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_geoid_trends,
                )
                ocean_mean_radial_displacement_component = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_radial_displacement_trends,
                )
                add_result_to_table(
                    table_name="harmonic_load_signal_trends",
                    result_caracteristics={
                        "ID": harmonic_load_signal_id,
                        "Love_numbers_ID": Love_numbers_id,
                        "elastic_load_signal_ID": elastic_load_signal_id,
                        "ocean_mean": ocean_mean,  # (L).
                        "scale_factor_component": scale_factor_component,  # (D).
                        "ocean_mean_geoid_component": ocean_mean_geoid_component,  # (G).
                        "ocean_mean_radial_displacement_component": ocean_mean_radial_displacement_component,  # (R).
                        "ocean_mean_radial_residuals": ocean_mean
                        + scale_factor_component
                        - (ocean_mean_geoid_component - ocean_mean_radial_displacement_component),
                    }  # (L + D - (G - R)).
                    | load_signal_hyper_parameters.__dict__,
                )

                if save_components:

                    # Load signal.
                    save_complex_array_to_binary(
                        input_array=harmonic_load_signal_trends,
                        name=harmonic_load_signal_id,
                        path=harmonic_load_signal_trends_path,
                    )

                    # Geoid height.
                    save_complex_array_to_binary(
                        input_array=harmonic_geoid_trends,
                        name=harmonic_load_signal_id,
                        path=harmonic_geoid_trends_path,
                    )

                    # Radial displacement.
                    save_complex_array_to_binary(
                        input_array=harmonic_radial_displacement_trends,
                        name=harmonic_load_signal_id,
                        path=harmonic_radial_displacement_trends_path,
                    )

                    # Computes residuals.
                    frequencial_harmonic_residuals = (
                        harmonic_load_signal_trends  # L
                        + scale_factor_component
                        * map_sampling(map=ocean_mask, n_max=load_signal_hyper_parameters.n_max, harmonic_domain=True)[
                            0
                        ]  # + D
                        - (harmonic_geoid_trends - harmonic_radial_displacement_trends)  # - (G - R)
                    )
                    save_complex_array_to_binary(
                        input_array=frequencial_harmonic_residuals,
                        name=harmonic_load_signal_id,
                        path=harmonic_residual_trends_path,
                    )
