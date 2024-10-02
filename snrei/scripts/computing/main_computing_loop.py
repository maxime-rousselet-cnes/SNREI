from copy import deepcopy
from itertools import product
from time import time

from numpy import array, ndarray, tensordot

from ...functions import make_grid, mean_on_mask
from ...utils import (
    C_2_1_PT_SE_ELASTIC_CORRECTION,
    ELASTIC_RUN_HYPER_PARAMETERS,
    S_2_1_PT_SE_ELASTIC_CORRECTION,
    SECONDS_PER_YEAR,
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
    base_format_load_signal_trends_path,
    build_elastic_load_signal_components,
    build_elastic_load_signal_history,
    compute_harmonic_signal_trends,
    compute_signal_trend,
    create_all_load_signal_hyper_parameters_variations,
    create_all_model_variations,
    dates_path,
    degree_one_inversion,
    elastic_load_signal_trends_path,
    find_minimal_computing_options,
    frequencies_path,
    generate_degrees_list,
    generate_log_frequency_initial_values,
    generate_new_id,
    harmonic_load_signal_trends_path,
    interpolate_anelastic_Love_numbers,
    interpolate_elastic_Love_numbers,
    leakage_correction,
    load_base_model,
    load_Love_numbers_hyper_parameters,
    map_sampling,
    parameters_path,
    polar_motion_correction,
    save_base_format,
    save_base_model,
    save_complex_array_to_binary,
    save_harmonics,
)


def compute_load_signal_trends_for_anelastic_Earth_models(
    elasticity_model_names: list[str],
    long_term_anelasticity_model_names: list[str],
    short_term_anelasticity_model_names: list[str],
    rheological_parameters: dict[ModelPart, dict[str, dict[str, list[list[float]]]]],
    load_signal_parameters: dict[str, list[str | bool]],
    options: list[RunHyperParameters],
) -> None:
    """
    Computes the load signal trends estimated with anelastic Earth hypothesis for several rheological models and load history
    models. This function:
        - I - Computes load signals with elastic Earth hypothesis for all load history models.
        - II - Computes Love numbers for all rheological models.
            - III - Computes load signals with anelastic Earth hypothesis for all Love numbers and load signals with elastic Earth
        hypothesis.
                - Interpolates Love numbers on load signal frequencies.
                - loops until anelastic load signal past trend matches source data:
                    - Defines elastic harmonic frequencial signal with wanted past trend.
                    - Computes anelastic load signal by frequencial filtering with Love number fractions.
                    - Re-estimates degree one coefficients by inversion on oceans.
                    - performs leakage correction.
                - Computes trends.
                - Saves.
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
    load_signal_hyper_parameter_variations = create_all_load_signal_hyper_parameters_variations(load_signal_parameters=load_signal_parameters)

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
        str,  # ID.
        tuple[
            LoadSignalHyperParameters,
            ndarray,  # Ocean/land mask.
            ndarray,  # Ocean/land buffered mask,
            ndarray,  # Spatial_component.
            ndarray,  # Initial signal dates.
            ndarray,  # Signal frequencies.
            ndarray,  # Initial load signal.
            ndarray,  # Latitudes.
            ndarray,  # Longitudes.
            float,  # Target past trend.
        ],
    ] = {}
    for load_signal_hyper_parameters in load_signal_hyper_parameter_variations:

        t_0 = time()

        load_signal_hyper_parameters.mean_signal_threshold = float(load_signal_hyper_parameters.mean_signal_threshold)
        load_signal_hyper_parameters.mean_signal_threshold = float(load_signal_hyper_parameters.mean_signal_threshold_past)

        # Builds the signal.
        (
            load_signal_hyper_parameters.n_max,
            initial_signal_dates,  # Temporal component's dates.
            harmonic_elastic_load_signal_spatial_component,
            initial_load_signal,
            ocean_land_mask,
            ocean_land_buffered_mask,
            latitudes,
            longitudes,
        ) = build_elastic_load_signal_components(load_signal_hyper_parameters=load_signal_hyper_parameters)

        # "RIES" correction back.
        harmonic_elastic_load_signal_spatial_component[0, 2, 1] += C_2_1_PT_SE_ELASTIC_CORRECTION
        harmonic_elastic_load_signal_spatial_component[1, 2, 1] += S_2_1_PT_SE_ELASTIC_CORRECTION

        (signal_dates, signal_frequencies, _, target_past_trend, recent_trend) = build_elastic_load_signal_history(
            initial_signal_dates=initial_signal_dates,
            initial_load_signal=initial_load_signal,
            load_signal_hyper_parameters=load_signal_hyper_parameters,
        )
        trends_factor = target_past_trend / recent_trend
        load_signal_hyper_parameters.signal_threshold_past = trends_factor * load_signal_hyper_parameters.signal_threshold
        load_signal_hyper_parameters.mean_signal_threshold_past = trends_factor * load_signal_hyper_parameters.mean_signal_threshold

        # Saves dates, frequencies, and load signal trends.
        elastic_load_signal_trends_id = generate_new_id(path=elastic_load_signal_trends_path)
        save_base_model(obj=signal_dates, name=elastic_load_signal_trends_id, path=dates_path)
        save_base_model(
            obj=signal_frequencies,
            name=elastic_load_signal_trends_id,
            path=frequencies_path,
        )
        save_complex_array_to_binary(
            input_array=harmonic_elastic_load_signal_spatial_component,
            name=elastic_load_signal_trends_id,
            path=elastic_load_signal_trends_path,
        )

        # Updates.
        elastic_load_signal_datas[elastic_load_signal_trends_id] = (
            load_signal_hyper_parameters,
            ocean_land_mask,
            ocean_land_buffered_mask,
            harmonic_elastic_load_signal_spatial_component,
            initial_signal_dates,
            signal_frequencies,
            initial_load_signal,
            latitudes,
            longitudes,
            target_past_trend,
        )

        print("Building elastic load signal:", time() - t_0)

    # II - Loops on rheological models to produce Love numbers.
    models_iterable = list(
        product(
            rheological_model_variations[ModelPart.elasticity],
            rheological_model_variations[ModelPart.long_term_anelasticity],
            rheological_model_variations[ModelPart.short_term_anelasticity],
        )
    )
    for (
        elasticity_model_name,
        long_term_anelasticity_model_name,
        short_term_anelasticity_model_name,
    ) in models_iterable:

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

            t_0 = time()

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
                    "frequencies": anelasticity_description.frequency_unit * 10.0**log_frequencies * SECONDS_PER_YEAR,
                },  # (yr^-1)
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
                | {key: value for key, value in Love_numbers_hyper_parameters.y_system_hyper_parameters.__dict__.items() if type(value) is bool}
                | run_hyper_parameters.__dict__,
            )

            print("Computing Love numbers:", time() - t_0)

            # III - Loops on elastic load signals to compute anelastic load signals.
            for elastic_load_signal_trends_id, (
                load_signal_hyper_parameters,
                ocean_land_mask,
                ocean_land_buffered_mask,
                harmonic_elastic_load_signal_spatial_component,
                initial_signal_dates,
                signal_frequencies,
                initial_load_signal,
                latitudes,
                longitudes,
                target_past_trend,
            ) in elastic_load_signal_datas.items():

                t_0 = time()

                # initializes.
                harmonic_load_signal_id = generate_new_id(path=harmonic_load_signal_trends_path.joinpath("step_5"))
                # Arbitrary intitial value.
                past_trend = target_past_trend + 2 * load_signal_hyper_parameters.past_trend_error
                elastic_past_trend = target_past_trend

                # Interpolates Love numbers.
                if run_hyper_parameters == ELASTIC_RUN_HYPER_PARAMETERS:
                    # Memorizes elastic Love numbers for other runs.
                    elastic_Love_numbers = interpolate_elastic_Love_numbers(
                        anelasticity_description_id=anelasticity_description.id,
                        n_max=load_signal_hyper_parameters.n_max,
                        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                    )
                    anelastic_Love_numbers = elastic_Love_numbers
                else:
                    # Interpolates anelastic Love numbers on signal degrees and frequencies as hermitian signal.
                    anelastic_Love_numbers: Result = interpolate_anelastic_Love_numbers(
                        n_max=load_signal_hyper_parameters.n_max,
                        anelasticity_description_id=anelasticity_description.id,
                        target_frequencies=signal_frequencies,  # (yr^-1).
                        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
                    )

                # Loops on elastic past trend while anelastic past trend is too far from data source past trend.
                while abs(past_trend - target_past_trend) > load_signal_hyper_parameters.past_trend_error:

                    t_1 = time()

                    # Updates.
                    elastic_past_trend = elastic_past_trend * target_past_trend / past_trend

                    # Builds an elastic load history with given past trend.
                    (signal_dates, _, elastic_unitless_load_signal, _, _) = build_elastic_load_signal_history(
                        initial_signal_dates=initial_signal_dates,
                        initial_load_signal=initial_load_signal,
                        load_signal_hyper_parameters=load_signal_hyper_parameters,
                        elastic_past_trend=elastic_past_trend,
                    )

                    # Generates an elastic load signal depending on space and time.
                    frequencial_harmonic_load_signal_step_1 = tensordot(
                        a=harmonic_elastic_load_signal_spatial_component,
                        b=elastic_unitless_load_signal,
                        axes=0,
                    )

                    frequencial_harmonic_load_signal_step_2 = frequencial_harmonic_load_signal_step_1.copy()

                    # Performs polar tide corrections.
                    if load_signal_hyper_parameters.polar_tide_correction:  # and run_hyper_parameters != ELASTIC_RUN_HYPER_PARAMETERS:
                        C_2_1_PM, S_2_1_PM = polar_motion_correction(
                            load_signal_hyper_parameters=load_signal_hyper_parameters,
                            Love_numbers=anelastic_Love_numbers,
                            signal_dates=signal_dates,
                            signal_frequencies=signal_frequencies,
                        )
                        frequencial_harmonic_load_signal_step_2[0, 2, 1, :] -= C_2_1_PM
                        frequencial_harmonic_load_signal_step_2[1, 2, 1, :] -= S_2_1_PM

                    # Performs product between Love number fraction and elastic load signal in frequencial harmonic domain.
                    frequencial_harmonic_load_signal_step_3 = (
                        frequencial_harmonic_load_signal_step_2.copy()
                        if run_hyper_parameters == ELASTIC_RUN_HYPER_PARAMETERS
                        else anelastic_frequencial_harmonic_load_signal_computing(
                            elastic_Love_numbers=elastic_Love_numbers,
                            anelastic_Love_numbers=anelastic_Love_numbers,
                            signal_frequencies=signal_frequencies,
                            frequencial_elastic_load_signal=frequencial_harmonic_load_signal_step_2,
                        )
                    )

                    t_2 = time()
                    # Computes needed trends.
                    harmonic_load_signal_step_3_trends = compute_harmonic_signal_trends(
                        signal_dates=signal_dates,
                        load_signal_hyper_parameters=load_signal_hyper_parameters,
                        frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_3,
                    )

                    # Derives degree one correction.
                    frequencial_harmonic_load_signal_step_4 = deepcopy(frequencial_harmonic_load_signal_step_3)
                    (
                        frequencial_harmonic_load_signal_step_4[:, 1, :2, :],
                        frequencial_scale_factor,
                        frequencial_harmonic_geoid,
                        frequencial_harmonic_radial_displacement,
                        frequencial_harmonic_residuals,
                    ) = degree_one_inversion(
                        anelastic_frequencial_harmonic_load_signal=frequencial_harmonic_load_signal_step_3,
                        anelastic_harmonic_load_signal_trends=harmonic_load_signal_step_3_trends,
                        Love_numbers=anelastic_Love_numbers,
                        ocean_land_mask=ocean_land_mask,
                        signal_threshold=load_signal_hyper_parameters.signal_threshold,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        n_max=load_signal_hyper_parameters.n_max,
                        compute_residuals=load_signal_hyper_parameters.compute_residuals,
                    )

                    print("Degree one inversion:", time() - t_2)
                    t_2 = time()

                    # Prepares arrays.
                    harmonic_load_signal_step_4_past_trends = compute_harmonic_signal_trends(
                        signal_dates=signal_dates,
                        load_signal_hyper_parameters=load_signal_hyper_parameters,
                        frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_4,
                        recent_trend=False,
                    )

                    # Leakage correction.
                    harmonic_load_signal_step_5_past_trends = leakage_correction(  # Past trend leakage correction.
                        harmonic_load_signal=harmonic_load_signal_step_4_past_trends,
                        ocean_land_mask=ocean_land_mask,
                        ocean_land_buffered_mask=ocean_land_buffered_mask,
                        latitudes=latitudes,
                        n_max=load_signal_hyper_parameters.n_max,
                        ddk_filter_level=load_signal_hyper_parameters.ddk_filter_level,
                        iterations=load_signal_hyper_parameters.leakage_correction_iterations,
                        signal_threshold=load_signal_hyper_parameters.signal_threshold_past,
                        mean_signal_threshold=load_signal_hyper_parameters.mean_signal_threshold_past,
                    )

                    print("Leakage correction:", time() - t_2)

                    # Normalizes so that the data previous to 2003 matches source datas.
                    past_trend = mean_on_mask(
                        mask=ocean_land_buffered_mask,
                        latitudes=latitudes,
                        n_max=load_signal_hyper_parameters.n_max,
                        harmonics=harmonic_load_signal_step_5_past_trends,
                        signal_threshold=load_signal_hyper_parameters.mean_signal_threshold_past,
                    )
                    print("past trend: ", past_trend, " / ", target_past_trend)
                    print("Iteration process:", time() - t_1)

                # Computes trends.

                # Signal.
                harmonic_load_signal_step_1_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_1,
                )
                ocean_mean_step_1 = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_load_signal_step_1_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # After correction of polar tide terms.
                harmonic_load_signal_step_2_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_2,
                )
                ocean_mean_step_2 = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_load_signal_step_2_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # After frequencial filering by Love number fractions.
                ocean_mean_step_3 = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_load_signal_step_3_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # After degree one inversion.
                harmonic_load_signal_step_4_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_4,
                )
                ocean_mean_step_4 = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_load_signal_step_4_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # After leakage correction.

                # Leakage correction.
                harmonic_load_signal_step_5_trends = leakage_correction(  # Recent trend leakage correction.
                    harmonic_load_signal=harmonic_load_signal_step_4_trends,
                    ocean_land_mask=ocean_land_mask,
                    ocean_land_buffered_mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    ddk_filter_level=load_signal_hyper_parameters.ddk_filter_level,
                    iterations=load_signal_hyper_parameters.leakage_correction_iterations,
                    signal_threshold=load_signal_hyper_parameters.signal_threshold,
                    mean_signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                ocean_mean_step_5 = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_load_signal_step_5_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # Inversion components.

                # Scale factor.
                scale_factor_component = compute_signal_trend(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    input_signal=frequencial_scale_factor,
                )

                # Geoid height.
                harmonic_geoid_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_geoid,
                )
                ocean_mean_geoid_component = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_geoid_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # Radial displacement.
                harmonic_radial_displacement_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_radial_displacement,
                )
                ocean_mean_radial_displacement_component = mean_on_mask(
                    mask=ocean_land_buffered_mask,
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_radial_displacement_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # Computes residuals.
                harmonic_residual_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_residuals,
                )
                ocean_mean_residuals = mean_on_mask(
                    mask=ocean_land_buffered_mask
                    * (
                        abs(make_grid(harmonics=harmonic_load_signal_step_3_trends, n_max=load_signal_hyper_parameters.n_max))
                        < load_signal_hyper_parameters.signal_threshold
                    ),
                    latitudes=latitudes,
                    n_max=load_signal_hyper_parameters.n_max,
                    harmonics=harmonic_residual_trends,
                    signal_threshold=load_signal_hyper_parameters.mean_signal_threshold,
                )

                # Saves.
                add_result_to_table(
                    table_name="harmonic_load_signal_trends",
                    result_caracteristics={
                        "ID": harmonic_load_signal_id,
                        "ocean_mean_step_1": ocean_mean_step_1,
                        "ocean_mean_step_2": ocean_mean_step_2,
                        "ocean_mean_step_3": ocean_mean_step_3,
                        "ocean_mean_step_4": ocean_mean_step_4,  # (L).
                        "ocean_mean_step_5": ocean_mean_step_5,
                        "leakage_correction_iterations": load_signal_hyper_parameters.leakage_correction_iterations,
                        "buffer_distance": load_signal_hyper_parameters.buffer_distance,
                        "phi_constant": load_signal_hyper_parameters.phi_constant,
                        "polar_tide_correction": load_signal_hyper_parameters.polar_tide_correction,
                        "mean_pole_convention": load_signal_hyper_parameters.mean_pole_convention,
                        "pole_case": load_signal_hyper_parameters.pole_case,
                        "elastic_load_signal_trends_id": elastic_load_signal_trends_id,
                        "Love_numbers_ID": Love_numbers_id,
                        "elastic_past_trend": past_trend,
                        "LIA": load_signal_hyper_parameters.LIA,
                        "case": load_signal_hyper_parameters.load_history_case,
                        "load_spatial_behaviour_file": load_signal_hyper_parameters.load_spatial_behaviour_file,
                        "opposite_load_on_continents": load_signal_hyper_parameters.opposite_load_on_continents,
                        "filter_wobble": load_signal_hyper_parameters.filter_wobble,
                        "signal_threshold": load_signal_hyper_parameters.signal_threshold,
                        "signal_threshold_past": load_signal_hyper_parameters.signal_threshold_past,
                        "scale_factor_component": scale_factor_component,  # (D).
                        "ocean_mean_geoid_component": ocean_mean_geoid_component,  # (G).
                        "ocean_mean_radial_displacement_component": ocean_mean_radial_displacement_component,  # (R).
                        "ocean_mean_radial_residuals": ocean_mean_residuals,  # (L + D - (G - R)).
                        "short-term": run_hyper_parameters.use_short_term_anelasticity,
                        "long-term": run_hyper_parameters.use_long_term_anelasticity,
                        "elasticity_model": elasticity_model_name,
                        "long_term_anelasticity_model": long_term_anelasticity_model_name,
                        "short_term_anelasticity_model": short_term_anelasticity_model_name,
                    },
                )

                # Eventually saves trends.
                for save_function, save_base_path, kind in zip(
                    [
                        save_harmonics,
                        save_base_format,
                    ],
                    [
                        harmonic_load_signal_trends_path,
                        base_format_load_signal_trends_path,
                    ],
                    ["harmonics", "base_formats"],
                ):
                    # Initial load signal.
                    if load_signal_hyper_parameters.save_parameters[kind].step_1:
                        save_function(
                            trends_array=harmonic_load_signal_step_1_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_1"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                    # Load signal after replacing C_2_1/S_2_1.
                    if load_signal_hyper_parameters.save_parameters[kind].step_2:
                        save_function(
                            trends_array=harmonic_load_signal_step_2_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_2"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                    # Anelastic load signal after frequencial filtering with Love number fractions.
                    if load_signal_hyper_parameters.save_parameters[kind].step_3:
                        save_function(
                            trends_array=harmonic_load_signal_step_3_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_3"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                    # Anelastic load signal after degree one inversion.
                    if load_signal_hyper_parameters.save_parameters[kind].step_4:
                        save_function(
                            trends_array=harmonic_load_signal_step_4_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_4"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                    # Anelastic load signal after leakage correction.
                    if load_signal_hyper_parameters.save_parameters[kind].step_5:
                        save_function(
                            trends_array=harmonic_load_signal_step_5_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_5"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                    # Degree one inversion components.
                    if load_signal_hyper_parameters.save_parameters[kind].inversion_components:
                        save_function(
                            trends_array=harmonic_geoid_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("geoid"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                        save_function(
                            trends_array=harmonic_radial_displacement_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("radial_displacement"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )
                        save_function(
                            trends_array=harmonic_residual_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("residual"),
                            n_max=load_signal_hyper_parameters.n_max,
                        )

                print("Computing anelastic induced signal:", time() - t_0)
