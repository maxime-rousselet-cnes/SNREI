from copy import deepcopy
from itertools import product

from numpy import array, ndarray, tensordot
from pyGFOToolbox import GRACE_collection_SH
from tqdm import tqdm

from ...functions import mean_on_mask
from ...utils import (
    ELASTIC_RUN_HYPER_PARAMETERS,
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
    print_status: bool = True,
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
        str,  # ID.
        tuple[
            LoadSignalHyperParameters,
            ndarray,  # Ocean mask.
            ndarray,  # Spatial_component.
            ndarray,  # Initial signal dates.
            ndarray,  # Signal frequencies.
            ndarray,  # Initial load signal.
            float,  # Target past trend.
        ],
    ] = {}
    for load_signal_hyper_parameters in tqdm(load_signal_hyper_parameter_variations, desc="Preparing load signals"):

        # Builds the signal.
        (
            load_signal_hyper_parameters.n_max,
            initial_signal_dates,  # Temporal component's dates.
            harmonic_elastic_load_signal_spatial_component,
            initial_load_signal,
            ocean_mask,
        ) = build_elastic_load_signal_components(load_signal_hyper_parameters=load_signal_hyper_parameters)

        (signal_dates, signal_frequencies, _, target_past_trend) = build_elastic_load_signal_history(
            initial_signal_dates=initial_signal_dates,
            initial_load_signal=initial_load_signal,
            load_signal_hyper_parameters=load_signal_hyper_parameters,
        )

        # Saves dates, frequencies, and load signal trends.
        elastic_load_signal_trends_id = generate_new_id(path=elastic_load_signal_trends_path)
        save_base_model(obj=signal_dates, name=elastic_load_signal_trends_id, path=dates_path)
        save_base_model(obj=signal_frequencies, name=elastic_load_signal_trends_id, path=frequencies_path)
        save_complex_array_to_binary(
            input_array=harmonic_elastic_load_signal_spatial_component,
            name=elastic_load_signal_trends_id,
            path=elastic_load_signal_trends_path,
        )

        # Updates.
        elastic_load_signal_datas[elastic_load_signal_trends_id] = (
            load_signal_hyper_parameters,
            ocean_mask,
            harmonic_elastic_load_signal_spatial_component,
            initial_signal_dates,
            signal_frequencies,
            initial_load_signal,
            target_past_trend,
        )

    # II - Loops on rheological models to produce Love numbers.
    for (
        elasticity_model_name,
        long_term_anelasticity_model_name,
        short_term_anelasticity_model_name,
    ) in tqdm(
        product(
            rheological_model_variations[ModelPart.elasticity],
            rheological_model_variations[ModelPart.long_term_anelasticity],
            rheological_model_variations[ModelPart.short_term_anelasticity],
        ),
        desc="Looping on models",
        position=0,
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
        for run_hyper_parameters in tqdm(options_to_compute, desc="    Looping on options", position=1, leave=False):

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
                | {
                    key: value
                    for key, value in Love_numbers_hyper_parameters.y_system_hyper_parameters.__dict__.items()
                    if type(value) is bool
                }
                | run_hyper_parameters.__dict__,
            )

            # III - Loops on elastic load signals to compute anelastic load signals.
            for elastic_load_signal_trends_id, (
                load_signal_hyper_parameters,
                ocean_mask,
                harmonic_elastic_load_signal_spatial_component,
                initial_signal_dates,
                signal_frequencies,
                initial_load_signal,
                target_past_trend,
            ) in tqdm(
                elastic_load_signal_datas.items(),
                desc="        Looping on elastic load signals",
                position=2,
                leave=False,
            ):

                # initializes.
                harmonic_load_signal_id = generate_new_id(path=harmonic_load_signal_trends_path.joinpath("step_3"))
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
                    Love_numbers = elastic_Love_numbers
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

                    # Updates.
                    elastic_past_trend = elastic_past_trend * target_past_trend / past_trend

                    # Builds an elastic load history with given past trend.
                    (signal_dates, _, elastic_unitless_load_signal, _) = build_elastic_load_signal_history(
                        initial_signal_dates=initial_signal_dates,
                        initial_load_signal=initial_load_signal,
                        load_signal_hyper_parameters=load_signal_hyper_parameters,
                        elastic_past_trend=elastic_past_trend,
                    )

                    # Generates an elastic load signal depending on space and time.
                    frequencial_harmonic_load_signal_step_0 = tensordot(
                        a=harmonic_elastic_load_signal_spatial_component,
                        b=elastic_unitless_load_signal,
                        axes=0,
                    )

                    # Performs product between Love number fraction and elastic load signal in frequencial harmonic domain.
                    frequencial_harmonic_load_signal_step_1 = (
                        frequencial_harmonic_load_signal_step_0
                        if run_hyper_parameters == ELASTIC_RUN_HYPER_PARAMETERS
                        else anelastic_frequencial_harmonic_load_signal_computing(
                            elastic_Love_numbers=elastic_Love_numbers,
                            anelastic_Love_numbers=anelastic_Love_numbers,
                            signal_frequencies=signal_frequencies,
                            frequencial_elastic_load_signal=frequencial_harmonic_load_signal_step_0,
                        )
                    )

                    # Derives degree one correction.
                    frequencial_harmonic_load_signal_step_2 = deepcopy(frequencial_harmonic_load_signal_step_1)
                    (
                        frequencial_harmonic_load_signal_step_2[:, 1, :2, :],
                        frequencial_scale_factor,
                        frequencial_harmonic_geoid,
                        frequencial_harmonic_radial_displacement,
                    ) = degree_one_inversion(
                        anelastic_frequencial_harmonic_load_signal=frequencial_harmonic_load_signal_step_1,
                        Love_numbers=Love_numbers,
                        ocean_mask=ocean_mask,
                    )

                    # Leakage correction.
                    frequencial_harmonic_load_signal_step_3 = leakage_correction(
                        signal_frequencies=signal_frequencies,
                        frequencial_harmonic_load_signal=frequencial_harmonic_load_signal_step_2,
                        ocean_mask=ocean_mask,
                        iterations=load_signal_hyper_parameters.leakage_correction_iterations,
                        ddk_filter_level=load_signal_hyper_parameters.ddk_filter_level,
                    )

                    # Normalizes so that the data previous to 2003 matches source datas.
                    past_trend = mean_on_mask(
                        mask=ocean_mask,
                        harmonics=compute_harmonic_signal_trends(
                            signal_dates=signal_dates,
                            load_signal_hyper_parameters=load_signal_hyper_parameters,
                            frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_3,
                            recent_trend=False,
                        ),
                    )

                # Computes trends.

                # Signal.

                # After frequencial filering by Love number fractions.
                harmonic_load_signal_step_1_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_1,
                )
                ocean_mean_step_1 = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_load_signal_step_1_trends,
                )

                # After degree one inversion.
                harmonic_load_signal_step_2_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_2,
                )
                ocean_mean_step_2 = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_load_signal_step_2_trends,
                )

                # After leakage correction.
                harmonic_load_signal_step_3_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_load_signal_step_3,
                )
                ocean_mean_step_3 = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_load_signal_step_3_trends,
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
                    mask=ocean_mask,
                    harmonics=harmonic_geoid_trends,
                )

                # Radial displacement.
                harmonic_radial_displacement_trends = compute_harmonic_signal_trends(
                    signal_dates=signal_dates,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    frequencial_harmonic_signal=frequencial_harmonic_radial_displacement,
                )
                ocean_mean_radial_displacement_component = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_radial_displacement_trends,
                )

                # Computes residuals.
                harmonic_residual_trends = (
                    harmonic_load_signal_step_2_trends  # L
                    + scale_factor_component
                    * map_sampling(map=ocean_mask, n_max=load_signal_hyper_parameters.n_max, harmonic_domain=True)[
                        0
                    ]  # + D
                    - (harmonic_geoid_trends - harmonic_radial_displacement_trends)  # - (G - R)
                )
                ocean_mean_residuals = mean_on_mask(
                    mask=ocean_mask,
                    harmonics=harmonic_residual_trends,
                )

                # Saves.
                add_result_to_table(
                    table_name="harmonic_load_signal_trends",
                    result_caracteristics={
                        "ID": harmonic_load_signal_id,
                        "Love_numbers_ID": Love_numbers_id,
                        "elastic_past_trend": past_trend,
                        "ocean_mean_step_1": ocean_mean_step_1,
                        "ocean_mean_step_2": ocean_mean_step_2,  # (L).
                        "ocean_mean_step_3": ocean_mean_step_3,
                        "scale_factor_component": scale_factor_component,  # (D).
                        "ocean_mean_geoid_component": ocean_mean_geoid_component,  # (G).
                        "ocean_mean_radial_displacement_component": ocean_mean_radial_displacement_component,  # (R).
                        "ocean_mean_radial_residuals": ocean_mean_residuals,  # (L + D - (G - R)).
                    }
                    | load_signal_hyper_parameters.__dict__,
                )

                # Eventually saves trends.
                for save_function, save_base_path, kind in zip(
                    [
                        save_harmonics,
                        save_base_format,
                    ],
                    [harmonic_load_signal_trends_path, base_format_load_signal_trends_path],
                    ["harmonics", "base_formats"],
                ):
                    # Anelastic load signal after frequencial filtering with Love number fractions.
                    if load_signal_hyper_parameters.save_parameters[kind].step_1:
                        save_function(
                            trends_array=harmonic_load_signal_step_1_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_1"),
                        )
                    # Anelastic load signal after degree one inversion.
                    if load_signal_hyper_parameters.save_parameters[kind].step_2:
                        save_function(
                            trends_array=harmonic_load_signal_step_2_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_2"),
                        )
                    # Anelastic load signal after leakage correction.
                    if load_signal_hyper_parameters.save_parameters[kind].step_3:
                        save_function(
                            trends_array=harmonic_load_signal_step_3_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("step_3"),
                        )
                    # Degree one inversion components.
                    if load_signal_hyper_parameters.save_parameters[kind].inversion_components:
                        save_function(
                            trends_array=harmonic_geoid_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("geoid"),
                        )
                        save_function(
                            trends_array=harmonic_radial_displacement_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("radial_displacement"),
                        )
                        save_function(
                            trends_array=harmonic_residual_trends,
                            id=harmonic_load_signal_id,
                            path=save_base_path.joinpath("residual"),
                        )
