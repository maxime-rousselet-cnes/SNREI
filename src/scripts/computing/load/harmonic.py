def anelastic_induced_harmonic_load_trend(
    real_description_id: str,
    figure_subpath_string: str,
    signal_hyper_parameters: SignalHyperParameters = load_base_model(
        name="signal_hyper_parameters", path=parameters_path, base_model_type=SignalHyperParameters
    ),
) -> None:
    """
    Computes anelastic-modified harmonic load signal for a given description and options.
    Gets already computed Love numbers, builds load signal from data, computes anelastic induced harmonic load signal and
    saves it.
    Saves the corresponding figures (spatial domain) in the specified subfolder.
    """
    # Builds frequential signal.
    dates, frequencies, (elastic_trend, _, frequencial_elastic_load_signal, harmonic_weights) = build_elastic_load_signal(
        signal_hyper_parameters=signal_hyper_parameters, get_harmonic_weights=True
    )

    # Computes anelastic induced harmonic load signal.
    (
        path,
        _,
        _,
        _,
        harmonic_trends,
        _,
    ) = anelastic_harmonic_induced_load_signal(
        harmonic_weights=harmonic_weights,
        real_description_id=real_description_id,
        signal_hyper_parameters=signal_hyper_parameters,
        dates=dates,
        frequencies=frequencies,
        frequencial_elastic_normalized_load_signal=frequencial_elastic_load_signal / elastic_trend,
    )

    # Saves the figures.
    figure_subpath = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id).joinpath(path.name)
    figure_subpath.mkdir(parents=True, exist_ok=True)

    # Results.
    for saturation in BOOLEANS:
        v_min = None if not saturation else -5.0
        v_max = None if not saturation else 5.0
        # Inputs.
        plot_harmonics_on_natural_projection(
            harmonics=harmonic_weights,
            title=signal_hyper_parameters.weights_map,
            figure_subpath=figure_subpath.parent.parent,
            n_max=signal_hyper_parameters.n_max,
            name=signal_hyper_parameters.weights_map + ("_saturated" if saturation else ""),
            ocean_mask_filename=signal_hyper_parameters.ocean_mask,
            v_min=v_min,
            v_max=v_max,
        )
        # Output.
        plot_harmonics_on_natural_projection(
            harmonics=harmonic_trends,
            title="anelastic induced loads : trends since " + str(signal_hyper_parameters.first_year_for_trend),
            figure_subpath=figure_subpath,
            n_max=signal_hyper_parameters.n_max,
            name=signal_hyper_parameters.weights_map
            + "_"
            + signal_hyper_parameters.signal
            + "_trend_anelastic"
            + ("_saturated" if saturation else ""),
            ocean_mask_filename=signal_hyper_parameters.ocean_mask,
            v_min=v_min,
            v_max=v_max,
        )
        # Difference.
        plot_harmonics_on_natural_projection(
            harmonics=harmonic_trends - harmonic_weights,
            title="anelastic induced loads differences with elastic : trends since "
            + str(signal_hyper_parameters.first_year_for_trend),
            figure_subpath=figure_subpath,
            n_max=signal_hyper_parameters.n_max,
            name=signal_hyper_parameters.weights_map
            + "_"
            + signal_hyper_parameters.signal
            + "_trend_diff_with_elastic"
            + ("_saturated" if saturation else ""),
            ocean_mask_filename=signal_hyper_parameters.ocean_mask,
            v_min=v_min,
            v_max=v_max,
        )
