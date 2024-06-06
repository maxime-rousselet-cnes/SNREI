from ...utils import (
    OPTIONS,
    LoadSignalHyperParameters,
    RunHyperParameters,
    anelastic_induced_load_signal_per_degree,
    build_elastic_load_signal,
    compute_anelastic_induced_harmonic_load_per_description_per_options,
    load_load_signal_hyper_parameters,
)


def compute_anelastic_induced_load_per_degree_per_description_per_options(
    anelasticity_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
) -> None:
    """
    Computes anelastic induced load signal per degree for given descriptions and options:
        - Gets already computed Love numbers and load signal from data.
        - Computes and saves anelastic induced load signal per degree.
    """
    # Builds load signal.
    signal_dates, frequencies, (frequencial_elastic_normalized_load_signal, elastic_load_signal_trend, _) = (
        build_elastic_load_signal(signal_hyper_parameters=load_signal_hyper_parameters, get_harmonic_weights=False)
    )

    # Loops on descriptions.
    for anelasticity_description_id in anelasticity_description_ids:
        # Loops on options.
        for run_hyper_parameters in options:
            load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters
            # Computes anelastic induced load signal per degree.
            anelastic_induced_load_signal_per_degree(
                anelasticity_description_id=anelasticity_description_id,
                load_signal_hyper_parameters=load_signal_hyper_parameters,
                signal_dates=signal_dates,
                frequencies=frequencies,
                frequencial_elastic_normalized_load_signal=frequencial_elastic_normalized_load_signal,
                elastic_load_signal_trend=elastic_load_signal_trend,
            )


def compute_anelastic_induced_harmonic_load_per_description(
    anelasticity_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
) -> None:
    """
    Computes anelastic induced harmonic load signal for given descriptions.
    """
    compute_anelastic_induced_harmonic_load_per_description_per_options(
        anelasticity_description_ids=anelasticity_description_ids,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        options=[load_signal_hyper_parameters.run_hyper_parameters],
    )
