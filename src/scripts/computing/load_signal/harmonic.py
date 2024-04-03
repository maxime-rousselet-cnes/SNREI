from ....utils import (
    OPTIONS,
    LoadSignalHyperParameters,
    RunHyperParameters,
    anelastic_harmonic_induced_load_signal,
    build_elastic_load_signal,
    load_load_signal_hyper_parameters,
)


def compute_anelastic_induced_harmonic_load_per_description_per_options(
    real_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
) -> None:
    """
    Computes anelastic induced harmonic load signal for given descriptions and options:
        - Gets already computed Love numbers and load signal from data.
        - Computes and saves anelastic induced load signal per degree.
        - Computes and saves anelastic induced harmonic load signal.
    """
    # Builds load signal.
    (
        dates,
        frequencies,
        (frequencial_elastic_load_signal, elastic_load_signal_trend, harmonic_weights),
    ) = build_elastic_load_signal(signal_hyper_parameters=load_signal_hyper_parameters, get_harmonic_weights=True)

    # Loops on descriptions.
    for real_description_id in real_description_ids:
        # Loops on options.
        for run_hyper_parameters in options:
            load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters
            # Computes anelastic induced harmonic load signal.
            anelastic_harmonic_induced_load_signal(
                harmonic_weights=harmonic_weights,
                real_description_id=real_description_id,
                signal_hyper_parameters=load_signal_hyper_parameters,
                dates=dates,
                frequencies=frequencies,
                frequencial_elastic_normalized_load_signal=frequencial_elastic_load_signal,
                elastic_load_signal_trend=elastic_load_signal_trend,
            )


def compute_anelastic_induced_harmonic_load_per_description(
    real_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
) -> None:
    """
    Computes anelastic induced harmonic load signal for given descriptions.
    """
    compute_anelastic_induced_harmonic_load_per_description_per_options(
        real_description_ids=real_description_ids,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        options=[load_signal_hyper_parameters.run_hyper_parameters],
    )


def compute_anelastic_induced_harmonic_load_per_options(
    real_description_id: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[tuple[bool, bool, bool]] = OPTIONS,
) -> None:
    """
    Computes anelastic induced harmonic load signal for a given description and options.
    """
    compute_anelastic_induced_harmonic_load_per_description_per_options(
        real_description_ids=[real_description_id], load_signal_hyper_parameters=load_signal_hyper_parameters, options=options
    )


def compute_anelastic_induced_harmonic_load(
    real_description_id: str,
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
) -> None:
    """
    Computes anelastic induced harmonic load signal for a given description.
    """
    compute_anelastic_induced_harmonic_load_per_description(
        real_description_ids=[real_description_id],
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )
