from ....utils import (
    OPTIONS,
    LoadSignalHyperParameters,
    compute_anelastic_induced_harmonic_load_per_description_per_options,
    load_load_signal_hyper_parameters,
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


def compute_anelastic_induced_harmonic_load_per_options(
    anelasticity_description_id: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[tuple[bool, bool, bool]] = OPTIONS,
) -> None:
    """
    Computes anelastic induced harmonic load signal for a given description and options.
    """
    compute_anelastic_induced_harmonic_load_per_description_per_options(
        anelasticity_description_ids=[anelasticity_description_id],
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        options=options,
    )


def compute_anelastic_induced_harmonic_load(
    anelasticity_description_id: str,
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
) -> None:
    """
    Computes anelastic induced harmonic load signal for a given description.
    """
    compute_anelastic_induced_harmonic_load_per_description(
        anelasticity_description_ids=[anelasticity_description_id],
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )
