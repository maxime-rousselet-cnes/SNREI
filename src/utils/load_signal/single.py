from pathlib import Path
from typing import Optional

from ..classes import (
    OPTIONS,
    LoadSignalHyperParameters,
    RunHyperParameters,
    load_load_signal_hyper_parameters,
)
from .harmonic import anelastic_harmonic_induced_load_signal
from .temporal import build_elastic_load_signal


def compute_anelastic_induced_harmonic_load_per_description_per_options(
    anelasticity_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
    do_elastic: bool = True,
    src_directory: Optional[Path] = None,
) -> list[Path]:
    """
    Computes anelastic induced harmonic load signal for given descriptions and options:
        - Gets already computed Love numbers and load signal from data.
        - Computes and saves anelastic induced load signal per degree.
        - Computes and saves anelastic induced harmonic load signal.

    If 'do_elastic' is False, 'src_directory' has to be specified.
    """
    # Builds load signal.
    (
        signal_dates,
        frequencies,
        load_signal_informations,
    ) = build_elastic_load_signal(load_signal_hyper_parameters=load_signal_hyper_parameters, get_harmonic_weights=True)

    # Loops on descriptions.
    load_result_folders = []
    for anelasticity_description_id in anelasticity_description_ids:
        # Loops on options.
        for run_hyper_parameters in options:
            load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters
            # Computes anelastic induced harmonic load signal.
            load_result_folders += [
                anelastic_harmonic_induced_load_signal(
                    anelasticity_description_id=anelasticity_description_id,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    signal_dates=signal_dates,
                    frequencies=frequencies,
                    load_signal_informations=load_signal_informations,
                    do_elastic=do_elastic,
                    src_directory=src_directory,
                )
            ]

    return load_result_folders
