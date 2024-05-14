from pathlib import Path
from shutil import rmtree
from typing import Optional

from numpy import ndarray, real, sqrt, zeros
from scipy.fft import ifft

from ..classes import LoadSignalHyperParameters, RunHyperParameters, results_path
from ..database import (
    get_run_folder_name,
    load_base_model,
    load_complex_array_from_binary,
    save_base_model,
)
from .data import get_ocean_mask, load_subpath, territorial_mean
from .harmonic import harmonic_name
from .temporal import get_trend_dates, signal_trend


def get_load_signal_harmonic_trends(
    do_elastic: bool,
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    anelasticity_description_id: str,
    src_diretory: Optional[Path],
    run_hyper_parameters: Optional[RunHyperParameters] = None,
) -> tuple[Path, str, str, dict[str, ndarray], dict[str, float], ndarray]:
    """
    Computes harmonic load signal trends. Saves territorial mean value.

    If 'do_elastic' is False, 'src_directory' has to be specified.
    """
    load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters
    # Gets already computed anelastic induced harmonic load signal.
    earth_models = (["elastic"] if do_elastic else []) + ["anelastic"]
    run_id = run_hyper_parameters.run_id()
    run_folder_name = get_run_folder_name(anelasticity_description_id=anelasticity_description_id, run_id=run_id)
    result_subpath = load_subpath(
        path=results_path.joinpath(run_folder_name), load_signal_hyper_parameters=load_signal_hyper_parameters
    )
    signal_dates = load_base_model(name="signal_dates", path=result_subpath)
    trend_indices, trend_dates = get_trend_dates(
        signal_dates=signal_dates, load_signal_hyper_parameters=load_signal_hyper_parameters
    )
    n_max = min(
        load_signal_hyper_parameters.n_max,
        int(sqrt(len([f for f in result_subpath.joinpath("anelastic_harmonic_frequencial_load_signal").iterdir()]))) - 1,
    )
    load_signal_harmonic_trends = {earth_model: zeros(shape=(2, n_max + 1, n_max + 1)) for earth_model in earth_models}
    for earth_model in earth_models:
        result_full_path = result_subpath.joinpath(earth_model + "_harmonic_frequencial_load_signal")
        # Loops on harmonics:
        for i_order_sign, coefficient in enumerate(["C", "S"]):
            for degree in range(i_order_sign, n_max + 1):
                for order in range(i_order_sign, degree + 1):
                    harmonic_frequencial_load_signal = load_complex_array_from_binary(
                        name=harmonic_name(coefficient=coefficient, degree=degree, order=order),
                        path=result_full_path,
                    )
                    # Computes harmonic trend.
                    temporal_anelastic_harmonic_signal = real(ifft(x=harmonic_frequencial_load_signal))
                    load_signal_harmonic_trends[earth_model][i_order_sign][degree][order] = signal_trend(
                        trend_dates=trend_dates,
                        signal=temporal_anelastic_harmonic_signal[trend_indices],
                    )[0]

    # Empty space.
    rmtree(path=result_subpath.joinpath("anelastic_harmonic_frequencial_load_signal"))
    rmtree(path=result_subpath.joinpath("anelastic_induced_frequencial_load_signal_per_degree"))

    # Preprocesses ocean mask.
    ocean_mask = get_ocean_mask(name=load_signal_hyper_parameters.ocean_mask, n_max=n_max)
    # Saves ocean rise mean trend.
    territorial_means = {
        earth_model: territorial_mean(harmonics=load_signal_harmonic_trends[earth_model], territorial_mask=ocean_mask)
        for earth_model in earth_models
    }
    if not do_elastic:
        territorial_means["elastic"] = load_base_model(name="barystatic_rise_mean_trend", path=src_diretory)["elastic"]
    save_base_model(
        obj=territorial_means,
        name="barystatic_rise_mean_trend",
        path=result_subpath,
    )

    return result_subpath, run_folder_name, run_id, load_signal_harmonic_trends, territorial_means, ocean_mask
