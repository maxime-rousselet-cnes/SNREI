from pathlib import Path
from typing import Optional

from numpy import arange, ndarray, transpose
from scipy import interpolate
from tqdm import tqdm

from ..classes import LoadSignalHyperParameters
from ..database import save_base_model, symlink
from .temporal import anelastic_induced_load_signal_per_degree


def harmonic_name(coefficient: str, degree: int, order: int) -> str:
    """
    Builds a conventional name for a harmonic given its 3 indices.
    """
    return "_".join((coefficient, str(degree), str(order)))


def anelastic_harmonic_induced_load_signal(
    anelasticity_description_id: str,
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    signal_dates: ndarray[float],
    frequencies: ndarray[float],  # (y^-1).
    load_signal_informations: Path | tuple[ndarray[complex], float, Optional[ndarray[float]]],
    do_elastic: bool = True,
    src_directory: Optional[Path] = None,
) -> Path:
    """
    Computes the anelastic induced harmonic load and saves it in (.JSON) files.
    If 'do_elastic' is False, 'src_directory' has to be specified.
    """
    if load_signal_hyper_parameters.load_signal == "ocean_load":
        # Unpack load signal informations.
        frequencial_elastic_normalized_load_signal, elastic_load_signal_trend, harmonic_weights = load_signal_informations
        # Gets Love numbers, computes anelastic induced load signal and saves.
        path: Path
        degrees: ndarray[int]
        frequencial_load_signal_per_degree: ndarray[complex]
        path, degrees, frequencial_load_signal_per_degree = anelastic_induced_load_signal_per_degree(
            anelasticity_description_id=anelasticity_description_id,
            load_signal_hyper_parameters=load_signal_hyper_parameters,
            signal_dates=signal_dates,  # (y).
            frequencies=frequencies,  # (y^-1).
            frequencial_elastic_normalized_load_signal=frequencial_elastic_normalized_load_signal,
            elastic_load_signal_trend=elastic_load_signal_trend,
        )

        # Preprocesses.
        all_degrees = arange(load_signal_hyper_parameters.n_max + 1)
        anelastic_subpath = path.joinpath("anelastic_harmonic_frequencial_load_signal")
        anelastic_subpath.mkdir(parents=True, exist_ok=True)
        elastic_subpath = path.joinpath("elastic_harmonic_frequencial_load_signal")
        elastic_subpath.mkdir(parents=True, exist_ok=True)

        # Interpolates on degrees, for each frequency.
        frequencial_load_signals = interpolate_on_degrees(
            load_signal_per_degree=frequencial_load_signal_per_degree.real, degrees=degrees, new_degrees=all_degrees
        ) + 1.0j * interpolate_on_degrees(
            load_signal_per_degree=frequencial_load_signal_per_degree.imag, degrees=degrees, new_degrees=all_degrees
        )
        # TODO.
        frequencial_load_signals[0] = frequencial_elastic_normalized_load_signal

        # Loops on harmonics:
        for coefficient, (i_order_sign, weights_per_degree) in zip(["C", "S"], enumerate(harmonic_weights)):
            for degree, weights_per_order in tqdm(
                total=len(weights_per_degree) - i_order_sign,
                desc="----" + ("C" if i_order_sign == 0 else "S"),
                iterable=zip(range(i_order_sign, len(weights_per_degree)), weights_per_degree[i_order_sign:]),
            ):  # Because S_00 does not exist.
                for order, harmonic_weight in zip(
                    range(i_order_sign, degree + 1), weights_per_order[i_order_sign : degree + 1]
                ):  # Because S_n0 does not exist.
                    name = harmonic_name(coefficient=coefficient, degree=degree, order=order)
                    complex_anelastic_result: ndarray[complex] = frequencial_load_signals[degree] * harmonic_weight
                    # Saves results in (.JSON) files.
                    save_base_model(
                        obj={"real": complex_anelastic_result.real, "imag": complex_anelastic_result.imag},
                        name=name,
                        path=anelastic_subpath,
                    )
                    if do_elastic:
                        complex_elastic_result: ndarray[complex] = frequencial_elastic_normalized_load_signal * harmonic_weight
                        save_base_model(
                            obj={"real": complex_elastic_result.real, "imag": complex_elastic_result.imag},
                            name=name,
                            path=elastic_subpath,
                        )
                    else:
                        symlink(
                            src=src_directory.joinpath("elastic_harmonic_frequencial_load_signal").joinpath(name + ".json"),
                            dst=elastic_subpath.joinpath(name + ".json"),
                        )

        return anelastic_subpath.parent
    else:  # TODO: manage GRACE's full data.
        pass


def interpolate_on_degrees(
    load_signal_per_degree: ndarray[complex], degrees: ndarray[int], new_degrees: ndarray[int]
) -> ndarray[complex]:
    """
    Interpolate a 2-D (degrees, frequencies/time) signal on its first dimension (degrees).
    """
    return transpose(
        a=[
            interpolate.splev(x=new_degrees, tck=interpolate.splrep(x=degrees, y=load_signal_for_time, k=3), ext=0.0)
            for load_signal_for_time in transpose(a=load_signal_per_degree)
        ]
    )
