from re import compile

from numpy import array, ndarray
from scipy import interpolate

from ..classes import EARTH_RADIUS, OMEGA, BoundaryCondition, Direction, LoadSignalHyperParameters, Result, pole_data_path
from ..load_signal import build_elastic_load_signal_history


def polar_motion_correction(
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    Love_numbers: Result,
    g: float,  # Earth surface mean gravitational acceleration.
    target_frequencies: ndarray[float],
):
    """"""
    load_signal_hyper_parameters.LIA_amplitude_effect = 0.0

    file = open(pole_data_path.joinpath(load_signal_hyper_parameters.pole_data + ".txt"), "r")
    lines = file.readlines()
    dates, signal = [], []
    for line in lines[1:]:
        p = compile(r"\d+\.\d+")  # Compiles a pattern to capture float values.
        items = [float(i) for i in p.findall(line)]
        dates += [float(items[0])]
        signal += [float(items[1]) - 1.0j * float(items[3])]  # m_1 - i m_2 (Unitless).

    _, frequencies, pole_complex_frequencial_signal, _ = build_elastic_load_signal_history(
        initial_signal_dates=array(object=dates, dtype=float),
        initial_load_signal=array(object=signal, dtype=complex),
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )

    PHI_CONSTANT = -EARTH_RADIUS * OMEGA**2 / (2.0 * g)
    Phi_SE_PT_complex: ndarray[complex]
    # Gets element in position 1 for degree 2.
    Phi_SE_PT_complex = (
        PHI_CONSTANT
        * (Love_numbers.values[Direction.potential][BoundaryCondition.potential][1] - 1.0)
        * interpolate.interp1d(
            x=frequencies,
            y=pole_complex_frequencial_signal,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )(x=target_frequencies)
    )

    # C_PT_SE_2_1, S_PT_SE_2_1.
    return Phi_SE_PT_complex.real, -Phi_SE_PT_complex.imag
