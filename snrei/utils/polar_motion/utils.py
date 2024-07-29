from re import compile

from numpy import array

from ..classes import LoadSignalHyperParameters, pole_data_path
from ..load_signal import build_elastic_load_signal_history


def polar_motion_correction(load_signal_hyper_parameters: LoadSignalHyperParameters, Love_numbers):
    """"""
    load_signal_hyper_parameters.LIA_amplitude_effect = 0.0

    file = open(pole_data_path.joinpath(load_signal_hyper_parameters.pole_data + ".txt"), "r")
    lines = file.readlines()
    dates, signal = [], []
    for line in lines[1:]:
        p = compile(r"\d+\.\d+")  # Compile a pattern to capture float values
        items = [float(i) for i in p.findall(line)]
        dates += [float(items[0])]
        signal += [float(items[1]) - 1.0j * float(items[3])]

    times, frequencies, pole_complex_frequencial_signal, _ = build_elastic_load_signal_history(
        initial_signal_dates=array(object=dates, dtype=float),
        initial_load_signal=array(object=signal, dtype=complex),
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )
    # TODO. return 2 frequencial series. ^PT
    return 0.0, 0.0
