# Defines realistic bounds to help fitting.
PARAMETER_BOUNDS = {
    "secular_slope": {"lower_bound": -50.0, "upper_bound": 50.0},
    "additive_constant": {"lower_bound": -200.0, "upper_bound": 200.0},
    "sinusoidal_amplitude": {"lower_bound": 0.0, "upper_bound": 50.0},
    "sinusoidal_delay": {"lower_bound": 0.0, "upper_bound": 1.0},
    "earthquake_t_0": {"lower_bound": 2002.0, "upper_bound": 2023.0},
    "earthquake_tau": {"lower_bound": 0.05, "upper_bound": 3.0},
    "co_seismic_amplitude": {"lower_bound": -400.0, "upper_bound": 400.0},
    "post_seismic_amplitude": {"lower_bound": -50.0, "upper_bound": 50.0},
    "seismic_relaxation_time": {"lower_bound": 1.0, "upper_bound": 10.0},
}

# Defines areas where to remove earthquakes signals
EARTHQUAKE_CORNERS = {
    "Chili": {"upper_left": (-31.546211, 360 - 79.554043), "lower_right": (-38.712215, 360 - 66.682230)},
    "Sumatra": {"upper_left": (8.945213, 86.928182), "lower_right": (-5.161619, 99.144977)},
    "Tohoku": {"upper_left": (41.455418, 134.301228), "lower_right": (31.142693, 148.100055)},
}

SEISMIC_PARAMETERS_NUMBER = len(
    [parameter for parameter in PARAMETER_BOUNDS.keys() if ("seismic" in parameter) or ("earthquake" in parameter)]
)
PERIODIC_PARAMETERS_NUMBER = len([parameter for parameter in PARAMETER_BOUNDS.keys() if "sinusoidal" in parameter])

OTHER_PARAMETERS_NUMBER = len(PARAMETER_BOUNDS) - PERIODIC_PARAMETERS_NUMBER - SEISMIC_PARAMETERS_NUMBER
