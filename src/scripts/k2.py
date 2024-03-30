from scipy import interpolate

from utils import (
    BoundaryCondition,
    Direction,
    Result,
    frequencies_to_periods,
    load_base_model,
    results_path,
)

path = results_path.joinpath("PREM_low-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio1.0-1.0")
sub_path = path.joinpath("runs").joinpath("__attenuation")
anelastic_Love_numbers = Result()
anelastic_Love_numbers.load(name="anelastic_Love_numbers", path=sub_path)
frequency_values = load_base_model(name="frequencies", path=sub_path)
T_values = frequencies_to_periods(frequencies=frequency_values)

anelastic_values = anelastic_Love_numbers.values[Direction.potential][BoundaryCondition.potential][1]

# Interpolates value for k_2(18.6 y tide)
f = interpolate.interp1d(x=T_values, y=anelastic_values, kind="linear")
f(18.6) / 2.0
