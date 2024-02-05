import matplotlib.pyplot as plt
from numpy import imag, linspace, real

from utils import (
    BoundaryCondition,
    Direction,
    Result,
    frequencies_to_periods,
    load_base_model,
    results_path,
)

use_attenuation = True
use_anelasticity = True
path = results_path.joinpath("f018-876f-a9e0-2f46")
sub_path = path.joinpath("runs").joinpath("anelasticity_" + str(use_anelasticity) + "__attenuation_" + str(use_attenuation))
anelastic_Love_numbers = Result()
anelastic_Love_numbers.load(name="anelastic_Love_numbers", path=sub_path)
omega_values = load_base_model(name="frequencies", path=sub_path)
T_values = frequencies_to_periods(frequencies=omega_values)
elastic_Love_numbers = Result()
elastic_Love_numbers.load(name="elastic_Love_numbers", path=path)
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

degrees_to_plot = [2, 3, 4, 5, 6]
degrees_indices = [degrees.index(degree) for degree in degrees_to_plot]
T = frequencies_to_periods(omega_values)


plt.figure()
min_values, max_values = [], []
for i_degree, degree in zip(degrees_indices, degrees):
    color = ((i_degree + 1.0) / 8.0, 0.0, 1.0)
    elastic_value = real(elastic_Love_numbers.values[Direction.radial][BoundaryCondition.load][i_degree][0])
    anelastic_values = real(anelastic_Love_numbers.values[Direction.radial][BoundaryCondition.load][i_degree])
    h_ratio = anelastic_values / elastic_value
    min_values += [min(h_ratio)]
    max_values += [max(h_ratio)]
    plt.plot(T, h_ratio, label="n = " + str(degree), color=color)
    plt.xscale("log")
    plt.legend()
plt.xlabel("T (y)")
plt.yticks(linspace(min(min_values), max(max_values), 10))
plt.ylabel("real part")
plt.title("$ h_n(T) / h_n^E$")
plt.show()

plt.figure()
min_values, max_values = [], []
for i_degree, degree in zip(degrees_indices, degrees):
    color = ((i_degree + 1.0) / 8.0, 0.0, 1.0)
    anelastic_values = imag(anelastic_Love_numbers.values[Direction.radial][BoundaryCondition.load][i_degree])
    min_values += [min(anelastic_values)]
    max_values += [max(anelastic_values)]
    plt.plot(T, anelastic_values, label="n = " + str(degree), color=color)
    plt.xscale("log")
    plt.legend()
plt.xlabel("T (y)")
plt.yticks(linspace(min(min_values), max(max_values), 10))
plt.ylabel("imaginary part")
plt.title("$ h_n(T)$")
plt.show()
