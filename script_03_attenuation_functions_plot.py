import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import linspace, log, ndarray, pi

from utils import Earth_radius, RealDescription, real_descriptions_path

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, required=True, help="wanted ID for the real description to load")
parser.add_argument("--figure_path_string", type=str, required=True, help="wanted path to save figure")

args = parser.parse_args()


def plot_attenuation_functions(real_description_id: str, figure_path_string: str):
    """
    Generates a figure of attenuation functions f_r and f_i.
    """
    real_description = RealDescription(
        id=real_description_id,
        below_ICB_layers=1,
        below_CMB_layers=2,
        splines_degree=1,
        radius_unit=Earth_radius,
        real_crust=False,
        n_splines_base=10,
        profile_precision=10000,
        radius=Earth_radius,
    )
    real_description.load(path=real_descriptions_path)
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)

    # Plots f_r and f_i.
    omega_0 = 1.0 / real_description.frequency_unit
    omega_m = 3.09e-4 / real_description.frequency_unit
    alpha = 0.25
    T_tab = 10 ** linspace(0, 10, 100)  # (s).
    f_tab = 1.0 / T_tab / real_description.frequency_unit
    high_frequency_domain = f_tab >= omega_m
    f_r_f_i: ndarray[complex] = ((2.0 / pi) * log(f_tab / omega_0) + 1.0j) * high_frequency_domain + (
        (2.0 / pi) * (log(omega_m / omega_0) + (1 / alpha) * (1 - (omega_m / f_tab) ** alpha))
        + (omega_m / f_tab) ** alpha * 1.0j
    ) * (1 - high_frequency_domain)
    plt.plot(T_tab, f_r_f_i.real, label="$f_r$")
    plt.plot(T_tab, f_r_f_i.imag, label="$f_i$")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Period (s)")
    plt.savefig(figure_path.joinpath("attenuation_functions.png"))
    plt.show()


if __name__ == "__main__":
    plot_attenuation_functions(real_description_id=args.real_description_id, figure_path_string=args.figure_path_string)
