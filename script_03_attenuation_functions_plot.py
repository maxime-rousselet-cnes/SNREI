import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import array, exp, linspace, log, ndarray, pi, round
from scipy import integrate

from utils import (
    SECONDS_PER_YEAR,
    Earth_radius,
    RealDescription,
    real_descriptions_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, required=True, help="wanted ID for the real description to load")
parser.add_argument("--figure_path_string", type=str, required=True, help="wanted path to save figure")
args = parser.parse_args()


def plot_attenuation_functions(
    real_description_id: str, figure_path_string: str, tau_M_values: list[float] = [1.0 / 12 / 1.0, 5.0, 20.0]
):
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
        profile_precision=10,
        radius=Earth_radius,
    )
    real_description.load(path=real_descriptions_path)
    figure_path = Path(figure_path_string)
    figure_path.mkdir(parents=True, exist_ok=True)

    # Gets values.
    omega_0 = 1.0 / real_description.frequency_unit
    omega_m = real_description.description_layers[2].evaluate(
        x=real_description.description_layers[2].x_inf, variable="omega_m"
    )
    tau_M_description = real_description.description_layers[2].evaluate(
        x=real_description.description_layers[2].x_inf, variable="tau_M"
    )
    alpha = real_description.description_layers[2].evaluate(x=real_description.description_layers[2].x_inf, variable="alpha")
    T_tab = 10 ** linspace(0, 10, 100)  # (s).
    f_tab = 1.0 / T_tab / real_description.frequency_unit
    high_frequency_domain: ndarray[bool] = f_tab >= omega_m
    tau_M_adim_values = [tau_M * SECONDS_PER_YEAR / real_description.period_unit for tau_M in tau_M_values]

    # Plots unbounded f.
    f_r_f_i: ndarray[complex] = ((2.0 / pi) * log(f_tab / omega_0) + 1.0j) * high_frequency_domain + (
        (2.0 / pi) * (log(omega_m / omega_0) + (1 / alpha) * (1 - (omega_m / f_tab) ** alpha))
        + (omega_m / f_tab) ** alpha * 1.0j
    ) * (1 - high_frequency_domain)
    plt.plot(T_tab, f_r_f_i.real, label="$f_r$")
    plt.plot(T_tab, f_r_f_i.imag, label="$f_i$")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Period (s)")
    plt.savefig(figure_path.joinpath(real_description_id + "_attenuation_functions.png"))
    plt.show()

    # Plots bounded f.
    _, plots = plt.subplots(2, 1, sharex=True, figsize=(12, 15))
    for tau_M, tau_M_years in zip(tau_M_adim_values + [tau_M_description], tau_M_values + ["model"]):
        tau = lambda tau_log: exp(tau_log)
        Y = lambda tau_log, alpha: (tau(tau_log=tau_log) ** alpha)
        denom = lambda tau_log, omega: (1.0 + (omega * tau(tau_log=tau_log) * 1.0j))
        integrand = lambda tau_log, omega, alpha: Y(tau_log=tau_log, alpha=alpha) / denom(tau_log=tau_log, omega=omega)
        f_r_f_i: ndarray[complex] = array(
            [
                -integrate.quad(
                    func=integrand, a=log(1.0 / omega_m), b=log(tau_M), args=(2.0 * pi * frequency, alpha), complex_func=True
                )[0]
                for frequency in f_tab
            ]
        )

        plots[0].plot(real_description.period_unit / f_tab, f_r_f_i.real)
        plots[1].plot(
            real_description.period_unit / f_tab,
            f_r_f_i.imag,
            label=(
                tau_M_years if isinstance(tau_M_years, str) else "$\\tau _M=$" + str(round(a=tau_M_years, decimals=4)) + " (y)"
            ),
        )
        plots[0].scatter(
            [tau_M * real_description.period_unit] * 15, linspace(start=min(f_r_f_i.real), stop=max(f_r_f_i.real), num=15), s=5
        )
        plots[1].scatter(
            [tau_M * real_description.period_unit] * 15, linspace(start=min(f_r_f_i.imag), stop=max(f_r_f_i.imag), num=15), s=5
        )

    f_r_f_i: ndarray[complex] = ((2.0 / pi) * log(f_tab / omega_0) + 1.0j) * high_frequency_domain + (
        (2.0 / pi) * (log(omega_m / omega_0) + (1 / alpha) * (1 - (omega_m / f_tab) ** alpha))
        + (omega_m / f_tab) ** alpha * 1.0j
    ) * (1 - high_frequency_domain)
    plots[0].plot(T_tab, f_r_f_i.real)
    plots[1].plot(T_tab, f_r_f_i.imag, label="unbounded")

    plots[0].set_ylabel("$f_r$")
    plots[0].set_xlim(1.0, 1e10)
    plots[0].set_ylim(-120.0, 0.0)
    plots[0].set_yticks(linspace(-120.0, 0.0, 25))
    plots[0].grid()

    plots[1].set_ylabel("$f_i$")
    plots[1].set_ylim(-5.0, 30.0)
    plots[1].set_yticks(linspace(-5.0, 30.0, 36))
    plots[1].grid()

    plt.xlabel("Period (s)")
    plt.xscale("log")
    plt.legend()
    plt.savefig(figure_path.joinpath(real_description_id + "_bounded_attenuation_functions.png"))
    plt.show()


if __name__ == "__main__":
    plot_attenuation_functions(
        real_description_id=args.real_description_id,
        figure_path_string=args.figure_path_string,
    )
