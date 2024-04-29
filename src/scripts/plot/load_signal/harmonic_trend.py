from numpy import array, real, sqrt, zeros
from scipy.fft import ifft

from ....utils import (
    OPTIONS,
    LoadSignalHyperParameters,
    RunHyperParameters,
    figures_path,
    get_ocean_mask,
    get_run_folder_name,
    get_trend_dates,
    harmonic_name,
    load_base_model,
    load_load_signal_hyper_parameters,
    load_subpath,
    results_path,
    save_base_model,
    signal_trend,
    territorial_mean,
)
from ..utils import plot_harmonics_on_natural_projection


def plot_anelastic_induced_spatial_load_trend_per_description_per_options(
    anelasticity_description_ids: list[str],
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    options: list[RunHyperParameters] = OPTIONS,
    min_saturation: float = -50,
    max_saturation: float = 50,
    figsize: tuple[int, int] = (10, 10),
    ndigits: int = 3,
    exp_scale_factor: float = 1.0,
    saturation_factor: float = 1.0,
    continents: bool = False,
) -> None:
    """
    Generates figures showing the anelastic induced spatial load signal trend for given descriptions and options:
        - Gets already computed anelastic induced harmonic frequencial load signals.
        - Compute spatial trends.
        - Generates and saves figures in the specified subfolder.
    """
    # Loops on descriptions.
    for anelasticity_description_id in anelasticity_description_ids:
        # Prints status.
        print("Description: " + anelasticity_description_id + ":")
        # Loops on options.
        for run_hyper_parameters in options:
            load_signal_hyper_parameters.run_hyper_parameters = run_hyper_parameters

            # Gets already computed anelastic induced harmonic load signal.
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
                int(sqrt(len([f for f in result_subpath.joinpath("elastic_harmonic_frequencial_load_signal").iterdir()]))) - 1,
            )
            load_signal_harmonic_trends = {
                "elastic": zeros(shape=(2, n_max + 1, n_max + 1)),
                "anelastic": zeros(shape=(2, n_max + 1, n_max + 1)),
            }
            for earth_model in ["elastic", "anelastic"]:
                # Loops on harmonics:
                for i_order_sign, coefficient in enumerate(["C", "S"]):
                    for degree in range(i_order_sign, n_max + 1):
                        for order in range(i_order_sign, degree + 1):
                            harmonic_frequencial_load_signal = load_base_model(
                                name=harmonic_name(coefficient=coefficient, degree=degree, order=order),
                                path=result_subpath.joinpath(earth_model + "_harmonic_frequencial_load_signal"),
                            )
                            # Computes harmonic trend.
                            temporal_anelastic_harmonic_signal = real(
                                ifft(
                                    x=array(object=harmonic_frequencial_load_signal["real"], dtype=float)
                                    + 1.0j * array(object=harmonic_frequencial_load_signal["imag"], dtype=float)
                                )
                            )
                            load_signal_harmonic_trends[earth_model][i_order_sign][degree][order] = signal_trend(
                                trend_dates=trend_dates,
                                signal=temporal_anelastic_harmonic_signal[trend_indices],
                            )[0]

            # Preprocesses ocean mask.
            ocean_mask = [[1.0]] if continents else get_ocean_mask(name=load_signal_hyper_parameters.ocean_mask, n_max=n_max)
            # Saves ocean rise mean trend.
            territorial_means = {
                earth_model: territorial_mean(harmonics=load_signal_harmonic_trends[earth_model], territorial_mask=ocean_mask)
                for earth_model in ["elastic", "anelastic"]
            }
            save_base_model(
                obj=territorial_means,
                name="ocean_rise_mean_trend",
                path=result_subpath,
            )

            # Saves the figures.
            figure_subpath = load_subpath(
                path=figures_path.joinpath(run_folder_name), load_signal_hyper_parameters=load_signal_hyper_parameters
            )
            figure_subpath.mkdir(parents=True, exist_ok=True)

            # Input elastic spatial load signal trend.
            plot_harmonics_on_natural_projection(
                harmonics=load_signal_harmonic_trends["elastic"],
                figure_subpath=figure_subpath,
                name=load_signal_hyper_parameters.weights_map + "_load_signal_trend",
                title=load_signal_hyper_parameters.weights_map + " load signal trend",
                label="(mm/yr): ocean mean = " + str(round(number=territorial_means["elastic"], ndigits=ndigits)),
                ocean_mask=ocean_mask,
                min_saturation=min_saturation,
                max_saturation=max_saturation,
                logscale=False,
                figsize=figsize,
                exp_scale_factor=1.0,
                saturation_factor=1.3,
                continents=continents,
            )

            # Output anelastic spatial load signal trend.
            plot_harmonics_on_natural_projection(
                harmonics=load_signal_harmonic_trends["anelastic"],
                figure_subpath=figure_subpath,
                name=load_signal_hyper_parameters.weights_map
                + "_anelastic_induced_load_signal_trend_since_"
                + str(load_signal_hyper_parameters.first_year_for_trend),
                title=load_signal_hyper_parameters.weights_map
                + " anelastic induced load signal trend since "
                + str(load_signal_hyper_parameters.first_year_for_trend),
                label="(mm/yr): ocean mean = " + str(round(number=territorial_means["anelastic"], ndigits=ndigits)),
                ocean_mask=ocean_mask,
                min_saturation=min_saturation,
                max_saturation=max_saturation,
                logscale=True,
                figsize=figsize,
                exp_scale_factor=exp_scale_factor,
                saturation_factor=saturation_factor,
                continents=continents,
            )

            # Differences between elastic and anelastic spatial load signal trend.
            plot_harmonics_on_natural_projection(
                harmonics=load_signal_harmonic_trends["anelastic"] - load_signal_harmonic_trends["elastic"],
                figure_subpath=figure_subpath,
                name=load_signal_hyper_parameters.weights_map
                + "_anelastic_induced_load_signal_trend_difference_with_elastic_since_"
                + str(load_signal_hyper_parameters.first_year_for_trend),
                title=load_signal_hyper_parameters.weights_map
                + " anelastic induced load signal trend difference with elastic since "
                + str(load_signal_hyper_parameters.first_year_for_trend),
                label="(mm/yr): ocean mean = "
                + str(round(number=territorial_means["anelastic"] - territorial_means["elastic"], ndigits=ndigits)),
                ocean_mask=ocean_mask,
                min_saturation=min_saturation,
                max_saturation=max_saturation,
                logscale=True,
                figsize=figsize,
                exp_scale_factor=exp_scale_factor,
                saturation_factor=saturation_factor,
                continents=continents,
            )

            # Load bar.
            print("----Run: " + run_id + ": Done.")
