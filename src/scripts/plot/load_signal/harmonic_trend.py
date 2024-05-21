from ....utils import (
    OPTIONS,
    LoadSignalHyperParameters,
    RunHyperParameters,
    figures_path,
    get_load_signal_harmonic_trends,
    load_load_signal_hyper_parameters,
    load_subpath,
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
            _, run_folder_name, run_id, load_signal_harmonic_trends, territorial_means, ocean_mask = (
                get_load_signal_harmonic_trends(
                    do_elastic=True,
                    load_signal_hyper_parameters=load_signal_hyper_parameters,
                    run_hyper_parameters=run_hyper_parameters,
                    anelasticity_description_id=anelasticity_description_id,
                    remove=False,
                )
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
                ocean_mask=[[1.0]] if continents else ocean_mask,
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
                ocean_mask=[[1.0]] if continents else ocean_mask,
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
                ocean_mask=[[1.0]] if continents else ocean_mask,
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
