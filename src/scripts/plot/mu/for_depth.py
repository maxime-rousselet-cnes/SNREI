from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
from numpy import linspace, log10


def plot_mu_profiles_for_options(
    forced_real_description_id: Optional[str] = None,
    elasticity_model_name: Optional[str] = None,
    long_term_anelasticity_model_name: Optional[str] = None,
    short_term_anelasticity_model_names: Optional[str] = None,
    load_description: bool = False,
    figure_subpath_string: str = "mu/for_depth",
    period_values: list[float] = [10, 100, 1000],
):
    """
    Generates figures of mu_0/Q_0, and real and imaginary parts of mu for several period values.
    """
    # Initializes.
    Love_numbers_hyper_parameters = load_Love_numbers_hyper_parameters()
    path = figures_path.joinpath(figure_subpath_string).joinpath(real_description_id)
    booleans = [True, False]
    options_list = list(product(booleans, booleans, booleans))
    integrations: dict[float, dict[tuple[bool, bool, bool], Integration]] = {}
    frequencies = frequencies_to_periods(period_values)  # It is OK to converts years like this. Tested.
    path.mkdir(parents=True, exist_ok=True)
    real_description = real_description_from_parameters(
        Love_numbers_hyper_parameters=Love_numbers_hyper_parameters,
        real_description_id=real_description_id,
        load_description=load_description,
        save=False,
    )

    # Preprocesses.
    for frequency in frequencies:
        integrations[frequency] = {}
        for use_anelasticity, use_attenuation, bounded_attenuation_functions in options_list:
            if (not use_attenuation) and bounded_attenuation_functions:
                continue
            integration = Integration(
                real_description=real_description,
                log_frequency=log10(frequency / real_description.frequency_unit),
                use_anelasticity=use_anelasticity,
                use_attenuation=use_attenuation,
                bounded_attenuation_functions=bounded_attenuation_functions,
            )
            integrations[frequency][use_anelasticity, use_attenuation, bounded_attenuation_functions] = integration

    # Plots mu_0 / Q_0.
    plt.figure(figsize=(8, 10))
    plot = plt.subplot()
    # Iterates on layers.
    for k_layer in range(2, len(integrations[frequency][True, True, True].description_layers)):
        # Iterates on options.
        for use_anelasticity, use_attenuation, bounded_attenuation_functions in options_list:
            if (not use_attenuation) and bounded_attenuation_functions:
                continue
            base_label = (
                "elastic"
                if (not use_anelasticity and not use_attenuation)
                else ("with anelasticity " if use_anelasticity else "")
                + ("with " + ("bounded " if bounded_attenuation_functions else "") + "attenuation" if use_attenuation else "")
            )
            layer = integrations[frequency][
                use_anelasticity, use_attenuation, bounded_attenuation_functions
            ].description_layers[k_layer]
            variable_values = real_description.variable_values_per_layer[k_layer]
            x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
            plot.semilogx(
                variable_values["mu_0"] / variable_values["Qmu"] * real_description.elasticity_unit,
                (1.0 - x) * real_description.radius_unit / 1e3,
                color=(
                    1.0 if use_anelasticity else 0.0,
                    1.0 if bounded_attenuation_functions else 0.0,
                    1.0 if use_attenuation else 0.0,
                ),
                linestyle=(
                    ":"
                    if (not use_anelasticity and not use_attenuation)
                    else ("-" if use_anelasticity else "") + ("-" if use_attenuation else ".")
                ),
                label=base_label,
            )
        if k_layer == 2:
            plot.legend(loc="lower left")
    plot.set_xlabel("$\mu_{real} / Q_{mu}$ (Pa)")
    plot.set_ylabel("Depth (km)")
    plot.invert_yaxis()
    plt.savefig(path.joinpath("mu_on_Q.png"))
    plt.show()

    # Plots mu_real and mu_imag.
    for part in ["real", "imag"]:
        _, plots = plt.subplots(1, len(frequencies), figsize=(18.3, 9), sharex=True)
        # Iterates on frequencies.
        for frequency, period, plot in zip(frequencies, period_values, plots):
            # Iterates on layers.
            for k_layer in range(2, len(integrations[frequency][True, True, True].description_layers)):
                # Iterates on options.
                for use_anelasticity, use_attenuation, bounded_attenuation_functions in options_list:
                    if (not use_attenuation) and bounded_attenuation_functions:
                        continue
                    if use_attenuation and not bounded_attenuation_functions:
                        continue
                    base_label = (
                        "elastic"
                        if (not use_anelasticity and not use_attenuation)
                        else ("long-term anelasticity " if use_anelasticity else "")
                        + (
                            "transient regime"  # ("bounded " if bounded_attenuation_functions else "") + "attenuation"
                            if use_attenuation
                            else ""
                        )
                    )
                    layer = integrations[frequency][
                        use_anelasticity, use_attenuation, bounded_attenuation_functions
                    ].description_layers[k_layer]
                    x = linspace(start=layer.x_inf, stop=layer.x_sup, num=real_description.profile_precision)
                    plot.plot(
                        layer.evaluate(x=x, variable="mu_" + part) * real_description.elasticity_unit,
                        (1.0 - x) * real_description.radius_unit / 1e3,
                        color=(
                            1.0 if use_anelasticity else 0.0,
                            0.5 if bounded_attenuation_functions else 0.0,
                            1.0 if use_attenuation else 0.0,
                        ),
                        linestyle=(
                            ":"
                            if (not use_anelasticity and not use_attenuation)
                            else ("-" if use_anelasticity else "") + ("-" if use_attenuation else ".")
                        ),
                        label=base_label,
                    )
                if k_layer == 2 and frequency == frequencies[0]:
                    plot.legend(loc="lower left")
            plot.set_xlabel("$\mu_{" + part + "}$ (Pa)")
            # plot.set_xscale("log")
            plot.set_ylabel("Depth (km)")
            plot.invert_yaxis()
            plot.grid()
            plot.set_title("$T=" + str(period) + "$ (y)")
        plt.savefig(path.joinpath("mu_" + part + ".png"))
        plt.show()
