from snrei import generate_load_signal_components_figure, load_load_signal_hyper_parameters

load_signal_hyper_parameters = load_load_signal_hyper_parameters()


generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("residual", None, 20.0, 2.0),
        ("residual", "C_1_0 C_1_1 S_1_1", 5.0, 1.0),
        ("residual", "C_2_1 S_2_1", 5.0, 1.0),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("step_1", None, 50.0, 5.0),
        ("step_2", None, 50.0, 1.0),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("step_2", None, 50.0, 5.0),
        ("step_3", None, 50.0, 5.0),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("step_3", None, 50.0, 5.0),
        ("step_3", "C_1_0 C_1_1 S_1_1", 5.0, 1.0),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("step_4", None, 50.0, 5.0),
        ("step_4", "C_1_0 C_1_1 S_1_1", 5.0, 1.0),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("geoid", None, 5.0, 0.3),
        ("radial_displacement", None, 5.0, 1.0),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id="6",
    anelastic_load_signal_id="28",
    rows=[
        ("step_4", None, 50.0, 5.0),
        ("step_5", None, 50.0, 5.0),
    ],
)
