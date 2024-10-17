from snrei import generate_load_signal_components_figure, load_load_signal_hyper_parameters

load_signal_hyper_parameters = load_load_signal_hyper_parameters()

elastic_load_signal_id = "0"
anelastic_load_signal_id = "2"


generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("residual", None, 2e-4, 2e-4, 2e-5, None),
        ("residual", None, 2e-4, 2e-4, 2e-5, 2),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("residual", None, 2e-4, 2e-1, 2e-1, None),
        ("residual", "C_1_0 C_1_1 S_1_1", 1e-4, 1e-1, 1e-1, None),
        ("residual", "C_2_1 S_2_1", 1e-4, 1e-1, 1e-1, None),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("step_1", None, 50.0, 50.0, 5.0, None),
        ("step_2", None, 50.0, 50.0, 1.0, None),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("step_2", None, 50.0, 50.0, 5.0, None),
        ("step_3", None, 50.0, 50.0, 5.0, None),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("step_3", None, 50.0, 50.0, 5.0, None),
        ("step_3", "C_1_0 C_1_1 S_1_1", 5.0, 5.0, 1.0, None),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("step_4", None, 50.0, 50.0, 5.0, None),
        ("step_4", "C_1_0 C_1_1 S_1_1", 5.0, 5.0, 1.0, None),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("geoid", None, 5.0, 5.0, 0.3, None),
        ("radial_displacement", None, 5.0, 5.0, 1.0, None),
    ],
)

generate_load_signal_components_figure(
    load_signal_hyper_parameters=load_signal_hyper_parameters,
    elastic_load_signal_id=elastic_load_signal_id,
    anelastic_load_signal_id=anelastic_load_signal_id,
    rows=[
        ("step_4", None, 50.0, 50.0, 5.0, None),
        ("step_5", None, 50.0, 50.0, 5.0, None),
    ],
)
