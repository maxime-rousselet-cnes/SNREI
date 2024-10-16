from typing import Any, Callable

from numpy import array, errstate, inf, ndarray, pi, shape, where, zeros
from scipy import integrate, interpolate
from scipy.integrate import OdeSolution

from ...rheological_formulas import (b_computing, build_cutting_omegas,
                                     delta_mu_computing,
                                     f_attenuation_computing,
                                     m_prime_computing, mu_computing)
from ...y_system import (fluid_system, fluid_to_solid, load_surface_solution,
                         potential_surface_solution, shear_surface_solution,
                         solid_homogeneous_system, solid_system,
                         solid_to_fluid)
from ..constants import INITIAL_Y_VECTOR
from ..description_layer import DescriptionLayer
from ..hyper_parameters import YSystemHyperParameters
from .anelasticity_description import AnelasticityDescription
from .description import Description


class Integration(Description):
    """
    "Applies" an anelasticity description to some frequency value.
    Describes the integration constants and all complex description layers at a given frequency.
    Handles elastic case for frequency = inf.
    Description layers variables include mu and lambda real and imaginary parts.
    """

    # Attributes from the anelasticity description.
    piG: float
    below_ICB_layers: int
    below_CMB_layers: int
    x_CMB: float

    # Proper attributes.
    frequency: float  # (Unitless frequency).
    omega: float  # (Unitless pulsation).
    omega_j: complex  # (Complex unitless pulsation).

    def __init__(
        self,
        # Proper field parameters.
        anelasticity_description: AnelasticityDescription,
        log_frequency: float,  # Base 10 logarithm of the unitless frequency.
        use_long_term_anelasticity: bool,
        use_short_term_anelasticity: bool,
        use_bounded_attenuation_functions: bool,
    ) -> None:
        """
        Creates an Integration instance.
        """

        super().__init__(
            radius_unit=anelasticity_description.radius_unit,
            real_crust=anelasticity_description.real_crust,
            spline_number=anelasticity_description.spline_number,
        )

        # Updates proper attributes.
        self.frequency = inf if log_frequency == inf else 10.0**log_frequency
        self.omega = inf if self.frequency == inf else 2 * pi * self.frequency
        self.omega_j = inf if self.omega == inf else self.omega * 1.0j

        # Updates attributes from the anelasticity description.
        self.piG = anelasticity_description.piG
        self.below_ICB_layers = anelasticity_description.below_ICB_layers
        self.below_CMB_layers = anelasticity_description.below_CMB_layers
        self.x_CMB = anelasticity_description.x_CMB

        # Initializes the needed description layers.
        for i_layer, (variables, layer) in enumerate(
            zip(
                anelasticity_description.variable_values_per_layer,
                anelasticity_description.description_layers,
            )
        ):

            # First gets the needed real variable splines.
            description_layer = DescriptionLayer(
                name=layer.name,
                x_inf=layer.x_inf,
                x_sup=layer.x_sup,
                splines={
                    variable_name: layer.splines[variable_name]
                    for variable_name in [
                        "g_0",
                        "rho_0",
                        "mu_0",
                        "lambda_0",
                        "Vs",
                        "Vp",
                    ]
                },
            )

            # Then gets lambda and mu splines.
            description_layer.splines.update(
                {  # Just copies lambda_0 and mu_0.
                    "mu_real": description_layer.splines["mu_0"],
                    "lambda_real": description_layer.splines["lambda_0"],
                    "mu_imag": (0.0, 0.0, 0),
                    "lambda_imag": (0.0, 0.0, 0),
                }
            )

            # Computes complex mu and lambda.
            if i_layer >= self.below_CMB_layers:

                # Default.
                variables["lambda"] = array(object=variables["lambda_0"], dtype=complex)
                variables["mu"] = array(object=variables["mu_0"], dtype=complex)

                # Attenuation.
                if use_short_term_anelasticity:

                    # Updates with attenuation functions f_r and f_i.
                    f = f_attenuation_computing(
                        omega_m_tab=variables["omega_m"],
                        tau_M_tab=variables["tau_M"],
                        alpha_tab=variables["alpha"],
                        omega=self.omega,
                        frequency=self.frequency,
                        frequency_unit=anelasticity_description.frequency_unit,
                        use_bounded_attenuation_functions=use_bounded_attenuation_functions,
                    )
                    description_layer.splines.update(
                        {
                            "f_r": interpolate.splrep(x=variables["x"], y=f.real),
                            "f_i": interpolate.splrep(x=variables["x"], y=f.imag),
                        }
                    )

                    # Adds delta mu, computed using f_r and f_i.
                    delta_mu = delta_mu_computing(
                        mu_0=variables["mu_0"],
                        Q_mu=variables["Q_mu"],
                        f=f,
                    )
                    variables["mu"] = variables["mu_0"] + delta_mu

                # Long-term anelasticity.
                if use_long_term_anelasticity:
                    # Complex cut frequency variables.
                    variables.update(build_cutting_omegas(variables=variables))
                    # Frequency filtering functions.
                    m_prime = m_prime_computing(omega_cut_m=variables["omega_cut_m"], omega_j=self.omega_j)
                    b = b_computing(
                        omega_cut_m=variables["omega_cut_m"],
                        omega_cut_k=variables["omega_cut_k"],
                        omega_cut_b=variables["omega_cut_b"],
                        omega_j=self.omega_j,
                    )
                    variables["mu"] = mu_computing(mu_complex=variables["mu"], m_prime=m_prime, b=b)

                variables["lambda"] = variables["lambda_0"] - 2.0 / 3.0 * (variables["mu"] - variables["mu_0"])

                # Updates.
                description_layer.splines.update(
                    {
                        "lambda_real": interpolate.splrep(x=variables["x"], y=variables["lambda"].real),
                        "lambda_imag": interpolate.splrep(x=variables["x"], y=variables["lambda"].imag),
                        "mu_real": interpolate.splrep(x=variables["x"], y=variables["mu"].real),
                        "mu_imag": interpolate.splrep(x=variables["x"], y=variables["mu"].imag),
                    },
                )

            # Updates.
            self.description_layers += [description_layer]

    def integration(
        self,
        Y_i: ndarray,
        integration_start: float,
        integration_stop: float,
        hyper_parameters: YSystemHyperParameters,
        n: int,
        n_layer: int,
        system: Callable[[Any], ndarray],
    ) -> tuple[ndarray, ndarray]:
        """
        Proceeds to the numerical integration of the wanted system along the planet's unitless radius.
        The 'system' input may corresponds to 'fluid_system' or 'solid_system'. It should always be a callable. Its first inputs
        are x and Y vector and its output is the dY/dx vector.
        """
        with errstate(divide="ignore", invalid="ignore"):
            solver: OdeSolution = integrate.solve_ivp(
                fun=system,
                t_span=(integration_start, integration_stop),
                y0=Y_i,
                method=hyper_parameters.method,
                t_eval=hyper_parameters.t_eval,
                args=(
                    (
                        n,
                        self.description_layers[n_layer],
                        self.piG,
                    )
                    if system == fluid_system
                    else (
                        n,
                        self.description_layers[n_layer],
                        self.piG,
                        self.omega,
                        hyper_parameters.dynamic_term,
                        hyper_parameters.inhomogeneity_gradients,
                    )
                ),
                rtol=hyper_parameters.rtol,
                atol=hyper_parameters.atol,
            )

        # TODO: catch exception.
        if solver.success == True:
            return solver.y, solver.t  # x corresponds to last dimension.
        else:
            print(":: RUNTINE ERROR ::")

    def y_system_integration(self, n: int, hyper_parameters: YSystemHyperParameters) -> ndarray[complex]:
        """
        Integrates the unitless gravito-elastic system from the Geocenter to the surface, at given n, omega and rheology.
        """

        # I - Integrate from Geocenter to CMB.
        if n <= hyper_parameters.n_max_for_sub_CMB_integration:
            # Integrate from Geocenter to CMB for low degrees.
            integration_start = hyper_parameters.minimal_radius / self.radius_unit

            Y = (
                INITIAL_Y_VECTOR
                if not hyper_parameters.homogeneous_solution
                else solid_homogeneous_system(
                    x=integration_start,
                    n=n,
                    layer=self.description_layers[0],
                    piG=self.piG,
                )
            )
            Y1 = Y[0, :].flatten()
            Y2 = Y[1, :].flatten()
            Y3 = Y[2, :].flatten()

            # Integrates in the Inner-Core.
            for n_layer in range(self.below_ICB_layers):
                integration_stop = self.description_layers[n_layer].x_sup
                Y1, _ = self.integration(
                    Y_i=Y1,
                    integration_start=integration_start,
                    integration_stop=integration_stop,
                    hyper_parameters=hyper_parameters,
                    n=n,
                    n_layer=n_layer,
                    system=solid_system,
                )
                Y2, _ = self.integration(
                    Y_i=Y2,
                    integration_start=integration_start,
                    integration_stop=integration_stop,
                    hyper_parameters=hyper_parameters,
                    n=n,
                    n_layer=n_layer,
                    system=solid_system,
                )
                Y3, _ = self.integration(
                    Y_i=Y3,
                    integration_start=integration_start,
                    integration_stop=integration_stop,
                    hyper_parameters=hyper_parameters,
                    n=n,
                    n_layer=n_layer,
                    system=solid_system,
                )
                Y1, Y2, Y3 = Y1[:, -1], Y2[:, -1], Y3[:, -1]
                integration_start = integration_stop

            # ICB Boundary conditions.
            Y = solid_to_fluid(
                Y1=Y1.real,
                Y2=Y2.real,
                Y3=Y3.real,
                x=integration_stop,
                first_fluid_layer=self.description_layers[self.below_ICB_layers],
                piG=self.piG,
            )

            # Integrates in the Outer-Core.
            for n_layer in range(self.below_ICB_layers, self.below_CMB_layers):
                integration_stop = self.description_layers[n_layer].x_sup
                Y, _ = self.integration(
                    Y_i=Y,
                    integration_start=integration_start,
                    integration_stop=integration_stop,
                    hyper_parameters=hyper_parameters,
                    n=n,
                    n_layer=n_layer,
                    system=fluid_system,
                )
                Y = Y[:, -1]
                integration_start = integration_stop

            # CMB Boundary conditions.
            Y1, Y2, Y3 = fluid_to_solid(
                Yf1=Y,
                x=integration_stop,
                last_fluid_layer=self.description_layers[self.below_CMB_layers - 1],
                piG=self.piG,
            )
            n_start_layer = self.below_CMB_layers

        else:
            # Integrate from Geocenter to CMB with high degrees approximation.
            n_start_layer: int = (
                where((array([layer.x_inf for layer in self.description_layers]) ** n) > (self.x_CMB**n))[0][0] + 1
            )

            Y = (
                INITIAL_Y_VECTOR
                if not hyper_parameters.homogeneous_solution
                else solid_homogeneous_system(
                    x=self.description_layers[n_start_layer].x_inf,
                    n=n,
                    layer=self.description_layers[n_start_layer],
                    piG=self.piG,
                )
            )
            Y1 = Y[0, :].flatten()
            Y2 = Y[1, :].flatten()
            Y3 = Y[2, :].flatten()

        # Integrates from the CMB to the surface.
        for n_layer in range(n_start_layer, len(self.description_layers)):
            integration_start = self.description_layers[n_layer].x_inf
            integration_stop = self.description_layers[n_layer].x_sup

            Y1, _ = self.integration(
                Y_i=Y1,
                integration_start=integration_start,
                integration_stop=integration_stop,
                hyper_parameters=hyper_parameters,
                n=n,
                n_layer=n_layer,
                system=solid_system,
            )
            Y2, _ = self.integration(
                Y_i=Y2,
                integration_start=integration_start,
                integration_stop=integration_stop,
                hyper_parameters=hyper_parameters,
                n=n,
                n_layer=n_layer,
                system=solid_system,
            )
            Y3, _ = self.integration(
                Y_i=Y3,
                integration_start=integration_start,
                integration_stop=integration_stop,
                hyper_parameters=hyper_parameters,
                n=n,
                n_layer=n_layer,
                system=solid_system,
            )
            Y1, Y2, Y3 = Y1[:, -1], Y2[:, -1], Y3[:, -1]

        # Surface Love numbers deriving.
        g_0_surface = self.description_layers[-1].evaluate(x=1.0, variable="g_0")
        h_load, l_load, k_load = load_surface_solution(
            n=n,
            Y1s=Y1,
            Y2s=Y2,
            Y3s=Y3,
            g_0_surface=g_0_surface,
            piG=self.piG,
        )
        h_shr, l_shr, k_shr = shear_surface_solution(
            n=n,
            Y1s=Y1,
            Y2s=Y2,
            Y3s=Y3,
            g_0_surface=g_0_surface,
            piG=self.piG,
        )

        h_pot, l_pot, k_pot = potential_surface_solution(
            n=n,
            Y1s=Y1,
            Y2s=Y2,
            Y3s=Y3,
            g_0_surface=g_0_surface,
        )

        return array(object=[h_load, l_load, k_load, h_shr, l_shr, k_shr, h_pot, l_pot, k_pot])
