from typing import Any, Callable, Optional

from numpy import Inf, array, errstate, ndarray, pi, shape, where, zeros
from scipy import integrate, interpolate
from scipy.integrate import OdeSolution

from ...formulas import (
    b_computing,
    delta_mu_computing,
    f_attenuation_computing,
    lambda_computing,
    m_prime_computing,
    mu_computing,
)
from ...y_system import (
    fluid_system,
    fluid_to_solid,
    load_surface_solution,
    potential_surface_solution,
    shear_surface_solution,
    solid_homogeneous_system,
    solid_system,
    solid_to_fluid,
)
from ..description_layer import DescriptionLayer
from ..hyper_parameters import YSystemHyperParameters
from ..spline import Spline
from .description import Description
from .real_description import RealDescription


class Integration(Description):
    """
    "Applies" a real description to some frequency value.
    Describes the integration constants and all complex description layers at a given frequency.
    Handles elastic case for frequency = Inf.
    Description layers variables include mu and lambda real and imaginary parts.
    """

    # Attributes from the real description.
    piG: float
    length_ratio: float
    below_ICB_layers: int
    below_CMB_layers: int
    CMB_x: float

    # Proper attributes.
    frequency: float  # (Unitless frequency).
    omega: float  # (Unitless pulsation).
    omega_j: complex  # (Complex unitless pulsation).

    def __init__(
        self,
        # Proper field parameters.
        real_description: RealDescription,
        log_frequency: float,  # Base 10 logarithm of the unitless frequency.
        use_anelasticity: bool,
        use_attenuation: bool,
        bounded_attenuation_functions: bool,
        # Other parameters.
        id: Optional[str] = None,
    ) -> None:
        """
        Creates an Integration instance.
        """
        super().__init__(
            radius_unit=real_description.radius_unit,
            real_crust=real_description.real_crust,
            n_splines_base=real_description.n_splines_base,
            id=id,
        )

        # Updates proper attributes.
        self.frequency = Inf if log_frequency == Inf else 10.0**log_frequency
        self.omega = Inf if self.frequency == Inf else 2 * pi * self.frequency
        self.omega_j = Inf if self.omega == Inf else self.omega * 1.0j

        # Initializes the needed description layers.
        for i_layer, (variables, layer) in enumerate(
            zip(real_description.variable_values_per_layer, real_description.description_layers)
        ):
            # First gets the needed real variable splines.
            description_layer = DescriptionLayer(
                name=layer.name,
                x_inf=layer.x_inf,
                x_sup=layer.x_sup,
                splines={
                    variable_name: layer.splines[variable_name]
                    for variable_name in ["g_0", "rho_0", "mu_0", "lambda_0", "Vs", "Vp"]
                },
            )

            # Then gets lambda and mu splines.
            description_layer.splines.update(
                {  # Just copies lambda_0 and mu_0.
                    "mu_real": description_layer.splines["mu_0"],
                    "lambda_real": description_layer.splines["lambda_0"],
                    "mu_imag": Spline((0.0, 0.0, 0)),
                    "lambda_imag": Spline((0.0, 0.0, 0)),
                }
            )

            # Computes complex mu and lambda.
            if i_layer >= real_description.below_CMB_layers:
                if i_layer >= real_description.below_CMB_layers:

                    # Attenuation.
                    if use_attenuation:
                        # Updates with attenuation functions f_r and f_i.
                        f = f_attenuation_computing(
                            omega_m_tab=variables["omega_m"],
                            tau_M_tab=variables["tau_M"],
                            alpha_tab=variables["alpha"],
                            frequency=self.frequency,
                            frequency_unit=real_description.frequency_unit,
                            bounded_attenuation_functions=bounded_attenuation_functions,
                        )
                        description_layer.splines.update(
                            {
                                "f_r": interpolate.splrep(x=variables["x"], y=f.real),
                                "f_i": interpolate.splrep(x=variables["x"], y=f.imag),
                            }
                        )
                        # Adds delta mu, computed using f_r and f_i.
                        delta_mu = delta_mu_computing(
                            mu_0=variables["mu_1"],
                            Qmu=variables["Qmu"],
                            f=f,
                        )
                        complex_lambda = variables["lambda_0"] - 2.0 / 3.0 * delta_mu
                        complex_mu = variables["mu_0"] + delta_mu
                    else:
                        # No attenuation: mu = mu_0 and lambda = lambda_0.
                        complex_mu = array(variables["mu_0"], dtype=complex)
                        complex_lambda = array(variables["lambda_0"], dtype=complex)

                    # Anelasticity.
                    if use_anelasticity:
                        m_prime = m_prime_computing(omega_cut_m=variables["omega_cut_m"], omega_j=self.omega_j)
                        b = b_computing(
                            omega_cut_m=variables["omega_cut_m"],
                            omega_cut_k=variables["omega_cut_k"],
                            omega_cut_b=variables["omega_cut_b"],
                            omega_j=self.omega_j,
                        )
                        complex_lambda = lambda_computing(
                            mu_complex=complex_mu,
                            lambda_complex=complex_lambda,
                            m_prime=m_prime,
                            b=b,
                        )
                        complex_mu = mu_computing(mu_complex=complex_mu, m_prime=m_prime, b=b)

                    # Updates.
                    description_layer.splines.update(
                        {
                            "lambda_real": interpolate.splrep(x=variables["x"], y=complex_lambda.real),
                            "lambda_imag": interpolate.splrep(x=variables["x"], y=complex_lambda.imag),
                            "mu_real": interpolate.splrep(x=variables["x"], y=complex_mu.real),
                            "mu_imag": interpolate.splrep(x=variables["x"], y=complex_mu.imag),
                        },
                    )

            # Updates.
            self.description_layers += [description_layer]

        # Updates attributes from the real description.
        self.piG = real_description.piG
        self.length_ratio = real_description.length_ratio
        self.below_ICB_layers = real_description.below_ICB_layers
        self.below_CMB_layers = real_description.below_CMB_layers
        self.CMB_x = real_description.CMB_x

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
        are x and Y and its output is dY/dx.
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
                        hyper_parameters.first_order_cross_terms,
                    )
                ),
                rtol=hyper_parameters.rtol,
                atol=hyper_parameters.atol,
            )
        if solver.success == True:
            return solver.y, solver.t  # x corresponds to last dimension.
        else:
            print("")
            print(":: ERROR: method 'scipy.integrate.solve_ivp' failed.")
            print("")

    # TODO: Vectorize.
    def y_system_integration(self, n: int, hyper_parameters: YSystemHyperParameters) -> ndarray[complex]:
        """
        Integrates the unitless gravito-elastic system from the geocenter to the surface, at given n, omega and fixed rheology.
        """

        # Integrate from geocenter to CMB.
        if n <= hyper_parameters.n_max_for_sub_CMB_integration:
            # Integrate from geocenter to CMB for low degrees.
            integration_start = hyper_parameters.minimal_radius / self.radius_unit
            if hyper_parameters.homogeneous_solution:
                # Gets analytical homogeneous solution from r = 0 m to r = minimal_radius...
                Y = solid_homogeneous_system(
                    x=integration_start,
                    n=n,
                    layer=self.description_layers[0],
                    piG=self.piG,
                )
                Y1i = Y[0, :].flatten()
                Y2i = Y[1, :].flatten()
                Y3i = Y[2, :].flatten()
            else:
                # ...Or starts to integrate from r = minimal_radius.
                Y = array(
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=complex,
                )
                Y1i = Y[0, :].flatten()
                Y2i = Y[1, :].flatten()
                Y3i = Y[2, :].flatten()

            # Integrates in the Inner-Core.
            Y1, Y2, Y3 = Y1i, Y2i, Y3i
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
            Yicb = solid_to_fluid(
                Y1=Y1.real,
                Y2=Y2.real,
                Y3=Y3.real,
                x=integration_stop,
                first_fluid_layer=self.description_layers[self.below_ICB_layers],
                piG=self.piG,
            )

            # Integrates in the Outer-Core.
            Y = Yicb
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
            Y1cmb, Y2cmb, Y3cmb = fluid_to_solid(
                Yf1=Y,
                x=integration_stop,
                last_fluid_layer=self.description_layers[self.below_CMB_layers - 1],
                piG=self.piG,
            )
            n_start_layer = self.below_CMB_layers

        else:
            # Integrate from geocenter to CMB with high degrees approximation.
            n_start_layer: int = (
                where((array([layer.x_inf for layer in self.description_layers]) ** n) > (self.CMB_x**n))[0][0] + 1
            )
            if hyper_parameters.homogeneous_solution:
                # 1) solution homogène jusqu'à r>=3480000m (CMB).
                Y = solid_homogeneous_system(
                    x=self.description_layers[n_start_layer].x_inf,
                    n=n,
                    layer=self.description_layers[n_start_layer],
                    piG=self.piG,
                )
                Y1cmb = Y[0, :].flatten()
                Y2cmb = Y[1, :].flatten()
                Y3cmb = Y[2, :].flatten()
            else:
                # 1) solution initiale.
                Y = array(
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=complex,
                )
                Y1cmb = Y[0, :].flatten()
                Y2cmb = Y[1, :].flatten()
                Y3cmb = Y[2, :].flatten()

        # Integrates from the CMB to the surface.
        Y1, Y2, Y3 = Y1cmb, Y2cmb, Y3cmb
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
        _, _, _, _, _, _, _, h_load, l_load, k_load = load_surface_solution(
            n=n,
            Y1s=Y1,
            Y2s=Y2,
            Y3s=Y3,
            g_0_surface=g_0_surface,
            piG=self.piG,
            length_ratio=self.length_ratio,
        )
        _, _, _, _, _, _, _, h_shr, l_shr, k_shr = shear_surface_solution(
            n=n,
            Y1s=Y1,
            Y2s=Y2,
            Y3s=Y3,
            g_0_surface=g_0_surface,
            piG=self.piG,
            length_ratio=self.length_ratio,
        )

        _, _, _, _, _, _, _, h_pot, l_pot, k_pot = potential_surface_solution(
            n=n,
            Y1s=Y1,
            Y2s=Y2,
            Y3s=Y3,
            g_0_surface=g_0_surface,
            length_ratio=self.length_ratio,
        )

        LOVE = zeros((9), dtype=complex)
        LOVE[0] = h_load
        LOVE[1] = l_load
        LOVE[2] = k_load
        LOVE[3] = h_shr
        LOVE[4] = l_shr
        LOVE[5] = k_shr
        if shape(h_pot) == (1, 1):
            LOVE[6] = h_pot[0, 0]
            LOVE[7] = l_pot[0, 0]
            LOVE[8] = k_pot[0, 0]
        else:
            LOVE[6] = h_pot
            LOVE[7] = l_pot
            LOVE[8] = k_pot

        return LOVE
