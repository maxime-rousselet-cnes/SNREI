from typing import Optional

from numpy import Inf, array, ndarray, round
from scipy import interpolate

from ...constants import SECONDS_PER_YEAR, Earth_radius
from ...paths import (
    anelasticity_descriptions_path,
    attenuation_descriptions_path,
    elasticity_descriptions_path,
    real_descriptions_path,
)
from ...rheological_formulas import find_tau_M, mu_k_computing
from ..description_layer import DescriptionLayer
from ..hyper_parameters import LoveNumbersHyperParameters
from ..spline import Spline
from .anelasticity_description import AnelasticityDescription
from .attenuation_description import AttenuationDescription
from .description import Description
from .elasticity_description import ElasticityDescription


class RealDescription(Description):
    """
    Describes the integration constants and all layer model descriptions, including anelastic parameters.
    """

    # Proper attributes.
    CMB_x: float
    length_ratio: float
    frequency_unit: float
    elasticity_unit: float
    viscosity_unit: float
    variable_values_per_layer: list[dict[str, ndarray]]

    # Attributes also present in the elasticity description part, but may differ if elasticity part is loaded.
    radius_unit: float
    period_unit: float
    density_unit: float
    speed_unit: float
    piG: float
    below_ICB_layers: int
    below_CMB_layers: int
    profile_precision: int
    splines_degree: int

    # Different parts descriptions.
    elasticity_model_name: str
    anelasticity_model_name: str
    attenuation_model_name: str
    elasticity_description: str  # Unitless.
    anelasticicty_description: str  # With units.
    attenuation_description: str  # With units.

    def __init__(
        self,
        # Parameters for elasticity description preprocessing only.
        below_ICB_layers: Optional[int],
        below_CMB_layers: Optional[int],
        splines_degree: int,
        # Base parameters.
        radius_unit: float,
        real_crust: bool,
        n_splines_base: int,
        profile_precision: int,
        # Real description parameters.
        radius: float,
        id: Optional[str] = None,
        # Loading parameters.
        elasticity_model_from_name: Optional[str] = None,
        anelasticity_model_from_name: Optional[str] = None,
        attenuation_model_from_name: Optional[str] = None,
        elasticity_description_from_id: Optional[str] = None,
        anelasticity_description_from_id: Optional[str] = None,
        attenuation_description_from_id: Optional[str] = None,
        # Whether to load the whole real description later or not.
        load_description: bool = True,
        # Whether to save when preprocessed or not.
        save: bool = True,
    ) -> None:
        super().__init__(
            id=id,
            radius_unit=radius_unit,
            real_crust=real_crust,
            n_splines_base=n_splines_base,
        )
        if load_description:
            return

        # Elasticity description.
        if elasticity_description_from_id:
            # Loads elasticity description.
            elasticity_description = ElasticityDescription(
                radius_unit=0.0,
                below_ICB_layers=None,
                below_CMB_layers=None,
                real_crust=False,
                n_splines_base=0,
                profile_precision=0,
                splines_degree=0,
                model_filename="",
                id=elasticity_description_from_id,
            )
            elasticity_description.load(path=elasticity_descriptions_path)
        else:
            # Builds elasticity description.
            elasticity_description = ElasticityDescription(
                id=id,
                radius_unit=radius_unit,
                below_ICB_layers=below_ICB_layers,
                below_CMB_layers=below_CMB_layers,
                real_crust=real_crust,
                n_splines_base=n_splines_base,
                profile_precision=profile_precision,
                splines_degree=splines_degree,
                model_filename=elasticity_model_from_name if elasticity_model_from_name else "PREM",
                load_description=False,
                save=save,
            )

        # Anelasticity description.
        if anelasticity_description_from_id:
            # Loads anelasticity description.
            anelasticity_description = AnelasticityDescription(
                radius_unit=0.0,
                below_ICB_layers=None,
                below_CMB_layers=None,
                real_crust=False,
                n_splines_base=0,
                profile_precision=0,
                splines_degree=0,
                model_filename="",
                id=anelasticity_description_from_id,
            )
            anelasticity_description.load(path=anelasticity_descriptions_path)
        else:
            # Builds anelasticity description.
            anelasticity_description = AnelasticityDescription(
                id=id,
                radius_unit=radius_unit,
                real_crust=real_crust,
                n_splines_base=n_splines_base,
                model_filename=(
                    anelasticity_model_from_name
                    if anelasticity_model_from_name
                    else "low-viscosity-asthenosphere-elastic-lithosphere"
                ),
                load_description=False,
                save=save,
            )

        # Attenuation description.
        if attenuation_description_from_id:
            # Loads attenuation description.
            attenuation_description = AttenuationDescription(
                radius_unit=0.0,
                real_crust=False,
                n_splines_base=0,
                model_filename="",
                id=attenuation_description_from_id,
            )
            attenuation_description.load(path=attenuation_descriptions_path)
        else:
            # Builds attenuation description.
            attenuation_description = AttenuationDescription(
                id=id,
                radius_unit=radius_unit,
                real_crust=real_crust,
                n_splines_base=n_splines_base,
                model_filename=attenuation_model_from_name if attenuation_model_from_name else "Benjamin",
                load_description=False,
                save=save,
            )

        # Builds ID as the concatenation of the description IDs.
        if not id:
            self.id = "_".join((elasticity_description.id, anelasticity_description.id, attenuation_description.id, self.id))

        # Updates fields from elasticity description.
        self.radius_unit = elasticity_description.radius_unit
        self.period_unit = elasticity_description.period_unit
        self.density_unit = elasticity_description.density_unit
        self.speed_unit = elasticity_description.speed_unit
        self.piG = elasticity_description.piG
        self.below_ICB_layers = elasticity_description.below_ICB_layers
        self.below_CMB_layers = elasticity_description.below_CMB_layers

        # Updates new fields.
        self.profile_precision = profile_precision
        self.splines_degree = splines_degree
        self.CMB_x = elasticity_description.description_layers[self.below_CMB_layers].x_inf
        self.length_ratio = self.radius_unit / radius
        self.frequency_unit = 1.0 / elasticity_description.period_unit
        self.viscosity_unit = self.density_unit * self.radius_unit**2 / self.period_unit
        self.elasticity_unit = self.viscosity_unit / self.period_unit

        # Builds common description layers.
        self.merge_descriptions(
            elasticity_description=elasticity_description,
            anelasticity_description=anelasticity_description,
            attenuation_description=attenuation_description,
        )

        # Computes explicit variable values for incoming lambda and mu complex computings.
        self.variable_values_per_layer = self.compute_variable_values()

        # Saves resulting real description in a (.JSON) file.
        self.elasticity_model_name = elasticity_description.model_filename
        self.anelasticity_model_name = anelasticity_description.model_filename
        self.attenuation_model_name = attenuation_description.model_filename
        self.elasticity_description = elasticity_description.id
        self.anelasticity_description = anelasticity_description.id
        self.attenuation_description = attenuation_description.id
        if save:
            self.real_description_save()

    def merge_descriptions(
        self,
        elasticity_description: ElasticityDescription,
        anelasticity_description: AnelasticityDescription,
        attenuation_description: AttenuationDescription,
    ):
        """
        Merges elasticity, anelasticity, and attenuation descriptions with unitless variables only.
        """
        # Initializes with Core elastic and liquid layers.
        self.description_layers = elasticity_description.description_layers[: self.below_CMB_layers]
        x_inf = self.CMB_x
        i_layer_elasticity = self.below_CMB_layers
        i_layer_anelasticity = 0
        i_layer_attenuation = 0

        # Checks all layers from CMB to surface and merges their descrptions.
        while x_inf < 1.0:
            # Checks which layer ends first.
            x_sup_elasticity = round(a=elasticity_description.description_layers[i_layer_elasticity].x_sup, decimals=8)
            x_sup_anelasticity = round(a=anelasticity_description.description_layers[i_layer_anelasticity].x_sup, decimals=8)
            x_sup_attenuation = round(a=attenuation_description.description_layers[i_layer_attenuation].x_sup, decimals=8)
            x_sup = min(x_sup_elasticity, x_sup_anelasticity, x_sup_attenuation)

            # Updates.
            self.description_layers += [
                self.merge_layers(
                    x_inf=x_inf,
                    x_sup=x_sup,
                    elasticity_layer=elasticity_description.description_layers[i_layer_elasticity],
                    anelasticity_layer=anelasticity_description.description_layers[i_layer_anelasticity],
                    attenuation_layer=attenuation_description.description_layers[i_layer_attenuation],
                )
            ]
            x_inf = x_sup
            i_layer_elasticity += 1 if x_sup_elasticity == x_sup else 0
            i_layer_anelasticity += 1 if x_sup_anelasticity == x_sup else 0
            i_layer_attenuation += 1 if x_sup_attenuation == x_sup else 0

    def merge_layers(
        self,
        x_inf: float,
        x_sup: float,
        elasticity_layer: DescriptionLayer,
        anelasticity_layer: DescriptionLayer,
        attenuation_layer: DescriptionLayer,
    ) -> DescriptionLayer:
        """
        Merges elasticity, anelasticity, and attenuation description layers with unitless variables only.
        """
        # Creates corresponding minimal length layer with elasticity variables.
        description_layer = DescriptionLayer(
            name="-".join(
                (
                    elasticity_layer.name,
                    anelasticity_layer.name,
                    attenuation_layer.name,
                )
            ),
            x_inf=x_inf,
            x_sup=x_sup,
            splines=elasticity_layer.splines.copy(),
        )

        # Adds anelasticity and attenuation unitless variables.
        description_layer.splines["c"] = anelasticity_layer.splines["c"]
        description_layer.splines["alpha"] = attenuation_layer.splines["alpha"]
        description_layer.splines["asymptotic_attenuation"] = attenuation_layer.splines["asymptotic_attenuation"]

        # Builds anelasticity and attenuation unitless variables from variables with units.
        for variable_name, unit, splines in [
            ("eta_m", self.viscosity_unit, anelasticity_layer.splines),
            ("eta_k", self.viscosity_unit, anelasticity_layer.splines),
            ("mu_K1", self.elasticity_unit, anelasticity_layer.splines),
            ("omega_m", self.frequency_unit, attenuation_layer.splines),
            ("tau_M", self.period_unit / SECONDS_PER_YEAR, attenuation_layer.splines),
        ]:
            description_layer.splines[variable_name] = Spline(
                (
                    splines[variable_name][0],
                    splines[variable_name][1] / unit,  # Gets unitless variable.
                    splines[variable_name][2],
                )
            )

        return description_layer

    def compute_variable_values(
        self,
    ) -> list[dict[str, ndarray]]:
        """
        Computes explicit variable values for all layers.
        """
        variable_values_per_layer = []
        for i_layer, layer in enumerate(self.description_layers):
            variable_values_per_layer += [self.compute_variable_values_per_layer(i_layer=i_layer, layer=layer)]
        return variable_values_per_layer

    def compute_variable_values_per_layer(self, i_layer: int, layer: DescriptionLayer) -> dict[str, ndarray]:
        """
        Computes explicit variable values for a single layer.
        """
        x = layer.x_profile(profile_precision=self.profile_precision)
        # Variables needed for all layers.
        variable_values: dict[str, ndarray] = {
            "x": x,
            "mu_0": layer.evaluate(x=x, variable="mu_0"),
            "lambda_0": layer.evaluate(x=x, variable="lambda_0"),
        }
        if i_layer >= self.below_CMB_layers:
            # Variables needed above the Core-Mantle Boundary.
            variable_values.update(
                {
                    "eta_m": layer.evaluate(x=x, variable="eta_m"),
                    "mu_k": mu_k_computing(
                        mu_K1=layer.evaluate(x=x, variable="mu_K1"),
                        c=layer.evaluate(x=x, variable="c"),
                        mu_0=layer.evaluate(x=x, variable="mu_0"),
                    ),
                    "eta_k": layer.evaluate(x=x, variable="eta_k"),
                    "Qmu": layer.evaluate(x=x, variable="Qmu"),
                    "alpha": layer.evaluate(x=x, variable="alpha"),
                    "omega_m": layer.evaluate(x=x, variable="omega_m"),
                    "tau_M": layer.evaluate(x=x, variable="tau_M"),
                    "asymptotic_attenuation": layer.evaluate(x=x, variable="asymptotic_attenuation"),
                }
            )
            # Eventually finds tau_M profile that constrains mu(omega -> Inf) = asymptotic_ratio * mu_0:
            if round(a=variable_values["asymptotic_attenuation"], decimals=4).any():
                for i_x, (omega_m, alpha, asymptotic_attenuation, Qmu) in enumerate(
                    zip(
                        variable_values["omega_m"],
                        variable_values["alpha"],
                        variable_values["asymptotic_attenuation"],
                        variable_values["Qmu"],
                    )
                ):
                    variable_values["tau_M"][i_x] = find_tau_M(
                        omega_m=omega_m,
                        alpha=alpha,
                        asymptotic_attenuation=asymptotic_attenuation,
                        Qmu=Qmu,
                    )
                self.description_layers[i_layer].splines.update(
                    {
                        "tau_M": interpolate.splrep(x=variable_values["x"], y=variable_values["tau_M"]),
                    }
                )
        return variable_values

    def real_description_save(self) -> None:
        """
        Replace carrefully infinite values by strings in fields for convenient (.JSON) writing, then save and replace back by
        infinite values.
        """
        # Replace infinite values by strings.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if Inf in values:
                    self.variable_values_per_layer[i_layer][variable_name] = array(["Inf"] * len(values))
        # Saves to (.JSON) file.
        self.save(path=real_descriptions_path)
        # Replace back strings by infinite values.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if "Inf" in values:
                    self.variable_values_per_layer[i_layer][variable_name] = array([Inf] * len(values))


def real_description_from_parameters(
    Love_numbers_hyper_parameters: LoveNumbersHyperParameters,
    real_description_id: Optional[str] = None,
    load_description: Optional[bool] = None,
    elasticity_model_from_name: Optional[str] = None,
    anelasticity_model_from_name: Optional[str] = None,
    attenuation_model_from_name: Optional[str] = None,
    save: bool = True,
) -> RealDescription:
    """
    Builds a real description instance given the needed hyper parameters.
    """
    real_description_parameters = Love_numbers_hyper_parameters.real_description_parameters
    real_description = RealDescription(
        id=real_description_id,
        below_ICB_layers=real_description_parameters.below_ICB_layers,
        below_CMB_layers=real_description_parameters.below_CMB_layers,
        splines_degree=real_description_parameters.splines_degree,
        radius_unit=real_description_parameters.radius_unit if real_description_parameters.radius_unit else Earth_radius,
        real_crust=real_description_parameters.real_crust,
        n_splines_base=real_description_parameters.n_splines_base,
        profile_precision=real_description_parameters.profile_precision,
        radius=real_description_parameters.radius if real_description_parameters.radius else Earth_radius,
        load_description=False if load_description is None else load_description,
        elasticity_model_from_name=elasticity_model_from_name,
        anelasticity_model_from_name=anelasticity_model_from_name,
        attenuation_model_from_name=attenuation_model_from_name,
        save=save,
    )

    if load_description:
        real_description.load(path=real_descriptions_path)

    return real_description
