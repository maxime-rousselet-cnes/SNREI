from typing import Optional

from numpy import Inf, array, min, ndarray, round
from scipy import interpolate

from ...database import load_base_model
from ...rheological_formulas import find_tau_M, mu_k_computing
from ..constants import SECONDS_PER_YEAR
from ..description_layer import DescriptionLayer
from ..hyper_parameters import AnelasticityDescriptionParameters
from ..model import ModelPart
from ..paths import parameters_path
from .description import Description, Spline
from .elasticity_description import ElasticityDescription


def anelasticity_description_id_from_part_names(
    elasticity_name: str, long_term_anelasticity_name: str, short_term_anelasticity_name: str
):
    """
    Builds an id for an anelasticity description given the names of its components.
    """
    return "_____".join(
        (
            part_name.replace("/", "____")
            for part_name in (elasticity_name, long_term_anelasticity_name, short_term_anelasticity_name)
        )
    )


class AnelasticityDescription(Description):
    """
    Describes the integration constants and all layer model descriptions, including anelastic parameters.
    """

    # Proper fields.
    frequency_unit: float
    elasticity_unit: float
    viscosity_unit: float
    variable_values_per_layer: list[dict[str, ndarray]]

    #  Fields also present in the elasticity description, but may differ if it is loaded.
    x_CMB: float
    period_unit: float
    density_unit: float
    speed_unit: float
    piG: float
    below_ICB_layers: int
    below_CMB_layers: int
    spline_degree: int

    # Different parts descriptions.
    elasticity_model_name: str
    long_term_anelasticity_model_name: str
    short_term_anelasticity_model_name: str
    elasticity_description: str  # Unitless.
    anelasticicty_description: str  # With units.
    attenuation_description: str  # With units.

    def load(self):
        """
        Loads an Anelasticity Description instance with correctly formatted fields.
        """
        super().load()
        # Formats variable array values.
        layer_values_list: list[dict[str, list[float]]] = self.variable_values_per_layer
        self.variable_values_per_layer: list[dict[str, ndarray]] = [
            {variable_name: array(values, dtype=float) for variable_name, values in layer_values.items()}
            for layer_values in layer_values_list
        ]

    def save(self, overwrite_description: bool = True) -> None:
        """
        Replace carrefully infinite values by strings in proper fields for convenient (.JSON) writing, then save and replace
        back by infinite values.
        """
        # Replace infinite values by strings.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if Inf in values:
                    self.variable_values_per_layer[i_layer][variable_name] = array(["Inf"] * len(values))
        # Saves to (.JSON) file.
        super().save(overwrite_description=overwrite_description)
        # Replace back strings by infinite values.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if "Inf" in values:
                    self.variable_values_per_layer[i_layer][variable_name] = array([Inf] * len(values))

    def __init__(
        self,
        anelasticity_description_parameters: AnelasticityDescriptionParameters = load_base_model(
            name="anelasticity_description_parameters", path=parameters_path, base_model_type=AnelasticityDescriptionParameters
        ),
        load_description: bool = False,
        id: Optional[str] = None,
        save: bool = True,
        overwrite_descriptions: bool = False,
        elasticity_name: Optional[str] = None,
        long_term_anelasticity_name: Optional[str] = None,
        short_term_anelasticity_name: Optional[str] = None,
    ) -> None:
        # Updates inherited fields.
        super().__init__(
            id=(
                id
                if not (id is None)
                else anelasticity_description_id_from_part_names(
                    elasticity_name=elasticity_name,
                    long_term_anelasticity_name=long_term_anelasticity_name,
                    short_term_anelasticity_name=short_term_anelasticity_name,
                )
            ),
            radius_unit=anelasticity_description_parameters.radius_unit,
            real_crust=anelasticity_description_parameters.real_crust,
            spline_number=anelasticity_description_parameters.spline_number,
        )
        # Eventually loads already preprocessed anelasticity description...
        if load_description and self.get_path().joinpath(self.id + ".json").is_file():
            self.load()
        # ... or builds the description.
        else:
            # Initializes all model description parts.
            description_parts: dict[ModelPart, Description | ElasticityDescription] = {}
            part_names: dict[ModelPart, str] = {
                ModelPart.elasticity: elasticity_name,
                ModelPart.long_term_anelasticity: long_term_anelasticity_name,
                ModelPart.short_term_anelasticity: short_term_anelasticity_name,
            }
            for model_part, (_, part_name) in zip(ModelPart, part_names.items()):
                # Initializes.
                if model_part == ModelPart.elasticity:
                    description_parts[model_part] = ElasticityDescription(
                        id=elasticity_name,
                        model_filename=elasticity_name,
                        radius_unit=anelasticity_description_parameters.radius_unit,
                        real_crust=anelasticity_description_parameters.real_crust,
                        spline_number=anelasticity_description_parameters.spline_number,
                        below_ICB_layers=anelasticity_description_parameters.below_ICB_layers,
                        below_CMB_layers=anelasticity_description_parameters.below_CMB_layers,
                        spline_degree=anelasticity_description_parameters.spline_degree,
                    )
                else:
                    description_parts[model_part] = Description(
                        id=part_name,
                        model_filename=part_name,
                        model_part=model_part,
                        radius_unit=anelasticity_description_parameters.radius_unit,
                        real_crust=anelasticity_description_parameters.real_crust,
                        spline_number=anelasticity_description_parameters.spline_number,
                    )
                # Eventually loads the model description part ...
                if (not overwrite_descriptions) and description_parts[model_part].get_path().joinpath(
                    description_parts[model_part].id
                ).is_file():

                    description_parts[model_part].load()
                # ... or builds it.
                else:
                    description_parts[model_part].build(overwrite_description=True, save=save)

            # Updates fields from elasticity description.
            self.x_CMB = description_parts[ModelPart.elasticity].x_CMB
            self.period_unit = description_parts[ModelPart.elasticity].period_unit
            self.density_unit = description_parts[ModelPart.elasticity].density_unit
            self.speed_unit = description_parts[ModelPart.elasticity].speed_unit
            self.piG = description_parts[ModelPart.elasticity].piG
            self.below_ICB_layers = description_parts[ModelPart.elasticity].below_ICB_layers
            self.below_CMB_layers = description_parts[ModelPart.elasticity].below_CMB_layers
            self.spline_degree = description_parts[ModelPart.elasticity].spline_degree

            # Updates new fields.
            self.frequency_unit = 1.0 / self.period_unit
            self.viscosity_unit = self.density_unit * self.radius_unit**2 / self.period_unit
            self.elasticity_unit = self.viscosity_unit / self.period_unit

            # Builds common description layers.
            self.merge_descriptions(description_parts=description_parts)

            # Computes explicit variable values for incoming lambda and mu complex computings.
            self.variable_values_per_layer = self.compute_variable_values()

            # Saves resulting anelasticity description in a (.JSON) file.
            if save:
                self.save(overwrite_description=overwrite_descriptions)

    def merge_descriptions(self, description_parts: dict[ModelPart, Description | ElasticityDescription]):
        """
        Merges all model description parts with unitless variables only.
        """
        # Initializes with Core elastic and liquid layers.
        self.description_layers = description_parts[ModelPart.elasticity].description_layers[: self.below_CMB_layers]
        # Initializes accumulators.
        x_inf: float = self.x_CMB
        layer_indices_per_part: dict[ModelPart, int] = {model_part: 0 for model_part in ModelPart}
        layer_indices_per_part[ModelPart.elasticity] = self.below_CMB_layers
        # Checks all layers from CMB to surface and merges their descrptions.
        while x_inf < 1.0:
            # Checks which layer ends first.
            x_sup_per_part: dict[ModelPart, float] = {
                model_part: round(
                    a=description_parts[model_part].description_layers[layer_indices_per_part[model_part]].x_sup, decimals=5
                )
                for model_part in ModelPart
            }
            x_sup: float = min([value for _, value in x_sup_per_part.items()])
            # Updates.
            self.description_layers += [
                self.merge_layers(
                    x_inf=x_inf,
                    x_sup=x_sup,
                    layers_per_part={
                        model_part: description_parts[model_part].description_layers[layer_indices_per_part[model_part]]
                        for model_part in ModelPart
                    },
                )
            ]
            x_inf = x_sup
            for model_part in ModelPart:
                if x_sup == x_sup_per_part[model_part]:
                    layer_indices_per_part[model_part] += 1

    def merge_layers(
        self,
        x_inf: float,
        x_sup: float,
        layers_per_part: dict[ModelPart, DescriptionLayer],
    ) -> DescriptionLayer:
        """
        Merges elasticity, anelasticity, and attenuation description layers with unitless variables only.
        """
        # Creates corresponding minimal length layer with elasticity variables.
        description_layer = DescriptionLayer(
            name="__".join((layer.name for _, layer in layers_per_part.items())),
            x_inf=x_inf,
            x_sup=x_sup,
            splines=layers_per_part[ModelPart.elasticity].splines.copy(),
        )

        # Adds other unitless variables.
        description_layer.splines["c"] = layers_per_part[ModelPart.long_term_anelasticity].splines["c"]
        description_layer.splines["alpha"] = layers_per_part[ModelPart.short_term_anelasticity].splines["alpha"]
        description_layer.splines["asymptotic_mu_ratio"] = layers_per_part[ModelPart.short_term_anelasticity].splines[
            "asymptotic_mu_ratio"
        ]
        description_layer.splines["Q_mu"] = layers_per_part[ModelPart.short_term_anelasticity].splines["Q_mu"]

        # Builds other unitless variables from variables with units.
        for variable_name, unit, splines in [
            ("eta_m", self.viscosity_unit, layers_per_part[ModelPart.long_term_anelasticity].splines),
            ("eta_k", self.viscosity_unit, layers_per_part[ModelPart.long_term_anelasticity].splines),
            ("mu_K1", self.elasticity_unit, layers_per_part[ModelPart.long_term_anelasticity].splines),
            ("omega_m", self.frequency_unit, layers_per_part[ModelPart.short_term_anelasticity].splines),
            ("tau_M", self.period_unit / SECONDS_PER_YEAR, layers_per_part[ModelPart.short_term_anelasticity].splines),
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
        Computes the needed explicit variable values for a single layer.
        """
        x = layer.x_profile(spline_number=self.spline_number)
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
                    "Q_mu": layer.evaluate(x=x, variable="Q_mu"),
                    "alpha": layer.evaluate(x=x, variable="alpha"),
                    "omega_m": layer.evaluate(x=x, variable="omega_m"),
                    "tau_M": layer.evaluate(x=x, variable="tau_M"),
                    "asymptotic_mu_ratio": layer.evaluate(x=x, variable="asymptotic_mu_ratio"),
                }
            )
            # Eventually finds tau_M profile that constrains mu(omega -> Inf) = asymptotic_ratio * mu_0:
            if round(a=1.0 - variable_values["asymptotic_mu_ratio"], decimals=5).any():
                for i_x, (omega_m, alpha, asymptotic_mu_ratio, Q_mu) in enumerate(
                    zip(
                        variable_values["omega_m"],
                        variable_values["alpha"],
                        variable_values["asymptotic_mu_ratio"],
                        variable_values["Q_mu"],
                    )
                ):
                    # Updates explicit variable.
                    variable_values["tau_M"][i_x] = find_tau_M(
                        omega_m=omega_m,
                        alpha=alpha,
                        asymptotic_mu_ratio=asymptotic_mu_ratio,
                        Q_mu=Q_mu,
                    )
                # Updates spline.
                self.description_layers[i_layer].splines.update(
                    {
                        "tau_M": interpolate.splrep(x=variable_values["x"], y=variable_values["tau_M"]),
                    }
                )
        return variable_values
