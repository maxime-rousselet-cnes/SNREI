from itertools import product

from script_02_Love_number_comparative import Love_number_comparative
from utils import (
    RealDescriptionParameters,
    load_base_model,
    parameters_path,
    save_base_model,
)


def Love_number_comparative_for_sampling(
    profile_precisions: dict[str, int] = {"low": 10, "mid": 100, "high": 1000},
    n_splines_bases: dict[str, int] = {"low": 10, "mid": 100, "high": 1000},
) -> None:
    """
    Computes anelastic Love numbers with and without anelasticity and with and without attenuation.
    """
    # Loads hyper parameters.
    real_description_parameters: RealDescriptionParameters = load_base_model(
        name="real_description_parameters", path=parameters_path, base_model_type=RealDescriptionParameters
    )
    for (profile_precision_name_part, profile_precision), (n_splines_base_name_part, n_splines_base) in product(
        profile_precisions.items(), n_splines_bases.items()
    ):
        real_description_parameters.profile_precision = profile_precision
        real_description_parameters.n_splines_base = n_splines_base
        save_base_model(obj=real_description_parameters, name="real_description_parameters", path=parameters_path)
        Love_number_comparative(
            real_description_id=profile_precision_name_part + "_p_" + n_splines_base_name_part + "_ns", load_description=False
        )


if __name__ == "__main__":
    Love_number_comparative_for_sampling()
