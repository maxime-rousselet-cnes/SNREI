# Computes Love numbers for a single Earth model, but iterates on run options: uses long term anelasticity or attenuation or
# both, with/without bounded functions when it is possible.
#
# See script_01 for details on parameter and result files.

import argparse

from utils import Love_number_comparative_for_options, id_from_model_names_string

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, required=True, help="Optional wanted ID for the real description")
parser.add_argument("--elasticity", type=str, help="Optional wanted ID for the elasticity model to use")
parser.add_argument("--anelasticity", type=str, help="Optional wanted ID for the anelasticity model to use")
parser.add_argument("--attenuation", type=str, help="Optional wanted ID for the attenuation model to use")
parser.add_argument("--load_description", action="store_true", help="Option to tell if the description should be loaded")
args = parser.parse_args()


if __name__ == "__main__":
    Love_number_comparative_for_options(
        real_description_id=(
            args.real_description_id
            if (args.elasticity is None) or (args.anelasticity is None) or (args.attenuation is None)
            else id_from_model_names_string(
                elasticity_model_name=args.elasticity,
                anelasticity_model_name=args.anelasticity,
                attenuation_model_name=args.attenuation,
            )
        ),
        load_description=args.load_description,
        elasticity_model_from_name=args.elasticity,
        anelasticity_model_from_name=args.anelasticity,
        attenuation_model_from_name=args.attenuation,
    )
