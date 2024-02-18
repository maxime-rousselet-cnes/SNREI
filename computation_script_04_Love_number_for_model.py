# Computes Love numbers for a multiple Earth models and iterates on run options: uses long term anelasticity or attenuation or
# both.
#
# See script_01 for details on parameter and result files.


import argparse

from utils import Love_number_comparative_for_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--initial_real_description_id", type=str, required=True, help="Optional wanted ID for the real description"
)
parser.add_argument(
    "--load_initial_description", action="store_true", help="Option to tell if the description should be loaded"
)

args = parser.parse_args()

if __name__ == "__main__":
    Love_number_comparative_for_model(
        initial_real_description_id=args.initial_real_description_id,
        load_initial_description=args.load_initial_description,
        anelasticity_model_names=["test", "test-low-viscosity-Asthenosphere"],
    )
