# Computes Love numbers for a multiple Earth models and iterates on run options: uses long term anelasticity or attenuation or
# both.
#
# See script_01 for details on parameter and result files.

import argparse

from utils import Love_number_comparative_for_asymptotic_ratio

parser = argparse.ArgumentParser()
parser.add_argument(
    "--initial_real_description_id", type=str, required=True, help="Optional wanted ID for the real description"
)
parser.add_argument(
    "--load_initial_description", action="store_true", help="Option to tell if the description should be loaded"
)
args = parser.parse_args()

if __name__ == "__main__":
    Love_number_comparative_for_asymptotic_ratio(
        initial_real_description_id=args.initial_real_description_id,
        asymptotic_ratios=[[1.0, 1.0], [0.5, 1.0], [0.2, 1.0], [0.1, 1.0], [0.05, 1.0]],
        load_initial_description=args.load_initial_description,
        anelasticity_model_names=["test-low-viscosity-Asthenosphere", "test"],
    )
