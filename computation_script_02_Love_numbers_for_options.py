# Computes Love numbers for a single Earth model, but iterates on run options: uses long term anelasticity or attenuation or
# both, with/without bounded functions when it is possible.
#
# See script_01 for details on parameter and result files.

import argparse

from utils import Love_number_comparative_for_options

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, required=True, help="Optional wanted ID for the real description")
parser.add_argument("--load_description", action="store_true", help="Option to tell if the description should be loaded")
args = parser.parse_args()


if __name__ == "__main__":
    Love_number_comparative_for_options(real_description_id=args.real_description_id, load_description=args.load_description)
