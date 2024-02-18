# Computes Love numbers for a single Earth model.
#
# Iterates on degrees and frequencies.
#
# Parameters are written as:
#   > parameters
#       > real_description_parameters.json: parameters describing the construction of the Earth's model description.
#       > Y_system_hyper_parameters.json: parameters describing the numerical integration for a given (n, omega).
#       > Love_numbers_hyper_parameters.json: parameters describing the loop on frequencies and anelasticities options.
#
# Results are saved as:
#   > 'real_description_id'
#       > degrees.json: list of degrees for which Love numbers were computed.
#       > elastic_Love_numbers.json: elastic values of Love numbers. See Result class.
#       > runs
#           > 'run_id'
#               > per_degree
#                   > 'n'
#                       > frequencies.json: frequencies for which Love numbers of degree 'n' were computed.
#                       > Love_numbers.json:computed Love numbers for degree 'n'.
#           > anelastic_Love_numbers.json: interpolated Love numbers at all frequencies and degrees. See Result class.
#           > frequencies.json: All frequencies.

import argparse

from utils import Love_numbers_from_models_to_result

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--run_id", type=str, help="Optional wanted ID for the run")
parser.add_argument("--load_description", action="store_true", help="Option to tell if the description should be loaded")

args = parser.parse_args()


if __name__ == "__main__":
    Love_numbers_from_models_to_result(
        real_description_id=args.real_description_id, run_id=args.run_id, load_description=args.load_description
    )
