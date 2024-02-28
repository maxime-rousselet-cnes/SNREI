# Computes viscoelastic-modified load signal for a given description and options.
# Gets already computed Love numbers, builds load signal from data, computes viscoelastic induced load signal and save it as
# a Result instance in (.JSON) file.
# May save corresponding figures in specified subfolder.

import argparse

from utils import single_viscoelastic_signal

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()

if __name__ == "__main__":
    single_viscoelastic_signal(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_high-viscosity-asthenosphere-elastic-lithosphere_Benjamin-variable-asymptotic_ratio0.1-1.0"
        ),
        figure_subpath_string=args.subpath if args.subpath else "load_signal",
    )
