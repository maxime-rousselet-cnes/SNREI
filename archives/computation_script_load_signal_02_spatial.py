# Computes spatially-variable viscoelastic-modified load signal for a given description and options.
# Gets already computed Love numbers, builds load signal from data, computes viscoelastic induced load signal and save it as
# a Result instance in (.JSON) file.
# Then, applies spacial dependency to this mean oceanic load signal in harmonic domain, saves harmonic and spatial datas.
# May save corresponding figures in specified subfolder.

import argparse

from utils import single_spatial_viscoelastic_load_signal

parser = argparse.ArgumentParser()
parser.add_argument("--real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("--subpath", type=str, help="wanted path to save figure")
args = parser.parse_args()

if __name__ == "__main__":
    single_spatial_viscoelastic_load_signal(
        real_description_id=(
            args.real_description_id
            if args.real_description_id
            else "PREM_low-viscosity-asthenosphere-anelastic-lithosphere_Benjamin-variable-asymptotic_ratio0.05-1.0"
        ),
        figure_subpath_string=args.subpath if args.subpath else "spatial_load_signal",
    )
# TODO: dependent on run ? and in 01 too ?
