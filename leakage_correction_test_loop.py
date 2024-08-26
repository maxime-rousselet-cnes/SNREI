from matplotlib.pyplot import imshow, plot, show
from numpy import arange, expand_dims, zeros

from snrei.functions import mean_on_mask
from snrei.utils import get_ocean_mask, map_sampling
from snrei.utils.data import extract_GRACE_data
from snrei.utils.filtering import leakage_correction

map_0 = extract_GRACE_data(
    # name="GRACE_MSSA_corrected_for_leakage_2003_2022.xyz",
    name="GRACE_MSSA_2003_2022.xyz",
    # name="MSSA",
    # name="CSR"
    # name="TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"
)[0]

imshow(map_0)
show()
harmonics_1 = map_sampling(
    map=map_0,
    n_max=89,
    harmonic_domain=True,
)[0]
harmonics_1_0 = expand_dims(harmonics_1, axis=-1)
l = []
m = []
n = []
max_pixels = 10
for pixels_to_coast in range(max_pixels):
    ocean_mask = get_ocean_mask(name="IMERG_land_sea_mask.nc", n_max=89, pixels_to_coast=pixels_to_coast)
    print(mean_on_mask(mask=ocean_mask, grid=map_0))
    l += [mean_on_mask(mask=ocean_mask, harmonics=harmonics_1)]
    harmonics_2_0 = leakage_correction(
        frequencial_harmonic_load_signal_initial=harmonics_1_0,
        frequencial_harmonic_geoid=harmonics_1_0,
        frequencial_scale_factor=zeros(shape=(1)),
        frequencial_harmonic_radial_displacement=harmonics_1_0,
        ocean_mask=ocean_mask,
        iterations=1,
        ddk_filter_level=7,
        n_max=89,
    )
    harmonics_2 = harmonics_2_0[:, :, :, 0]
    m += [mean_on_mask(mask=ocean_mask, harmonics=harmonics_2)]
    harmonics_3_0 = leakage_correction(
        frequencial_harmonic_load_signal_initial=harmonics_1_0,
        frequencial_harmonic_geoid=harmonics_1_0,
        frequencial_scale_factor=zeros(shape=(1)),
        frequencial_harmonic_radial_displacement=harmonics_1_0,
        ocean_mask=ocean_mask,
        iterations=2,
        ddk_filter_level=7,
        n_max=89,
    )
    harmonics_3 = harmonics_3_0[:, :, :, 0]
    n += [mean_on_mask(mask=ocean_mask, harmonics=harmonics_3)]

d = 6378 * 3.14159 / 180 * arange(max_pixels)
plot(d, l)
plot(d, m)
plot(d, n)
show()
