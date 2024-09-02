from cartopy.crs import Robinson
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import expand_dims, zeros

from snrei.functions import mean_on_mask
from snrei.scripts.plot.utils import natural_projection
from snrei.utils import get_ocean_mask
from snrei.utils.data import extract_GRACE_data, map_sampling
from snrei.utils.filtering import leakage_correction

fig = figure(layout="compressed")
ax: list[GeoAxes] = []
current_ax: GeoAxes = fig.add_subplot(
    2,
    1,
    1,
    projection=Robinson(central_longitude=180),
)
ocean_mask = get_ocean_mask(name="0", n_max=89, pixels_to_coast=0)
map_0 = extract_GRACE_data(
    # name="GRACE_MSSA_corrected_for_leakage_2003_2022.xyz",
    # name="GRACE_MSSA_2003_2022.xyz",
    # name="MSSA",
    # name="CSR"
    name="TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"
)[0]

harmonics_1 = map_sampling(
    map=map_0,
    n_max=89,
    harmonic_domain=True,
)[0]
contour = natural_projection(
    ax=current_ax,
    harmonics=harmonics_1.real,
    saturation_threshold=10.0,
    n_max=89,
    mask=ocean_mask,
)
ax += [current_ax]
current_ax.set_title("before leakage correction")
print(mean_on_mask(mask=ocean_mask, harmonics=harmonics_1))

current_ax: GeoAxes = fig.add_subplot(
    2,
    1,
    2,
    projection=Robinson(central_longitude=180),
)
current_ax.set_title("after leakage correction")
harmonics_1_0 = expand_dims(harmonics_1, axis=-1)
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
harmonics_2 = harmonics_2_0[:, :, :, 0].real
contour = natural_projection(
    ax=current_ax,
    harmonics=harmonics_2,
    saturation_threshold=10.0,
    n_max=89,
    mask=ocean_mask,
)
ax += [current_ax]
print(mean_on_mask(mask=ocean_mask, harmonics=harmonics_2))

cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", shrink=0.5, extend="both")
cbar.set_label(label="(mm/yr)")

show()
