from pathlib import Path
from typing import Optional

from cartopy.crs import Robinson
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.pyplot import figure, show
from numpy import arange, expand_dims, ndarray, zeros
from numpy.random import random

from snrei.functions import mean_on_mask
from snrei.scripts.plot.utils import natural_projection
from snrei.utils import elastic_load_signal_trends_path, get_ocean_mask, load_complex_array_from_binary
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
harmonics_1 = load_complex_array_from_binary(name="0", path=elastic_load_signal_trends_path)
contour = natural_projection(
    ax=current_ax,
    harmonics=harmonics_1,
    saturation_threshold=10.0,
    n_max=89,
    mask=ocean_mask,
)
ax += [current_ax]
print(mean_on_mask(mask=ocean_mask, harmonics=harmonics_1))

current_ax: GeoAxes = fig.add_subplot(
    2,
    1,
    2,
    projection=Robinson(central_longitude=180),
)
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
harmonics_2 = harmonics_2_0[:, :, :, 0]
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
