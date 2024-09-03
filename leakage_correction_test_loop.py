from matplotlib.pyplot import grid, imshow, legend, plot, show, xlabel, ylabel
from numpy import expand_dims, zeros

from snrei.functions import generate_continents_buffered_reprojected_grid, geopandas_oceanic_mean
from snrei.utils import build_elastic_load_signal_components, load_load_signal_hyper_parameters, map_sampling
from snrei.utils.data import extract_GRACE_data
from snrei.utils.filtering import leakage_correction

load_signal_hyper_parameters = load_load_signal_hyper_parameters()
map_0, latitudes, longitudes = extract_GRACE_data(
    # name="GRACE_MSSA_corrected_for_leakage_2003_2022.xyz",
    # name="GRACE_MSSA_2003_2022.xyz",
    # name="MSSA",
    # name="CSR",
    name="TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"
)

harmonics_1 = map_sampling(
    map=map_0,
    n_max=89,
    harmonic_domain=True,
)[0]
harmonics_1_0 = expand_dims(harmonics_1, axis=-1)
l = []
m = []
n = []
d = []
(
    load_signal_hyper_parameters.n_max,
    initial_signal_dates,  # Temporal component's dates.
    harmonic_elastic_load_signal_spatial_component,
    initial_load_signal,
    ocean_mask,
    continents,
    latitudes,
    longitudes,
) = build_elastic_load_signal_components(load_signal_hyper_parameters=load_signal_hyper_parameters)
from matplotlib.pyplot import imshow

imshow(ocean_mask)

continents_reprojected = generate_continents_buffered_reprojected_grid(
    EWH=ocean_mask,
    latitudes=latitudes,
    longitudes=longitudes,
    n_max=load_signal_hyper_parameters.n_max,
    continents=continents,
    buffer_distance=0.0,
)

for buffer_distance in [50, 100, 200, 300]:
    # Buffer to coast.
    continents_buffered_reprojected = generate_continents_buffered_reprojected_grid(
        EWH=ocean_mask,
        latitudes=latitudes,
        longitudes=longitudes,
        n_max=load_signal_hyper_parameters.n_max,
        continents=continents,
        buffer_distance=buffer_distance,
    )
    l += [geopandas_oceanic_mean(continents=continents_buffered_reprojected, latitudes=latitudes, longitudes=longitudes, harmonics=harmonics_1)]
    harmonics_2_0 = leakage_correction(
        frequencial_harmonic_load_signal_initial=harmonics_1_0,
        frequencial_harmonic_geoid=harmonics_1_0,
        frequencial_scale_factor=zeros(shape=(1)),
        frequencial_harmonic_radial_displacement=harmonics_1_0,
        ocean_mask=ocean_mask,
        continents_buffered_reprojected=continents_reprojected,
        latitudes=latitudes,
        longitudes=longitudes,
        iterations=1,
        ddk_filter_level=7,
        n_max=89,
    )
    harmonics_2 = harmonics_2_0[:, :, :, 0].real
    m += [geopandas_oceanic_mean(continents=continents_buffered_reprojected, latitudes=latitudes, longitudes=longitudes, harmonics=harmonics_2)]
    harmonics_3_0 = leakage_correction(
        frequencial_harmonic_load_signal_initial=harmonics_1_0,
        frequencial_harmonic_geoid=harmonics_1_0,
        frequencial_scale_factor=zeros(shape=(1)),
        frequencial_harmonic_radial_displacement=harmonics_1_0,
        ocean_mask=ocean_mask,
        continents_buffered_reprojected=continents_reprojected,
        latitudes=latitudes,
        longitudes=longitudes,
        iterations=2,
        ddk_filter_level=7,
        n_max=89,
    )
    harmonics_3 = harmonics_3_0[:, :, :, 0].real
    n += [geopandas_oceanic_mean(continents=continents_buffered_reprojected, latitudes=latitudes, longitudes=longitudes, harmonics=harmonics_3)]
    print(l[-1], m[-1], n[-1])
    d += [buffer_distance]

plot(d, l, label="no leakage correction")
plot(d, m, label="after leakage correction - 1 iteration")
plot(d, n, label="after leakage correction - 2 iteration")
grid()
xlabel("buffer (km)")
ylabel("ocean mean (mm/yr)")
legend()
show()
