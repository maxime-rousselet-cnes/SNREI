from matplotlib.pyplot import grid, legend, plot, show, xlabel, ylabel
from numpy import expand_dims, zeros

from snrei.functions import geopandas_oceanic_mean
from snrei.utils import build_elastic_load_signal_components, load_load_signal_hyper_parameters, map_sampling
from snrei.utils.filtering import leakage_correction

load_signal_hyper_parameters = load_load_signal_hyper_parameters()
load_signal_hyper_parameters.load_spatial_behaviour_file = "TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"
# "GRACE_MSSA_corrected_for_leakage_2003_2022.xyz",
# "GRACE_MSSA_2003_2022.xyz",
# "MSSA",
# "CSR",


l = []
m = []
n = []
d = []
from matplotlib.pyplot import show

for buffer_distance in [50, 100, 200, 300]:
    # Buffer to coast.
    load_signal_hyper_parameters.buffer_distance = buffer_distance
    (
        load_signal_hyper_parameters.n_max,
        initial_signal_dates,
        harmonics_1,
        initial_load_signal,
        ocean_land_geopandas_buffered_reprojected,
        ocean_land_mask,
        ocean_land_buffered_mask,
        latitudes,
        longitudes,
    ) = build_elastic_load_signal_components(load_signal_hyper_parameters=load_signal_hyper_parameters)
    harmonics_1_0 = expand_dims(harmonics_1, axis=-1)
    l += [
        geopandas_oceanic_mean(
            signal_threshold=load_signal_hyper_parameters.signal_threshold,
            ocean_land_geopandas_buffered_reprojected=ocean_land_geopandas_buffered_reprojected,
            latitudes=latitudes,
            longitudes=longitudes,
            harmonics=harmonics_1,
        )
    ]
    harmonics_2_0 = leakage_correction(
        frequencial_harmonic_load_signal_initial=harmonics_1_0,
        frequencial_harmonic_geoid=harmonics_1_0,
        frequencial_scale_factor=zeros(shape=(1)),
        frequencial_harmonic_radial_displacement=harmonics_1_0,
        ocean_land_mask=ocean_land_mask,
        ocean_land_geopandas_buffered_reprojected=ocean_land_geopandas_buffered_reprojected,
        latitudes=latitudes,
        longitudes=longitudes,
        iterations=1,
        ddk_filter_level=7,
        n_max=89,
    )
    harmonics_2 = harmonics_2_0[:, :, :, 0].real
    m += [
        geopandas_oceanic_mean(
            signal_threshold=load_signal_hyper_parameters.signal_threshold,
            ocean_land_geopandas_buffered_reprojected=ocean_land_geopandas_buffered_reprojected,
            latitudes=latitudes,
            longitudes=longitudes,
            harmonics=harmonics_2,
        )
    ]
    harmonics_3_0 = leakage_correction(
        frequencial_harmonic_load_signal_initial=harmonics_1_0,
        frequencial_harmonic_geoid=harmonics_1_0,
        frequencial_scale_factor=zeros(shape=(1)),
        frequencial_harmonic_radial_displacement=harmonics_1_0,
        ocean_land_mask=ocean_land_mask,
        ocean_land_geopandas_buffered_reprojected=ocean_land_geopandas_buffered_reprojected,
        latitudes=latitudes,
        longitudes=longitudes,
        iterations=2,
        ddk_filter_level=7,
        n_max=89,
    )
    harmonics_3 = harmonics_3_0[:, :, :, 0].real
    n += [
        geopandas_oceanic_mean(
            signal_threshold=load_signal_hyper_parameters.signal_threshold,
            ocean_land_geopandas_buffered_reprojected=ocean_land_geopandas_buffered_reprojected,
            latitudes=latitudes,
            longitudes=longitudes,
            harmonics=harmonics_3,
        )
    ]
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
