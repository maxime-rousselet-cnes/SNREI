from typing import Optional

from geopandas import GeoDataFrame
from numpy import arange, array, ceil, concatenate, flip, inf, linspace, log2, meshgrid, ndarray, round, where, zeros
from scipy import interpolate
from scipy.fft import fft, fftfreq, ifft
from shapely.geometry import Point

from ...functions import EARTH_EQUAL_PROJECTION, LAT_LON_PROJECTION, map_normalizing, mean_on_mask, signal_trend, surface_ponderation
from ..classes import LoadSignalHyperParameters, load_load_signal_hyper_parameters
from ..data import extract_GRACE_data, extract_temporal_load_signal, get_continents, get_ocean_mask, map_sampling


def get_trend_dates(
    signal_dates: ndarray[float] | list[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    recent_trend: bool = True,
    shift: bool = True,
) -> tuple[ndarray[float], ndarray[int]]:
    """
    Returns trend indices and trend dates. Works for 1900-2003 trend or 2003 - 2022 depending on the boolean value.
    """

    shifted_dates = (
        signal_dates
        if not shift
        else (
            array(object=signal_dates, dtype=float)
            + load_signal_hyper_parameters.spline_time_years
            + load_signal_hyper_parameters.last_year_for_trend
        )
    )
    trend_indices = where(
        (shifted_dates < (load_signal_hyper_parameters.last_year_for_trend if recent_trend else load_signal_hyper_parameters.first_year_for_trend))
        * (
            shifted_dates
            >= (load_signal_hyper_parameters.first_year_for_trend if recent_trend else load_signal_hyper_parameters.load_history_start_date)
        )
    )[0]
    return trend_indices, shifted_dates[trend_indices]


def compute_signal_trend(
    signal_dates: ndarray,
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    input_signal: ndarray[complex],
    recent_trend: bool = True,
) -> ndarray[float]:
    """
    Computes trend from frequencial data.
    """

    # Initializes.
    trend_indices, trend_dates = get_trend_dates(
        signal_dates=signal_dates, load_signal_hyper_parameters=load_signal_hyper_parameters, recent_trend=recent_trend
    )

    # Computes trend for all harmonics.
    signal: ndarray = ifft(input_signal)[trend_indices]

    return signal_trend(
        trend_dates=trend_dates,
        signal=signal.real,
    )[0]


def build_elastic_load_signal_history(
    initial_signal_dates: ndarray[float],
    initial_load_signal: ndarray[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
    elastic_past_trend: Optional[float] = None,  # (mm/yr).
) -> tuple[ndarray[float], ndarray[float], ndarray[complex], float]:
    """
    Builds a unitless load signal history function suitable for Fourier analysis. It has zero mean value, antisymetry and no
    Gibbs effect.
    """

    # Sets wanted past trend without modifying recent trend.
    past_trend_indices, past_trend_dates = get_trend_dates(
        signal_dates=initial_signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        recent_trend=False,
        shift=False,
    )
    recent_trend_indices, recent_trend_dates = get_trend_dates(
        signal_dates=initial_signal_dates,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
        recent_trend=True,
        shift=False,
    )
    past_trend = signal_trend(
        trend_dates=past_trend_dates,
        signal=initial_load_signal[past_trend_indices],
    )[0]
    past_trend_ratio = (elastic_past_trend if not (elastic_past_trend is None) else past_trend) / past_trend
    load_signal = concatenate(
        (
            initial_load_signal[past_trend_indices] * past_trend_ratio,  # Rescaled past time serie.
            initial_load_signal[recent_trend_indices]  # Unnmodified recent time serie.
            + initial_load_signal[past_trend_indices][-1] * (past_trend_ratio - 1),  # Additive constant for continuity.
        )
    )

    # Linearly extends the signal for last years.
    elastic_load_signal_recent_trend, elastic_load_signal_additive_constant = signal_trend(
        trend_dates=recent_trend_dates,
        signal=load_signal[recent_trend_indices],
    )
    extend_part_dates = arange(
        initial_signal_dates[-1] + 1,
        load_signal_hyper_parameters.last_year_for_trend + 1,
    )
    extend_part_load_signal = elastic_load_signal_recent_trend * extend_part_dates + elastic_load_signal_additive_constant

    # Eventually includes a LIA effect.
    if load_signal_hyper_parameters.LIA:
        LIA_value = load_signal[-1] * load_signal_hyper_parameters.LIA_amplitude_effect
        # zero plateau after LIA.
        load_signal = (
            concatenate(
                (
                    linspace(
                        start=LIA_value,
                        stop=0.0,
                        num=load_signal_hyper_parameters.LIA_time_years,
                    ),
                    zeros(shape=(initial_signal_dates[0] - load_signal_hyper_parameters.LIA_end_date)),
                    load_signal,
                )
            )
            - LIA_value
        )
        extend_part_load_signal -= LIA_value

    # Creates cubic spline for antisymetry.
    mean_slope = extend_part_load_signal[-1] / load_signal_hyper_parameters.spline_time_years
    spline = lambda T: mean_slope / (2.0 * load_signal_hyper_parameters.spline_time_years**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T

    # Builds signal history / Creates an initial plateau at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(load_signal_hyper_parameters.initial_plateau_time_years)),
            load_signal,
            extend_part_load_signal,
            spline(T=arange(start=-load_signal_hyper_parameters.spline_time_years, stop=0)),
        )
    )

    # Applies antisymetry.
    extended_time_serie = concatenate((extended_time_serie_past, [0], -flip(m=extended_time_serie_past)))

    # Deduces dates axis.
    n_extended_signal = len(extended_time_serie)
    extended_dates = arange(stop=n_extended_signal) - n_extended_signal // 2

    # Interpolates at sufficient sampling for no Gibbs effect.
    n_log_min_no_Gibbs = round(ceil(log2(n_extended_signal)))
    half_signal_period = max(extended_dates)
    n_signal = int(2 ** (n_log_min_no_Gibbs + load_signal_hyper_parameters.anti_Gibbs_effect_factor))
    signal_dates = linspace(-half_signal_period, stop=half_signal_period, num=n_signal)

    # Gets frequencial component.
    time_step = 2.0 * half_signal_period / n_signal
    frequencies = fftfreq(n=n_signal, d=time_step)

    return (
        signal_dates,
        frequencies,
        fft(
            x=interpolate.splev(
                x=signal_dates,
                tck=interpolate.splrep(x=extended_dates, y=extended_time_serie, k=3),
            )
        )
        / elastic_load_signal_recent_trend,  # Unitless signal.
        past_trend,
    )


def build_elastic_load_signal_components(
    load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
) -> tuple[
    int,
    ndarray[float],  # Temporal component's dates.
    ndarray[float],  # Spatial component in harmonic domain.
    ndarray[float],  # Unnormalized temporal component.
    ndarray[float],  # Ocean/land mask.
    ndarray[float],  # Ocean/land buffered mask,
    ndarray[float],  # Latitudes.
    ndarray[float],  # Longitudes.
]:
    """
    Builds the elastic load signal components.
    The spatial component comes from GRACE solution recent trends.
    The temporal componet comes from Frederikse barystatic sea level from 1900.
    """

    # Builds the signal's temporal component.
    dates, temporal_component = extract_temporal_load_signal(
        name=load_signal_hyper_parameters.case,
        filename=load_signal_hyper_parameters.load_history,
    )

    # Gets harmonic component.
    if load_signal_hyper_parameters.load_spatial_behaviour_data == "GRACE":
        map, latitudes, longitudes = extract_GRACE_data(
            name=load_signal_hyper_parameters.load_spatial_behaviour_file,
        )
    else:  # Considered as ocean/land repartition only.
        map, latitudes, longitudes = get_ocean_mask(
            name=load_signal_hyper_parameters.load_spatial_behaviour_file,
            n_max=load_signal_hyper_parameters.n_max,
        )
        map = map_normalizing(map=map)

    map, n_max = map_sampling(map=map, n_max=load_signal_hyper_parameters.n_max)

    if n_max != load_signal_hyper_parameters.n_max:
        latitudes = linspace(90, -90, 2 * (n_max + 1) + 1)
        latitudes = linspace(0, 360, 4 * (n_max + 1) + 1)

    # Loads ocean mask.
    ocean_land_geopandas = get_continents(name=load_signal_hyper_parameters.continents).to_crs(epsg=EARTH_EQUAL_PROJECTION)
    lon_grid, lat_grid = meshgrid(longitudes, latitudes)
    gdf = GeoDataFrame(geometry=[Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())])
    gdf.set_crs(epsg=LAT_LON_PROJECTION, inplace=True)
    gdf = gdf.to_crs(epsg=EARTH_EQUAL_PROJECTION)
    ocean_land_geopandas_buffered_reprojected: GeoDataFrame = ocean_land_geopandas.buffer(load_signal_hyper_parameters.buffer_distance * 1e3)
    oceanic_mask = ~gdf.intersects(ocean_land_geopandas.unary_union)
    oceanic_mask_buffered = ~gdf.intersects(ocean_land_geopandas_buffered_reprojected.unary_union)
    ocean_land_mask = oceanic_mask.to_numpy().reshape(lon_grid.shape).astype(int)
    ocean_land_buffered_mask = oceanic_mask_buffered.to_numpy().reshape(lon_grid.shape).astype(int)

    # Loads the continents with opposite value, such that global mean is null.
    if load_signal_hyper_parameters.opposite_load_on_continents:
        map = map * ocean_land_mask - (1 - ocean_land_mask) * (
            mean_on_mask(
                signal_threshold=inf,
                grid=map,
                latitudes=latitudes,
                mask=ocean_land_mask,
                n_max=n_max,
            )
            * sum(surface_ponderation(mask=ocean_land_mask, latitudes=latitudes).flatten())
            / sum(surface_ponderation(mask=(1 - ocean_land_mask), latitudes=latitudes).flatten())
        )

    # Harmonic component.
    harmonic_component, _ = map_sampling(map=map, n_max=n_max, harmonic_domain=True)
    return (
        n_max,
        dates,
        harmonic_component,
        temporal_component,
        ocean_land_mask,
        ocean_land_buffered_mask,
        latitudes,
        longitudes,
    )
